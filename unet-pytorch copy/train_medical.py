import os
import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch_no_val

'''
Important notes for training your own semantic segmentation model:
1. This dataset is a special training file I created based on medical datasets found online, just as an example to show how to train when the dataset is not in VOC format.
   
   Cannot calculate metrics like mIoU. Only for observing training effects on medical datasets.
   Cannot calculate metrics like mIoU.
   Cannot calculate metrics like mIoU.

   If you have your own medical dataset to train, there are two cases:
   a. Medical dataset without labels:
      Please follow the dataset annotation tutorial in the video, first use labelme to annotate images, convert to VOC format, then use train.py for training.
   b. Medical dataset with labels:
      Convert the label format so each pixel's value represents its class.
      Therefore, dataset labels need to be modified so background pixels=0 and target pixels=1.
      Reference: https://github.com/bubbliiiing/segmentation-format-fix

2. Loss value is used to judge convergence. What's important is the convergence trend, i.e., validation loss keeps decreasing.
   The absolute loss value doesn't mean much - large or small depends on the loss calculation method, not that closer to 0 is better.
   If you want better-looking loss values, you can divide by 10000 in the loss function.
   Training loss values are saved in logs/loss_%Y_%m_%d_%H_%M_%S folder.
   
3. Trained weights are saved in the logs folder. Each training epoch contains several steps, with gradient descent performed each step.
   Weights won't be saved if only a few steps are trained. Understand the difference between Epoch and Step.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda     Whether to use CUDA
    #            Set to False if no GPU
    #---------------------------------#
    Cuda = True
    #----------------------------------------------#
    #   Seed     For fixing random seed
    #            Ensures reproducible results across independent runs
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Whether to use single-machine multi-GPU distributed training
    #                  Terminal commands only supported on Ubuntu. CUDA_VISIBLE_DEVICES specifies GPUs on Ubuntu.
    #                  Windows defaults to DP mode using all GPUs, doesn't support DDP.
    #   DP mode:
    #       Set            distributed = False
    #       Run in terminal: CUDA_VISIBLE_DEVICES=0,1 python train_medical.py
    #   DDP mode:
    #       Set            distributed = True
    #       Run in terminal: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_medical.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn (available for multi-GPU DDP mode)
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Reduces ~50% VRAM, requires pytorch 1.7.1+
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     Must modify for your own dataset
    #                   Number of classes + 1, e.g. 2+1
    #-----------------------------------------------------#
    num_classes = 2
    #-----------------------------------------------------#
    #   Backbone network choices:
    #   vgg
    #   resnet50
    #-----------------------------------------------------#
    backbone    = "vgg"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use backbone's pretrained weights (loaded during model construction)
    #                   If model_path is set, backbone weights don't need loading, pretrained is meaningless.
    #                   If model_path not set and pretrained=True, only backbone weights are loaded for training.
    #                   If model_path not set and pretrained=False, Freeze_Train=False, training starts from scratch without freezing backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Weight file download instructions in README, available via cloud storage.
    #   Pretrained weights are general across different datasets because features are general.
    #   The important part is the backbone feature extraction network weights.
    #   Pretrained weights are necessary in 99% of cases - without them, backbone weights are too random, feature extraction ineffective, training results poor.
    #   Dimension mismatch warnings when training your own dataset are normal since predictions differ.
    #
    #   If training is interrupted, set model_path to weights in logs folder to resume training.
    #   Adjust freeze/unfreeze phase parameters to ensure epoch continuity.
    #   
    #   When model_path = '', no model weights are loaded.
    #
    #   Here we use the entire model's weights, loaded in train.py, pretrain doesn't affect this.
    #   To train from backbone's pretrained weights: model_path = '', pretrain = True (only loads backbone)
    #   To train from scratch: model_path = '', pretrain = False, Freeze_Train = False (no backbone freezing)
    #   
    #   Generally, training from scratch performs poorly due to random weights and ineffective feature extraction - strongly not recommended!
    #   If you must train from scratch, study imagenet dataset, first train classification model to get backbone weights (shared with this model).
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = ""
    #-----------------------------------------------------#
    #   input_shape     Input image size (multiple of 32)
    #-----------------------------------------------------#
    input_shape = [512, 512]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two phases: freeze phase and unfreeze phase. Freeze phase is for users with limited hardware.
    #   Freeze training requires less VRAM. For very weak GPUs, set Freeze_Epoch=UnFreeze_Epoch for freeze-only training.
    #   
    #   Some parameter suggestions (adjust flexibly based on needs):
    #   (1) Training from full model's pretrained weights:
    #       Adam:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (unfreeze)
    #       SGD:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (unfreeze)
    #       UnFreeze_Epoch can be adjusted between 100-300.
    #   (2) Training from backbone's pretrained weights:
    #       Adam:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (unfreeze)
    #       SGD:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=120, Freeze_Train=True, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=120, Freeze_Train=False, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (unfreeze)
    #       Since backbone weights may not suit segmentation, more training is needed to escape local optima.
    #       UnFreeze_Epoch can be 120-300.
    #       Adam converges faster than SGD, so UnFreeze_Epoch can theoretically be smaller (but still recommend more epochs).
    #   (3) batch_size settings:
    #       As large as GPU can handle. OOM/CUDA out of memory means reduce batch_size.
    #       When backbone is resnet50, batch_size cannot be 1 due to BatchNormalization.
    #       Normally Freeze_batch_size should be 1-2x Unfreeze_batch_size. Don't set too large a gap (affects LR auto-adjustment).
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freeze phase training parameters
    #   Backbone is frozen, feature extraction network unchanged
    #   Uses less VRAM, only fine-tunes network
    #   Init_Epoch          Current starting epoch (can be > Freeze_Epoch)
    #                       e.g. Init_Epoch=60, Freeze_Epoch=50, UnFreeze_Epoch=100
    #                       skips freeze phase, starts from epoch 60 with adjusted LR
    #                       (for resuming training)
    #   Freeze_Epoch        Number of freeze training epochs
    #                       (invalid when Freeze_Train=False)
    #   Freeze_batch_size   Freeze phase batch_size
    #                       (invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    #------------------------------------------------------------------#
    #   Unfreeze phase training parameters
    #   Backbone is unfrozen, feature extraction network changes
    #   Uses more VRAM, all parameters change
    #   UnFreeze_Epoch      Total training epochs
    #   Unfreeze_batch_size Unfrozen batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to do freeze training
    #                   Default: freeze backbone first, then unfreeze
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   Other training params: learning rate, optimizer, LR scheduling
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Max learning rate
    #                   Adam optimizer: Init_lr=1e-4
    #                   SGD optimizer: Init_lr=1e-2
    #   Min_lr          Min learning rate (default: 0.01 * Init_lr)
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Optimizer choices: adam, sgd
    #                   Adam: Init_lr=1e-4
    #                   SGD: Init_lr=1e-2
    #   momentum        Optimizer momentum parameter
    #   weight_decay    Weight decay (prevents overfitting)
    #                   Adam may have weight_decay issues - recommend 0 for Adam
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay: 'step', 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     Save weights every N epochs
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        Folder to save weights and logs
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    
    #------------------------------#
    #   Dataset path
    #------------------------------#
    VOCdevkit_path  = 'Medical_Datasets'
    #------------------------------------------------------------------#
    #   Recommended settings:
    #   Few classes (several): True
    #   Many classes (dozens) + large batch_size (>10): True
    #   Many classes (dozens) + small batch_size (<10): False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   Whether to use focal loss to address class imbalance
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   Whether to weight different classes differently (default: balanced)
    #   If set, must be numpy array with length=num_classes
    #   e.g.:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     Whether to use multi-threaded data loading (1=off)
    #                   Speeds up data loading but uses more memory
    #                   Sometimes slower in keras when enabled
    #                   Only enable when I/O is bottleneck (GPU much faster than image loading)
    #------------------------------------------------------------------#
    num_workers         = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   Set GPU devices
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   Download pretrained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   Weight file instructions in README (Baidu cloud)
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load weights based on matching keys between pretrained and model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Display unmatched keys
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: Head not loading is normal, backbone not loading is wrong.\033[0m")

    #----------------------#
    #   Loss logging
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape, val_loss_flag = False)
    else:
        loss_history = None

    #------------------------------------------------------------------#
    #   torch 1.2 doesn't support amp - recommend torch 1.7.1+ for fp16
    #   So torch1.2 shows "could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU sync BatchNorm
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   Read dataset txt files
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    num_train   = len(train_lines)
    
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
        )
    #------------------------------------------------------#
    #   Backbone feature extraction is general - freezing speeds up training
    #   Also prevents weight damage in early training.
    #   Init_Epoch is starting epoch
    #   Interval_Epoch is freeze training epochs
    #   Epoch is total training epochs
    #   OOM/VRAM issues - reduce Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze part of training
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
                
        #-------------------------------------------------------------------#
        #   If not freezing, directly set batch_size=Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Auto-adjust learning rate based on current batch_size
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select optimizer based on type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   Get learning rate decay function
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine steps per epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        
        if epoch_step == 0:
            raise ValueError("Dataset too small for training, please expand dataset.")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            shuffle         = True
            
        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If model has frozen parts
            #   Unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Auto-adjust learning rate based on current batch_size
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get learning rate decay function
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size

                if epoch_step == 0:
                    raise ValueError("Dataset too small for training, please expand dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()