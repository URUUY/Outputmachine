import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Important notes for training your own semantic segmentation model:
1. Before training, carefully check if your format meets requirements. This library requires VOC format dataset with:
   - Input images as .jpg (automatically resized during training)
   - Grayscale images will be automatically converted to RGB
   - If input images aren't .jpg, convert them before training

   Labels should be png images (automatically resized during training)
   Many online datasets have incorrect label formats - each pixel's value must represent its class.
   Common online datasets have two classes: background=0, target=255. This will run but predictions won't work!
   Must modify to: background=0, target=1
   For format fixes see: https://github.com/bubbliiiing/segmentation-format-fix

2. Loss value indicates convergence - what matters is the trend (validation loss decreasing)
   Absolute loss value doesn't matter (large/small depends on calculation method). 
   To make loss "look better", you can divide by 10000 in the loss function.
   Training loss values are saved in logs/loss_%Y_%m_%d_%H_%M_%S folder
   
3. Trained weights are saved in logs folder. Each epoch contains multiple steps (gradient descent per step)
   Weights aren't saved if only a few steps are trained. Understand Epoch vs Step difference.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use CUDA
    #           Set False if no GPU
    #---------------------------------#
    Cuda = True
    #----------------------------------------------#
    #   Seed    For fixing random seed
    #           Ensures reproducible results
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Whether to use single-machine multi-GPU distributed training
    #                  Terminal commands only supported on Ubuntu. CUDA_VISIBLE_DEVICES specifies GPUs.
    #                  Windows defaults to DP mode using all GPUs (no DDP support)
    #   DP mode:
    #       Set            distributed = False
    #       Run:          CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True  
    #       Run:          CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn (DDP multi-GPU only)
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Reduces ~50% VRAM, requires pytorch 1.7.1+
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     Must modify for your dataset
    #                   Number of classes + 1 (e.g. 2+1)
    #-----------------------------------------------------#
    num_classes = 104
    #-----------------------------------------------------#
    #   Backbone choices:
    #   vgg
    #   resnet50  
    #-----------------------------------------------------#
    backbone    = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use backbone's pretrained weights (loaded during model construction)
    #                   If model_path is set, backbone weights aren't loaded (pretrained meaningless)
    #                   If model_path not set and pretrained=True: only loads backbone
    #                   If model_path not set and pretrained=False, Freeze_Train=False: trains from scratch
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Weight file download instructions in README (cloud storage)
    #   Pretrained weights are general across datasets (features are general)
    #   Important part is backbone feature extraction weights
    #   Pretrained weights are needed in 99% cases - without them, backbone weights are too random (poor feature extraction/results)
    #   Dimension mismatch warnings when training your dataset are normal
    #
    #   If training is interrupted, set model_path to weights in logs folder to resume
    #   Adjust freeze/unfreeze phase parameters to maintain epoch continuity
    #
    #   When model_path='', no weights are loaded
    #
    #   Here we use the full model's weights (loaded in train.py), pretrain doesn't affect this
    #   To train from backbone's pretrained weights: model_path='', pretrain=True
    #   To train from scratch: model_path='', pretrain=False, Freeze_Train=False
    #
    #   Training from scratch generally performs poorly (random weights, poor feature extraction) - strongly discouraged!
    #   If you must train from scratch, study imagenet dataset first (train classification model to get backbone weights)
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = ""
    #-----------------------------------------------------#
    #   input_shape     Input image size (multiple of 32)
    #-----------------------------------------------------#
    input_shape = [672, 672]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two phases: freeze and unfreeze. Freeze phase helps users with limited hardware.
    #   Freeze training uses less VRAM (only fine-tunes network)
    #   For very weak GPUs, set Freeze_Epoch=UnFreeze_Epoch (freeze-only training)
    #
    #   Parameter suggestions (adjust as needed):
    #   (1) Training from full model's pretrained weights:
    #       Adam:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (unfreeze)
    #       SGD:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (unfreeze)
    #       UnFreeze_Epoch can be 100-300
    #   (2) Training from backbone's pretrained weights:
    #       Adam:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=100, Freeze_Train=True, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=100, Freeze_Train=False, optimizer_type='adam', Init_lr=1e-4, weight_decay=0 (unfreeze)
    #       SGD:
    #           Init_Epoch=0, Freeze_Epoch=50, UnFreeze_Epoch=120, Freeze_Train=True, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (freeze)
    #           Init_Epoch=0, UnFreeze_Epoch=120, Freeze_Train=False, optimizer_type='sgd', Init_lr=1e-2, weight_decay=1e-4 (unfreeze)
    #       Since backbone weights may not suit segmentation, more training is needed (UnFreeze_Epoch 120-300)
    #       Adam converges faster than SGD (can use smaller UnFreeze_Epoch but still recommend more epochs)
    #   (3) batch_size settings:
    #       As large as GPU can handle (reduce if OOM occurs)
    #       When backbone is resnet50, batch_size cannot be 1 (due to BatchNormalization)
    #       Normally Freeze_batch_size should be 1-2x Unfreeze_batch_size
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freeze phase training parameters
    #   Backbone is frozen (feature extraction unchanged)
    #   Uses less VRAM (only fine-tunes)
    #   Init_Epoch      Starting epoch (can be > Freeze_Epoch to skip freezing)
    #   Freeze_Epoch    Number of freeze training epochs
    #   Freeze_batch_size   Freeze phase batch_size
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 25
    Freeze_batch_size   = 4
    #------------------------------------------------------------------#
    #   Unfreeze phase training parameters  
    #   Backbone is unfrozen (feature extraction changes)
    #   Uses more VRAM (all parameters change)
    #   UnFreeze_Epoch      Total training epochs
    #   Unfreeze_batch_size Unfrozen batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 50
    Unfreeze_batch_size = 4
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
    #                   Adam: Init_lr=1e-4
    #                   SGD: Init_lr=1e-2
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
    #                   Set to 0 for Adam (issues with weight_decay)
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
    #------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate during training (on val set)
    #   eval_period     Evaluate every N epochs (don't evaluate too frequently)
    #                   Evaluation is time-consuming (slows training)
    #   Note: mAP here differs from get_map.py because:
    #   (1) This evaluates on validation set
    #   (2) Conservative evaluation params for speed
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    
    #------------------------------#
    #   Dataset path
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    #   Recommended settings:
    #   Few classes (several): True
    #   Many classes (dozens) + large batch_size (>10): True  
    #   Many classes (dozens) + small batch_size (<10): False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   Whether to use focal loss for class imbalance
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   Whether to weight classes differently (default: balanced)
    #   If set, must be numpy array with length=num_classes
    #   e.g.:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     Whether to use multi-threaded data loading (1=off)
    #                   Speeds up loading but uses more memory
    #                   Sometimes slower in keras when enabled
    #                   Only enable when I/O is bottleneck (GPU >> loading)
    #------------------------------------------------------------------#
    num_workers     = 4

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
        #   Load weights based on key matching
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
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
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
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    #   Backbone feature extraction is general - freezing speeds up training
    #   Also prevents weight damage in early training
    #   Init_Epoch is starting epoch
    #   Interval_Epoch is freeze training epochs
    #   Epoch is total training epochs
    #   Reduce Batch_size if OOM occurs
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
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small for training, please expand dataset.")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        #----------------------#
        #   Record eval mAP curve
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
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
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small for training, please expand dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()