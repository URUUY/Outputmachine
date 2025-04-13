import datetime
import os
from functools import partial
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
torch.backends.cudnn.benchmark = True  # Let cuDNN auto-optimize
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matrix calculations
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Important notes for training your own semantic segmentation model:
1. Before training, carefully check if your data format meets requirements. This library requires VOC format:
   - Input images should be .jpg (automatically resized during training)
   - Grayscale images will be automatically converted to RGB
   - If images have non-jpg extensions, convert them to jpg first
   
   - Label images should be .png (automatically resized during training)
   - Each pixel value in label images represents its class
   - Common online datasets often use 0 for background and 255 for target - this won't work for prediction!
     You need to change to 0 for background and 1 for target
   - For format issues, refer to: https://github.com/bubbliiiing/segmentation-format-fix

2. Loss values indicate convergence - focus on the trend (validation loss decreasing)
   - Absolute loss values don't matter much (depends on calculation method)
   - Training losses are saved in logs/loss_%Y_%m_%d_%H_%M_%S

3. Trained weights are saved in logs folder:
   - Each Epoch contains multiple Steps
   - Weights are saved per Epoch, not per Step
'''

if __name__ == "__main__":
    #---------------------------------#
    #   Cuda     Whether to use CUDA
    #            Set to False if no GPU
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed     Fixed random seed for reproducibility
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Whether to use distributed training (multi-GPU)
    #                   Only supported on Ubuntu via command line
    #                   Windows defaults to DP mode (uses all GPUs)
    #   DP mode:
    #       Set            distributed = False
    #       Run           CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Run           CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn (only for DDP multi-GPU)
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Reduces VRAM usage, requires pytorch 1.7.1+
    #---------------------------------------------------------------------#
    fp16            = True
    #-----------------------------------------------------#
    #   num_classes     Must modify for your dataset
    #                   Number of classes + 1 (e.g. 2+1)
    #-----------------------------------------------------#
    num_classes     = 104
    #---------------------------------#
    #   Backbone network:
    #   mobilenet
    #   xception
    #---------------------------------#
    backbone        = "mobilenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use pretrained backbone weights
    #                   If model_path is set, pretrained is ignored
    #                   If model_path='' and pretrained=True: loads only backbone
    #                   If model_path='' and pretrained=False: trains from scratch
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   model_path      Pretrained weights path (see README)
    #                   When model_path='', doesn't load any weights
    #                   For training from backbone pretrained weights: set model_path='', pretrained=True
    #                   For training from scratch: set model_path='', pretrained=False, Freeze_Train=False
    #                   Note: Training from scratch is not recommended due to poor performance
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #---------------------------------------------------------#
    #   downsample_factor   Downsampling factor (8 or 16)
    #                       8 gives better results but needs more VRAM
    #---------------------------------------------------------#
    downsample_factor   = 8
    #------------------------------#
    #   Input image size
    #------------------------------#
    input_shape         = [680, 680]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two phases: Freeze and Unfreeze
    #   Freeze phase uses less VRAM for weaker hardware
    #   Set Freeze_Epoch=UnFreeze_Epoch for freeze-only training
    #
    #   Training parameter suggestions:
    #   (1) From full model pretrained weights:
    #       Adam: Init_lr=5e-4, weight_decay=0
    #       SGD: Init_lr=7e-3, weight_decay=1e-4
    #       UnFreeze_Epoch between 100-300
    #   (2) From backbone pretrained weights:
    #       Adam: Init_lr=5e-4, weight_decay=0
    #       SGD: Init_lr=7e-3, weight_decay=1e-4
    #       UnFreeze_Epoch between 120-300
    #   (3) batch_size: As large as your GPU can handle
    #       Minimum batch_size is 2 (due to BatchNorm)
    #       Freeze_batch_size should be 1-2x Unfreeze_batch_size
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freeze phase training parameters
    #   Freezes backbone (feature extraction doesn't change)
    #   Uses less VRAM, fine-tunes network
    #   Init_Epoch      Starting epoch (can be > Freeze_Epoch for resuming)
    #   Freeze_Epoch    Number of freeze training epochs
    #   Freeze_batch_size   Batch size during freeze phase
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 20
    Freeze_batch_size   = 4
    #------------------------------------------------------------------#
    #   Unfreeze phase training parameters
    #   Unfreezes backbone (feature extraction changes)
    #   Uses more VRAM, trains all parameters
    #   UnFreeze_Epoch      Total training epochs
    #   Unfreeze_batch_size Batch size after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 40
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to do freeze training first
    #                   Default: freeze then unfreeze
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   Other training params: learning rate, optimizer, LR scheduling
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Max learning rate
    #                   Adam: Init_lr=5e-4
    #                   SGD: Init_lr=7e-3
    #   Min_lr          Min learning rate (default: 0.01*Init_lr)
    #------------------------------------------------------------------#
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Optimizer type: adam or sgd
    #   momentum        Optimizer momentum
    #   weight_decay    Weight decay (L2 regularization)
    #                   Set to 0 for Adam (causes issues)
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay: 'step' or 'cos'
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
    #   eval_flag       Whether to evaluate during training
    #   eval_period     Evaluate every N epochs
    #                   Evaluation slows down training
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5

    #------------------------------------------------------------------#
    #   VOCdevkit_path  Dataset path
    #------------------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    #   dice_loss       Recommended:
    #                   True for few classes (<10) or large batch_size (>10)
    #                   False for many classes with small batch_size
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   focal_loss      Whether to use focal loss for class imbalance
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   cls_weights     Class weights for imbalance (numpy array)
    #                   Length should match num_classes
    #                   Default is balanced (all ones)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     Number of workers for data loading
    #                   1 = no multiprocessing
    #                   Increase if I/O is bottleneck
    #------------------------------------------------------------------#
    num_workers         = 2

    seed_everything(seed)
    #------------------------------------------------------#
    #   GPU settings
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

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   Load pretrained weights
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
        #   Print loading results
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: Not loading head weights is normal, but backbone weights must load correctly.\033[0m")

    #----------------------#
    #   Init loss history
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   FP16 initialization (requires torch 1.7.1+)
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Sync BN for multi-GPU
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
        #---------------------------------------------------------#
        #   Total steps = total data / batch_size * epochs
        #   Minimum recommended steps:
        #   SGD: 15,000 steps
        #   Adam: 5,000 steps
        #----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset too small for training, please expand dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] With %s optimizer, recommended total steps >%d\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] Current settings: %d data, %d batch_size, %d epochs = %d steps\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Recommended epochs: %d\033[0m"%(wanted_epoch))
        
    #------------------------------------------------------#
    #   Freeze backbone for faster training
    #   Also prevents early weight damage
    #   Reduce batch_size if OOM occurs
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze backbone if needed
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   Set batch_size based on freeze status
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Auto-adjust learning rate based on batch_size
        #-------------------------------------------------------------------#
        nbs             = 16  # nominal batch size
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select optimizer
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   Get learning rate scheduler
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Calculate steps per epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small for training, please expand dataset.")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

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
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   Eval callback
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   Unfreeze if needed
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Recalculate learning rate based on new batch_size
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Update learning rate scheduler
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small for training, please expand dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler, 
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