import datetime
import os
from functools import partial
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for matrix computations
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
   - Input images should be in .jpg format (automatically resized during training)
   - Grayscale images will be automatically converted to RGB
   - Labels should be in .png format (automatically resized)
   - Each pixel value in the label represents its class (0 for background, 1 for object, etc.)

2. Loss values indicate convergence - focus on the trend of validation loss decreasing
   - Absolute loss values aren't meaningful (can be scaled by dividing by 10000 if desired)
   - Training losses are saved in the logs folder

3. Trained weights are saved in the logs folder
   - Each Epoch contains multiple Steps (gradient descent iterations)
   - Weights are only saved after completing an Epoch, not individual Steps
'''

if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use CUDA
    #           Set to False if no GPU
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Fixed random seed for reproducibility
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Whether to use multi-GPU distributed training
    #                   Only supported on Ubuntu for DDP mode
    #   DP mode (DataParallel):
    #       Set distributed = False
    #       Run: CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode (DistributedDataParallel):
    #       Set distributed = True  
    #       Run: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use synchronized batch normalization (for DDP)
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training (requires PyTorch 1.7.1+)
    #               Reduces GPU memory usage by ~50%
    #---------------------------------------------------------------------#
    fp16            = True
    #-----------------------------------------------------#
    #   num_classes     Must modify for your dataset
    #                   Number of classes + 1 (background)
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
    #                   If no model_path and pretrained=True, loads only backbone
    #                   If no model_path and pretrained=False, trains from scratch
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   model_path      Path to pretrained model weights
    #                   Set to '' to train from scratch (not recommended)
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #---------------------------------------------------------#
    #   downsample_factor   Downsampling rate (8 or 16)
    #                       8 gives better results but needs more memory
    #---------------------------------------------------------#
    downsample_factor   = 8
    #------------------------------#
    #   Input image size
    #------------------------------#
    input_shape         = [680, 680]
    
    # Training parameters
    Init_Epoch          = 0
    Freeze_Epoch        = 20
    Freeze_batch_size   = 4
    UnFreeze_Epoch      = 40
    Unfreeze_batch_size = 4
    Freeze_Train        = True  # Whether to freeze backbone initially

    # Optimization parameters
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    lr_decay_type       = 'cos'
    
    save_period         = 5  # Save weights every N epochs
    save_dir            = 'logs'
    
    eval_flag           = True  # Whether to evaluate during training
    eval_period         = 5     # Evaluate every N epochs
    
    VOCdevkit_path      = 'VOCdevkit'  # Dataset path
    
    # Loss function settings
    dice_loss       = False  # Whether to use Dice loss
    focal_loss      = False  # Whether to use Focal loss
    cls_weights     = np.ones([num_classes], np.float32)  # Class weights
    
    num_workers     = 2  # Data loading threads

    # Set random seed
    seed_everything(seed)
    
    # Initialize distributed training if enabled
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # Download pretrained weights if needed
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    # Initialize model
    model = DeepLab(num_classes=num_classes, backbone=backbone, 
                   downsample_factor=downsample_factor, pretrained=pretrained)
    
    if not pretrained:
        weights_init(model)  # Random initialization if not using pretrained
        
    if model_path != '':
        # Load pretrained weights
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        
        # Filter out incompatible keys
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
                
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: It's normal if head parts aren't loaded, but backbone should load correctly.\033[0m")

    # Initialize loss history logger
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # Mixed precision training
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    
    # Convert to SyncBatchNorm for distributed training
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    # Move model to GPU
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, 
                              device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    # Load dataset
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # Show training configuration
    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, 
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, 
            Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, 
            momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir, 
            num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        
        # Check if training steps are sufficient
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset too small for training, please expand dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using %s optimizer, recommended total steps >%d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] Training data: %d, Batch size: %d, Epochs: %d, Total steps: %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Recommended total epochs: %d.\033[0m"%(wanted_epoch))
        
    # Training preparation
    if True:
        UnFreeze_flag = False
        
        # Freeze backbone if enabled
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # Set batch size
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # Adjust learning rate based on batch size
        nbs = 16  # Nominal batch size
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
            
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # Initialize optimizer
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        # Get learning rate scheduler
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        # Calculate steps per epoch
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small for training, please expand dataset.")

        # Create datasets
        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        # Distributed sampler if enabled
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # Create data loaders
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, 
                         pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate, 
                         sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                            sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # Initialize evaluation callback
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, 
                                       log_dir, Cuda, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        
        # Start training loop
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # Unfreeze backbone if reaching Freeze_Epoch
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # Recalculate learning rate
                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                    
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                # Update learning rate scheduler
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                # Unfreeze backbone
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                # Recalculate steps per epoch
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small for training, please expand dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                # Recreate data loaders with new batch size
                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                               pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                               sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                   pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                   sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            # Set epoch for distributed sampler
            if distributed:
                train_sampler.set_epoch(epoch)

            # Update learning rate
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # Train for one epoch
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                         epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, 
                         dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, 
                         save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()  # Synchronize processes

        if local_rank == 0:
            loss_history.writer.close()  # Close log file