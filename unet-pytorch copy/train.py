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

"""
Key considerations for training your semantic segmentation model:
1. Dataset format must be VOC format:
   - Input images: .jpg format (automatically converted to RGB if grayscale)
   - Label images: .png format where pixel values represent class indices
     (0=background, 1=class1, etc. Not 255 for objects!)
   
2. Training convergence:
   - Monitor validation loss trend (decreasing = good)
   - Absolute loss values depend on calculation method
   - Loss values saved in logs/loss_YYYY_MM_DD_HH_MM_SS

3. Model saving:
   - Weights saved per epoch (not per step)
   - Complete epochs required for saving
"""

if __name__ == "__main__":
    #-------------------------#
    #   Hardware Configuration
    #-------------------------#
    Cuda = True  # Set False if no GPU available
    seed = 11    # Random seed for reproducibility
    
    # Distributed Training Options
    distributed = False  # DDP mode if True, DP mode if False
    sync_bn = False      # Sync BatchNorm for DDP
    fp16 = False         # Mixed precision training
    
    #-------------------------#
    #   Model Configuration
    #-------------------------#
    num_classes = 104    # Number of classes (including background)
    backbone = "resnet50" # Backbone network (vgg or resnet50)
    pretrained = False   # Use pretrained backbone weights
    model_path = ""      # Path to load full model weights
    
    # Input image size (must be multiple of 32)
    input_shape = [672, 672]  
    
    #-------------------------#
    #   Training Schedule
    #-------------------------#
    # Freeze Phase (feature extractor frozen)
    Init_Epoch = 0         # Starting epoch
    Freeze_Epoch = 25      # Freeze training epochs
    Freeze_batch_size = 4  # Batch size during freeze
    
    # Unfreeze Phase (full network training)
    UnFreeze_Epoch = 50    # Total training epochs
    Unfreeze_batch_size = 4
    Freeze_Train = True    # Whether to use freeze phase
    
    #-------------------------#
    #   Optimization Parameters
    #-------------------------#
    Init_lr = 1e-4         # Initial learning rate
    Min_lr = Init_lr * 0.01 # Minimum learning rate
    optimizer_type = "adam" # "adam" or "sgd"
    momentum = 0.9         # For SGD
    weight_decay = 0       # L2 regularization (0 for Adam)
    lr_decay_type = 'cos'  # Learning rate decay ('cos' or 'step')
    
    #-------------------------#
    #   Logging & Saving
    #-------------------------#
    save_period = 5        # Save weights every N epochs
    save_dir = 'logs'      # Directory for weights and logs
    
    #-------------------------#
    #   Validation Settings
    #-------------------------#
    eval_flag = True       # Enable validation
    eval_period = 5       # Validate every N epochs
    
    #-------------------------#
    #   Dataset Paths
    #-------------------------#
    VOCdevkit_path = 'VOCdevkit'
    
    #-------------------------#
    #   Loss Configuration
    #-------------------------#
    dice_loss = False      # Dice loss for class imbalance
    focal_loss = False     # Focal loss for hard examples
    cls_weights = np.ones([num_classes], np.float32) # Class weights
    
    #-------------------------#
    #   Data Loading
    #-------------------------#
    num_workers = 4        # Data loading threads

    # Initialize random seeds
    seed_everything(seed)
    
    #-------------------------#
    #   Device Setup
    #-------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("GPU Device Count: ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    #-------------------------#
    #   Model Initialization
    #-------------------------#
    # Download pretrained weights if needed
    if pretrained and local_rank == 0:
        download_weights(backbone)
    if distributed:
        dist.barrier()

    # Create model
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    
    # Initialize weights if not using pretrained
    if not pretrained:
        weights_init(model)
    
    # Load model weights if specified
    if model_path:
        if local_rank == 0:
            print(f'Loading weights from {model_path}')
        
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        
        # Filter out incompatible keys
        load_keys = [k for k in pretrained_dict.keys() 
                    if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape]
        
        model_dict.update({k: pretrained_dict[k] for k in load_keys})
        model.load_state_dict(model_dict)
        
        if local_rank == 0:
            print(f"\nLoaded {len(load_keys)}/{len(pretrained_dict)} parameters")
            print("Note: Head mismatch is normal, backbone mismatch indicates error")

    #-------------------------#
    #   Training Setup
    #-------------------------#
    # Initialize loss history
    if local_rank == 0:
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, f"loss_{time_str}")
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    # Convert to training mode
    model_train = model.train()
    
    # Sync BatchNorm if needed
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("SyncBatchNorm requires multiple GPUs with DDP")
    
    # Parallelize model
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #-------------------------#
    #   Data Loading
    #-------------------------#
    # Load dataset splits
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt")) as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt")) as f:
        val_lines = f.readlines()
    
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    # Display config
    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path,
            input_shape=input_shape, Init_Epoch=Init_Epoch, 
            Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type
        )
    
    #-------------------------#
    #   Training Loop
    #-------------------------#
    # Initial setup
    UnFreeze_flag = False
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    
    # Adaptive learning rate based on batch size
    nbs = 16  # nominal batch size
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size/nbs*Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size/nbs*Min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)
    
    # Initialize optimizer
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]
    
    # Learning rate scheduler
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    
    # Prepare datasets
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
    # Distributed sampler if needed
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    # Data loaders
    gen = DataLoader(
        train_dataset, shuffle=shuffle, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=unet_dataset_collate, sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
    )
    
    gen_val = DataLoader(
        val_dataset, shuffle=shuffle, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=unet_dataset_collate, sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
    )
    
    # Initialize evaluation callback
    eval_callback = EvalCallback(
        model, input_shape, num_classes, val_lines, VOCdevkit_path,
        log_dir, Cuda, eval_flag=eval_flag, period=eval_period
    ) if local_rank == 0 else None
    
    # Calculate steps per epoch
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset too small for current batch size")
    
    # Freeze backbone if needed
    if Freeze_Train:
        model.freeze_backbone()
    
    #-------------------------#
    #   Main Training Loop
    #-------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # Unfreeze backbone if reaching unfreeze epoch
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            
            # Recalculate learning rate
            Init_lr_fit = min(max(batch_size/nbs*Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size/nbs*Min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            
            # Unfreeze and recreate data loaders
            model.unfreeze_backbone()
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            
            if distributed:
                batch_size = batch_size // ngpus_per_node
            
            gen = DataLoader(
                train_dataset, shuffle=shuffle, batch_size=batch_size,
                num_workers=num_workers, pin_memory=True, drop_last=True,
                collate_fn=unet_dataset_collate, sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
            )
            
            gen_val = DataLoader(
                val_dataset, shuffle=shuffle, batch_size=batch_size,
                num_workers=num_workers, pin_memory=True, drop_last=True,
                collate_fn=unet_dataset_collate, sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
            )
            
            UnFreeze_flag = True
        
        # Set epoch for distributed sampler
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Update learning rate
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        # Train for one epoch
        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
            dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
            save_period, save_dir, local_rank
        )
        
        if distributed:
            dist.barrier()
    
    # Clean up
    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()