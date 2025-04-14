import os
import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                 epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, 
                 dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, 
                 save_period, save_dir, local_rank=0):
    """
    Train and validate model for one epoch
    
    Args:
        model_train: Model in training mode
        model: Original model (for saving weights)
        loss_history: Loss tracking object
        eval_callback: Evaluation callback
        optimizer: Training optimizer
        epoch: Current epoch number
        epoch_step: Number of training steps per epoch
        epoch_step_val: Number of validation steps per epoch
        gen: Training data generator
        gen_val: Validation data generator
        Epoch: Total number of epochs
        cuda: Whether to use CUDA
        dice_loss: Whether to use Dice loss
        focal_loss: Whether to use Focal loss
        cls_weights: Class weights for loss calculation
        num_classes: Number of classes
        fp16: Whether to use mixed precision training
        scaler: Gradient scaler for fp16 training
        save_period: Save model every N epochs
        save_dir: Directory to save models
        local_rank: Local rank for distributed training
    """
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Training')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', 
                   postfix=dict, mininterval=0.3)
    
    # Training phase
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
            
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        
        if not fp16:
            # Forward pass
            outputs = model_train(imgs)
            
            # Loss calculation
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # Calculate F-score
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # Forward pass
                outputs = model_train(imgs)
                
                # Loss calculation
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # Calculate F-score
                    _f_score = f_score(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1), 
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Training Completed')
        
        # Validation phase
        print('Starting Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',
                   postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
            
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # Forward pass
            outputs = model_train(imgs)
            
            # Loss calculation
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
                
            # Calculate F-score
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'f_score': val_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Validation Completed')
        
        # Record losses and metrics
        loss_history.append_loss(epoch + 1, total_loss/epoch_step, val_loss/epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print(f'Epoch: {epoch+1}/{Epoch}')
        print(f'Total Loss: {total_loss/epoch_step:.3f} || Val Loss: {val_loss/epoch_step_val:.3f}')
        
        # Save model weights
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(
                save_dir, f'ep{epoch+1:03d}-loss{total_loss/epoch_step:.3f}-val_loss{val_loss/epoch_step_val:.3f}.pth'
            ))

        # Save best model
        if len(loss_history.val_loss) <= 1 or (val_loss/epoch_step_val) <= min(loss_history.val_loss):
            print('Saving best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        # Save last epoch model
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, 
                        cls_weights, num_classes, fp16, scaler, save_period, 
                        save_dir, local_rank=0):
    """
    Train model for one epoch without validation
    
    Args: (similar to fit_one_epoch, excluding validation-related parameters)
    """
    total_loss = 0
    total_f_score = 0
    
    if local_rank == 0:
        print('Start Training')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',
                   postfix=dict, mininterval=0.3)
    
    # Training phase
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
            
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        
        if not fp16:
            # Forward pass
            outputs = model_train(imgs)
            
            # Loss calculation
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # Calculate F-score
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # Forward pass
                outputs = model_train(imgs)
                
                # Loss calculation
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # Calculate F-score
                    _f_score = f_score(outputs, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1), 
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        
        # Record losses
        loss_history.append_loss(epoch + 1, total_loss/epoch_step)
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total Loss: {total_loss/epoch_step:.3f}')
        
        # Save model weights
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(
                save_dir, f'ep{epoch+1:03d}-loss{total_loss/epoch_step:.3f}.pth'
            ))

        # Save best model
        if len(loss_history.losses) <= 1 or (total_loss/epoch_step) <= min(loss_history.losses):
            print('Saving best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        # Save last epoch model
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))