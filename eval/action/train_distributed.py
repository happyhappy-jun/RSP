#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.optim import Optimizer

from optimizers.lars import LARS
import math
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        return rank, world_size, gpu
    else:
        logger.info('Using single GPU mode')
        return 0, 1, 0

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer:
    best_acc = 0.0

def main_worker(cfg):
    rank, world_size, gpu = init_distributed_mode()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # Create model and move it to GPU
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint into backbone if specified
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')
        msg = model.backbone.load_state_dict(checkpoint['model'], strict=False)
        if rank == 0:
            logger.info(f"Loaded checkpoint with message: {msg}")
    
    model = model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    
    # Create datasets and samplers
    train_dataset = hydra.utils.instantiate(cfg.train_dataset)
    val_dataset = hydra.utils.instantiate(cfg.val_dataset)
    
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=True
    )
    
    # Create optimizer with LARS
    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Different parameter settings based on layer type
            if any(k in name for k in ('bias', 'bn', 'norm')):
                param_groups.append({'params': [param], 'weight_decay': 0.0, 'lars_exclude': True})
            else:
                param_groups.append({'params': [param], 'weight_decay': cfg.weight_decay, 'lars_exclude': False})

    optimizer = LARS(
        param_groups,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        trust_coefficient=0.001,
        eps=1e-8,
        nesterov=False
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Cosine learning rate schedule
    def adjust_learning_rate(optimizer, epoch, cfg):
        """Decay the learning rate with cosine schedule"""
        if epoch < cfg.warmup_epochs:
            lr = cfg.lr * epoch / cfg.warmup_epochs 
        else:
            lr = cfg.lr * 0.5 * (1. + math.cos(math.pi * (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    # Initialize tensorboard writer for rank 0
    writer = SummaryWriter(cfg.output_dir) if rank == 0 else None
    
    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        
        # Initialize meters
        losses = AverageMeter()
        top1 = AverageMeter()
        
        train_pbar = tqdm(train_loader, disable=rank != 0)
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, cfg)
        if rank == 0:
            writer.add_scalar('Learning_rate', lr, epoch)
            
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            
            if rank == 0:
                train_pbar.set_description(
                    f'Train Epoch: {epoch} | Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}%'
                )
        
        # Validation phase
        model.eval()
        val_losses = AverageMeter()
        val_top1 = AverageMeter()
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, disable=rank != 0)
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Measure accuracy and record loss
                acc1 = accuracy(output, target)
                val_losses.update(loss.item(), data.size(0))
                val_top1.update(acc1[0].item(), data.size(0))
                
                if rank == 0:
                    val_pbar.set_description(
                        f'Val Epoch: {epoch} | Loss: {val_losses.avg:.4f} | Acc: {val_top1.avg:.2f}%'
                    )
        
        if rank == 0:
            # Log to tensorboard
            writer.add_scalar('Loss/train', losses.avg, epoch)
            writer.add_scalar('Accuracy/train', top1.avg, epoch)
            writer.add_scalar('Loss/val', val_losses.avg, epoch)
            writer.add_scalar('Accuracy/val', val_top1.avg, epoch)
            
            # Log to console
            logger.info(
                f'Epoch: {epoch} | '
                f'Train Loss: {losses.avg:.4f} | Train Acc: {top1.avg:.2f}% | '
                f'Val Loss: {val_losses.avg:.4f} | Val Acc: {val_top1.avg:.2f}%'
            )
            
            # Save checkpoint if best validation accuracy
            if val_top1.avg > Trainer.best_acc:
                Trainer.best_acc = val_top1.avg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': Trainer.best_acc,
                }, os.path.join(cfg.output_dir, 'best_model.pth'))


def accuracy(output, target):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

@hydra.main(config_path="config", config_name="linear_probing", version_base="1.1")
def main(cfg: DictConfig):
    try:
        main_worker(cfg)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
