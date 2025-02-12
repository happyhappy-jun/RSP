# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modified for action recognition linear probing

import os
import sys
import math
import datetime
import time
import numpy as np
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

from timm.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from eval.action.optimizers.lars import LARS
import modeling

class LinearProbing(torch.nn.Module):
    """Linear probing model that freezes backbone and trains only the head"""
    def __init__(self, backbone, num_classes=400, pool_type='mean'):
        super().__init__()
        self.backbone = backbone
        self.pool_type = pool_type
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Get embedding dimension from the model
        embed_dim = 384
        # Replace head with BN + Linear
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6),
            torch.nn.Linear(embed_dim, num_classes)
        )
        # Initialize head
        trunc_normal_(self.head[-1].weight, std=0.01)
        
    def forward(self, x):
        x = self.backbone.forward_encoder(x)[0]
        # Apply pooling based on specified type
        if self.pool_type == 'mean':
            x = x[:, 1:].mean(dim=1)  # mean pool over patches
        elif self.pool_type == 'cls':
            x = x[:, 0]  # use CLS token
        x = self.head(x)
        return x

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = getattr(args, 'min_lr', 0.0)  # Default to 0.0 if not specified
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = min_lr + (args.lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        
        # Log gradients before scaling
        if log_writer is not None and (data_iter_step + 1) % print_freq == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    log_writer.add_histogram(f'gradients/{name}', param.grad, data_iter_step)
                    log_writer.add_histogram(f'weights/{name}', param, data_iter_step)

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                   parameters=model.parameters(), create_graph=False,
                   update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Log learning rate
        if log_writer is not None and (data_iter_step + 1) % print_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                log_writer.add_scalar(f'learning_rate/group_{i}', param_group['lr'], data_iter_step)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

@hydra.main(config_path="config", config_name="linear_probing", version_base="1.2")
def main(cfg: DictConfig):
    misc.init_distributed_mode(cfg)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(OmegaConf.to_yaml(cfg)))

    device = torch.device(cfg.device)

    # fix the seed for reproducibility
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Dataset setup
    dataset_train = hydra.utils.instantiate(cfg.train_dataset)
    dataset_val = hydra.utils.instantiate(cfg.val_dataset)
    
    global_rank = 0  # Default for non-distributed
    if cfg.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if cfg.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )

    # Model setup
    model_params = {k: v for k, v in cfg.model.items() if k not in ['_target_', 'num_classes']}
    model = modeling.__dict__[cfg.model_name](**model_params)

    if cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % cfg.checkpoint_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # Create linear probing model
    model = LinearProbing(model, num_classes=cfg.num_classes, pool_type='mean')
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = cfg.batch_size * cfg.accum_iter * misc.get_world_size()
    
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        model_without_ddp = model.module

    # following LARS optimizer setup from MoCo v3
    # Initialize optimizer with larger learning rate and proper weight decay
    optimizer = LARS(model_without_ddp.head.parameters(), 
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    momentum=0.9,
                    trust_coefficient=0.001)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(cfg.epochs):
        if cfg.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=cfg
        )

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        if cfg.output_dir and misc.is_main_process():
            # Save checkpoint
            checkpoint_path = os.path.join(cfg.output_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': OmegaConf.to_container(cfg),
            }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()
