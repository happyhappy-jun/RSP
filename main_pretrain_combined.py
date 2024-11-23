import os
import time
import json
import datetime
import argparse
import wandb

from pathlib import Path

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_rsp

@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    misc.init_distributed_mode(cfg)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(cfg).replace(", ", ",\n"))

    device = torch.device(cfg.device)

    # fix the seed for reproducibility
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = instantiate(cfg.dataset)

    print(dataset_train)

    if cfg.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0:
        if cfg.log_dir is not None:
            os.makedirs(cfg.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=cfg.log_dir)
        else:
            log_writer = None
            
        # Initialize wandb
        wandb.init(
            project="rsp-training",
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    # define the model
    model = models_rsp.__dict__[cfg.model](
        context_emb_dim=cfg.context_emb_dim,
        norm_pix_loss=cfg.norm_pix_loss,
        kl_scale=cfg.kl_scale,
        kl_balance=cfg.kl_balance,
        kl_freebit=cfg.kl_freebit,
        stoch=cfg.stoch,
        discrete=cfg.discrete,
        mask_ratio=cfg.mask_ratio,
        noise_scale=cfg.noise_scale
    )

    model.to(device)
    model_without_ddp = model

    eff_batch_size = cfg.batch_size * cfg.accum_iter * misc.get_world_size()

    if cfg.lr is None:  # only base_lr is specified
        cfg.lr = cfg.blr * eff_batch_size / 256

    print("base lr: %.2e" % (cfg.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % cfg.lr)

    print("accumulate grad iterations: %d" % cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=cfg,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    cfg.epochs = cfg.epochs // cfg.repeated_sampling
    cfg.warmup_epochs = cfg.warmup_epochs // cfg.repeated_sampling

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    train_loop = hydra.utils.instantiate(cfg.train_loop)
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_loop(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=cfg,
        )
        
        # Log training stats for this epoch
        if misc.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            wandb.log(log_stats)
        if cfg.output_dir and (epoch % 10 == 0 or epoch in [cfg.epochs - 2, cfg.epochs - 1, cfg.epochs]):
            misc.save_model(
                args=cfg,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # Log metrics to wandb
            wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    
    if misc.is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()
