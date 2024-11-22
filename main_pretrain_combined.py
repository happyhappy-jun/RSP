import os
import time
import json
import datetime
import argparse

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
from util.kinetics import PairedKinetics
from util.kinetics_caption import PairedKineticsWithCaption

import models_rsp

from engine_pretrain_repsamp import train_one_epoch as train_one_epoch_rsp
from engine_pretrain_repsamp_llm import train_one_epoch as train_one_epoch_llm

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument("--mode", default="rsp", type=str, choices=["rsp", "llm"],
                        help="Mode of operation: rsp or llm")
    # Add other arguments here as needed
    return parser

@hydra.main(config_path="config", config_name="config_combined")
def main(cfg: DictConfig):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if cfg.mode == "rsp":
        dataset_train = instantiate(cfg.dataset_rsp)
        train_one_epoch = train_one_epoch_rsp
    else:
        dataset_train = instantiate(cfg.dataset_llm)
        train_one_epoch = train_one_epoch_llm

    print(dataset_train)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    # define the model
    model = models_rsp.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        kl_scale=args.kl_scale,
        kl_balance=args.kl_balance,
        kl_freebit=args.kl_freebit,
        stoch=args.stoch,
        discrete=args.discrete,
        mask_ratio=args.mask_ratio,
        noise_scale=args.noise_scale
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    args.epochs = args.epochs // args.repeated_sampling
    args.warmup_epochs = args.warmup_epochs // args.repeated_sampling

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % 10 == 0 or epoch in [args.epochs - 2, args.epochs - 1, args.epochs]):
            misc.save_model(
                args=args,
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

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
