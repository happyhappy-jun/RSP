import os
import time
import json
import argparse
import datetime

from pathlib import Path

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from DeBERTa import deberta
from util.kinetics_global_caption import PairedKineticsWithGlobalCaption
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.kinetics import PairedKinetics

import models_rsp

from engine_pretrain_repsamp import train_one_epoch



def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)

    # Training
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_dir")

    # Model parameters
    parser.add_argument("--model", default="rsp_vit_small_patch16", type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--kl-scale", type=float, default=0.01)
    parser.add_argument("--kl-balance", type=float, default=0.2)
    parser.add_argument("--kl-freebit", type=float, default=0.1)
    parser.add_argument("--stoch", default=32, type=int)
    parser.add_argument("--discrete", default=32, type=int)

    # Optimizer
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1.5e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=40)

    # Dataset parameters
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--max_distance", default=48, type=int)
    parser.add_argument("--repeated_sampling", type=int, default=2)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--prefetch_factor", default=2, type=int)

    # Distributed training
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser



def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Fix seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Dataset & dataloader
    dataset_train = PairedKineticsWithGlobalCaption(
        video_root=args.video_root,
        json_path=args.json_path,
        embeddings_path=args.embeddings_path,
        max_distance=args.max_distance,
        repeated_sampling=args.repeated_sampling
    )

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # Setup logging
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

    # Model
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

    # Distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Optimizer
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                   optimizer=optimizer, loss_scaler=loss_scaler)

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

        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)