import sys
import math

from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        lm_logits = batch["input_ids"].to(device, non_blocking=True)
        
        src_samples = src_samples.reshape(-1, *src_samples.shape[2:])
        tgt_samples = tgt_samples.reshape(-1, *tgt_samples.shape[2:])

        with torch.amp.autocast("cuda"):
            loss, _, (loss_post, loss_prior, loss_kl, value_kl, loss_mae, context_kl) = model(
                src_samples, tgt_samples, lm_logits, 
                data_iter_step / len(data_loader) + epoch
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_post=loss_post.item())
        metric_logger.update(loss_prior=loss_prior.item())
        metric_logger.update(loss_kl=loss_kl.item())
        metric_logger.update(kl=value_kl.item())
        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(context_kl=context_kl_loss.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            log_writer.add_scalar("loss_post", loss_post.item(), epoch_1000x)
            log_writer.add_scalar("loss_prior", loss_prior.item(), epoch_1000x)
            log_writer.add_scalar("loss_kl", loss_kl.item(), epoch_1000x)
            log_writer.add_scalar("loss_mae", loss_mae.item(), epoch_1000x)
            log_writer.add_scalar("kl", value_kl.item(), epoch_1000x)
            log_writer.add_scalar("context_kl", context_kl_loss.item(), epoch_1000x)
            log_writer.add_scalar("context_kl", context_kl.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_llm(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        lm_logits = batch["embeddings"].to(device, non_blocking=True)
        
        src_samples = src_samples.reshape(-1, *src_samples.shape[2:])
        tgt_samples = tgt_samples.reshape(-1, *tgt_samples.shape[2:])

        with torch.amp.autocast("cuda"):
            loss, _, (loss_post, loss_prior, loss_kl, value_kl, loss_mae, context_kl_loss) = model(
                src_samples, tgt_samples, lm_logits,
                data_iter_step / len(data_loader) + epoch
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_post=loss_post.item())
        metric_logger.update(loss_prior=loss_prior.item())
        metric_logger.update(loss_kl=loss_kl.item())
        metric_logger.update(kl=value_kl.item())
        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(context_kl=context_kl.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            log_writer.add_scalar("loss_post", loss_post.item(), epoch_1000x)
            log_writer.add_scalar("loss_prior", loss_prior.item(), epoch_1000x)
            log_writer.add_scalar("loss_kl", loss_kl.item(), epoch_1000x)
            log_writer.add_scalar("loss_mae", loss_mae.item(), epoch_1000x)
            log_writer.add_scalar("kl", value_kl.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
