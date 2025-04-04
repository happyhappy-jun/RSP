import sys
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

import wandb

import util.misc as misc
import util.lr_sched as lr_sched

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def visualize_reconstruction(
    model,
    src_imgs: torch.Tensor,
    tgt_imgs: torch.Tensor,
    artifacts: dict,
    device: torch.device,
    step: int = 0,
    log_to_wandb: bool = True,
    revert_norm_pix_loss: bool = True
):
    """
    Visualize reconstruction by showing source images, target images, masked targets,
    and predictions (posterior & prior) from the model artifacts.

    Args:
        model: Your RSP model (providing patchify, unpatchify, etc.).
        src_imgs (torch.Tensor): [B, 3, H, W], ImageNet-normalized source images.
        tgt_imgs (torch.Tensor): [B, 3, H, W], ImageNet-normalized target images.
        artifacts (dict): 
            "masks": the binary mask used on target images (shape [B, N]),
            "tgt_pred": posterior reconstruction in patch form (shape [B, N, patchdim]),
            "tgt_pred_prior": prior reconstruction in patch form (shape [B, N, patchdim]).
        device (torch.device): device used for tensors (e.g., CPU or GPU).
        step (int): Step/epoch index for logging.
        log_to_wandb (bool): Whether to log visualizations to wandb.
        revert_norm_pix_loss (bool): If True, revert the patch normalization used during training.

    Returns:
        A torchvision grid image of shape [3, H_total, W_total] suitable for logging or saving.
    """

    # Extract from artifacts
    tgt_pred_post = artifacts["tgt_pred"]            # Posterior path prediction (patch-level)
    tgt_pred_prior = artifacts["tgt_pred_prior"]      # Prior path prediction (patch-level)
    tgt_masked_pred = artifacts["tgt_masked_pred"]    # Masked prediction (patch-level)
    mask = artifacts["masks"]                         # Mask used on target patches, shape [B, N]

    # 1) Possibly revert norm_pix_loss on both predictions
    if revert_norm_pix_loss:
        with torch.no_grad():
            # Need the ground-truth patchified target
            tgt_patch_gt = model.patchify(tgt_imgs)  # [B, N, patchdim]
            mean_patch   = tgt_patch_gt.mean(dim=-1, keepdim=True)
            var_patch    = tgt_patch_gt.var(dim=-1, keepdim=True)
            eps          = 1.0e-6

            # Revert for posterior
            tgt_pred_post = tgt_pred_post * (var_patch + eps).sqrt() + mean_patch
            tgt_masked_pred = tgt_masked_pred * (var_patch + eps).sqrt() + mean_patch
            tgt_pred_prior = tgt_pred_prior * (var_patch + eps).sqrt() + mean_patch

    # 2) Unpatchify the predictions
    img_pred_post  = model.unpatchify(tgt_pred_post)    # [B, 3, H, W]
    img_masked_pred = model.unpatchify(tgt_masked_pred) # [B, 3, H, W]
    img_pred_prior = model.unpatchify(tgt_pred_prior)   # [B, 3, H, W]

    # 3) Revert ImageNet normalization on inputs (src + tgt)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    src_denorm = src_imgs * imagenet_std + imagenet_mean
    tgt_denorm = tgt_imgs * imagenet_std + imagenet_mean

    # Revert ImageNet normalization on both predictions
    img_pred_post_denorm  = img_pred_post  * imagenet_std + imagenet_mean
    img_masked_pred_denorm = img_masked_pred * imagenet_std + imagenet_mean
    img_pred_prior_denorm = img_pred_prior * imagenet_std + imagenet_mean

    # (Optional) also visualize the masked target
    # We can produce a "masked" version of the ground truth patch (tgt_patch_gt),
    # zeroing out masked patches, then unpatchify.
    masked_patch = tgt_patch_gt.clone()
    masked_patch[mask.bool()] = 0.0  # zero out masked patches
    masked_img = model.unpatchify(masked_patch)
    masked_img_denorm = masked_img * imagenet_std + imagenet_mean

    # 4) Clamp [0,1] for safe visualization
    def clamp_zero_one(x):
        return torch.clamp(x, 0.0, 1.0)

    src_denorm              = clamp_zero_one(src_denorm)
    tgt_denorm              = clamp_zero_one(tgt_denorm)
    masked_img_denorm       = clamp_zero_one(masked_img_denorm)
    img_pred_post_denorm    = clamp_zero_one(img_pred_post_denorm)
    img_masked_pred_denorm  = clamp_zero_one(img_masked_pred_denorm)
    img_pred_prior_denorm   = clamp_zero_one(img_pred_prior_denorm)

    # 5) Build a grid: For each example, we want to line them up horizontally:
    #    src | tgt | masked tgt | pred (posterior) | pred (prior)
    # We'll just stack them all in a single batch dimension: [5B, 3, H, W]
    B = src_imgs.size(0)
    n_vis = min(B, 16)  # visualize up to 4
    combined_list = []
    # Gather the first n_vis items from each set
    combined_list.extend([src_denorm[i] for i in range(n_vis)])
    combined_list.extend([tgt_denorm[i] for i in range(n_vis)])
    combined_list.extend([masked_img_denorm[i] for i in range(n_vis)])
    combined_list.extend([img_pred_post_denorm[i] for i in range(n_vis)])
    combined_list.extend([img_masked_pred_denorm[i] for i in range(n_vis)])
    combined_list.extend([img_pred_prior_denorm[i] for i in range(n_vis)])

    # Convert list back to a single tensor [ (5*n_vis), 3, H, W ]
    to_show = torch.stack(combined_list, dim=0)

    # By default, vutils.make_grid groups by row. Let's have row = n_vis. 
    # Then we get 5 rows (for 5 types) if we pass nrow = n_vis.
    grid = vutils.make_grid(to_show, nrow=n_vis, padding=2)

    # 6) Optionally log to wandb
    if log_to_wandb:
        # Each row in the grid: 1..n_vis are src, n_vis+1..2*n_vis are tgt, etc.
        caption = "Rows: [src, tgt, masked_tgt, pred_post, pred_masked_prior, pred_prior], Columns: per example"
        wandb.log({"reconstruction": [wandb.Image(grid, caption=caption)]}, step=step)

    return grid

# ---------------------------
# Training Loop Definitions
# ---------------------------
def run_train_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    batch_step_fn=None,
):
    """
    Generic training loop for one epoch.

    Args:
        model: The model to train.
        data_loader: Iterable data loader.
        optimizer: Optimizer for model parameters.
        device: Device to run the training on.
        epoch: Current epoch number.
        loss_scaler: Scaler for mixed precision training.
        log_writer: Optional logger for writing logs.
        args: Additional arguments.
        batch_step_fn: Function to process each batch and return loss and detailed loss.

    Returns:
        Dictionary of averaged metrics.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        out = batch_step_fn(batch, model, device, epoch, data_iter_step, data_loader, args)
        if isinstance(out, tuple) and len(out) == 3:
            loss, detailed_loss, artifacts = out
        else:
            loss, detailed_loss = out
            artifacts = None

        if artifacts is not None and data_iter_step % 100 == 0:
            visualize_reconstruction(model, batch["src_images"], batch["tgt_images"], artifacts, device, step=int(epoch * len(data_loader) + data_iter_step))
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=1.0,
            parameters=model.parameters(),
            update_grad=((data_iter_step + 1) % accum_iter == 0),
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(**detailed_loss)
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            for k, v in detailed_loss.items():
                log_writer.add_scalar(k, v.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_online_token(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    wandb_writer=None,
    args=None,
):
    def batch_step_fn(batch, model, device, epoch, data_iter_step, data_loader, args):
        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_map = batch["attention_map"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=args.amp):
            loss, tgt_pred_post, detailed_loss, artifacts = model(
                src_samples,
                tgt_samples,
                input_ids,
                attention_map,
                epoch + data_iter_step / len(data_loader)
            )
        if data_iter_step % 100 == 0:
            visualize_reconstruction(model, src_samples, tgt_samples, artifacts, device, step=int(epoch * len(data_loader) + data_iter_step))
        return loss, detailed_loss

    return run_train_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        args=args,
        batch_step_fn=batch_step_fn,
    )


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
    def batch_step_fn(batch, model, device, epoch, data_iter_step, data_loader, args):
        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        lm_logits = batch["embeddings"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=args.amp):
            loss, _, detailed_loss = model(
                src_samples, tgt_samples, lm_logits,
                data_iter_step / len(data_loader) + epoch
            )
        return loss, detailed_loss

    return run_train_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        args=args,
        batch_step_fn=batch_step_fn,
    )

def train_one_epoch_m3ae(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    def batch_step_fn(batch, model, device, epoch, data_iter_step, data_loader, args):
        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        lm_logits = batch["embeddings"].to(device, non_blocking=True)
        future_lm = batch["future_embeddings"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=args.amp):
            loss, _, detailed_loss = model(
                src_samples, tgt_samples, lm_logits, future_lm, 
                data_iter_step / len(data_loader) + epoch
            )
        return loss, detailed_loss

    return run_train_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        args=args,
        batch_step_fn=batch_step_fn,
    )

def train_one_epoch_online(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    wandb_writer=None,
    args=None,
):
    def batch_step_fn(batch, model, device, epoch, data_iter_step, data_loader, args):
        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        captions = batch["captions"].to(device, non_blocking=True)  # tokenized texts
        with torch.amp.autocast('cuda', enabled=args.amp):                
            (
                loss,
                tgt_pred_post,            # The reconstruction from posterior path
                detailed_loss,
                artifacts
            ) = model(
                src_samples, 
                tgt_samples, 
                captions,
                epoch + data_iter_step / len(data_loader)
            )
        if data_iter_step % 100 == 0:
            visualize_reconstruction(model, src_samples, tgt_samples, artifacts, device, step=int(epoch * len(data_loader) + data_iter_step))
        return loss, detailed_loss

    return run_train_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        args=args,
        batch_step_fn=batch_step_fn,
    )

def train_one_epoch_self(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    def batch_step_fn(batch, model, device, epoch, data_iter_step, data_loader, args):
        src_samples = batch["src_images"].to(device, non_blocking=True)
        tgt_samples = batch["tgt_images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_map = batch["attention_map"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=args.amp):
            loss, _, detailed_loss = model(
                src_samples, tgt_samples, input_ids, attention_map,
                data_iter_step / len(data_loader) + epoch
            )
        return loss, detailed_loss

    return run_train_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        args=args,
        batch_step_fn=batch_step_fn,
    )
# ---------------------------
# General Training Loop Definition
# ---------------------------
def train_one_epoch_general(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    """
    General training loop that processes a batch dictionary by moving each tensor
    in the batch to the specified device and passing the entire dictionary to the model.
    The model should extract the necessary inputs from the dictionary.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (Iterable): Data loader returning a dictionary of batch items.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Target device.
        epoch (int): Current epoch number.
        loss_scaler: Scaler for mixed precision training.
        log_writer: Optional logger.
        args: Additional arguments, expected to contain 'accum_iter' and 'amp'.

    Returns:
        Dict: Averaged metrics computed over the epoch.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # Move each tensor in batch to the target device if possible.
        for key, value in batch.items():
            if hasattr(value, "to"):
                batch[key] = value.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=args.amp):
            outputs = model(batch, epoch + data_iter_step / len(data_loader))
            if isinstance(outputs, tuple) and len(outputs) == 4:
                loss, _, detailed_loss, artifacts = outputs
            else:
                loss, _, detailed_loss = outputs
                artifacts = None
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=1.0,
            parameters=model.parameters(),
            update_grad=((data_iter_step + 1) % accum_iter == 0),
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        metric_logger.update(**detailed_loss)
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            for k, v in detailed_loss.items():
                log_writer.add_scalar(k, v.item(), epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
