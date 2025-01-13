import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
import modeling
from eval.action.linear_probe_model import LinearProbing
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import subprocess

from eval.action.optimizers.lars import LARS


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    correct = torch.zeros(1).to(device)
    total = torch.zeros(1).to(device)

    if dist.is_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch in tqdm(dataloader, desc=f"Training epoch {epoch}", disable=not is_main_process()):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum()

    # Gather metrics from all processes
    if dist.is_initialized():
        dist.all_reduce(total_loss)
        dist.all_reduce(correct)
        dist.all_reduce(total)

    accuracy = 100. * correct.item() / total.item()
    avg_loss = total_loss.item() / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = torch.zeros(1).to(device)
    correct = torch.zeros(1).to(device)
    total = torch.zeros(1).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum()

    # Gather metrics from all processes
    if dist.is_initialized():
        dist.all_reduce(total_loss)
        dist.all_reduce(correct)
        dist.all_reduce(total)

    accuracy = 100. * correct.item() / total.item()
    avg_loss = total_loss.item() / len(dataloader)
    return avg_loss, accuracy


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


@hydra.main(config_path="config", config_name="linear_probing", version_base="1.2")
def main(cfg: DictConfig):
    init_distributed_mode(cfg)

    if cfg.use_wandb and is_main_process():
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Set device
    device = torch.device(cfg.gpu if cfg.distributed else cfg.device)

    # Load pretrained model
    model = modeling.__dict__[cfg.model_name](**cfg.model_params)
    checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')["model"]

    # remove prefixes
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # state_dict = {k.replace("base_encoder.", ""): v for k, v in state_dict.items()}
    interpolate_pos_embed(model, checkpoint)
    msg = model.load_state_dict(checkpoint, strict=False)
    if is_main_process():
        print(msg)

    # Create linear probing model and move to device
    linear_probe = LinearProbing(model, num_classes=cfg.model.num_classes)
    linear_probe = linear_probe.to(device)

    if cfg.distributed:
        linear_probe = torch.nn.parallel.DistributedDataParallel(
            linear_probe,
            device_ids=[cfg.gpu],
        )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = LARS(linear_probe.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0)

    # Instantiate datasets
    train_dataset = hydra.utils.instantiate(cfg.train_dataset, _convert_="all")
    val_dataset = hydra.utils.instantiate(cfg.val_dataset, _convert_="all")

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset) if cfg.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if cfg.distributed else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        sampler=train_sampler,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        sampler=val_sampler
    )

    # Training loop
    best_acc = 0
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_epoch(
            linear_probe, train_loader, criterion, optimizer, device, epoch
        )

        # Step the scheduler
        scheduler.step()
        
        val_loss, val_acc = evaluate(
            linear_probe, val_loader, criterion, device
        )

        if is_main_process():
            print(f"Epoch: {epoch + 1}/{cfg.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if cfg.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': scheduler.get_last_lr()[0]
                })

            # Save best model
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     if cfg.output_dir:
            #         state_dict = linear_probe.module.state_dict() if cfg.distributed else linear_probe.state_dict()
            #         torch.save({
            #             'epoch': epoch,
            #             'model_state_dict': state_dict,
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'best_acc': best_acc,
            #         }, os.path.join(cfg.output_dir, 'best_model.pth'))

    if is_main_process():
        print(f"Best validation accuracy: {best_acc:.2f}%")
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
