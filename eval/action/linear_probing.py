import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import modeling
from eval.action.linear_probe_model import LinearProbing
import hydra
from omegaconf import DictConfig, OmegaConf

def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
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
            
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
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
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

@hydra.main(config_path="config", config_name="linear_probing", version_base="1.2")
def main(cfg: DictConfig):
    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Load pretrained model
    model = modeling.__dict__[cfg.model_name](**cfg.model_params)
    checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("base_encoder.", ""): v for k, v in state_dict.items()}
    interpolate_pos_embed(model, state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    
    # Create linear probing model
    linear_probe = LinearProbing(model, num_classes=cfg.num_classes).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_probe.classifier.parameters(), lr=cfg.lr)
    
    # Instantiate datasets
    train_dataset = hydra.utils.instantiate(cfg.train_dataset, _convert_="all")
    val_dataset = hydra.utils.instantiate(cfg.val_dataset, _convert_="all")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_epoch(
            linear_probe, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc = evaluate(
            linear_probe, val_loader, criterion, device
        )
        
        print(f"Epoch: {epoch+1}/{cfg.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if cfg.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            if cfg.output_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': linear_probe.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(cfg.output_dir, 'best_model.pth'))
    
    print(f"Best validation accuracy: {best_acc:.2f}%")
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
