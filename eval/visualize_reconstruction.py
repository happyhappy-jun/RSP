import os
import argparse

import hydra.utils
from hydra import compose, initialize
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import util.misc as misc
import modeling
from util.kinetics import PairedKineticsFixed
from util.kinetics_caption import PairedKineticsWithCaption, collate_fn

def get_args_parser():
    parser = argparse.ArgumentParser('Visualization script', add_help=False)
    parser.add_argument('--config', default='path/to/your/config.yaml', type=str,
                        help='Path to config file used for training')
    parser.add_argument('--checkpoint', default='path/to/checkpoint.pth', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_samples', default=5, type=int,
                        help='Number of samples to visualize')
    return parser

def prepare_model(args, cfg):
    # Define the model
    model = modeling.__dict__[cfg.model](
        norm_pix_loss=cfg.norm_pix_loss,
        kl_scale=cfg.kl_scale,
        kl_balance=cfg.kl_balance,
        kl_freebit=cfg.kl_freebit,
        stoch=cfg.stoch,
        discrete=cfg.discrete,
        mask_ratio=cfg.mask_ratio,
        noise_scale=cfg.noise_scale
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    
    model.to(args.device)
    model.eval()
    return model

def visualize_reconstruction(model, src_imgs, tgt_imgs, device, cfg):
    # Move images to device
    src_imgs = src_imgs.to(device)
    tgt_imgs = tgt_imgs.to(device)

    num_samples = src_imgs.size(0)

    with torch.no_grad():
        # Get model predictions
        src_h, _, _ = model.forward_encoder(src_imgs, mask_ratio=0)
        
        # Get prior distribution and sample
        prior_h = src_h[:, 0]
        prior_logits = model.to_prior(prior_h)
        prior_dist = model.make_dist(prior_logits)
        prior_z = prior_dist.rsample()
        
        # Get reconstruction from prior
        tgt_pred_prior = model.forward_decoder_fut(src_h, prior_z)
        reconstructed_imgs = model.unpatchify(tgt_pred_prior)
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        
        src_imgs = src_imgs * std + mean
        tgt_imgs = tgt_imgs * std + mean
        reconstructed_imgs = reconstructed_imgs * std + mean
        
        # Plot results
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        for i in range(num_samples):
            # Source image
            axes[i, 0].imshow(src_imgs[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 0].set_title(f'Source - {cfg.exp_name}')
            axes[i, 0].axis('off')
            
            # Target image
            axes[i, 1].imshow(tgt_imgs[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].set_title('Target')
            axes[i, 1].axis('off')
            
            # Reconstructed image
            axes[i, 2].imshow(reconstructed_imgs[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 2].set_title('Reconstruction')
            axes[i, 2].axis('off')
        
        # Add experiment name from config as a figure suptitle
        fig.suptitle(f'Reconstruction Results - {cfg.exp_name}', y=1.02, fontsize=14)
        plt.tight_layout()
        save_path = f'artifacts/{cfg.exp_name}_viz.png'
        plt.savefig(save_path)
        plt.close()
        print(f'Saved visualization to {save_path}')

def main(args):
    # Load config
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name=args.config)

    # Fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # Create dataset and dataloader
    dataset = PairedKineticsFixed(root="/data/kinetics400")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
    )

    # Prepare model
    model = prepare_model(args, cfg)
    
    # Visualize reconstructions
    batch = next(iter(dataloader))
    if isinstance(batch, dict):
        src_imgs, tgt_imgs = batch['src_images'], batch['tgt_images']
    else:
        src_imgs, tgt_imgs, _ = batch
    indices = np.random.choice(len(src_imgs), args.num_samples, replace=False)
    indices = indices * 2

    src_imgs = src_imgs.view(-1, 3, src_imgs.size(-2), src_imgs.size(-1))
    tgt_imgs = tgt_imgs.view(-1, 3, tgt_imgs.size(-2), tgt_imgs.size(-1))

    src_imgs = src_imgs[indices]
    tgt_imgs = tgt_imgs[indices]
    visualize_reconstruction(model, src_imgs, tgt_imgs, args.device, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
