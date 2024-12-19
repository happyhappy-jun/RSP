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

import modeling
from util.kinetics import PairedKineticsFixed, PairedKinetics
import os
import random
import pickle
import numpy as np
from decord import VideoReader, cpu
import torch
from torch.utils.data import Dataset
from util.misc import seed_everything



class PairedKineticsFixedViz(PairedKinetics):
    def __init__(
            self,
            root,
            max_distance=48,
            repeated_sampling=2,
            seed=42
    ):
        super().__init__(root, max_distance, repeated_sampling, seed)
        self.presampled_indices = {}
        # Presample multiple pairs of indices for all videos
        for idx in range(len(self.samples)):
            sample = os.path.join(self.root, self.samples[idx][1])
            try:
                vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
            except Exception as e:
                print(f"Error loading video {sample}: {str(e)}")
                # Return a default/empty sample
                return torch.zeros(self.repeated_sampling, 3, 224, 224), \
                    torch.zeros(self.repeated_sampling, 3, 224, 224), 0
            seg_len = len(vr)
            least_frames_num = self.max_distance + 1

            # Sample repeated_sampling pairs of frames
            pairs = []
            for _ in range(self.repeated_sampling):
                if seg_len >= least_frames_num:
                    idx_cur = random.randint(0, seg_len - least_frames_num)
                    interval = random.randint(4, self.max_distance)
                    idx_fut = idx_cur + interval
                else:
                    indices = random.sample(range(seg_len), 2)
                    indices.sort()
                    idx_cur, idx_fut = indices
                pairs.append((idx_cur, idx_fut))

            self.presampled_indices[idx] = pairs

    def load_frames(self, vr, index, pair_idx):
        idx_cur, idx_fut = self.presampled_indices[index][pair_idx]
        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()
        return frame_cur, frame_fut

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))

        # Load and transform each pair of presampled frames
        src_images = []
        tgt_images = []
        for pair_idx in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr, index, pair_idx)
            src_images.append(src_image)
            tgt_images.append(tgt_image)

        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0



def prepare_model(args, cfg):
    # Define the model
    model = modeling.__dict__[cfg.model_name](
       **cfg.model_params
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
        
        # If norm_pix_loss was used during training, we need to revert it
        if True:
        # if cfg.norm_pix_loss:
            target = model.patchify(tgt_imgs)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            tgt_pred_prior = tgt_pred_prior * (var + 1.0e-6) ** 0.5 + mean
            
        reconstructed_imgs = model.unpatchify(tgt_pred_prior)
        
        # Denormalize images (ImageNet normalization)
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
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='If set, the model uses pixel-wise normalization')
    return parser

def main(args):
    # Load config
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name=args.config)

    # Fix the seed
    seed_everything(args.seed)
    cudnn.benchmark = True
    
    # Create dataset and dataloader
    dataset = PairedKineticsFixed(root=f"{os.environ['HOME']}/kinetics400", seed=args.seed)
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
