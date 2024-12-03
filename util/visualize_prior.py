import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import modeling
from util.misc import load_model

def get_args_parser():
    parser = argparse.ArgumentParser('Prior visualization script', add_help=False)
    parser.add_argument('--model', default='rsp_vit_large_patch16', type=str)
    parser.add_argument('--checkpoint_path', default='', type=str, help='Path to checkpoint')
    parser.add_argument('--input_path', default='', type=str, help='Path to input image')
    parser.add_argument('--output_dir', default='visualizations', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    return parser

def prepare_model(args):
    # Create model
    print(f"Creating model: {args.model}")
    model = getattr(modeling, args.model)()
    model.to(args.device)
    
    # Load checkpoint
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
        print(msg)
    return model

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare model
    model = prepare_model(args)
    model.eval()
    
    # Load and process image
    src_img = load_image(args.input_path).to(args.device)
    
    with torch.no_grad():
        # Get source image embeddings
        src_h, _, _ = model.forward_encoder(src_img)
        
        # Get prior distribution and sample
        prior_h = src_h[:, 0]
        prior_logits = model.to_prior(prior_h)
        prior_dist = model.make_dist(prior_logits)
        prior_z = prior_dist.rsample()
        
        # Generate reconstruction from prior
        prior_recon = model.forward_decoder_fut(src_h, prior_z)
        prior_recon = model.unpatchify(prior_recon)
        
        # Save original and reconstructed images
        save_image(
            torch.cat([
                src_img.cpu(),
                prior_recon.cpu()
            ], dim=0),
            os.path.join(args.output_dir, 'prior_reconstruction.png'),
            nrow=2,
            normalize=True,
            range=(-1, 1)
        )
        print(f"Saved visualization to {args.output_dir}/prior_reconstruction.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prior visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
