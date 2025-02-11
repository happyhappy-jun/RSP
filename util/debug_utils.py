import torch
import torchvision.transforms as transforms
import numpy as np

def create_debug_image(size=224):
    """Create a random noise image for debugging"""
    # Create random noise
    noise = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(noise)
