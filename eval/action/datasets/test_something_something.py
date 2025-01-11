import torch
import torchvision.transforms as transforms
from something_something_v2 import SomethingSomethingV2

def test_dataset():
    # Define basic transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Create dataset instance
    dataset = SomethingSomethingV2(
        split="validation",
        transform=transform,
        frames_per_video=1
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading a single sample
    frames, label = dataset[0]
    print(f"\nSample info:")
    if isinstance(frames, torch.Tensor):
        print(f"Single frame tensor shape: {frames.shape}")
        print(f"Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
    elif isinstance(frames, list):
        print(f"Number of frames: {len(frames)}")
        print(f"Frame tensor shape: {frames[0].shape}")
    else:
        print(f"Frames tensor shape: {frames.shape}")
    print(f"Label: {label} ({dataset.classes[label]})")
    
    # Test DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )
    
    print("\nTesting DataLoader...")
    batch = next(iter(dataloader))
    frames, labels = batch
    print(f"Batch frames shape: {frames.shape}")
    print(f"Batch labels shape: {labels.shape}")

if __name__ == "__main__":
    test_dataset()
