from importlib.readers import FileReader
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from util.transform import PairedRandomResizedCrop
from decord import VideoReader, cpu


class PairedRLBench(Dataset):
    """Dataset class for paired frames from RLBench videos."""

    def __init__(
        self,
        root: str,
        max_distance: int = 48,
        repeated_sampling: int = 2,
        seed: int = 42
    ):
        """Initialize the dataset.
        
        Args:
            root: Root directory containing the videos
            max_distance: Maximum frame distance between pairs
            repeated_sampling: Number of times to sample from each video
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        random.seed(seed)
        self.root = root
        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling
        
        # Collect all MP4 files
        self.samples = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.mp4'):
                    self.samples.append(os.path.join(dirpath, filename))

        # Initialize transforms
        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a pair of frames from a video.
        
        Args:
            index: Index of the video
            
        Returns:
            Tuple of (source frames, target frames, label)
        """
        video_path = self.samples[index]
        
        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a zero tensor if video loading fails
            zero_tensor = torch.zeros(self.repeated_sampling, 3, 224, 224)
            return zero_tensor, zero_tensor, 0

        src_images = []
        tgt_images = []
        
        for _ in range(self.repeated_sampling):
            try:
                src_image, tgt_image = self.load_frames(vr)
                src_image, tgt_image = self.transform(src_image, tgt_image)
                src_images.append(src_image)
                tgt_images.append(tgt_image)
            except Exception as e:
                print(f"Error processing frames from {video_path}: {str(e)}")
                # Return zero tensors if frame processing fails
                zero_tensor = torch.zeros(self.repeated_sampling, 3, 224, 224)
                return zero_tensor, zero_tensor, 0

        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        
        return src_images, tgt_images, 0

    def load_frames(self, vr: VideoReader) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a pair of frames from the video with temporal distance.
        
        Args:
            vr: VideoReader object
            
        Returns:
            Tuple of (source frame, target frame)
        """
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1

        if seg_len >= least_frames_num:
            idx_cur = random.randint(0, seg_len - least_frames_num)
            interval = random.randint(4, self.max_distance)
            idx_fut = min(idx_cur + interval, seg_len - 1)
        else:
            indices = random.sample(range(seg_len), 2)
            indices.sort()
            idx_cur, idx_fut = indices

        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()

        return frame_cur, frame_fut

    def transform(self, src_image: torch.Tensor, tgt_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to the image pair.
        
        Args:
            src_image: Source image
            tgt_image: Target image
            
        Returns:
            Tuple of transformed (source image, target image)
        """
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image


def test_dataset(root_path: str):
    """Test function to verify dataset behavior."""
    # Create two instances with same seed
    dataset1 = PairedRLBench(root_path, seed=42)
    dataset2 = PairedRLBench(root_path, seed=42)
    print(len(dataset1))
    
    print("Testing dataset consistency...")
    
    # Compare first few items
    for idx in range(min(5, len(dataset1))):
        src1, tgt1, _ = dataset1[idx]
        src2, tgt2, _ = dataset2[idx]
        
        print(f"\nVideo {idx}:")
        print(f"Source shapes: {src1.shape} | {src2.shape}")
        print(f"Target shapes: {tgt1.shape} | {tgt2.shape}")
        
        # Check if tensors have expected properties
        assert not torch.isnan(src1).any(), f"NaN values in source tensor 1, video {idx}"
        assert not torch.isnan(tgt1).any(), f"NaN values in target tensor 1, video {idx}"
        assert src1.shape == tgt1.shape, f"Shape mismatch in video {idx}"

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_dataset("/data/RSP/rlbench/demo")  # Replace with actual path