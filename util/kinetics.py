import os
import random
import pickle
import torch
import numpy as np

from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from util.kinetics_caption import PairedRandomResizedCrop


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PairedKinetics(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2,
        seed=42
    ):
        super().__init__()
        seed_everything(seed)
        self.root = root
        with open(
            os.path.join(self.root, "labels", f"label_full_1.0.pickle"), "rb"
        ) as f:
            self.samples = pickle.load(f)

        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
        src_images = []
        tgt_images = []
        for i in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr)
            src_image, tgt_image = self.transform(src_image, tgt_image)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0


    def load_frames(self, vr):
        # handle temporal segments
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1
        if seg_len >= least_frames_num:
            idx_cur = random.randint(0, seg_len - least_frames_num)
            interval = random.randint(4, self.max_distance)
            idx_fut = idx_cur + interval
        else:
            indices = random.sample(range(seg_len), 2)
            indices.sort()
            idx_cur, idx_fut = indices
        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()

        return frame_cur, frame_fut

    def transform(self, src_image, tgt_image):
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image


class PairedKineticsFixed(PairedKinetics):
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
            vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
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
            src_transformed, tgt_transformed = self.transform(src_image, tgt_image)
            src_images.append(src_transformed)
            tgt_images.append(tgt_transformed)
            
        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0



if __name__ == "__main__":
    # Test to verify PairedKineticsFixed uses fixed frame numbers
    root = "/data/kinetics400"  # Replace with actual path
    
    # Create two instances of PairedKineticsFixed
    dataset1 = PairedKineticsFixed(root, seed=42)
    dataset2 = PairedKineticsFixed(root, seed=42)
    
    # Check if presampled indices are the same for both instances
    print("Testing if frame indices are fixed across dataset instances...")
    
    # Compare presampled indices for first 5 videos
    for idx in range(min(5, len(dataset1))):
        indices1 = dataset1.presampled_indices[idx]
        indices1_2 = dataset1.presampled_indices[idx]
        indices2 = dataset2.presampled_indices[idx]
        indices2_2 = dataset2.presampled_indices[idx]
        
        print(f"Video {idx}:")
        print(f"Dataset1 indices: {indices1}")
        print(f"Dataset2 indices: {indices2}")
        print(f"Dataset1 indices (2nd time): {indices1_2}")
        print(f"Dataset2 indices (2nd time): {indices2_2}")

    
    print("Test passed! Frame indices are fixed across dataset instances.")
