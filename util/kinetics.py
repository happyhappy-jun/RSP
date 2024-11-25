import os
import random
import pickle

from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from util.misc import PairedRandomResizedCrop


class PairedKinetics(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2
    ):
        super().__init__()
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
        repeated_sampling=2
    ):
        super().__init__(root, max_distance, repeated_sampling)

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
        
        # Load frames once
        src_image, tgt_image = self.load_frames(vr)
        
        # Apply same transformation multiple times
        src_images = []
        tgt_images = []
        for i in range(self.repeated_sampling):
            src_transformed, tgt_transformed = self.transform(src_image.copy(), tgt_image.copy())
            src_images.append(src_transformed)
            tgt_images.append(tgt_transformed)
            
        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0

