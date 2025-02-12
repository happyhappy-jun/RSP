import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
from PIL import Image
import requests
import base64
import io

from util.transform import PairedRandomResizedCrop

class RLBenchOnlineCaption(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2,
    ):
        super().__init__()
        self.root = root
        
        # Find all task directories
        self.video_paths = glob.glob(os.path.join(root, "*_*{overhead,front}.mp4"))

        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling
        
        # LLM server endpoint
        self.llm_url = "http://0.0.0.0:23333/v1/chat/completions"

    def get_caption(self, frame1, frame2):
        """Generate caption comparing two frames using LLM"""
        # Convert frames to base64
        def frame_to_base64(frame):
            img = Image.fromarray(frame)
            return base64.b64encode(img.tobytes()).decode('utf-8')
            
        img1_b64 = frame_to_base64(frame1)
        img2_b64 = frame_to_base64(frame2)
        
        # Prepare request
        payload = {
            "model": "OpenGVLab/InternVL2_5-8B",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the main differences between these two frames from a video."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
                ]
            }],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        # Send request to local server
        response = requests.post(self.llm_url, json=payload)
        response_json = response.json()
        caption = response_json['choices'][0]['message']['content']
        print(caption)
        
        # Convert response to embedding using mean pooling
        # This is a placeholder - replace with actual text-to-embedding logic
        # For now, returning random embedding of expected size
        embedding = np.random.randn(384).astype(np.float32)
        return torch.from_numpy(embedding)

    def __len__(self):
        return len(self.video_paths)

    def transform(self, src_image, tgt_image):
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

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

    def __getitem__(self, index):
        video = self.video_paths[index]
        vr = VideoReader(video, num_threads=1, ctx=cpu(0))
        embeddings = []
        src_images = []
        tgt_images = []

        for i in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr)
            embedding = self.get_caption(src_image, tgt_image)
            src_image, tgt_image = self.transform(src_image, tgt_image)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(embedding)
        
        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0)
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Initialize dataset
    dataset = RLBenchOnlineCaption(
        root="/data/RSP/rlbench/demo",
        repeated_sampling=2
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}")
        print("Source image shape:", batch["src_images"].shape)
        print("Target image shape:", batch["tgt_images"].shape)
        print("Embedding shape:", batch["embeddings"].shape)
        
        if i >= 2:  # Test first 3 batches only
            break

