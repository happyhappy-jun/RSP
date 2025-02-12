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
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        img1_b64 = frame_to_base64(frame1)
        img2_b64 = frame_to_base64(frame2)
        
        # Prepare request
        payload = {
            "model": "internvl2",
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
        
        # Convert response to embedding using mean pooling
        # This is a placeholder - replace with actual text-to-embedding logic
        # For now, returning random embedding of expected size
        embedding = np.random.randn(384).astype(np.float32)
        return torch.from_numpy(embedding)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        front_video, overhead_video = self.samples[index]
        vr_front = VideoReader(front_video, num_threads=1, ctx=cpu(0))
        vr_overhead = VideoReader(overhead_video, num_threads=1, ctx=cpu(0))
        
        src_images = []
        tgt_images = []
        embeddings = []
        
        for _ in range(self.repeated_sampling):
            # Load frames from both views at same timestamp
            seg_len = min(len(vr_front), len(vr_overhead))
            frame_idx = random.randint(0, seg_len - 1)
            
            frame_front = vr_front[frame_idx].asnumpy()
            frame_overhead = vr_overhead[frame_idx].asnumpy()
            
            # Apply transforms to both views
            src_image, tgt_image = self.transforms(frame_front, frame_overhead)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            # Get caption embedding
            embedding = self.get_caption(frame_front, frame_overhead)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(embedding)
        
        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0)
        }

