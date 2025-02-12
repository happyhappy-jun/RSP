import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
from PIL import Image
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64

from util.transform import PairedRandomResizedCrop

class RLBenchOnlineCaption(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2,
        model_path='OpenGVLab/InternVL2-8B'
    ):
        super().__init__()
        self.root = root
        
        # Find all task directories
        task_dirs = glob.glob(os.path.join(root, "*"))
        self.samples = []
        
        # Collect front and overhead video pairs
        for task_dir in task_dirs:
            front_videos = glob.glob(os.path.join(task_dir, "*_front.mp4"))
            for front_video in front_videos:
                overhead_video = front_video.replace("_front.mp4", "_overhead.mp4")
                if os.path.exists(overhead_video):
                    self.samples.append((front_video, overhead_video))

        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling
        
        # Initialize LLM pipeline
        print(f"Initializing LLM pipeline with model: {model_path}")
        self.pipe = pipeline(model_path, log_level='INFO')

    def get_caption(self, frame1, frame2):
        """Generate caption comparing two frames using LLM"""
        # Convert frames to PIL Images
        img1 = Image.fromarray(frame1)
        img2 = Image.fromarray(frame2)
        
        # Construct prompt
        question = f'Frame1: {IMAGE_TOKEN}\nFrame2: {IMAGE_TOKEN}\nDescribe the main differences between these two frames from a video.'
        
        # Prepare content for LLM
        content = [{'type': 'text', 'text': question}]
        for img in [img1, img2]:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'max_dynamic_patch': 1,
                    'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'
                }
            })
        
        # Generate caption
        messages = [dict(role='user', content=content)]
        response = self.pipe(messages, gen_config=GenerationConfig(top_k=1))
        
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
            least_frames_num = self.max_distance + 1
            if seg_len >= least_frames_num:
                idx_cur = random.randint(0, seg_len - least_frames_num)
                interval = random.randint(4, self.max_distance)
                idx_fut = idx_cur + interval
            else:
                indices = random.sample(range(seg_len), 2)
                indices.sort()
                idx_cur, idx_fut = indices
                
            frame_front = vr_front[idx_cur].asnumpy()
            frame_overhead = vr_overhead[idx_cur].asnumpy()
            
            # Apply transforms to both views
            src_image, tgt_image = self.transforms(frame_front, frame_overhead)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            # Get caption embedding
            embedding = self.get_caption(frame_cur, frame_fut)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(embedding)
        
        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0)
        }

