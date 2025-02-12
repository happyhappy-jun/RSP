import os
import random
import pickle
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

class PairedKineticsOnlineCaption(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2,
        model_path='OpenGVLab/InternVL2-8B'
    ):
        super().__init__()
        self.root = root
        with open(
            os.path.join(self.root, "labels", f"label_1.0.pickle"), "rb"
        ) as f:
            self.samples = pickle.load(f)

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
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
        
        src_images = []
        tgt_images = []
        embeddings = []
        
        for _ in range(self.repeated_sampling):
            # Load frames
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
            
            # Apply transforms
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
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

