import os
import random
import glob
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
from PIL import Image
import requests
import base64
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kinetics_online_caption.log')
    ]
)
logger = logging.getLogger(__name__)

from util.transform import PairedRandomResizedCrop

class RLBenchOnlineCaption(Dataset):
    def __init__(
        self,
        root,
        max_distance=48,
        repeated_sampling=2,
        llm=None,
        max_length=8192,
        embedding_model=None
    ):
        from transformers import AutoTokenizer
        super().__init__()
        self.root = root
        
        # Find all task directories
        self.video_paths = (glob.glob(os.path.join(root, "*_front.mp4")) +
                            glob.glob(os.path.join(root, "*_overhead.mp4")))


        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling
        
        # LLM configuration
        self.llm = llm or {
            "model": self.llm["model"],
            "host": "0.0.0.0",
            "port": 23333,
            "postfix": "/v1/chat/completions"
        }
        self.llm_url = f"http://{self.llm['host']}:{self.llm['port']}{self.llm['postfix']}"
        
        # Initialize tokenizer from config
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.max_length = max_length

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy array frame to base64 string."""
        try:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(frame)

            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')

            # Convert to base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to convert frame to base64: {e}", exc_info=True)
            raise

    def get_caption(self, frame1, frame2, max_retries=3):
        """Generate caption comparing two frames using LLM"""
        # Convert frames to base64
        img1_b64 = self.frame_to_base64(frame1)
        img2_b64 = self.frame_to_base64(frame2)
        
        # Prepare request
        payload = {
            "model": self.llm["model"],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the main differences between these two frames from a video."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
                ]
            }],
            "temperature": 1.0,
        }
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Send request to local server and measure time
                start_time = time.time()
                response = requests.post(self.llm_url, json=payload)
                response.raise_for_status()  # Raise exception for bad status codes
                response_json = response.json()
                caption = response_json['choices'][0]['message']['content']
                request_time = time.time() - start_time
                return caption
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise Exception(f"Failed to get caption after {max_retries} retries: {str(e)}")
                logger.warning(f"Request failed (attempt {retry_count}/{max_retries})", exc_info=True)
                # Exponential backoff with jitter
                wait_time = min(60 * (2 ** retry_count) + random.uniform(0, 10), 300)  # Cap at 5 minutes
                logger.info(f"Waiting {wait_time:.1f} seconds before retry {retry_count + 1}")
                time.sleep(wait_time)

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
        captions = []
        src_images = []
        tgt_images = []

        for i in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr)
            caption = self.get_caption(src_image, tgt_image)
            src_image, tgt_image = self.transform(src_image, tgt_image)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            captions.append(caption)
        
        # Tokenize captions individually to avoid collation issues
        tokenized_list = []
        for caption in captions:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Squeeze out the batch dimension since we're processing one at a time
            tokenized_list.append({
                k: v.squeeze(0) for k, v in tokens.items()
            })
        
        # Stack all tokenized outputs
        tokenized_batch = {
            k: torch.stack([item[k] for item in tokenized_list])
            for k in tokenized_list[0].keys()
        }
        
        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "captions": tokenized_batch
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
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # logging.info(f"Dataset size: {len(dataset)}")
    # logging.info(f"Number of batches: {len(dataloader)}")
    #
    # # Test batch loading
    # start_time = time.time()
    # for batch_idx, batch in enumerate(dataloader):
    #     logging.info(f"\nProcessing batch {batch_idx + 1}")
    #     logging.info(f"Batch shapes:")
    #
    #     if batch_idx >= 2:  # Test first 3 batches only
    #         break
    #
    # total_time = time.time() - start_time
    # logging.info(f"\nProcessed 3 batches in {total_time:.2f} seconds")
    # logging.info(f"Average time per batch: {total_time/3:.2f} seconds")
    logging.info(len(dataset))
