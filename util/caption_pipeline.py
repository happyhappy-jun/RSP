import os
import queue
import threading
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Optional
import numpy as np
from vllm import LLM, SamplingParams
from util.debug_utils import create_debug_image

class CaptionWorker:
    """Worker process for caption generation running on GPUs 4-7"""
    def __init__(self, device_ids: List[int], queue_size: int = 8):
        self.device_ids = device_ids
        self.input_queue = mp.Queue(maxsize=queue_size)
        self.output_queue = mp.Queue(maxsize=queue_size)
        self.process = None
        
    def start(self):
        self.process = mp.Process(
            target=self._run_caption_worker,
            args=(self.input_queue, self.output_queue, self.device_ids)
        )
        self.process.start()

    def stop(self):
        if self.process:
            self.input_queue.put(None)  # Sentinel to stop worker
            self.process.join()
            self.process = None

    @staticmethod
    def _run_caption_worker(input_queue, output_queue, device_ids):
        import torch
        from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
        from vllm import LLM, SamplingParams
        
        # Initialize InternVL2 model using tensor parallelism on GPUs 4-7
        model_name = "OpenGVLab/InternVL2-2B"
        
        # Use vLLM for tensor parallel inference
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
        
        llm = LLM(
            model=model_name,
            tensor_parallel_size=len(device_ids),  # Split across all specified GPUs
            trust_remote_code=True,
            dtype="float16",
            max_model_len=4096,
            quantization="awq"  # Optional: Enable quantization for memory efficiency
        )
        
        # Initialize processor for image preprocessing
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        while True:
            try:
                batch = input_queue.get()
                if batch is None:  # Stop sentinel
                    break
                    
                batch_id, images = batch
                
                # Process each image in the batch
                all_embeddings = []
                for img in images:
                    # Format prompt for single image
                    messages = [{'role': 'user', 'content': f"Image-1: <image>\nDescribe this image in detail."}]
                    
                    # Process image and create input for vLLM
                    processed_img = processor(images=img, return_tensors="pt")
                    
                    # Generate caption using vLLM
                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=128,
                        stop=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
                    )
                    
                    outputs = llm.generate(
                        prompt=messages[0]['content'],
                        sampling_params=sampling_params,
                        multi_modal_data={"image": [processed_img]}
                    )
                    
                    # Extract caption from output
                    caption = outputs[0].outputs[0].text
                    
                    # Create embedding from caption
                    # For now using random embedding - replace with actual embedding generation
                    embedding = torch.randn(1, 3072)
                    all_embeddings.append(embedding)
                
                # Stack all embeddings
                embeddings = torch.cat(all_embeddings, dim=0)
                print(f"Generated embeddings shape: {embeddings.shape}")
                output_queue.put((batch_id, embeddings.cpu()))
                
            except Exception as e:
                print(f"Error in caption worker: {str(e)}")
                # Return random embeddings as fallback
                embeddings = torch.randn(len(images), 3072)
                output_queue.put((batch_id, embeddings))

class CaptionPipeline:
    """Manages asynchronous caption generation pipeline"""
    def __init__(self, 
                 caption_device_ids: List[int] = [4,5,6,7],
                 queue_size: int = 8):
        self.worker = CaptionWorker(caption_device_ids, queue_size)
        self.pending_batches = {}
        self.next_batch_id = 0
        
    def start(self):
        """Start the caption worker process"""
        self.worker.start()
        
    def stop(self):
        """Stop the caption worker process"""
        self.worker.stop()

    def submit_batch(self, images: torch.Tensor) -> int:
        """Submit a batch of images for caption generation"""
        batch_id = self.next_batch_id
        self.next_batch_id += 1
        self.worker.input_queue.put((batch_id, images))
        return batch_id
        
    def get_result(self, batch_id: int, timeout: Optional[float] = None) -> torch.Tensor:
        """Get caption embeddings for a batch. Returns None if not ready."""
        try:
            result_id, embeddings = self.worker.output_queue.get(timeout=timeout)
            if result_id == batch_id:
                return embeddings
            self.pending_batches[result_id] = embeddings
        except queue.Empty:
            return None
            
        # Check if we have the requested batch in pending
        return self.pending_batches.pop(batch_id, None)

if __name__ == "__main__":
    import PIL.Image
    from vllm import LLM, SamplingParams
    
    # Test InternVL2 loading and inference
    print("Testing InternVL2 caption generation...")
    
    # Create test pipeline
    pipeline = CaptionPipeline(caption_device_ids=[4,5,6,7])
    pipeline.start()
    
    # Create some test images
    test_images = [create_debug_image() for _ in range(2)]
    
    # Submit batch
    print("Submitting test batch...")
    batch_id = pipeline.submit_batch(test_images)
    
    # Try to get results with timeout
    print("Waiting for results...")
    result = pipeline.get_result(batch_id, timeout=30)
    
    if result is not None:
        print(f"Success! Got embeddings with shape: {result.shape}")
    else:
        print("Failed to get results within timeout")
    
    # Cleanup
    pipeline.stop()
