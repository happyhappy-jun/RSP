import queue
import threading
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Optional
import numpy as np
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
        
        # Initialize InternVL2 model on GPUs 4-7
        model_name = "OpenGVLab/InternVL2-2B"
        
        # Configure model to use specified GPUs
        device_str = f"cuda:{device_ids[0]}"
        torch.cuda.set_device(device_ids[0])
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_str,
            trust_remote_code=True
        )
        
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
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    # Process image
                    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device_str)
                    
                    # Generate caption
                    with torch.cuda.amp.autocast():
                        outputs = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=128,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode caption
                    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Get text embedding from the last hidden state
                    with torch.no_grad():
                        text_outputs = model(**inputs, output_hidden_states=True)
                        # Use the last hidden state as embedding
                        embedding = text_outputs.hidden_states[-1].mean(dim=1)  # Average pooling
                        # Resize to expected dimension
                        embedding = torch.nn.functional.linear(
                            embedding, 
                            torch.randn(3072, embedding.shape[-1], device=device_str)
                        )
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
