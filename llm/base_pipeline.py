import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

# --- Data Classes for Interfaces ---

@dataclass
class Step1Output:
    frame_path: Path               # Path to the sampled image frame.
    scene: str                     # Scene description extracted from the response.
    objects: str                   # Detected objects extracted from the response.

@dataclass
class BoundingBox:
    x: float                       # X coordinate (top-left).
    y: float                       # Y coordinate (top-left).
    width: float                   # Width of the box.
    height: float                  # Height of the box.

@dataclass
class Step2Detection:
    bounding_box: BoundingBox  # Detected bounding box.
    logit: float              # Corresponding logit value.
    phrase: str               # Detected phrase.

@dataclass
class Step2Output:
    detections: List[Step2Detection]  # List of detection results.

@dataclass
class Step3Output:
    closed_bboxes: List[BoundingBox]            # List of bounding boxes from closed detection in the future frame.
    movement_captions: List[str] = None           # List of movement captions comparing original and future detections.

# --- Abstract Interface Definitions for Each Step ---

class Step1Sampler(abc.ABC):
    @abc.abstractmethod
    def sample_frame_and_generate_caption(self, video_path: Path) -> Step1Output:
        """
        Given a video file, sample an image frame and generate a descriptive caption 
        (with chain-of-thought) using an OpenAI-compatible backend.
        """
        pass

class Step2Grounding(abc.ABC):
    @abc.abstractmethod
    def detect_bounding_boxes(self, frame_path: Path, caption: str) -> Step2Output:
        """
        Given the input image frame (from step1) and its caption, perform open-vocabulary detection 
        using Grounding Dino and return the detected bounding boxes.
        """
        pass

class Step3FutureDetection(abc.ABC):
    @abc.abstractmethod
    def detect_in_future_frame(self, video_path: Path, bounding_boxes: List[BoundingBox]) -> Step3Output:
        """
        Given the original video and a list of bounding boxes detected in step2, sample a future frame 
        and perform closed bounding box detection for each bounding box.
        """
        pass

# --- High-level Pipeline Class ---

class DatasetGenerationPipeline:
    def __init__(self, step1: Step1Sampler, step2: Step2Grounding, step3: Step3FutureDetection):
        self.step1 = step1
        self.step2 = step2
        self.step3 = step3

    def generate_pipeline(self, video_path: Path) -> Dict[str, object]:
        """
        Executes the full pipeline:
          • Step 1: Sample a frame from the video and generate a caption.
          • Step 2: Use the sampled frame and caption to detect bounding boxes.
          • Step 3: From one of the bounding boxes, sample a future frame and perform closed detection.
          
        Returns a dictionary with keys 'step1', 'step2', and 'step3' holding their respective outputs.
        """
        step1_output = self.step1.sample_frame_and_generate_caption(video_path)
        step2_output = self.step2.detect_bounding_boxes(step1_output.frame_path, step1_output.objects)
        
        if not step2_output.detections:
            raise ValueError("No detections detected in step2.")
        
        # Process all detections from step2.
        bounding_boxes = [d.bounding_box for d in step2_output.detections]
        step3_output = self.step3.detect_in_future_frame(video_path, bounding_boxes)

        return {
            "step1": step1_output,
            "step2": step2_output,
            "step3": step3_output,
        }
