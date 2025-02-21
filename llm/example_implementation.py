# Example Implementation for the DatasetGenerationPipeline.
# This file provides dummy implementations for each pipeline step.

from pathlib import Path
from llm.dataset_pipeline import (
    Step1Sampler,
    Step2Grounding,
    Step3FutureDetection,
    Step1Output,
    Step2Output,
    Step3Output,
    BoundingBox,
    DatasetGenerationPipeline
)

class DummyStep1Sampler(Step1Sampler):
    def sample_frame_and_generate_caption(self, video_path: Path) -> Step1Output:
        # Dummy implementation: Returns a fixed frame path, caption, and chain-of-thought reasoning.
        return Step1Output(
            frame_path=Path("/tmp/dummy_frame.jpg"),
            caption=f"Dummy caption for {video_path}",
            chain_of_thought="Dummy chain-of-thought reasoning."
        )

class DummyStep2Grounding(Step2Grounding):
    def detect_bounding_boxes(self, frame_path: Path, caption: str) -> Step2Output:
        # Dummy implementation: Returns a fixed bounding box.
        dummy_bbox = BoundingBox(x=50.0, y=50.0, width=200.0, height=200.0)
        return Step2Output(bounding_boxes=[dummy_bbox])

class DummyStep3FutureDetection(Step3FutureDetection):
    def detect_in_future_frame(self, video_path: Path, bounding_box: BoundingBox) -> Step3Output:
        # Dummy implementation: Returns a bounding box slightly offset from the input.
        new_bbox = BoundingBox(
            x=bounding_box.x + 10.0,
            y=bounding_box.y + 10.0,
            width=bounding_box.width,
            height=bounding_box.height
        )
        return Step3Output(closed_bbox=new_bbox)

if __name__ == '__main__':
    # Create instances of each dummy step.
    step1 = DummyStep1Sampler()
    step2 = DummyStep2Grounding()
    step3 = DummyStep3FutureDetection()
    
    # Assemble the pipeline.
    pipeline = DatasetGenerationPipeline(step1, step2, step3)
    
    # Example usage with a dummy video file path.
    example_video_path = Path("/path/to/example_video.mp4")
    results = pipeline.generate_pipeline(example_video_path)
    
    print("Step 1 Output:", results["step1"])
    print("Step 2 Output:", results["step2"])
    print("Step 3 Output:", results["step3"])
