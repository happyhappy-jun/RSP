# Example Implementation for the DatasetGenerationPipeline.
# This file provides dummy implementations for each pipeline step.

from pathlib import Path
from llm.base_pipeline import (
    Step1Sampler,
    Step2Grounding,
    Step3FutureDetection,
    Step1Output,
    Step2Output,
    Step3Output,
    BoundingBox,
    DatasetGenerationPipeline
)
import openai

class GPT4OMiniStep1Sampler(Step1Sampler):
    def sample_frame_and_generate_caption(self, video_path: Path) -> Step1Output:
        import random
        import decord
        import cv2
        import base64

        # Read the video and select a random frame using decord.
        vr = decord.VideoReader(str(video_path))
        num_frames = len(vr)
        random_index = random.randint(0, num_frames - 1)
        frame = vr[random_index].asnumpy()

        # Save the extracted frame to a temporary file.
        temp_frame_path = "/tmp/extracted_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        frame_path = Path(temp_frame_path)

        # Encode the image to Base64.
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("Failed to encode the frame to JPEG format.")
        image_bytes = encoded_image.tobytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{base64_image}"

        # Prepare the prompt for describing the scene.
        prompt_text = "Briefly describe the scene depicted in the image, focusing on spatial relationships among objects."
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data, "detail": "high"}}
            ]
        }

        try:
            response = openai.ChatCompletion.create(
                model="gpt4o-mini",
                messages=[message],
                max_tokens=300
            )
            generated_text = response.choices[0].message.content.strip()
        except Exception as e:
            generated_text = f"Error generating caption: {str(e)}"

        # Assume the response format is: "Caption: <caption>\nChain-of-thought: <reasoning>"
        parts = generated_text.split("\nChain-of-thought:")
        if len(parts) == 2:
            caption = parts[0].replace("Caption:", "").strip()
            chain_of_thought = parts[1].strip()
        else:
            caption = generated_text
            chain_of_thought = ""

        return Step1Output(
            frame_path=frame_path,
            caption=caption,
            chain_of_thought=chain_of_thought
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
    # Create instances of each step.
    step1 = GPT4OMiniStep1Sampler()
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
