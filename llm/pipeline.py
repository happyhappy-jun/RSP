# Example Implementation for the DatasetGenerationPipeline.
# This file provides dummy implementations for each pipeline step.
from dataclasses import asdict
from pathlib import Path
from typing import List
from pprint import pprint

from base_pipeline import (
    Step1Sampler,
    Step2Grounding,
    Step3FutureDetection,
    Step1Output,
    Step2Output,
    Step2Detection,
    Step3Output,
    BoundingBox,
    DatasetGenerationPipeline
)
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

class GPT4OMiniStep1Sampler(Step1Sampler):
    def sample_frame_and_generate_caption(self, image_paths: List[Path]) -> Step1Output:
        import cv2
        import base64
        from openai import OpenAI

        if len(image_paths) != 2:
            raise ValueError("Expected exactly 2 images")

        # Read first image
        img1 = cv2.imread(str(image_paths[0]))
        if img1 is None:
            raise ValueError("Failed to read the first image")
        success, encoded_image1 = cv2.imencode('.jpg', img1)
        if not success:
            raise ValueError("Failed to encode the first image")
        base64_image1 = base64.b64encode(encoded_image1.tobytes()).decode("utf-8")
        image_data1 = f"data:image/jpeg;base64,{base64_image1}"

        # Read second image
        img2 = cv2.imread(str(image_paths[1]))
        if img2 is None:
            raise ValueError("Failed to read the second image")
        success, encoded_image2 = cv2.imencode('.jpg', img2)
        if not success:
            raise ValueError("Failed to encode the second image")
        base64_image2 = base64.b64encode(encoded_image2.tobytes()).decode("utf-8")
        image_data2 = f"data:image/jpeg;base64,{base64_image2}"

        prompt_text = (
            "Describe the spatial change of the primary object between the two images provided.\n"
            "Mention differences in position, size, or orientation.\n"
            "Provide the result in the following format:\n"
            "    - Separate objects with space-wrapped period\" . \"\n"
            "    - Add \" .\" and the end of detection\n"
            "Example1:\n"
            "<SceneChange>a kid with yellow hat is riding bike moving forward</SceneChange>\n"
            "<Objects>a kid with yellow hat . bike .</Objects>"
        )
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data1, "detail": "high"}},
                {"type": "image_url", "image_url": {"url": image_data2, "detail": "high"}}
            ]
        }

        client = OpenAI()

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[message],
                max_tokens=300
            )
            generated_text = response.choices[0].message.content.strip()
        except Exception as e:
            generated_text = f"Error generating caption: {str(e)}"

        try:
            scene_start = generated_text.index("<SceneChange>")
            scene_end = generated_text.index("</SceneChange>")
            objects_start = generated_text.index("<Objects>")
            objects_end = generated_text.index("</Objects>")
            scene = generated_text[scene_start + len("<SceneChange>"):scene_end].strip()
            objects = generated_text[objects_start + len("<Objects>"):objects_end].strip()
        except Exception as e:
            scene = "No scene change info"
            objects = "No objects info"

        return Step1Output(
            frame_path=image_paths[0],
            scene=scene,
            objects=objects
        )

class DummyStep2Grounding(Step2Grounding):
    def __init__(self):
        self.model = load_model(
            "/slurm_home/byungjun/RSP/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "/slurm_home/byungjun/RSP/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        )
        self.box_threshold = 0.3
        self.text_threshold = 0.2

    def detect_bounding_boxes(self, frame_path: Path, caption: str) -> Step2Output:
        image_source, image = load_image(str(frame_path))
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=caption ,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        output_path = "annotated_" + frame_path.stem + ".jpg"
        cv2.imwrite(output_path, annotated_frame)
        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            x1, y1, x2, y2 = box
            bbox = BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1))
            detection = Step2Detection(bounding_box=bbox, logit=logit, phrase=phrase)
            detections.append(detection)
        return Step2Output(detections=detections)

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
    example_video_path = Path("/data/kinetics400/cheerleading/zYbQPUCn8Bw_000003_000013.mp4")
    results = pipeline.generate_pipeline(example_video_path)
    
    pprint(asdict(results["step1"]))
    pprint(asdict(results["step2"]))
    pprint(asdict(results["step3"]))
