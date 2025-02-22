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
            "When describing objects, use descriptive language, so that agent can identify and distinguish object only with text descriptions.\n"
            "Provide the result in the following format:\n"
            "    - Separate objects with semi-colon ; \n"
            "Example1:\n"
            "<SceneChange>a kid with yellow hat is riding bike moving forward</SceneChange>\n"
            "<Objects>a kid with yellow hat; bike</Objects>"
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
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.box_threshold = 0.3
        self.text_threshold = 0.2

    def detect_bounding_boxes(self, frame_path: Path, caption: str) -> Step2Output:
        from PIL import Image
        import numpy as np
        import cv2
        import torch

        img = Image.open(frame_path).convert("RGB")
        objects = [o.strip() for o in caption.split(",")]
        print(objects)
        inputs = self.processor(images=img, text=[objects], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]]
        )

        # Convert PIL image to numpy array for annotation
        image_np = np.array(img)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        detections = []
        if len(results) > 0:
            res = results[0]
            boxes = res["boxes"]
            scores = res["scores"]
            text_labels = res["text_labels"]
            for box, score, text_label in zip(boxes, scores, text_labels):
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, f"{text_label[:15]}: {score:.2f}", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                bbox = BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1))
                detection = Step2Detection(bounding_box=bbox, logit=score.item(), phrase=text_label)
                detections.append(detection)

        output_path = "annotated_" + frame_path.stem + ".jpg"
        cv2.imwrite(output_path, image_np)
        return Step2Output(detections=detections)

class DummyStep3FutureDetection(Step3FutureDetection):
    def compare_detections(self, detections1: List[Step2Detection], detections2: List[Step2Detection]) -> Step3Output:
        threshold = 0.03
        common_bboxes = []
        movement_captions = []
        for det1 in detections1:
            for det2 in detections2:
                if det1.phrase == det2.phrase:
                    bbox1 = det1.bounding_box
                    bbox2 = det2.bounding_box
                    delta_x = bbox2.x - bbox1.x
                    delta_y = bbox2.y - bbox1.y
                    norm_dx = delta_x / bbox1.width
                    norm_dy = delta_y / bbox1.height
                    horiz_direction = ""
                    vert_direction = ""
                    if abs(norm_dx) >= threshold:
                        horiz_direction = "right" if norm_dx > 0 else "left"
                    if abs(norm_dy) >= threshold:
                        vert_direction = "down" if norm_dy > 0 else "up"
                    if horiz_direction or vert_direction:
                        movement = f"move {vert_direction} {horiz_direction}".strip()
                        movement = " ".join(movement.split())
                    else:
                        movement = "no significant movement"
                    caption = f"{det1.phrase} {movement}"
                    common_bboxes.append(bbox2)
                    movement_captions.append(caption)
        return Step3Output(closed_bboxes=common_bboxes, movement_captions=movement_captions)

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
