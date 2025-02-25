# Example Implementation for the DatasetGenerationPipeline.
# This file provides dummy implementations for each pipeline step.
from dataclasses import asdict
from pathlib import Path
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
    def sample_frame_and_generate_caption(self, video_path: Path) -> Step1Output:
        import random
        import decord
        import cv2
        import base64
        from openai import OpenAI

        client = OpenAI()

        # Read the video and select a random frame using decord.
        vr = decord.VideoReader(str(video_path))
        num_frames = len(vr)
        random_index = random.randint(0, num_frames - 1)
        frame = vr[random_index].asnumpy()

        # Encode the image to Base64 and save frame to a temporary file.
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("Failed to encode the frame to JPEG format.")
        tmp_image_path = Path("tmp_frame.jpg")
        cv2.imwrite(str(tmp_image_path), frame)
        frame_path = tmp_image_path
        image_bytes = encoded_image.tobytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{base64_image}"

        # Prepare the prompt for describing the scene.
        prompt_text = (
            "Describe the main action happening in this scene, focusing on the primary moving subject.\n"
            "Separate objects with space-wrapped period\" . \"\n"
            "Add \" .\" and the end of detection"
            "Response in following format:\n"
            "<Scene>a kid with yellow hat is riding bike in a park</Scene>"
            "<Objects>a red bicycle . a kid with yellow hat .</Objects>"

                       )
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data, "detail": "high"}}
            ]
        }

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[message],
                max_tokens=300
            )
            generated_text = response.choices[0].message.content.strip()
        except Exception as e:
            generated_text = f"Error generating caption: {str(e)}"

        # Parse the response format with <Scene> and <Objects> tags
        try:
            scene_start = generated_text.index("<Scene>")
            scene_end = generated_text.index("</Scene>")
            objects_start = generated_text.index("<Objects>")
            objects_end = generated_text.index("</Objects>")
            scene = generated_text[scene_start + len("<Scene>"):scene_end].strip()
            objects = generated_text[objects_start + len("<Objects>"):objects_end].strip()
        except Exception as e:
            scene = "No scene info"
            objects = "No objects info"

        return Step1Output(
            frame_path=frame_path,
            scene=scene,
            objects=objects
        )

class DummyStep2Grounding(Step2Grounding):
    def __init__(self):
        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "GroundingDINO/weights/groundingdino_swint_ogc.pth"
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
