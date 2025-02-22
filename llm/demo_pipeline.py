#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from dataclasses import asdict

from pipeline import GPT4OMiniStep1Sampler

def load_trajectories(dataset_root: str) -> dict:
    """
    Traverse the dataset directory and find all trajectories with images.
    Returns a dictionary mapping trajectory folder (as string) to a list of image file paths (Path).
    """
    traj_dict = {}
    pattern = os.path.join(dataset_root, "*", "*", "raw", "traj_group0", "traj0", "images0")
    for images_dir in glob.glob(pattern):
        images_path = Path(images_dir)
        # The trajectory folder is the parent directory of the images folder.
        traj_folder = images_path.parent
        images = sorted(list(images_path.glob("*.jpg")))
        if images:
            traj_dict[str(traj_folder)] = images
    return traj_dict

def main():
    dataset_root = "/shared/common/bridge_dataset/raw/rss/toykitchen2/set_table"
    trajectories = load_trajectories(dataset_root)
    print("Found trajectories:")
    for traj, images in trajectories.items():
        print(f"{traj}: {len(images)} images")
    
    # For demo, pick one trajectory that has at least 2 images.
    selected_images = None
    for images in trajectories.values():
        if len(images) >= 2:
            selected_images = images
            break
    if not selected_images:
        raise ValueError("No trajectory with at least 2 images found.")
    
    # Sample two images from the selected trajectory with random distance.
    import random
    idx1 = random.randint(0, len(selected_images) - 2)
    idx2 = random.randint(idx1 + 1, len(selected_images) - 1)
    demo_images = [selected_images[idx1], selected_images[idx2]]
    print("Using demo images:")
    for img in demo_images:
        print(img)
    
    # Instantiate the pipeline step1 sampler and process the two images.
    sampler = GPT4OMiniStep1Sampler()
    output = sampler.sample_frame_and_generate_caption(demo_images)
    print("Pipeline Step1 output:")
    print(asdict(output))
    
    # Run step2 detection on both images and visualize outputs.
    from pipeline import DummyStep2Grounding
    import matplotlib.pyplot as plt
    import cv2
    step2 = DummyStep2Grounding()
    orig_images = []
    annotated_images = []
    detections_outputs = []
    for img in demo_images:
        # Load original image
        orig = cv2.imread(str(img))
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_images.append(orig)
        # Run step2 on the image and store detection output.
        det_out = step2.detect_bounding_boxes(img, output.objects)
        detections_outputs.append(det_out)
        annotated_path = "annotated_" + img.stem + ".jpg"
        ann_img = cv2.imread(annotated_path)
        if ann_img is not None:
            ann_img = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
        annotated_images.append(ann_img)
    
    # Compare detections from the two demo images using step3's new method.
    from pipeline import DummyStep3FutureDetection
    step3 = DummyStep3FutureDetection()
    print("Image1 Detection output:", detections_outputs[0].detections)
    print("Image2 Detection output:", detections_outputs[1].detections)
    step3_output = step3.compare_detections(detections_outputs[0].detections, detections_outputs[1].detections)
    print("Step3 Movement Captions:")
    for cap in step3_output.movement_captions:
        print(cap)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(orig_images[0])
    axs[0, 0].set_title("Original Image 1")
    axs[0, 0].axis("off")
    axs[0, 1].imshow(orig_images[1])
    axs[0, 1].set_title("Original Image 2")
    axs[0, 1].axis("off")
    axs[1, 0].imshow(annotated_images[0])
    axs[1, 0].set_title("Annotated Image 1")
    axs[1, 0].axis("off")
    axs[1, 1].imshow(annotated_images[1])
    axs[1, 1].set_title("Annotated Image 2")
    axs[1, 1].axis("off")
    plt.figtext(0.5, 0.98, f"Caption: {output.scene}", wrap=True, horizontalalignment='center', fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.94, f"Objects: {output.objects}\nMovements: {'; '.join(step3_output.movement_captions)}", wrap=True, horizontalalignment='center', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig("output.png")

if __name__ == '__main__':
    main()
