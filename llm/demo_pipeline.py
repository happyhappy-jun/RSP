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
    pattern = os.path.join(dataset_root, "*", "*", "*", "*", "*", "raw", "traj_group0", "*", "images0")
    for images_dir in glob.glob(pattern):
        images_path = Path(images_dir)
        # The trajectory folder is the parent directory of the images folder.
        traj_folder = images_path.parent
        images = sorted(list(images_path.glob("*.jpg")))
        if images:
            traj_dict[str(traj_folder)] = images
    return traj_dict


def main():
    import os
    import random
    import cv2
    import matplotlib.pyplot as plt
    from pipeline import GPT4OMiniStep1Sampler, DummyStep2Grounding, DummyStep3FutureDetection
    from dataclasses import asdict

    # Use a broader dataset root that lists all trajectories.
    dataset_root = "/shared/common/bridge_dataset/raw"
    trajectories = load_trajectories(dataset_root)
    print("Found trajectories:")
    for traj, images in trajectories.items():
        print(f"{traj}: {len(images)} images")

    # Filter trajectories that have at least 9 images (to allow 8-step difference).
    valid_traj = {k: v for k, v in trajectories.items() if len(v) >= 9}
    print(f"Found {len(valid_traj)} trajectories with enough images")

    # Randomly sample 20 trajectories (or all if less than 20).
    traj_keys = list(valid_traj.keys())
    num_traj = min(20, len(traj_keys))
    sampled_keys = random.sample(traj_keys, num_traj)

    # Instantiate pipeline components.
    sampler = GPT4OMiniStep1Sampler()
    step2 = DummyStep2Grounding()
    step3 = DummyStep3FutureDetection()

    for traj_key in sampled_keys:
        images = valid_traj[traj_key]
        images = sorted(images)
        # Sample two images with at least 8 steps difference.
        i = random.randint(0, len(images) - 9)
        j = i + 8
        demo_images = [images[i], images[j]]
        print(f"\nProcessing trajectory: {traj_key}")
        print(f"Selected images: {demo_images[0]}, {demo_images[1]}")

        # Run step1: generate caption for the two images.
        output = sampler.sample_frame_and_generate_caption(demo_images)
        print("Step1 output:")
        print(asdict(output))

        # Run step2 on each image.
        orig_images = []
        annotated_images = []
        detections_outputs = []
        for img in demo_images:
            orig = cv2.imread(str(img))
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig_images.append(orig)
            det_out = step2.detect_bounding_boxes(img, output.objects)
            detections_outputs.append(det_out)
            annotated_path = "annotated_" + img.stem + ".jpg"
            ann_img = cv2.imread(annotated_path)
            if ann_img is not None:
                ann_img = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
            annotated_images.append(ann_img)

        # Run step3: compare detections between the two images.
        step3_output = step3.compare_detections(detections_outputs[0].detections, detections_outputs[1].detections)
        print("Step3 Movement Captions:")
        for cap in step3_output.movement_captions:
            print(cap)

        # Visualize results.
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
        plt.figtext(0.5, 0.98, f"Trajectory: {traj_key}", wrap=True, horizontalalignment='center', fontsize=16, fontweight='bold')
        plt.figtext(0.5, 0.02, f"Caption: {output.scene}\nObjects: {output.objects}\nMovements: {'; '.join(step3_output.movement_captions)}", wrap=True, horizontalalignment='center', fontsize=14)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        out_filename = f"output_{traj_key.replace(os.sep, '_')}.png"
        plt.savefig(out_filename)
        plt.close(fig)


if __name__ == '__main__':
    main()
