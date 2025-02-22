#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from dataclasses import asdict

from llm.pipeline import GPT4OMiniStep1Sampler

def load_trajectories(dataset_root: str) -> dict:
    """
    Traverse the dataset directory and find all trajectories with images.
    Returns a dictionary mapping trajectory folder (as string) to a list of image file paths (Path).
    """
    traj_dict = {}
    pattern = os.path.join(dataset_root, "*", "*", "raw", "traj_group0", "traj", "*", "images0")
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
    
    # Sample two images from the selected trajectory (here, simply take the first two).
    demo_images = selected_images[:2]
    print("Using demo images:")
    for img in demo_images:
        print(img)
    
    # Instantiate the pipeline step1 sampler and process the two images.
    sampler = GPT4OMiniStep1Sampler()
    output = sampler.sample_frame_and_generate_caption(demo_images)
    print("Pipeline Step1 output:")
    print(asdict(output))

if __name__ == '__main__':
    main()
