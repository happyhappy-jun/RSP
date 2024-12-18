import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Directory containing video frames
DATA_DIR = "data/"  # Adjust this path as needed

def get_random_video_dir(data_dir):
    """Get a random video directory from the data folder"""
    video_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    return os.path.join(data_dir, random.choice(video_dirs))

def load_frames(video_dir):
    """Load all frames from a video directory"""
    frames = []
    frame_files = sorted([f for f in os.listdir(video_dir) 
                         if f.endswith(('.jpg', '.png'))])
    
    for frame_file in frame_files:
        img_path = os.path.join(video_dir, frame_file)
        img = Image.open(img_path)
        frames.append(np.array(img))
    
    return frames

def create_figure(frames, num_samples=10):
    """Create figure with evenly sampled frames"""
    total_frames = len(frames)
    indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        ax.imshow(frames[indices[idx]])
        ax.axis('off')
        ax.set_title(f'Frame {indices[idx]}')
    
    plt.tight_layout()
    return fig

def main():
    # Get random video
    video_dir = get_random_video_dir(DATA_DIR)
    print(f"Selected video: {os.path.basename(video_dir)}")
    
    # Load frames
    frames = load_frames(video_dir)
    print(f"Loaded {len(frames)} frames")
    
    # Create and save figure
    fig = create_figure(frames)
    output_path = "paper_figure.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
