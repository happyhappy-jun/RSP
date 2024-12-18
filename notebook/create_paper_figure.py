import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Directory containing video files
DATA_DIR = "/home/junyoon/kinetics400/train2"  # Adjust this path as needed

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def get_random_video_path(data_dir):
    """Get a random video path from the data folder"""
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    
    # Select random class
    random_class = random.choice(class_dirs)
    class_path = os.path.join(data_dir, random_class)
    
    # Get all videos in the class
    videos = [f for f in os.listdir(class_path) 
             if f.endswith('.mp4')]
    
    # Select random video
    random_video = random.choice(videos)
    return os.path.join(class_path, random_video), random_class

def load_frames(video_path):
    """Load frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def create_figure(frames, num_samples=10):
    """Create figure with evenly sampled frames in one row"""
    total_frames = len(frames)
    indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(25, 3))
    
    for idx, ax in enumerate(axes):
        frame = frames[indices[idx]]
        h, w = frame.shape[:2]
        
        # Calculate center crop coordinates
        center_x, center_y = w // 2, h // 2
        size = min(h, w)
        half_size = size // 2
        
        # Crop to square from center
        cropped = frame[
            center_y - half_size:center_y + half_size,
            center_x - half_size:center_x + half_size
        ]
        
        ax.imshow(cropped)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Get random video
    video_path, class_name = get_random_video_path(DATA_DIR)
    print(f"Selected class: {class_name}")
    print(f"Selected video: {os.path.basename(video_path)}")
    
    # Load frames
    frames = load_frames(video_path)
    print(f"Loaded {len(frames)} frames")
    
    # Create and save figure
    fig = create_figure(frames)
    output_path = f"paper_figure_{class_name}.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
