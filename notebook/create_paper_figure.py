import os
import random
import numpy as np
from PIL import Image
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


def create_figure(frames, num_samples=10, frame_size=256, spacing=10):
    """Create figure with evenly sampled frames in one row"""
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    # Calculate total width including spacing
    total_width = (frame_size * num_samples) + (spacing * (num_samples - 1))
    
    # Create new image with white background
    combined_image = Image.new('RGB', (total_width, frame_size), 'white')
    
    for idx in range(num_samples):
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
        
        # Convert to PIL Image and resize
        pil_image = Image.fromarray(cropped)
        pil_image = pil_image.resize((frame_size, frame_size), Image.Resampling.LANCZOS)
        
        # Calculate position to paste
        x_position = idx * (frame_size + spacing)
        combined_image.paste(pil_image, (x_position, 0))
    
    return combined_image
def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Sample 10 different videos
    for i in range(10):
        # Get random video
        video_path, class_name = get_random_video_path(DATA_DIR)
        print(f"\nSample {i+1}/10")
        print(f"Selected class: {class_name}")
        print(f"Selected video: {os.path.basename(video_path)}")
        
        # Load frames
        frames = load_frames(video_path)
        print(f"Loaded {len(frames)} frames")
        
        # Create and save figure
        combined_image = create_figure(frames)
        output_path = f"paper_figure_{i+1}_{class_name}.png"
        combined_image.save(output_path, quality=95)
        print(f"Saved figure to {output_path}")
    
    print("\nDone! Please check the generated figures and choose the best one.")

if __name__ == "__main__":
    main()
