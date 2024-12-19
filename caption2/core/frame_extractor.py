import cv2
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from caption2.core.frame_sampler import UniformFrameSampler, PairedFrameSampler
from caption2.core.config import Config

def extract_frames(
    video_paths: List[str],
    output_dir: Path,
    sampler_type: str = "uniform",
    config: Config = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Extract frames using configured sampler"""
    if config is None:
        config = Config()
        
    # Create sampler with seed
    if sampler_type == "uniform":
        sampler = UniformFrameSampler(seed=seed)
        frame_config = config.frame_config['uniform']
    else:
        sampler = PairedFrameSampler(seed=seed, **config.frame_config['paired'])
        frame_config = config.frame_config['paired']
    
    frame_config['seed'] = seed  # Track seed in config
    
    frame_info = {
        'config': frame_config,
        'videos': []
    }
    
    for video_idx, video_path in enumerate(tqdm(video_paths)):
        try:
            # Get video metadata
            video_path = Path(video_path)
            video_name = video_path.stem
            class_label = video_path.parent.name
            # Create class directory first, then video directory
            class_dir = output_dir / class_label
            class_dir.mkdir(exist_ok=True, parents=True)
            video_dir = class_dir / video_name
            video_dir.mkdir(exist_ok=True)
            
            # Sample frames
            frames = sampler.sample_frames(str(video_path))
            frame_paths = []
            
            # Save frames using OpenCV
            cap = cv2.VideoCapture(str(video_path))
            
            for frame_idx, video_frame_idx in enumerate(frames):
                frame_path = video_dir / f"frame_{frame_idx}.jpg"
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    cv2.imwrite(str(frame_path), frame)
                    if frame_path.exists():
                        frame_paths.append(str(frame_path))
                    else:
                        raise FileNotFoundError(f"Failed to save frame to {frame_path}")
                else:
                    raise ValueError(f"Could not read frame {video_frame_idx} from video")
            
            cap.release()
                
            frame_info['videos'].append({
                'video_idx': video_idx,
                'video_path': str(video_path),
                'video_name': video_name,
                'class_label': class_label,
                'frame_indices': frames,
                'frame_paths': frame_paths,
                'sampling_seed': seed
            })
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            continue
            
    return frame_info
