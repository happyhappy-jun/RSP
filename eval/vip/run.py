import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import subprocess
from typing import Optional

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function for VIP dataset using Hydra configuration
    """
    # Get the current working directory and project root
    cwd = os.getcwd()
    project_root = hydra.utils.get_original_cwd()
    
    # Construct checkpoint path (using absolute path)
    ckpt_path = os.path.abspath(os.path.join(project_root, cfg.artifact, f"checkpoint-{cfg.epochs}.pth"))
    
    # Construct save path with all parameters
    save_dir = os.path.abspath(os.path.join(
        project_root, cfg.artifact,
        f"vip_560_560_seg_{cfg.epochs}"
    ))
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # First command: VIP video segmentation
    cmd1 = [
        "python",
        os.path.join(project_root, "eval/vip/main_video_segmentation_vip.py"),
        "--finetune", ckpt_path,
        "--output_dir", save_dir,
        "--data_path", cfg.paths.data_path,
        "--topk", str(cfg.eval.topk),
        "--size_mask_neighborhood", str(cfg.eval.neighbor),
        "--n_last_frames", str(cfg.eval.queue),
        "--model", f"{cfg.model}"
    ]

    # Second command: ATEN evaluation
    cmd2 = [
        "python",
        os.path.join(project_root, "eval/vip/ATEN/evaluate/test_parsing_ours.py")
        "--pre_dir", save_dir
    ]

    # Execute commands
    print("Running VIP video segmentation...")
    print(f"Command: {' '.join(cmd1)}")
    subprocess.run(cmd1, env=env, check=True)

    print("\nRunning ATEN evaluation...")
    print(f"Command: {' '.join(cmd2)}")
    subprocess.run(cmd2, env=env, check=True)

    print("\nRunning DAVIS evaluation...")
    print(f"Command: {' '.join(cmd3)}")
    subprocess.run(cmd3, env=env, check=True)

    # Print configuration
    print("\nConfiguration used:")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()