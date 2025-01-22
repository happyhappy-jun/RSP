import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import subprocess
from typing import Optional

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function for JHMDB dataset using Hydra configuration
    """
    # Get the current working directory and project root
    cwd = os.getcwd()
    project_root = hydra.utils.get_original_cwd()
    
    # Construct checkpoint path (using absolute path)
    ckpt_path = os.path.abspath(os.path.join(project_root, cfg.artifact, f"checkpoint-{cfg.epochs}.pth"))
    
    # Construct save path with all parameters
    save_dir = os.path.abspath(os.path.join(project_root, cfg.artifact, f"jhmdb_seg{cfg.epochs}"))
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # First command: JHMDB evaluation (using absolute paths)
    cmd1 = [
        "python",
        os.path.join(project_root, "eval/jhmdb/jhmdb_evaluation.py"),  # Update path to script
        "--resume", ckpt_path,
        "--save-path", save_dir,
        "--root", cfg.paths.data_root,
        "--filelist", cfg.paths.val_list,
        "--topk", str(cfg.eval.topk),
        "--radius", str(cfg.eval.neighbor),
        "--videoLen", str(cfg.eval.queue)
    ]

    # Second command: PCK evaluation
    cmd2 = [
        "python",
        os.path.join(project_root, "eval/jhmdb/jhmdb_evaluation/eval_pck.py"),  # Update path to script
        "--src-folder", save_dir,
        "--root", cfg.paths.data_root,
        "--filelist", cfg.paths.val_list
    ]

    # Execute commands
    print("Running JHMDB evaluation...")
    print(f"Command: {' '.join(cmd1)}")  # Print command for debugging
    subprocess.run(cmd1, env=env, check=True)

    print("\nRunning PCK evaluation...")
    print(f"Command: {' '.join(cmd2)}")  # Print command for debugging
    subprocess.run(cmd2, env=env, check=True)

    # Print configuration
    print("\nConfiguration used:")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()