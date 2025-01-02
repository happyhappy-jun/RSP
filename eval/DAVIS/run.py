import os
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import os
from pprint import pprint

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ckpt = f"checkpoint-{cfg.epoch}.pth"
    eval_dir = f"davis_seg{cfg.epoch}"

    cmd1 = f"CUDA_VISIBLE_DEVICES={cfg.device} python eval/DAVIS/eval_video_segmentation_davis.py \
       --finetune {os.path.join(cfg.artifact, ckpt)} \
       --output_dir {os.path.join(cfg.artifact, eval_dir)} \
       --data_path /data/DAVIS_480_880 \
       --topk 7 --size_mask_neighborhood 30 --n_last_frames 30 \
       --model vit_small"
    os.system(cmd1)

    cmd2 = (f"CUDA_VISIBLE_DEVICES={cfg.device} python eval/DAVIS/davis2017-evaluation/evaluation_method.py \
       --task semi-supervised \
       --results_path {os.path.join(cfg.artifact, eval_dir)} \
       --davis_path /data/DAVIS_480_880")
    os.system(cmd2)

    pprint(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()