#!/bin/bash
#SBATCH --job-name=rsp-cosine
#SBATCH --output=/shared/sjw_alinlab/RSP/slurm-logs/test_%j.out  # log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gpus=4
#SBATCH --partition=batch
#SBATCH --time=72:00:00      # 최대 48시간 실행

source /slurm_home/byungjun_alinlab/miniconda3/bin/activate
conda activate rsp2

echo $(which python)

cd /shared/sjw_alinlab/RSP
torchrun --nproc_per_node=4 main_pretrain.py \
    -cn joint_emb_self model_params.kl_scale=0.00001 model_params.cos_scale=0.001 dataset.data_dir=/shared/sjw_alinlab/RSP/npy_dataset 