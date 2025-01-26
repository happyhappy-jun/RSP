#!/bin/bash

#SBATCH --job-name=rsp-linprobe
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00

# Version 1: Using command line arguments
while getopts "d:m:" opt; do
    case $opt in
        d) DATASET=$OPTARG ;;
        m) MODEL=$OPTARG ;;
        *) echo "Usage: $0 -d dataset -m model" && exit 1 ;;
    esac
done

# Correct way to assign command output to PORT
PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf 2>/dev/null | head -n 1)

cd /mnt/nas/slurm_account/junyoon/RSP

source /mnt/nas/slurm_account/junyoon/.bashrc
# Use source instead of shell command for pipenv
pipenv shell

# Run training
torchrun \
    --nproc_per_node=4 \
    --master_port=$PORT \
    eval/action/main_linprobe.py \
    dataset=${DATASET} \
    model=${MODEL} \
    dataset.data_root=/mnt/nas/slurm_account/junyoon/data/ImageNet100 \
    finetune=/mnt/nas/slurm_account/junyoon/RSP/outputs/rsp_m3ae_2025-01-24_14-52-52/checkpoint-199.pth \
    batch_size=256