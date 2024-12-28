#!/bin/bash
#SBATCH --job-name=rsp-cosine
#SBATCH --output=/home/junyoon/slurm-logs/test-%j.out  # log
#SBATCH --error=/home/junyoon/slurm-logs/test-%j.err   # log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gres=gpu:a6000:8   # GPU 1개 사용
#SBATCH --cpus-per-task=2    # GPU당 CPU 사용 수
#SBATCH --mem-per-gpu=8G     # GPU당 mem 사용량
#SBATCH --time=72:00:00      # 최대 48시간 실행


