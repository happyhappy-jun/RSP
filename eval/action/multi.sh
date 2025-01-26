#!/bin/bash

# Define tasks
declare -a DATASETS=("imagenet100")
declare -a MODELS=("cos_cut_8_001" "cos_cut_8_imgq_0.005")

# Submit each job
for i in "${!DATASETS[@]}"; do
    sbatch train.sh -d "${DATASETS[$i]}" -m "${MODELS[$i]}"
done