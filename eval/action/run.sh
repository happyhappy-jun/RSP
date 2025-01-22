#!/bin/bash

# Kill existing session if it exists
if tmux has-session -t "training" 2>/dev/null; then
    echo "Killing existing training session..."
    tmux kill-session -t "training"
fi

# Define tasks as arrays of dataset and model pairs
declare -a tasks=(
    "ssv2 cos_cut_8_001"
    "ssv2 cos_cut_8_imgq_0.005"
    # "imagenet100 cos_cut_8_001"
    # "imagenet100 rsp_cos_proj"
)

# Create a temporary file to track assigned tasks
TASK_TRACKER="/tmp/task_tracker.$$"
printf "%s\n" "${tasks[@]}" > "$TASK_TRACKER"

# Function to get next available task atomically
get_next_task() {
    # Use flock to ensure atomic read and removal
    (
        flock -x 200
        if [ -s "$TASK_TRACKER" ]; then
            head -n 1 "$TASK_TRACKER"
            sed -i '1d' "$TASK_TRACKER"
            return 0
        else
            return 1
        fi
    ) 200>"$TASK_TRACKER.lock"
}

# Create a new tmux session
tmux new-session -d -s "training"

# Create the first split (horizontal)
tmux split-window -v
# Split the top pane (vertical)
tmux select-pane -t 0
tmux split-window -h
# Split the bottom pane (vertical)
tmux select-pane -t 2
tmux split-window -h

# Configure each pane with different ports and GPUs
declare -a gpu_pairs=("0,1" "2,3" "4,5" "6,7")
declare -a ports=(30600 30700 30800 30900)
declare -a pane_numbers=(1 2 3 4)

# Loop through panes
for i in {0..3}; do
    if task=$(get_next_task); then
        if [[ $task =~ ^([^ ]+)[[:space:]]+([^ ]+)$ ]]; then
            dataset="${BASH_REMATCH[1]}"
            model="${BASH_REMATCH[2]}"
            tmux select-pane -t ${pane_numbers[$i]}
            tmux send-keys "export CUDA_VISIBLE_DEVICES=${gpu_pairs[$i]}" C-m
#            tmux send-keys "echo Processing task: $dataset $model on GPUs ${gpu_pairs[$i]}" C-m
            tmux send-keys "torchrun --nproc_per_node=2 --master_port ${ports[$i]} eval/action/main_linprobe.py dataset=$dataset model=$model" C-m
        else
            echo "Error: Invalid task format: $task"
        fi
    fi
done

# Clean up temporary files
rm -f "$TASK_TRACKER" "$TASK_TRACKER.lock"

# Even resize
tmux select-layout tiled

# Attach to the tmux session
tmux attach-session -t "training"