#!/bin/bash

# Script to run 100 oak log gathering evaluations
# Make sure to update base_url and model_local_path before running

base_url="http://localhost:3000/v1"  # VLLM server URL
workers=10
max_frames=500
temperature=0.6
history_num=2
action_chunk_len=1
instruction_type="normal"
#model_local_path="jarvis_vla_qwen2_vl_7b_sft"  # Update this to your model name
model_local_path="CraftJarvis/JarvisVLA-Qwen2-VL-7B"

env_config="mine/mine_oak_log"

echo "Running oak log gathering evaluation with $workers workers..."

cd /home/minjune/JarvisVLA-oak

# Run the evaluation
num_iterations=$(($workers / 5 + 1))
for ((i = 0; i < num_iterations; i++)); do
    python3 jarvisvla/evaluate/evaluate.py \
        --workers $workers \
        --env-config $env_config \
        --max-frames $max_frames \
        --temperature $temperature \
        --checkpoints $model_local_path \
        --video-main-fold "logs/" \
        --base-url "$base_url" \
        --history-num $history_num \
        --instruction-type $instruction_type \
        --action-chunk-len $action_chunk_len \
        --split-number 5

    # If Python script executes successfully, exit the loop
    if [[ $? -eq 0 ]]; then
        echo "Iteration $i completed successfully, exiting loop."
        break
    fi

    if [[ $i -lt $((num_iterations - 1)) ]]; then
        echo "Waiting 10 seconds..."
        sleep 10
    fi
done

echo "Evaluation completed!"
echo "Results saved to: logs/$model_local_path-mine_oak_log/"
echo "Check logs/$model_local_path-mine_oak_log/end.json for success rates"
echo "Check logs/$model_local_path-mine_oak_log/image.png for visualization"
