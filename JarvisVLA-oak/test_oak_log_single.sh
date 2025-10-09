#!/bin/bash

# Test script for single oak log evaluation
# Make sure to update base_url and model_local_path before running

base_url="http://localhost:3000/v1"  # VLLM server URL
workers=1
max_frames=500
temperature=0.6
history_num=2
action_chunk_len=1
instruction_type="normal"
#model_local_path="jarvis_vla_qwen2_vl_7b_sft"  # Update this to your model name
model_local_path="CraftJarvis/JarvisVLA-Qwen2-VL-7B"

env_config="mine/mine_oak_log"

echo "Testing oak log gathering with 1 worker..."

cd /home/minjune/JarvisVLA-oak

# Run with verbose output to see what's happening
python3 jarvisvla/evaluate/evaluate.py \
    --workers 0 \
    --env-config $env_config \
    --max-frames $max_frames \
    --temperature $temperature \
    --checkpoints $model_local_path \
    --video-main-fold "logs/" \
    --base-url "$base_url" \
    --history-num $history_num \
    --instruction-type $instruction_type \
    --action-chunk-len $action_chunk_len \
    --verbos True

echo "Test completed!"
