#!/bin/bash
# JarvisVLA vLLM Server Startup Script

set -e

echo "🚀 Starting JarvisVLA vLLM Server..."

# Check if model exists locally
MODEL_PATH="./models/JarvisVLA-Qwen2-VL-7B"
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Found local model at $MODEL_PATH"
    MODEL_TO_SERVE="$MODEL_PATH"
else
    echo "📥 Local model not found, will download from Hugging Face..."
    MODEL_TO_SERVE="CraftJarvis/JarvisVLA-Qwen2-VL-7B"
fi

# Set default values
PORT=${PORT:-3000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}

echo "🔧 Configuration:"
echo "  Model: $MODEL_TO_SERVE"
echo "  Port: $PORT"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  GPU: $GPU_ID"
echo ""

echo "🌟 Starting vLLM server..."
echo "📝 Access the API at: http://localhost:$PORT"
echo "📖 API docs at: http://localhost:$PORT/docs"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the vLLM server
CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve \
    "$MODEL_TO_SERVE" \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --served-model-name JarvisVLA-Qwen2-VL-7B
