#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

# export MODEL_NAME="/dscilab_hiennx/workspace/llm-serving/lmdeploy-serving/InternVL3-8B-hf-traffic-accident-chat-4bit/"
export MODEL_NAME="OpenGVLab/InternVL3-8B-AWQ"

# Enable parallel processing for tokenizers to improve performance
export TOKENIZERS_PARALLELISM="true"

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES="0"

# Optimized CUDA memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Use v1 API for vLLM (more efficient for single requests)
export VLLM_USE_V1=1

# no CPU oversubscription
export OMP_NUM_THREADS=1                    

# Optimized vLLM server configuration for single-image inference
vllm serve $MODEL_NAME \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 \
    --port 8003 \
    --max-num-seqs 2 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    --dtype half \
    --quantization awq \
    --enable-prefix-caching \
    --disable-mm-preprocessor-cache \
    --limit-mm-per-prompt '{"images": 1}'


# --block-size 32 \
# --swap-space 2 \
# --cpu-offload-gb 0 \
# --disable-log-stats \
# --disable-fastapi-docs \
# --max-seq-len-to-capture 4096 \
# --scheduler-delay-factor 0.0 \
# --num-scheduler-steps 1 \
# --compilation-config '{"level": 3, "cudagraph_capture_sizes": [1, 2]}' \
# --generation-config vllm \
# --override-generation-config '{"temperature": 0.0, "top_p": 1.0, "max_tokens": 50}'
