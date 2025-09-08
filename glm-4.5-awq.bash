#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

export MODEL_NAME="cpatonn/GLM-4.5-Air-AWQ-4bit"

# Enable parallel processing for tokenizers to improve performance
export TOKENIZERS_PARALLELISM="true"

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES="4,5"

# Cuda architecture for a100, https://developer.nvidia.com/cuda-gpus
export TORCH_CUDA_ARCH_LIST="8.0"

# Optimized CUDA memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Use v1 API for vLLM (more efficient for single requests)
export VLLM_USE_V1=1

# no CPU oversubscription
export OMP_NUM_THREADS=1                    

# Optimized vLLM server configuration for single-image inference
vllm serve $MODEL_NAME \
    --trust-remote-code \
    --gpu-memory-utilization 0.98 \
    --host 0.0.0.0 \
    --port 8003 \
    --tensor-parallel-size 2 \
    --enable-chunked-prefill \
    --dtype half \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"images": 1}' \
    --tool-call-parser glm45 \
    --cpu-offload-gb 16 \
    --enable-auto-tool-choice \
    --chat-template glm-4.5-nothink.jinja \
    --max-model-len 81920 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --max-num-seqs 2 \
    --served-model-name "GLM-4.5-Air"

# --block-size 32 \
# --swap-space 2 \
# --cpu-offload-gb 0 \
# --disable-log-requests \
# --disable-log-stats \
# --disable-fastapi-docs \
# --max-seq-len-to-capture 4096 \
# --scheduler-delay-factor 0.0 \
# --num-scheduler-steps 1 \
# --compilation-config '{"level": 3, "cudagraph_capture_sizes": [1, 2]}' \
# --generation-config vllm \
# --override-generation-config '{"temperature": 0.0, "top_p": 1.0, "max_tokens": 50}'
