#!/bin/bash

# ========================================================================================
# NVIDIA Nemotron Nano 12B v2 - Deployment Script for vLLM
# ========================================================================================
# ========================================================================================

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

# ========================================================================================
# Model Configuration
# ========================================================================================

# Model name
export MODEL_NAME="nvidia/NVIDIA-Nemotron-Nano-12B-v2"

# Use the original model for tokenizer (recommended by vLLM)
export TOKENIZER_MODEL="nvidia/NVIDIA-Nemotron-Nano-12B-v2"

# Optional: HuggingFace config path if needed
export HF_CONFIG_PATH="nvidia/NVIDIA-Nemotron-Nano-12B-v2"

# ========================================================================================
# Environment Variables
# ========================================================================================

# Cuda architecture for a100, https://developer.nvidia.com/cuda-gpus
export TORCH_CUDA_ARCH_LIST="8.0"

# Enable parallel processing for tokenizers
export TOKENIZERS_PARALLELISM="true"

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"1,2,3,4"}

export DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-4}

# Optimized CUDA memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

export VLLM_ATTENTION_BACKEND=FLASHINFER

# Use v1 API for vLLM
export VLLM_USE_V1=1

# Prevent CPU oversubscription
export OMP_NUM_THREADS=1

# ========================================================================================
# vLLM Server Launch
# ========================================================================================

echo "Starting vLLM server with model..."

vllm serve "$MODEL_NAME" \
    --tokenizer "$TOKENIZER_MODEL" \
    --hf-config-path "$HF_CONFIG_PATH" \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8080 \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --enforce-eager \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-stats

# ========================================================================================
# Parameter Explanations for Deployment
# ========================================================================================
#
# --tokenizer $TOKENIZER_MODEL    : Use original model's tokenizer (recommended)
# --hf-config-path $HF_CONFIG_PATH: Provide HF config if automatic detection fails
# --max-num-seqs 16               : Reduced from 64 due to experimental limitations
# --gpu-memory-utilization 0.90   : Conservative memory usage for
# --enforce-eager                 : Required for stability with experimental features
# --dtype auto                    : Let vLLM decide the best dtype for
# --disable-log-stats             : Reduce overhead in experimental mode
#
# ========================================================================================

echo ""
echo "vLLM server should now be running on http://localhost:8003"
echo ""
echo "Test with:"
echo "curl -X POST http://localhost:8003/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"model\": \"nemotron-nano-12b-v2\","
echo "    \"messages\": ["
echo "      {\"role\": \"user\", \"content\": \"Hello, how are you?\"}"
echo "    ],"
echo "    \"max_tokens\": 100"
echo "  }'"
echo ""
