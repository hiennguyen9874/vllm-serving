#!/bin/bash

# ========================================================================================
# Qwen3-30B-A3B-Thinking-2507 - GGUF Deployment Script for vLLM
# ========================================================================================
# 
# WARNING: GGUF support in vLLM is highly experimental and under-optimized!
# This script is provided for educational purposes and testing only.
# For production use, please use the native format script instead.
#
# Prerequisites:
# 1. Download a GGUF version of the model (not officially available yet)
# 2. Ensure vLLM >= 0.10.1 is installed
# 3. Model must be a single-file GGUF (use gguf-split to merge if multi-file)
#
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

# Path to your downloaded GGUF model file
# Replace this with the actual path to your GGUF file
export GGUF_MODEL_PATH="/root/.cache/llama.cpp/BasedBase_Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill_Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill-Q5_K_M.gguf"

# Use the original model for tokenizer (recommended by vLLM)
export TOKENIZER_MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"

# Optional: HuggingFace config path if needed
export HF_CONFIG_PATH="Qwen/Qwen3-30B-A3B-Thinking-2507"

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

# Use v1 API for vLLM
export VLLM_USE_V1=1

# Prevent CPU oversubscription
export OMP_NUM_THREADS=1

# ========================================================================================
# GGUF Model Validation
# ========================================================================================

if [ ! -f "$GGUF_MODEL_PATH" ]; then
    echo "ERROR: GGUF model file not found at: $GGUF_MODEL_PATH"
    echo ""
    echo "To obtain a GGUF version of NVIDIA Nemotron Nano 12B v2:"
    echo "1. Check if community quantizations are available on HuggingFace"
    echo "2. Convert the model yourself using llama.cpp tools"
    echo "3. Use the native format script instead (recommended)"
    echo ""
    exit 1
fi

echo "Found GGUF model at: $GGUF_MODEL_PATH"
echo "Model size: $(du -h "$GGUF_MODEL_PATH" | cut -f1)"

# ========================================================================================
# vLLM Server Launch (GGUF Mode - Experimental)
# ========================================================================================

echo "Starting vLLM server with GGUF model (EXPERIMENTAL)..."
echo "WARNING: This is experimental and may not work optimally!"

vllm serve "$GGUF_MODEL_PATH" \
    --tokenizer "$TOKENIZER_MODEL" \
    --hf-config-path "$HF_CONFIG_PATH" \
    --trust-remote-code \
    --load-format gguf \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8003 \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --enforce-eager \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-stats

# ========================================================================================
# Parameter Explanations for GGUF Deployment
# ========================================================================================
#
# --load-format gguf              : Explicitly specify GGUF format
# --tokenizer $TOKENIZER_MODEL    : Use original model's tokenizer (recommended)
# --hf-config-path $HF_CONFIG_PATH: Provide HF config if automatic detection fails
# --max-num-seqs 16               : Reduced from 64 due to experimental GGUF limitations
# --gpu-memory-utilization 0.90   : Conservative memory usage for GGUF
# --enforce-eager                 : Required for stability with experimental features
# --dtype auto                    : Let vLLM decide the best dtype for GGUF
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
