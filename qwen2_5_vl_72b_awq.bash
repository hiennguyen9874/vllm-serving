#!/bin/bash

# Enable debug mode to print each command before execution
set -x
# Exit immediately if a command exits with a non-zero status
set -e

# Activate Python virtual environment
source .venv/bin/activate

# Enable parallel processing for tokenizers to improve performance
export TOKENIZERS_PARALLELISM="true"
# Specify which GPUs to use (GPUs 3 and 5 in this case)
export CUDA_VISIBLE_DEVICES="2,3,5,6"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start the vLLM server with Qwen2.5-VL model with the following configuration:
vllm serve PointerHQ/Qwen2.5-VL-72B-Instruct-Pointer-AWQ \
    --trust-remote-code \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.98 \
    --host 0.0.0.0 \
    --port 8003 \
    --limit-mm-per-prompt image=1,video=0 \
    --max-num-seqs 4 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-chunked-prefill \
    --enforce-eager

# TODO: --kv-cache-dtype fp8, fp8_e5m2

# Parameter explanations:
# --trust-remote-code             : Allow execution of custom code from the model repository for model-specific operations
# --quantization awq_marlin       : Use Activation-aware Weight Quantization (AWQ) with Marlin optimizations to reduce model size and memory footprint
# --dtype half                    : Use FP16 (half precision) for model weights to balance accuracy and memory usage
# --kv-cache-dtype fp8            : Store key-value cache in 8-bit floating point to further reduce memory usage during inference
# --tensor-parallel-size 2        : Distribute model computation across 2 GPUs for parallel processing
# --gpu-memory-utilization 0.95   : Allocate up to 95% of available GPU memory for model operations
# --max-model-len 16384           : Maximum context length the model can process (16K tokens)
# --max-num-batched-tokens 512    : Maximum number of tokens to process in a single batch for efficient GPU utilization
# --max-num-seqs 1                : Process only one sequence/request at a time to maintain quality
# --host 0.0.0.0                  : Bind server to all network interfaces (accessible from external machines)
# --port 8003                     : Listen for incoming requests on port 8003
# --enable-chunked-prefill        : Process long inputs in chunks to avoid OOM errors with lengthy contexts
# --limit-mm-per-prompt image=1,video=0 : Restrict multimodal inputs to at most 1 image and 0 videos per prompt
# --enforce-eager                 : Use PyTorch's eager execution mode instead of CUDA graphs for better compatibility
