#!/usr/bin/env bash
# ========================================================================================
# Nemotron-Nano-12B-v2 on vLLM (V1 engine) - A100 40GB optimized launcher
# ========================================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# --- Activate venv (optional) -----------------------------------------------------------
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

# --- Model & naming ---------------------------------------------------------------------
export MODEL_NAME="${MODEL_NAME:-cpatonn/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B-Thinking-2507}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_NAME}"
export HF_CONFIG_PATH="${HF_CONFIG_PATH:-$MODEL_NAME}"

# --- Hardware / runtime envs ------------------------------------------------------------
# A100 (SM80)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

# vLLM V1 engine & attention backend (FlashInfer, Torch SDPA, FlashAttention, etc.)
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# Avoid CPU oversubscription from tokenizers
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Tuning CUDA allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# --- GPU selection & parallelism --------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra _GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#_GPU_ARR[@]}"

# By default: replicate model per GPU for throughput (DP); keep TP=1 for 12B on 40GB.
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-$NUM_GPUS}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# --- Profiles ---------------------------------------------------------------------------
# THROUGHPUT: higher batch and batched tokens; LOW_LATENCY: smaller token bucket for lower ITL
PROFILE="${PROFILE:-LOW_LATENCY}"   # THROUGHPUT | LOW_LATENCY | LONG_CONTEXT

case "$PROFILE" in
  THROUGHPUT)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"          # NVIDIA model card suggests starting at 64
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"  # >= 8192 for throughput
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    ;;
  LOW_LATENCY)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    ;;
  LONG_CONTEXT)
    # Raise context, but keep batch modest to avoid KV OOM. Adjust as needed.
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"      # Model supports up to 128k if you have the memory
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE" >&2; exit 1
    ;;
esac

# Choose how to resolve mismatches:
#   cap-model-len: prefer throughput, keep bucket small
#   expand-batched-tokens: prefer long context, raise bucket (uses more VRAM)
: "${LEN_STRATEGY:=cap-model-len}"

if (( MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
  case "$LEN_STRATEGY" in
    cap-model-len)
      echo "[vLLM] Capping MAX_MODEL_LEN to ${MAX_NUM_BATCHED_TOKENS} to satisfy constraint"
      MAX_MODEL_LEN="$MAX_NUM_BATCHED_TOKENS"
      ;;
    expand-batched-tokens)
      echo "[vLLM] Raising MAX_NUM_BATCHED_TOKENS to ${MAX_MODEL_LEN} to satisfy constraint"
      MAX_NUM_BATCHED_TOKENS="$MAX_MODEL_LEN"
      ;;
    *)
      echo "Unknown LEN_STRATEGY=$LEN_STRATEGY"; exit 1;
      ;;
  esac
fi

# --- KV cache & memory ------------------------------------------------------------------
# FP8 KV cache reduces GPU memory pressure and boosts throughput.
# On NVIDIA, default to E5M2 (good range); allow override to E4M3 for accuracy needs.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"   # fp8_e5m2 | fp8_e4m3 | auto | fp16 | bf16
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}" # Pre-allocate more KV to reduce preemption

# --- API server scaling (parses requests on CPU) ----------------------------------------
# Parallelize request parsing if you have spare CPU cores
export API_SERVER_COUNT="${API_SERVER_COUNT:-1}"

# --- Networking -------------------------------------------------------------------------
export HOST="${HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"

# --- Sanity info ------------------------------------------------------------------------
echo "Using GPUs: $CUDA_VISIBLE_DEVICES (NUM_GPUS=$NUM_GPUS)"
python - <<'PY' || true
try:
    import vllm, sys
    print(f"vLLM version: {vllm.__version__}")
except Exception as e:
    print("WARNING: Could not import vllm:", e, file=sys.stderr)
PY

echo "Profile=${PROFILE}  DP=${DATA_PARALLEL_SIZE}  TP=${TENSOR_PARALLEL_SIZE}"
echo "KV dtype=${KV_CACHE_DTYPE}  max_num_seqs=${MAX_NUM_SEQS}  max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}  max_model_len=${MAX_MODEL_LEN}"

# --- Launch -----------------------------------------------------------------------------
set -x
vllm serve "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tokenizer "$TOKENIZER_MODEL" \
  --hf-config-path "$HF_CONFIG_PATH" \
  --trust-remote-code \
  --mamba_ssm_cache_dtype float32 \
  --dtype auto \
  --host "$HOST" \
  --port "$API_PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enforce-eager \
  --reasoning-parser deepseek_r1 \
  --enable-chunked-prefill \
  --api-server-count "$API_SERVER_COUNT" \
  --enable-prefix-caching

# Notes:
# * V1 engine enables chunked prefill by default; no need for --enable-chunked-prefill.
# * Use --disable-log-stats if you want to minimize logging overhead.
# * Metrics are exposed on /metrics of the same server port.
set +x

echo
echo "vLLM server is running at: http://${HOST}:${API_PORT}"
echo
echo "Test (Chat Completions):"
cat <<'CURL'
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-30B-A3B-Thinking-2507",
    "messages": [
      {"role": "system", "content": "/think"},
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 128,
    "temperature": 0.6,
    "top_p": 0.95
  }' | jq .
CURL

echo
echo "Prometheus metrics: curl http://localhost:${API_PORT}/metrics | head -n 20"
