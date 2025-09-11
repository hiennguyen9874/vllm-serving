# NVIDIA Nemotron-Nano-12B-v2 on vLLM (V1 engine)

Got it—here’s a tightened, production-ready launch script for **NVIDIA Nemotron-Nano-12B-v2 on vLLM (V1 engine)**, tuned for A100 40 GB and aligned with the latest vLLM + model guidance. I’ve also pulled in the relevant doc notes so you can see _why_ each change helps.

---

### ✅ What I changed (high level)

- **Align with model card guidance**: Nemotron-Nano-12B-v2 is a hybrid **Mamba2-Transformer**; NVIDIA explicitly recommends `--mamba_ssm_cache_dtype float32` and starting with `--max-num-seqs 64`. Context length supports up to **128K**; we default to 32K to keep KV memory sane and let you bump via `MAX_MODEL_LEN`. ([Hugging Face][1])
- **vLLM V1 defaults**: V1 has **chunked prefill always on**; the old `--enable-chunked-prefill` flag is no longer needed and `--no-enable-chunked-prefill` is not valid in V1. ([VLLM Docs][2])
- **Attention backend**: Allow selecting **FlashInfer** via env (`VLLM_ATTENTION_BACKEND=FLASHINFER`)—vLLM supports this env var to force a backend. ([VLLM Docs][3])
- **KV cache FP8**: Default KV cache to **FP8 E5M2** on NVIDIA (best range), with option to switch to E4M3. Both formats are supported in recent vLLM; E5M2 has historically been the NVIDIA default for KV. ([VLLM Docs][4])
- **Consistency fixes**: Unify **port** (default 8000) so the echo + curl match vLLM’s OpenAI-compatible server defaults. ([VLLM Docs][5])
- **Profiles & auto-sizing**: Add simple **THROUGHPUT** vs **LOW_LATENCY** presets, automatic DP sizing from `CUDA_VISIBLE_DEVICES`, and safe defaults for `max_num_batched_tokens` per vLLM guidance. ([VLLM Docs][2])
- **Prefix caching**: Enable **Automatic Prefix Caching** (APC) which V1 supports and is useful for repeated prefixes (RAG / multi-turn). ([VLLM Docs][6])

---

### 🚀 Optimized launch script

```bash
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
export MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-Nano-12B-v2}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-NVIDIA-Nemotron-Nano-12B-v2}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL_NAME}"
export HF_CONFIG_PATH="${HF_CONFIG_PATH:-$MODEL_NAME}"

# --- Hardware / runtime envs ------------------------------------------------------------
# A100 (SM80)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

# vLLM V1 engine & attention backend (FlashInfer, Torch SDPA, FlashAttention, etc.)
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

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
PROFILE="${PROFILE:-THROUGHPUT}"   # THROUGHPUT | LOW_LATENCY | LONG_CONTEXT

case "$PROFILE" in
  THROUGHPUT)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"          # NVIDIA model card suggests starting at 64
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"  # >= 8192 for throughput
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    ;;
  LOW_LATENCY)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    ;;
  LONG_CONTEXT)
    # Raise context, but keep batch modest to avoid KV OOM. Adjust as needed.
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"      # Model supports up to 128k if you have the memory
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE" >&2; exit 1
    ;;
esac

# --- KV cache & memory ------------------------------------------------------------------
# FP8 KV cache reduces GPU memory pressure and boosts throughput.
# On NVIDIA, default to E5M2 (good range); allow override to E4M3 for accuracy needs.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"   # fp8_e5m2 | fp8_e4m3 | auto | fp16 | bf16
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
    "model": "NVIDIA-Nemotron-Nano-12B-v2",
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
```

---

### Why these choices work

- **Nemotron-Nano-12B-v2 specifics:** It’s a **Mamba2-Transformer hybrid**, and NVIDIA’s model card shows vLLM usage with `--mamba_ssm_cache_dtype float32` and suggests `--max-num-seqs 64`. The card also lists **128K** context support—raise `MAX_MODEL_LEN` only if you’ve got KV budget. ([Hugging Face][1])
- **Chunked prefill in V1:** In vLLM **V1**, chunked prefill is **always enabled** and tuned by `max_num_batched_tokens`; you don’t need the old flag and disabling it isn’t supported in V1. Lower values improve **inter-token latency**, higher values improve **TTFT** and throughput. The script exposes `MAX_NUM_BATCHED_TOKENS` per profile. ([VLLM Docs][2])
- **KV cache FP8:** vLLM supports **FP8 E5M2** and **E4M3** KV cache formats. On NVIDIA GPUs, E5M2 is typically preferred for range; E4M3 is available and often has minimal accuracy drop. Toggle via `KV_CACHE_DTYPE`. ([VLLM Docs][4])
- **Reduce preemptions:** If you see scheduler warnings about preemption/recompute, **increase** `gpu_memory_utilization`, **decrease** `max_num_seqs` or `max_num_batched_tokens`, or consider **TP**/**PP** to free KV space—doc-backed options are noted in your pasted guidance and echoed here. ([VLLM Docs][2])
- **Attention backend:** You can force **FlashInfer** via `VLLM_ATTENTION_BACKEND=FLASHINFER`; vLLM recognizes this env var and will pick it if available. If kernels aren’t present, vLLM falls back to another backend. ([VLLM Docs][3])
- **Prefix caching (APC):** Enabling it helps a ton with repeated prefixes (multi-turn chats / RAG) by reusing KV blocks. ([VLLM Docs][6])
- **Metrics:** vLLM exposes **Prometheus metrics** at `/metrics` on the same port; no extra flag needed. ([VLLM Docs][7])

---

### Quick tuning guide (rules of thumb)

- **OOM / preemption seen** → drop `MAX_NUM_SEQS` (64 → 48/32), or drop `MAX_NUM_BATCHED_TOKENS` (e.g., 16384 → 12288/8192), or bump `GPU_MEMORY_UTILIZATION` (0.90 → 0.95). ([VLLM Docs][2])
- **Need lower ITL (snappier tokens)** → choose `PROFILE=LOW_LATENCY` (smaller `MAX_NUM_BATCHED_TOKENS`). ([VLLM Docs][2])
- **Very long prompts** → use `PROFILE=LONG_CONTEXT` and consider **TP=2** if you start hitting memory limits per GPU. ([VLLM Docs][8])
- **Throughput scaling** → keep **TP=1**, **DP=#GPUs** for 12B on 40GB A100s if the model fits easily and you want maximum parallel request processing. ([VLLM Docs][9])

If you want, I can add an auto-probe to try **TP=2** when `MAX_MODEL_LEN>=65536` and fall back to **TP=1** if it detects insufficient GPUs.

[1]: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2 "nvidia/NVIDIA-Nemotron-Nano-12B-v2 · Hugging Face"
[2]: https://docs.vllm.ai/en/latest/configuration/optimization.html?utm_source=chatgpt.com "Optimization and Tuning - vLLM"
[3]: https://docs.vllm.ai/en/latest/getting_started/quickstart.html?utm_source=chatgpt.com "Quickstart - vLLM"
[4]: https://docs.vllm.ai/en/v0.6.1/quantization/fp8_e5m2_kvcache.html?utm_source=chatgpt.com "FP8 E5M2 KV Cache - vLLM"
[5]: https://docs.vllm.ai/en/latest/cli/serve.html?utm_source=chatgpt.com "JSON CLI Arguments - vLLM"
[6]: https://docs.vllm.ai/en/stable/design/prefix_caching.html?utm_source=chatgpt.com "Automatic Prefix Caching - vLLM"
[7]: https://docs.vllm.ai/en/v0.9.2/usage/metrics.html?utm_source=chatgpt.com "Production Metrics - vLLM"
[8]: https://docs.vllm.ai/en/stable/serving/parallelism_scaling.html?utm_source=chatgpt.com "Parallelism and Scaling - vLLM"
[9]: https://docs.vllm.ai/en/latest/configuration/engine_args.html?utm_source=chatgpt.com "Engine Arguments - vLLM"
