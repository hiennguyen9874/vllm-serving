# Qwen3-Embedding & Qwen3-Reranking

https://github.com/vllm-project/vllm/pull/19260

## Qwen3 Embedding

```bash
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST="8.0" vllm serve alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC \
    --served-model-name Qwen3-VL-Embedding-2B \
    --trust-remote-code \
    --runner pooling \
    --max-model-len 8192 \
    --convert embed \
    --gpu-memory-utilization 0.15 \
    --host 0.0.0.0 \
    --port 8001 \
    --hf-overrides '{"is_matryoshka": true}'

curl http://localhost:8001/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "Qwen3-VL-Embedding-2B",
    "encoding_format": "float",
    "dimensions": 512
  }'
```

## Qwen3 Reranking

```bash
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST="8.0" vllm serve Qwen/Qwen3-VL-Reranker-2B \
    --trust-remote-code \
    --hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true,"quantization_config": {"ignored_layers": ["score"]}}' \
    --runner pooling \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.2 \
    --host 0.0.0.0 \
    --port 8002

curl http://localhost:8002/rerank \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "ping",
    "documents": ["pong"],
    "model": "Qwen/Qwen3-VL-Reranker-2B"
  }'
```
