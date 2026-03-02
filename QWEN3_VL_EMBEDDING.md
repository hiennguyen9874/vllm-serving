# Qwen3-Embedding & Qwen3-Reranking

https://github.com/vllm-project/vllm/pull/19260

## Qwen3 Embedding

```bash
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST="8.0" uv run vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --trust-remote-code \
    --runner pooling \
    --max-model-len 8192 \
    --convert embed \
    --gpu-memory-utilization 0.25 \
    --host 0.0.0.0 \
    --port 8003 \
    --hf_overrides '{"matryoshka_dimensions":[1024]}'

curl http://localhost:8003/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "Qwen/Qwen3-VL-Embedding-2B",
    "encoding_format": "float"
  }'
```

## Qwen3 Reranking

```bash
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST="8.0" uv run vllm serve Qwen/Qwen3-VL-Reranker-2B \
    --trust-remote-code \
    --hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
    --runner pooling \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.35 \
    --host 0.0.0.0 \
    --port 8004

curl http://localhost:8004/score \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text_1": "ping",
    "text_2": "pong",
    "model": "Qwen/Qwen3-VL-Reranker-2B"
  }'

curl http://localhost:8004/rerank \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "ping",
    "documents": ["pong"],
    "model": "Qwen/Qwen3-VL-Reranker-2B"
  }'
```
