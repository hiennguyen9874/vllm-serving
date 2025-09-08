# Qwen3-Embedding & Qwen3-Reranking

https://github.com/vllm-project/vllm/pull/19260

## Qwen3 Embedding

```bash
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST="8.0" uv run vllm serve Qwen/Qwen3-Embedding-0.6B \
    --trust-remote-code \
    --gpu-memory-utilization 0.25 \
    --host 0.0.0.0 \
    --port 8003

curl http://localhost:8003/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "encoding_format": "float"
  }'
```

## Qwen3 Reranking

```bash
CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST="8.0" uv run vllm serve tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
    --trust-remote-code \
    --gpu-memory-utilization 0.25 \
    --host 0.0.0.0 \
    --port 8004

curl http://localhost:8004/score \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text_1": "ping",
    "text_2": "pong",
    "model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
  }'

curl http://localhost:8004/rerank \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "ping",
    "documents": ["pong"],
    "model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
  }'
```
