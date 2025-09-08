## Qwen3 Embedding

```bash
curl http://127.0.0.1:8000/v1/embeddings \
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
curl http://127.0.0.1:8000/score \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text_1": "ping",
    "text_2": "pong",
    "model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
  }'


curl http://127.0.0.1:8000/rerank \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "ping",
    "documents": ["pong"],
    "model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
  }'

```
