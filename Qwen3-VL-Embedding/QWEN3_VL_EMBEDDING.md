# Qwen3-Embedding & Qwen3-Reranking

https://github.com/vllm-project/vllm/pull/19260

## Qwen3 Embedding

```bash
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST="8.0" vllm serve alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC \
    --served-model-name Qwen3-VL-Embedding-2B \
    --trust-remote-code \
    --runner pooling \
    --convert embed \
    --max-model-len 8192 \
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

curl https://llama-cpp.dscilab.com:20007/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "Qwen3-VL-Embedding-2B",
    "encoding_format": "float",
    "dimensions": 512
  }'
```

### Benchmark
- Run
  ```bash
  vllm serve alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC --runner pooling --convert embed

  # wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
  vllm bench serve \
    --model alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC \
    --backend openai-embeddings \
    --endpoint /v1/embeddings \
    --dataset-name sharegpt \
    --dataset-path ./data/ShareGPT_V3_unfiltered_cleaned_split.json
  ```
- Result:
  ```
  ============ Serving Benchmark Result ============
  Successful requests:                     1000      
  Failed requests:                         0         
  Benchmark duration (s):                  6.02      
  Total input tokens:                      218089    
  Request throughput (req/s):              166.18    
  Total token throughput (tok/s):          36241.12  
  ----------------End-to-end Latency----------------
  Mean E2EL (ms):                          3430.42   
  Median E2EL (ms):                        3353.79   
  P99 E2EL (ms):                           5864.62   
  ==================================================
  ```

- Run:
  ```bash
  vllm bench serve \
    --model alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC \
    --backend openai-embeddings-clip \
    --endpoint /v1/embeddings \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat
  ```

- Result:
  ```
  ============ Serving Benchmark Result ============
  Successful requests:                     1000      
  Failed requests:                         0         
  Benchmark duration (s):                  59.43     
  Total input tokens:                      517084    
  Request throughput (req/s):              16.83     
  Total token throughput (tok/s):          8700.62   
  ----------------End-to-end Latency----------------
  Mean E2EL (ms):                          41846.64  
  Median E2EL (ms):                        41631.30  
  P99 E2EL (ms):                           58585.37  
  ==================================================
  ```

- Run:
  ```bash
  vllm bench serve \
    --model alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC \
    --backend openai-embeddings-vlm2vec \
    --endpoint /v1/embeddings \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat
  ```

- Result:
  ```
  ============ Serving Benchmark Result ============
  Successful requests:                     1000      
  Failed requests:                         0         
  Benchmark duration (s):                  40.87     
  Total input tokens:                      574422    
  Request throughput (req/s):              24.47     
  Total token throughput (tok/s):          14055.47  
  ----------------End-to-end Latency----------------
  Mean E2EL (ms):                          23319.12  
  Median E2EL (ms):                        25498.49  
  P99 E2EL (ms):                           39993.24  
  ==================================================
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
