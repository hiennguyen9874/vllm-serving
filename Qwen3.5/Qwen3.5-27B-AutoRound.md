## Run

```bash
vllm serve kaitchup/Qwen3.5-27B-autoround-W4A16 \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype "fp8_e5m2" \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

## Bench
```bash
vllm bench serve \
  --backend vllm \
  --model kaitchup/Qwen3.5-27B-autoround-W4A16 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 50
```

## Result

```
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Benchmark duration (s):                  51.58     
Total input tokens:                      12852     
Total generated tokens:                  10645     
Request throughput (req/s):              0.97      
Output token throughput (tok/s):         206.37    
Peak output token throughput (tok/s):    1106.00   
Peak concurrent requests:                50.00     
Total token throughput (tok/s):          455.54    
---------------Time to First Token----------------
Mean TTFT (ms):                          31100.37  
Median TTFT (ms):                        31177.04  
P99 TTFT (ms):                           33001.02  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          77.31     
Median TPOT (ms):                        38.16     
P99 TPOT (ms):                           508.83    
---------------Inter-token Latency----------------
Mean ITL (ms):                           36.15     
Median ITL (ms):                         27.75     
P99 ITL (ms):                            537.89    
==================================================
```
