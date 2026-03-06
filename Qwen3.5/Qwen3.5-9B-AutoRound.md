## Run

```bash
vllm serve kaitchup/Qwen3.5-9B-autoround-W4A16 \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype "fp8_e5m2" \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

## Bench
```bash
vllm bench serve \
  --backend vllm \
  --model kaitchup/Qwen3.5-9B-autoround-W4A16 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ./data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 50
```

## Result

```
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Benchmark duration (s):                  13.28     
Total input tokens:                      12852     
Total generated tokens:                  10451     
Request throughput (req/s):              3.77      
Output token throughput (tok/s):         787.04    
Peak output token throughput (tok/s):    2542.00   
Peak concurrent requests:                50.00     
Total token throughput (tok/s):          1754.89   
---------------Time to First Token----------------
Mean TTFT (ms):                          4842.48   
Median TTFT (ms):                        4842.27   
P99 TTFT (ms):                           5556.50   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.06     
Median TPOT (ms):                        16.03     
P99 TPOT (ms):                           167.40    
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.39     
Median ITL (ms):                         11.15     
P99 ITL (ms):                            220.16    
==================================================
```
