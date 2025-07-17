# vLLM Serving

A production-ready deployment setup for serving Large Language Models (LLMs) using vLLM, with optimized configurations for various model sizes and types.

## 🚀 Features

- **Multi-Model Support**: Pre-configured scripts for popular models including Qwen2.5-VL, InternVL3, and DeepSeek-R1
- **AWQ Quantization**: Optimized for memory efficiency with Activation-aware Weight Quantization
- **GPU Optimization**: Tensor parallelism and memory utilization tuning
- **Production Ready**: Robust configurations with error handling and logging
- **Vision-Language Models**: Support for multimodal inference with image processing

## 📋 Prerequisites

- Python 3.10
- CUDA-compatible GPU(s)
- UV package manager
- Virtual environment support

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hiennguyen9874/vllm-serving
cd vllm-serving
```

### 2. Install Dependencies

```bash
# Install setuptools first
uv pip install setuptools

# Install base dependencies (without flash-attn for faster setup)
uv sync --no-dev

# Install all dependencies including development tools
uv sync --dev
```

### 3. Activate Virtual Environment

```bash
source .venv/bin/activate
```

## 📦 Supported Models

### Text Models

| Model                            | Script                                  | GPUs | Context Length | Tensor Parallel |
| -------------------------------- | --------------------------------------- | ---- | -------------- | --------------- |
| Qwen3-14B-AWQ                    | `qwen3_72b_awq.bash`                    | 2    | 16K            | 2               |
| DeepSeek-R1-Distill-Qwen-32B-AWQ | `deepseek-r1-distill-qwen-32b-awq.bash` | 4    | 32K            | 4               |

### Vision-Language Models

| Model                       | Script                                      | GPUs | Context Length | Features         |
| --------------------------- | ------------------------------------------- | ---- | -------------- | ---------------- |
| Qwen2.5-VL-32B-Instruct-AWQ | `qwen2_5_vl_32b_awq.bash`                   | 1    | 16K            | Image+Text       |
| Qwen2.5-VL-72B-Instruct-AWQ | `qwen2_5_vl_72b_awq.bash`                   | 4    | 16K            | Image+Text       |
| InternVL3-8B (Fine-tuned)   | `internvl3-8b-hf-traffic-accident.bash`     | 1    | 4K             | Traffic Analysis |
| InternVL3-8B-AWQ            | `internvl3-8b-hf-traffic-accident-awq.bash` | 1    | 4K             | Quantized Vision |

## 🚀 Quick Start

### 1. Choose Your Model

Select the appropriate script based on your hardware and requirements:

```bash
# For single GPU setup with vision capabilities
./qwen2_5_vl_32b_awq.bash

# For multi-GPU setup with larger model
./qwen2_5_vl_72b_awq.bash

# For specialized traffic accident analysis
./internvl3-8b-hf-traffic-accident.bash
```

### 2. Make Scripts Executable

```bash
chmod +x *.bash
```

### 3. Run the Server

```bash
# Example: Start Qwen2.5-VL-32B
./qwen2_5_vl_32b_awq.bash
```

The server will be available at `http://localhost:8003` (or `8004` for some models).

## ⚙️ Configuration

### Environment Variables

All scripts support these key environment variables:

- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use
- `TOKENIZERS_PARALLELISM`: Enable parallel tokenizer processing
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation settings
- `VLLM_USE_V1`: Use vLLM v1 API (when applicable)

### Key Parameters

| Parameter                  | Description                 | Default       |
| -------------------------- | --------------------------- | ------------- |
| `--gpu-memory-utilization` | GPU memory usage percentage | 0.95-0.98     |
| `--max-model-len`          | Maximum context length      | 4K-32K        |
| `--max-num-seqs`           | Concurrent sequences        | 1-4           |
| `--tensor-parallel-size`   | GPU parallelism             | 1-4           |
| `--quantization`           | Quantization method         | awq_marlin    |
| `--dtype`                  | Model precision             | half/bfloat16 |

## 🔧 Performance Optimization

### Memory Optimization

- **AWQ Quantization**: Reduces model size by ~4x with minimal quality loss
- **FP16 Precision**: Halves memory usage compared to FP32
- **Chunked Prefill**: Prevents OOM on long sequences
- **Expandable Segments**: Reduces memory fragmentation

### Speed Optimization

- **Tensor Parallelism**: Distributes computation across multiple GPUs
- **Batching**: Process multiple requests simultaneously
- **Prefix Caching**: Cache common prefixes for faster inference
- **CUDA Graphs**: Reduced kernel launch overhead (when `--enforce-eager` is disabled)

## 📡 API Usage

### OpenAI-Compatible API

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8003/v1",
    api_key="dummy"  # vLLM doesn't require authentication
)

# Text completion
response = client.completions.create(
    model="model-name",
    prompt="Hello, world!",
    max_tokens=100
)

# Chat completion
response = client.chat.completions.create(
    model="model-name",
    messages=[
        {"role": "user", "content": "What is the weather like?"}
    ]
)
```

### Vision-Language Models

```python
# For vision models, include image in the content
response = client.chat.completions.create(
    model="qwen2.5-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
    ]
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**

   - Reduce `--gpu-memory-utilization`
   - Decrease `--max-model-len`
   - Lower `--max-num-seqs`

2. **Slow Inference**

   - Increase `--max-num-batched-tokens`
   - Enable `--enable-prefix-caching`
   - Remove `--enforce-eager` for CUDA graphs

3. **Model Loading Errors**
   - Ensure `--trust-remote-code` is set
   - Check GPU memory availability
   - Verify model path/name

### Performance Monitoring

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check vLLM logs
tail -f vllm.log

# Monitor API requests
curl http://localhost:8003/metrics
```

## 📊 Benchmarks

### Typical Performance (RTX 4090)

| Model               | Memory Usage | Tokens/sec | Batch Size |
| ------------------- | ------------ | ---------- | ---------- |
| Qwen2.5-VL-32B-AWQ  | ~20GB        | 15-25      | 1-4        |
| InternVL3-8B-AWQ    | ~8GB         | 30-50      | 2-4        |
| DeepSeek-R1-32B-AWQ | ~40GB        | 10-20      | 1-2        |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model configuration
4. Test with your hardware setup
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [Model Hub](https://huggingface.co/)
- [AWQ Quantization](https://github.com/casper-hansen/AutoAWQ)

## 💡 Tips

- Start with smaller models to test your setup
- Monitor GPU memory usage during first runs
- Use `--dry-run` flag to validate configurations
- Keep model weights on fast storage (NVMe SSD)
- Consider model-specific chat templates for best results
