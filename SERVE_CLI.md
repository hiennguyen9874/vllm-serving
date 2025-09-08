```
INFO 08-11 13:48:54 [__init__.py:241] Automatically detected platform cuda.
usage: vllm serve [model_tag] [options]

Start the vLLM OpenAI Compatible API server.

positional arguments:
  model_tag             The model tag to serve (optional if specified in config) (default: None)

options:
  --api-server-count API_SERVER_COUNT, -asc API_SERVER_COUNT
                        How many API server processes to run. (default: 1)
  --config CONFIG       Read CLI options from a config file. Must be a YAML with the following options: https://docs.vllm.ai/en/latest/configuration/serve_args.html (default: None)
  --disable-log-requests, --no-disable-log-requests
                        [DEPRECATED] Disable logging requests. (default: True)
  --disable-log-stats   Disable logging statistics. (default: False)
  --enable-log-requests, --no-enable-log-requests
                        Enable logging requests. (default: False)
  --enable-prompt-adapter
                        [DEPRECATED] Prompt adapter has been removed. Setting this flag to True or False has no effect on vLLM behavior. (default: False)
  --headless            Run in headless mode. See multi-node data parallel documentation for more details. (default: False)
  -h, --help            show this help message and exit

Frontend:
  Arguments for the OpenAI-compatible frontend server.

  --allow-credentials, --no-allow-credentials
                        Allow credentials. (default: False)
  --allowed-headers ALLOWED_HEADERS
                        Allowed headers. (default: ['*'])
  --allowed-methods ALLOWED_METHODS
                        Allowed methods. (default: ['*'])
  --allowed-origins ALLOWED_ORIGINS
                        Allowed origins. (default: ['*'])
  --api-key API_KEY [API_KEY ...]
                        If provided, the server will require one of these keys to be presented in the header. (default: None)
  --chat-template CHAT_TEMPLATE
                        The file path to the chat template, or the template in single-line form for the specified model. (default: None)
  --chat-template-content-format {auto,openai,string}
                        The format to render message content within a chat template.
                        * "string" will render the content as a string. Example: `"Hello World"` * "openai" will render the content as a list of dictionaries, similar to OpenAI
                        schema. Example: `[{"type": "text", "text": "Hello world!"}]` (default: auto)
  --disable-fastapi-docs, --no-disable-fastapi-docs
                        Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint. (default: False)
  --disable-frontend-multiprocessing, --no-disable-frontend-multiprocessing
                        If specified, will run the OpenAI frontend server in the same process as the model serving engine. (default: False)
  --disable-uvicorn-access-log, --no-disable-uvicorn-access-log
                        Disable uvicorn access log. (default: False)
  --enable-auto-tool-choice, --no-enable-auto-tool-choice
                        If specified, exclude tool definitions in prompts when tool_choice='none'. (default: False)
  --enable-force-include-usage, --no-enable-force-include-usage
                        If set to True, including usage on every request. (default: False)
  --enable-log-outputs, --no-enable-log-outputs
                        If set to True, enable logging of model outputs (generations)  in addition to the input logging that is enabled by default. (default: False)
  --enable-prompt-tokens-details, --no-enable-prompt-tokens-details
                        If set to True, enable prompt_tokens_details in usage. (default: False)
  --enable-request-id-headers, --no-enable-request-id-headers
                        If specified, API server will add X-Request-Id header to responses. Caution: this hurts performance at high QPS. (default: False)
  --enable-server-load-tracking, --no-enable-server-load-tracking
                        If set to True, enable tracking server_load_metrics in the app state. (default: False)
  --enable-ssl-refresh, --no-enable-ssl-refresh
                        Refresh SSL Context when SSL certificate files change (default: False)
  --enable-tokenizer-info-endpoint, --no-enable-tokenizer-info-endpoint
                        Enable the /get_tokenizer_info endpoint. May expose chat templates and other tokenizer configuration. (default: False)
  --exclude-tools-when-tool-choice-none, --no-exclude-tools-when-tool-choice-none
                        Enable auto tool choice for supported models. Use `--tool-call-parser` to specify which parser to use. (default: False)
  --host HOST           Host name. (default: None)
  --log-config-file LOG_CONFIG_FILE
                        Path to logging config JSON file for both vllm and uvicorn (default: None)
  --lora-modules LORA_MODULES [LORA_MODULES ...]
                        LoRA modules configurations in either 'name=path' format or JSON format or JSON list format. Example (old format): `'name=path'` Example (new format):
                        `{"name": "name", "path": "lora_path", "base_model_name": "id"}` (default: None)
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt ID numbers being printed in log. The default of None means unlimited. (default: None)
  --middleware MIDDLEWARE
                        Additional ASGI middleware to apply to the app. We accept multiple --middleware arguments. The value should be an import path. If a function is provided,
                        vLLM will add it to the server using `@app.middleware('http')`. If a class is provided, vLLM will add it to the server using `app.add_middleware()`.
                        (default: [])
  --port PORT           Port number. (default: 8000)
  --response-role RESPONSE_ROLE
                        The role name to return if `request.add_generation_prompt=true`. (default: assistant)
  --return-tokens-as-token-ids, --no-return-tokens-as-token-ids
                        When `--max-logprobs` is specified, represents single tokens as strings of the form 'token_id:{token_id}' so that tokens that are not JSON-encodable can be
                        identified. (default: False)
  --root-path ROOT_PATH
                        FastAPI root_path when app is behind a path based routing proxy. (default: None)
  --ssl-ca-certs SSL_CA_CERTS
                        The CA certificates file. (default: None)
  --ssl-cert-reqs SSL_CERT_REQS
                        Whether client certificate is required (see stdlib ssl module's). (default: 0)
  --ssl-certfile SSL_CERTFILE
                        The file path to the SSL cert file. (default: None)
  --ssl-keyfile SSL_KEYFILE
                        The file path to the SSL key file. (default: None)
  --tool-call-parser {deepseek_v3,glm45,granite-20b-fc,granite,hermes,hunyuan_a13b,internlm,jamba,kimi_k2,llama4_pythonic,llama4_json,llama3_json,minimax,mistral,phi4_mini_json,pythonic,qwen3_coder,step3,xlam} or name registered in --tool-parser-plugin
                        Select the tool call parser depending on the model that you're using. This is used to parse the model-generated tool call into OpenAI API format. Required
                        for `--enable-auto-tool-choice`. You can choose any option from the built-in parsers or register a plugin via `--tool-parser-plugin`. (default: None)
  --tool-parser-plugin TOOL_PARSER_PLUGIN
                        Special the tool parser plugin write to parse the model-generated tool into OpenAI API format, the name register in this plugin can be used in `--tool-call-
                        parser`. (default: )
  --tool-server TOOL_SERVER
                        Comma-separated list of host:port pairs (IPv4, IPv6, or hostname). Examples: 127.0.0.1:8000, [::1]:8000, localhost:1234. Or `demo` for demo purpose.
                        (default: None)
  --uds UDS             Unix domain socket path. If set, host and port arguments are ignored. (default: None)
  --uvicorn-log-level {critical,debug,error,info,trace,warning}
                        Log level for uvicorn. (default: info)

ModelConfig:
  Configuration for the model.

  --allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH
                        Allowing API requests to read local images or videos from directories specified by the server file system. This is a security risk. Should only be enabled in
                        trusted environments. (default: )
  --code-revision CODE_REVISION
                        The specific revision to use for the model code on the Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the
                        default version. (default: None)
  --config-format {auto,hf,mistral}
                        The format of the model config to load:
                        - "auto" will try to load the config in hf format if available else it will try to load in mistral format.
                        - "hf" will load the config in hf format.
                        - "mistral" will load the config in mistral format. (default: auto)
  --convert {auto,classify,embed,none,reward}
                        Convert the model using adapters defined in [vllm.model_executor.models.adapters][]. The most common use case is to adapt a text generation model to be used
                        for pooling tasks. (default: auto)
  --disable-async-output-proc
                        Disable async output processing. This may result in lower performance. (default: False)
  --disable-cascade-attn, --no-disable-cascade-attn
                        Disable cascade attention for V1. While cascade attention does not change the mathematical correctness, disabling it could be useful for preventing potential
                        numerical issues. Note that even if this is set to False, cascade attention will be only used when the heuristic tells that it's beneficial. (default: False)
  --disable-sliding-window, --no-disable-sliding-window
                        Whether to disable sliding window. If True, we will disable the sliding window functionality of the model, capping to sliding window size. If the model does
                        not support sliding window, this argument is ignored. (default: False)
  --dtype {auto,bfloat16,float,float16,float32,half}
                        Data type for model weights and activations:
                        - "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
                        - "half" for FP16. Recommended for AWQ quantization.
                        - "float16" is the same as "half".
                        - "bfloat16" for a balance between precision and range.
                        - "float" is shorthand for FP32 precision.
                        - "float32" for FP32 precision. (default: auto)
  --enable-prompt-embeds, --no-enable-prompt-embeds
                        If `True`, enables passing text embeddings as inputs via the `prompt_embeds` key. Note that enabling this will double the time required for graph
                        compilation. (default: False)
  --enable-sleep-mode, --no-enable-sleep-mode
                        Enable sleep mode for the engine (only cuda platform is supported). (default: False)
  --enforce-eager, --no-enforce-eager
                        Whether to always use eager-mode PyTorch. If True, we will disable CUDA graph and always execute the model in eager mode. If False, we will use CUDA graph
                        and eager execution in hybrid for maximal performance and flexibility. (default: False)
  --generation-config GENERATION_CONFIG
                        The folder path to the generation config. Defaults to `"auto"`, the generation config will be loaded from model path. If set to `"vllm"`, no generation
                        config is loaded, vLLM defaults will be used. If set to a folder path, the generation config will be loaded from the specified folder path. If
                        `max_new_tokens` is specified in generation config, then it sets a server-wide limit on the number of output tokens for all requests. (default: auto)
  --hf-config-path HF_CONFIG_PATH
                        Name or path of the Hugging Face config to use. If unspecified, model name or path will be used. (default: None)
  --hf-overrides HF_OVERRIDES
                        If a dictionary, contains arguments to be forwarded to the Hugging Face config. If a callable, it is called to update the HuggingFace config. (default: {})
  --hf-token [HF_TOKEN]
                        The token to use as HTTP bearer authorization for remote files . If `True`, will use the token generated when running `huggingface-cli login` (stored in
                        `~/.huggingface`). (default: None)
  --logits-processor-pattern LOGITS_PROCESSOR_PATTERN
                        Optional regex pattern specifying valid logits processor qualified names that can be passed with the `logits_processors` extra completion argument. Defaults
                        to `None`, which allows no processors. (default: None)
  --logprobs-mode {processed_logits,processed_logprobs,raw_logits,raw_logprobs}
                        Indicates the content returned in the logprobs and prompt_logprobs. Supported mode: 1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4)
                        processed_logits. Raw means the values before applying logit processors, like bad words. Processed means the values after applying such processors. (default:
                        raw_logprobs)
  --max-logprobs MAX_LOGPROBS
                        Maximum number of log probabilities to return when `logprobs` is specified in `SamplingParams`. The default value comes the default for the OpenAI Chat
                        Completions API. -1 means no cap, i.e. all (output_length * vocab_size) logprobs are allowed to be returned and it may cause OOM. (default: 20)
  --max-model-len MAX_MODEL_LEN
                        Model context length (prompt and output). If unspecified, will be automatically derived from the model config.
                        When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable format. Examples:
                        - 1k -> 1000
                        - 1K -> 1024
                        - 25.6k -> 25,600 (default: None)
  --max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE
                        Maximum sequence len covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-
                        decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode. (default: 8192)
  --model-impl {auto,vllm,transformers}
                        Which implementation of the model to use:
                        - "auto" will try to use the vLLM implementation, if it exists, and fall back to the Transformers implementation if no vLLM implementation is available.
                        - "vllm" will use the vLLM model implementation.
                        - "transformers" will use the Transformers model implementation. (default: auto)
  --override-attention-dtype OVERRIDE_ATTENTION_DTYPE
                        Override dtype for attention (default: None)
  --override-generation-config OVERRIDE_GENERATION_CONFIG
                        Overrides or sets generation config. e.g. `{"temperature": 0.5}`. If used with `--generation-config auto`, the override parameters will be merged with the
                        default config from the model. If used with `--generation-config vllm`, only the override parameters are used.
                        Should either be a valid JSON string or JSON keys passed individually. (default: {})
  --override-neuron-config OVERRIDE_NEURON_CONFIG
                        Initialize non-default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the
                        neuron config that can not be gathered from the vllm arguments. e.g. `{"cast_logits_dtype": "bfloat16"}`.
                        Should either be a valid JSON string or JSON keys passed individually. (default: {})
  --override-pooler-config OVERRIDE_POOLER_CONFIG
                        Initialize non-default pooling config or override default pooling config for the pooling model. e.g. `{"pooling_type": "mean", "normalize": false}`.
                        (default: None)
  --quantization QUANTIZATION, -q QUANTIZATION
                        Method used to quantize the weights. If `None`, we first check the `quantization_config` attribute in the model config file. If that is `None`, we assume the
                        model weights are not quantized and use `dtype` to determine the data type of the weights. (default: None)
  --revision REVISION   The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. (default: None)
  --rope-scaling ROPE_SCALING
                        RoPE scaling configuration. For example, `{"rope_type":"dynamic","factor":2.0}`.
                        Should either be a valid JSON string or JSON keys passed individually. (default: {})
  --rope-theta ROPE_THETA
                        RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE theta improves the performance of the scaled model. (default: None)
  --runner {auto,draft,generate,pooling}
                        The type of model runner to use. Each vLLM instance only supports one model runner, even if the same model can be used for multiple types. (default: auto)
  --seed SEED           Random seed for reproducibility. Initialized to None in V0, but initialized to 0 in V1. (default: None)
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        The model name(s) used in the API. If multiple names are provided, the server will respond to any of the provided names. The model name in the model field of
                        a response will be the first name in this list. If not specified, the model name will be the same as the `--model` argument. Noted that this name(s) will
                        also be used in `model_name` tag content of prometheus metrics, if multiple names provided, metrics tag will take the first one. (default: None)
  --skip-tokenizer-init, --no-skip-tokenizer-init
                        Skip initialization of tokenizer and detokenizer. Expects valid `prompt_token_ids` and `None` for prompt from the input. The generated output will contain
                        token ids. (default: False)
  --task {auto,classify,draft,embed,embedding,generate,reward,score,transcription,None}
                        [DEPRECATED] The task to use the model for. If the model supports more than one model runner, this is used to select which model runner to run.
                        Note that the model may support other tasks using the same model runner. (default: None)
  --tokenizer TOKENIZER
                        Name or path of the Hugging Face tokenizer to use. If unspecified, model name or path will be used. (default: None)
  --tokenizer-mode {auto,custom,mistral,slow}
                        Tokenizer mode:
                        - "auto" will use the fast tokenizer if available.
                        - "slow" will always use the slow tokenizer.
                        - "mistral" will always use the tokenizer from `mistral_common`.
                        - "custom" will use --tokenizer to select the preregistered tokenizer. (default: auto)
  --tokenizer-revision TOKENIZER_REVISION
                        The specific revision to use for the tokenizer on the Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the
                        default version. (default: None)
  --trust-remote-code, --no-trust-remote-code
                        Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer. (default: False)

LoadConfig:
  Configuration for loading the model weights.

  --download-dir DOWNLOAD_DIR
                        Directory to download and load the weights, default to the default cache directory of Hugging Face. (default: None)
  --ignore-patterns IGNORE_PATTERNS [IGNORE_PATTERNS ...]
                        The list of patterns to ignore when loading the model. Default to "original/**/*" to avoid repeated loading of llama's checkpoints. (default: None)
  --load-format LOAD_FORMAT
                        The format of the model weights to load:
                        - "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.
                        - "pt" will load the weights in the pytorch bin format.
                        - "safetensors" will load the weights in the safetensors format.
                        - "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading.
                        - "dummy" will initialize the weights with random values, which is mainly for profiling.
                        - "tensorizer" will use CoreWeave's tensorizer library for fast weight loading. See the Tensorize vLLM Model script in the Examples section for more
                        information.
                        - "runai_streamer" will load the Safetensors weights using Run:ai Model Streamer.
                        - "bitsandbytes" will load the weights using bitsandbytes quantization.
                        - "sharded_state" will load weights from pre-sharded checkpoint files, supporting efficient loading of tensor-parallel models.
                        - "gguf" will load weights from GGUF format files (details specified in https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
                        - "mistral" will load weights from consolidated safetensors files used by Mistral models. - Other custom values can be supported via plugins. (default: auto)
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format. (default: {})
  --pt-load-map-location PT_LOAD_MAP_LOCATION
                        pt_load_map_location: the map location for loading pytorch checkpoint, to support loading checkpoints can only be loaded on certain devices like "cuda", this
                        is equivalent to {"": "cuda"}. Another supported format is mapping from different devices like from GPU 1 to GPU 0: {"cuda:1": "cuda:0"}. Note that when
                        passed from command line, the strings in dictionary needs to be double quoted for json parsing. For more details, see original doc for `map_location` in
                        https://pytorch.org/docs/stable/generated/torch.load.html (default: cpu)
  --use-tqdm-on-load, --no-use-tqdm-on-load
                        Whether to enable tqdm for showing progress bar when loading model weights. (default: True)

DecodingConfig:
  Dataclass which contains the decoding strategy of the engine.

  --guided-decoding-backend {auto,guidance,outlines,xgrammar}
                        Which engine will be used for guided decoding (JSON schema / regex etc) by default. With "auto", we will make opinionated choices based on request contents
                        and what the backend libraries currently support, so the behavior is subject to change in each release. (default: auto)
  --guided-decoding-disable-additional-properties, --no-guided-decoding-disable-additional-properties
                        If `True`, the `guidance` backend will not use `additionalProperties` in the JSON schema. This is only supported for the `guidance` backend and is used to
                        better align its behaviour with `outlines` and `xgrammar`. (default: False)
  --guided-decoding-disable-any-whitespace, --no-guided-decoding-disable-any-whitespace
                        If `True`, the model will not generate any whitespace during guided decoding. This is only supported for xgrammar and guidance backends. (default: False)
  --guided-decoding-disable-fallback, --no-guided-decoding-disable-fallback
                        If `True`, vLLM will not fallback to a different backend on error. (default: False)
  --reasoning-parser {deepseek_r1,glm45,GptOss,granite,hunyuan_a13b,mistral,qwen3,step3}
                        Select the reasoning parser depending on the model that you're using. This is used to parse the reasoning content into OpenAI API format. (default: )

ParallelConfig:
  Configuration for the distributed execution.

  --data-parallel-address DATA_PARALLEL_ADDRESS, -dpa DATA_PARALLEL_ADDRESS
                        Address of data parallel cluster head-node. (default: None)
  --data-parallel-backend DATA_PARALLEL_BACKEND, -dpb DATA_PARALLEL_BACKEND
                        Backend for data parallel, either "mp" or "ray". (default: mp)
  --data-parallel-hybrid-lb, --no-data-parallel-hybrid-lb
                        Whether to use "hybrid" DP LB mode. Applies only to online serving and when data_parallel_size > 0. Enables running an AsyncLLM and API server on a "per-
                        node" basis where vLLM load balances between local data parallel ranks, but an external LB balances between vLLM nodes/replicas. Set explicitly in
                        conjunction with --data-parallel-start-rank. (default: False)
  --data-parallel-rank DATA_PARALLEL_RANK, -dpn DATA_PARALLEL_RANK
                        Data parallel rank of this instance. When set, enables external load balancer mode. (default: None)
  --data-parallel-rpc-port DATA_PARALLEL_RPC_PORT, -dpp DATA_PARALLEL_RPC_PORT
                        Port for data parallel RPC communication. (default: None)
  --data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE
                        Number of data parallel groups. MoE layers will be sharded according to the product of the tensor parallel size and data parallel size. (default: 1)
  --data-parallel-size-local DATA_PARALLEL_SIZE_LOCAL, -dpl DATA_PARALLEL_SIZE_LOCAL
                        Number of data parallel replicas to run on this node. (default: None)
  --data-parallel-start-rank DATA_PARALLEL_START_RANK, -dpr DATA_PARALLEL_START_RANK
                        Starting data parallel rank for secondary nodes. (default: None)
  --disable-custom-all-reduce, --no-disable-custom-all-reduce
                        Disable the custom all-reduce kernel and fall back to NCCL. (default: False)
  --distributed-executor-backend {external_launcher,mp,ray,uni,None}
                        Backend to use for distributed model workers, either "ray" or "mp" (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size is
                        less than or equal to the number of GPUs available, "mp" will be used to keep processing on a single host. Otherwise, this will default to "ray" if Ray is
                        installed and fail otherwise. Note that tpu only support Ray for distributed inference. (default: None)
  --enable-eplb, --no-enable-eplb
                        Enable expert parallelism load balancing for MoE layers. (default: False)
  --enable-expert-parallel, --no-enable-expert-parallel
                        Use expert parallelism instead of tensor parallelism for MoE layers. (default: False)
  --enable-multimodal-encoder-data-parallel, --no-enable-multimodal-encoder-data-parallel
                        Use data parallelism instead of tensor parallelism for vision encoder. Only support LLama4 for now (default: False)
  --eplb-log-balancedness, --no-eplb-log-balancedness
                        Log the balancedness each step of expert parallelism. This is turned off by default since it will cause communication overhead. (default: False)
  --eplb-step-interval EPLB_STEP_INTERVAL
                        Interval for rearranging experts in expert parallelism.
                        Note that if this is greater than the EPLB window size, only the metrics of the last `eplb_window_size` steps will be used for rearranging experts. (default:
                        3000)
  --eplb-window-size EPLB_WINDOW_SIZE
                        Window size for expert load recording. (default: 1000)
  --max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS
                        Maximum number of parallel loading workers when loading model sequentially in multiple batches. To avoid RAM OOM when using tensor parallel and large models.
                        (default: None)
  --num-redundant-experts NUM_REDUNDANT_EXPERTS
                        Number of redundant experts to use for expert parallelism. (default: 0)
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Number of pipeline parallel groups. (default: 1)
  --ray-workers-use-nsight, --no-ray-workers-use-nsight
                        Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.
                        (default: False)
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Number of tensor parallel groups. (default: 1)
  --worker-cls WORKER_CLS
                        The full name of the worker class to use. If "auto", the worker class will be determined based on the platform. (default: auto)
  --worker-extension-cls WORKER_EXTENSION_CLS
                        The full name of the worker extension class to use. The worker extension class is dynamically inherited by the worker class. This is used to inject new
                        attributes and methods to the worker class for use in collective_rpc calls. (default: )

CacheConfig:
  Configuration for the KV cache.

  --block-size {1,8,16,32,64,128}
                        Size of a contiguous cache block in number of tokens. This is ignored on neuron devices and set to `--max-model-len`. On CUDA devices, only block sizes up to
                        32 are supported. On HPU devices, block size defaults to 128.
                        This config has no static default. If left unspecified by the user, it will be set in `Platform.check_and_update_config()` based on the current platform.
                        (default: None)
  --calculate-kv-scales, --no-calculate-kv-scales
                        This enables dynamic calculation of `k_scale` and `v_scale` when kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model checkpoint if
                        available. Otherwise, the scales will default to 1.0. (default: False)
  --cpu-offload-gb CPU_OFFLOAD_GB
                        The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading. Intuitively, this argument can be seen as a virtual way to increase the
                        GPU memory size. For example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU. Then you can load a 13B model with
                        BF16 weight, which requires at least 26GB GPU memory. Note that this requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to
                        GPU memory on the fly in each model forward pass. (default: 0)
  --enable-prefix-caching, --no-enable-prefix-caching
                        Whether to enable prefix caching. Disabled by default for V0. Enabled by default for V1. (default: None)
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
                        utilization. If unspecified, will use the default value of 0.9. This is a per-instance limit, and only applies to the current vLLM instance. It does not
                        matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU
                        memory utilization to 0.5 for each instance. (default: 0.9)
  --kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2,fp8_inc}
                        Data type for kv cache storage. If "auto", will use model data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports fp8
                        (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc). (default: auto)
  --kv-sharing-fast-prefill, --no-kv-sharing-fast-prefill
                        This feature is work in progress and no prefill optimization takes place with this flag enabled currently.
                        In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254), some layers can skip tokens corresponding to prefill. This flag enables attention
                        metadata for eligible layers to be overriden with metadata necessary for implementating this optimization in some models (e.g. Gemma3n) (default: False)
  --num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE
                        Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks` if specified. Does nothing if `None`. Used for testing preemption. (default: None)
  --prefix-caching-hash-algo {builtin,sha256,sha256_cbor_64bit}
                        Set the hash algorithm for prefix caching:
                        - "builtin" is Python's built-in hash.
                        - "sha256" is collision resistant but with certain overheads. This option uses Pickle for object serialization before hashing.
                        - "sha256_cbor_64bit" provides a reproducible, cross-language compatible hash. It serializes objects using canonical CBOR and hashes them with SHA-256. The
                        resulting hash consists of the lower 64 bits of the SHA-256 digest. (default: builtin)
  --swap-space SWAP_SPACE
                        Size of the CPU swap space per GPU (in GiB). (default: 4)

MultiModalConfig:
  Controls the behavior of multimodal models.

  --disable-mm-preprocessor-cache DISABLE_MM_PREPROCESSOR_CACHE
  --interleave-mm-strings, --no-interleave-mm-strings
                        Enable fully interleaved support for multimodal prompts. (default: False)
  --limit-mm-per-prompt LIMIT_MM_PER_PROMPT
                        The maximum number of input items allowed per prompt for each modality. Defaults to 1 (V0) or 999 (V1) for each modality.
                        For example, to allow up to 16 images and 2 videos per prompt: `{"image": 16, "video": 2}`
                        Should either be a valid JSON string or JSON keys passed individually. (default: {})
  --media-io-kwargs MEDIA_IO_KWARGS
                        Additional args passed to process media inputs, keyed by modalities. For example, to set num_frames for video, set `--media-io-kwargs '{"video":
                        {"num_frames": 40} }'`
                        Should either be a valid JSON string or JSON keys passed individually. (default: {})
  --mm-processor-cache-gb MM_PROCESSOR_CACHE_GB
                        The size (in GiB) of the multi-modal processor cache, which is used to
                        This cache is duplicated for each API process and engine core process, resulting in a total memory usage of `mm_processor_cache_gb * (api_server_count +
                        data_parallel_size)`.
                        Set to `0` to disable this cache completely (not recommended). (default: 4)
  --mm-processor-kwargs MM_PROCESSOR_KWARGS
                        Overrides for the multi-modal processor obtained from `transformers.AutoProcessor.from_pretrained`.
                        The available overrides depend on the model that is being run.
                        For example, for Phi-3-Vision: `{"num_crops": 4}`.
                        Should either be a valid JSON string or JSON keys passed individually. (default: None)

LoRAConfig:
  Configuration for LoRA.

  --default-mm-loras DEFAULT_MM_LORAS
                        Dictionary mapping specific modalities to LoRA model paths; this field is only applicable to multimodal models and should be leveraged when a model always
                        expects a LoRA to be active when a given modality is present. Note that currently, if a request provides multiple additional modalities, each of which have
                        their own LoRA, we do NOT apply default_mm_loras because we currently only support one lora adapter per prompt. When run in offline mode, the lora IDs for n
                        modalities will be automatically assigned to 1-n with the names of the modalities in alphabetic order.
                        Should either be a valid JSON string or JSON keys passed individually. (default: None)
  --enable-lora, --no-enable-lora
                        If True, enable handling of LoRA adapters. (default: None)
  --enable-lora-bias, --no-enable-lora-bias
                        Enable bias for LoRA adapters. (default: False)
  --fully-sharded-loras, --no-fully-sharded-loras
                        By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling this will use the fully sharded layers. At high sequence length,
                        max rank or tensor parallel size, this is likely faster. (default: False)
  --lora-dtype {auto,bfloat16,float16}
                        Data type for LoRA. If auto, will default to base model dtype. (default: auto)
  --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
                        Maximum size of extra vocabulary that can be present in a LoRA adapter (added to the base model vocabulary). (default: 256)
  --max-cpu-loras MAX_CPU_LORAS
                        Maximum number of LoRAs to store in CPU memory. Must be >= than `max_loras`. (default: None)
  --max-lora-rank MAX_LORA_RANK
                        Max LoRA rank. (default: 16)
  --max-loras MAX_LORAS
                        Max number of LoRAs in a single batch. (default: 1)

ObservabilityConfig:
  Configuration for observability - metrics and tracing.

  --collect-detailed-traces {all,model,worker,None} [{all,model,worker,None} ...]
                        It makes sense to set this only if `--otlp-traces-endpoint` is set. If set, it will collect detailed traces for the specified modules. This involves use of
                        possibly costly and or blocking operations and hence might have a performance impact.
                        Note that collecting detailed timing information for each request can be expensive. (default: None)
  --otlp-traces-endpoint OTLP_TRACES_ENDPOINT
                        Target URL to which OpenTelemetry traces will be sent. (default: None)
  --show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION
                        Enable deprecated Prometheus metrics that have been hidden since the specified version. For example, if a previously deprecated metric has been hidden since
                        the v0.7.0 release, you use `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while you migrate to new metrics. The metric is likely to be
                        removed completely in an upcoming release. (default: None)

SchedulerConfig:
  Scheduler configuration.

  --async-scheduling, --no-async-scheduling
                        EXPERIMENTAL: If set to True, perform async scheduling. This may help reduce the CPU overheads, leading to better latency and throughput. However, async
                        scheduling is currently not supported with some features such as structured outputs, speculative decoding, and pipeline parallelism. (default: False)
  --cuda-graph-sizes CUDA_GRAPH_SIZES [CUDA_GRAPH_SIZES ...]
                        Cuda graph capture sizes 1. if none provided, then default set to [min(max_num_seqs * 2, 512)] 2. if one value is provided, then the capture list would
                        follow the pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)] 3. more than one value (e.g. 1 2 128) is provided, then the capture list will
                        follow the provided list. (default: [])
  --disable-chunked-mm-input, --no-disable-chunked-mm-input
                        If set to true and chunked prefill is enabled, we do not want to partially schedule a multimodal item. Only used in V1 This ensures that if a request has a
                        mixed prompt (like text tokens TTTT followed by image tokens IIIIIIIIII) where only some image tokens can be scheduled (like TTTTIIIII, leaving IIIII), it
                        will be scheduled as TTTT in one step and IIIIIIIIII in the next. (default: False)
  --disable-hybrid-kv-cache-manager, --no-disable-hybrid-kv-cache-manager
                        If set to True, KV cache manager will allocate the same size of KV cache for all attention layers even if there are multiple type of attention layers like
                        full attention and sliding window attention. (default: False)
  --enable-chunked-prefill, --no-enable-chunked-prefill
                        If True, prefill requests can be chunked based on the remaining max_num_batched_tokens. (default: None)
  --long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD
                        For chunked prefill, a request is considered long if the prompt is longer than this number of tokens. (default: 0)
  --max-long-partial-prefills MAX_LONG_PARTIAL_PREFILLS
                        For chunked prefill, the maximum number of prompts longer than long_prefill_token_threshold that will be prefilled concurrently. Setting this less than
                        max_num_partial_prefills will allow shorter prompts to jump the queue in front of longer prompts in some cases, improving latency. (default: 1)
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
                        Maximum number of tokens to be processed in a single iteration.
                        This config has no static default. If left unspecified by the user, it will be set in `EngineArgs.create_engine_config` based on the usage context. (default:
                        None)
  --max-num-partial-prefills MAX_NUM_PARTIAL_PREFILLS
                        For chunked prefill, the maximum number of sequences that can be partially prefilled concurrently. (default: 1)
  --max-num-seqs MAX_NUM_SEQS
                        Maximum number of sequences to be processed in a single iteration.
                        This config has no static default. If left unspecified by the user, it will be set in `EngineArgs.create_engine_config` based on the usage context. (default:
                        None)
  --multi-step-stream-outputs, --no-multi-step-stream-outputs
                        If False, then multi-step will stream outputs at the end of all steps (default: True)
  --num-lookahead-slots NUM_LOOKAHEAD_SLOTS
                        The number of slots to allocate per sequence per step, beyond the known token ids. This is used in speculative decoding to store KV activations of tokens
                        which may or may not be accepted.
                        NOTE: This will be replaced by speculative config in the future; it is present to enable correctness tests until then. (default: 0)
  --num-scheduler-steps NUM_SCHEDULER_STEPS
                        Maximum number of forward steps per scheduler call. (default: 1)
  --preemption-mode {recompute,swap,None}
                        Whether to perform preemption by swapping or recomputation. If not specified, we determine the mode as follows: We use recomputation by default since it
                        incurs lower overhead than swapping. However, when the sequence group has multiple sequences (e.g., beam search), recomputation is not currently supported.
                        In such a case, we use swapping instead. (default: None)
  --scheduler-cls SCHEDULER_CLS
                        The scheduler class to use. "vllm.core.scheduler.Scheduler" is the default scheduler. Can be a class directly or the path to a class of form
                        "mod.custom_class". (default: vllm.core.scheduler.Scheduler)
  --scheduler-delay-factor SCHEDULER_DELAY_FACTOR
                        Apply a delay (of delay factor multiplied by previous prompt latency) before scheduling next prompt. (default: 0.0)
  --scheduling-policy {fcfs,priority}
                        The scheduling policy to use:
                        - "fcfs" means first come first served, i.e. requests are handled in order of arrival.
                        - "priority" means requests are handled based on given priority (lower value means earlier handling) and time of arrival deciding any ties). (default: fcfs)

VllmConfig:
  Dataclass which contains all vllm-related configuration. This
      simplifies passing around the distinct configurations in the codebase.


  --additional-config ADDITIONAL_CONFIG
                        Additional config for specified platform. Different platforms may support different configs. Make sure the configs are valid for the platform you are using.
                        Contents must be hashable. (default: {})
  --compilation-config COMPILATION_CONFIG, -O COMPILATION_CONFIG
                        `torch.compile` and cudagraph capture configuration for the model.
                        As a shorthand, `-O<n>` can be used to directly specify the compilation level `n`: `-O3` is equivalent to `-O.level=3` (same as `-O='{"level":3}'`).
                        Currently, -O <n> and -O=<n> are supported as well but this will likely be removed in favor of clearer -O<n> syntax in the future.
                        NOTE: level 0 is the default level without any optimization. level 1 and 2 are for internal testing only. level 3 is the recommended level for production,
                        also default in V1.
                        You can specify the full compilation config like so: `{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
                        Should either be a valid JSON string or JSON keys passed individually. (default: {"level":null,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":
                        [],"splitting_ops":[],"use_inductor":true,"compile_sizes":null,"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cu
                        dagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":null,"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture
                        _size":null,"local_cache_dir":null})
  --kv-events-config KV_EVENTS_CONFIG
                        The configurations for event publishing.
                        Should either be a valid JSON string or JSON keys passed individually. (default: None)
  --kv-transfer-config KV_TRANSFER_CONFIG
                        The configurations for distributed KV cache transfer.
                        Should either be a valid JSON string or JSON keys passed individually. (default: None)
  --speculative-config SPECULATIVE_CONFIG
                        Speculative decoding configuration.
                        Should either be a valid JSON string or JSON keys passed individually. (default: None)

When passing JSON CLI arguments, the following sets of arguments are equivalent:
   --json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'
   --json-arg.key1 value1 --json-arg.key2.key3 value2

Additionally, list elements can be passed individually using +:
   --json-arg '{"key4": ["value3", "value4", "value5"]}'
   --json-arg.key4+ value3 --json-arg.key4+='value4,value5'

Tip: Use `vllm [serve|run-batch|bench <bench_type>] --help=<keyword>` to explore arguments from help.
   - To view a argument group:     --help=ModelConfig
   - To view a single argument:    --help=max-num-seqs
   - To search by keyword:         --help=max
   - To list all groups:           --help=listgroup
   - To view help with pager:      --help=page
```
