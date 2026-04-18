# Adding Models

## Config file format

Create a JSON file in `configs/` with one or more model configurations:

```json
{
  "configs": [
    { ... model 1 ... },
    { ... model 2 ... }
  ],
  "prompt_dirs": ["prompts/"],
  "max_tokens": 16384,
  "temperature": 0.0
}
```

## GGUF models (llama-server)

```json
{
  "label": "Qwen3-8B-Q4_K_M",
  "hf_repo": "unsloth/Qwen3-8B-GGUF",
  "hf_file": "Qwen3-8B-Q4_K_M.gguf",
  "quantization": "Q4_K_M",
  "param_count": "8B",
  "ctx_size": 8192,
  "flash_attn": true
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `label` | yes | — | Unique name for this config (used by `--skip-existing`) |
| `hf_repo` | yes* | `""` | HuggingFace repo ID (e.g. `"unsloth/Qwen3-8B-GGUF"`) |
| `hf_file` | no | `""` | Specific GGUF file in the repo. Supports partial match (e.g. `"Q4_K_M"`) |
| `model_path` | yes* | `""` | Alternative: local path to a `.gguf` file |
| `quantization` | no | `""` | For display/reporting only |
| `param_count` | no | `""` | For display/reporting only |
| `ctx_size` | no | `4096` | Context window size |
| `n_gpu_layers` | no | `-1` | GPU layers to offload (`-1` = all) |
| `threads` | no | `0` | CPU threads (`0` = server default) |
| `batch_size` | no | `2048` | Processing batch size |
| `flash_attn` | no | `true` | Enable flash attention |
| `port` | no | `8999` | Server port |
| `extra_args` | no | `{}` | Additional CLI flags (see [Server Settings](server-settings.md)) |

*Either `hf_repo` or `model_path` must be specified.

Models are downloaded automatically on first use. The runner prefetches the next model in the background while the current one is benchmarking.

## MLX models (Apple Silicon)

```json
{
  "label": "Gemma4-E4B-MLX-4bit",
  "hf_repo": "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
  "quantization": "MLX-4bit",
  "param_count": "E4B",
  "backend": "mlx",
  "port": 8999
}
```

Set `"backend": "mlx"` to use `mlx_vlm.server` instead of `llama-server`.

MLX configs ignore `ctx_size`, `n_gpu_layers`, `batch_size`, `flash_attn`, and `threads` — MLX manages these automatically.

MLX models use safetensors format. Look for repos with `MLX` in the name on HuggingFace (e.g. from `unsloth` or `mlx-community`).

## Plan-level settings

| Field | Default | Description |
|---|---|---|
| `prompt_dirs` | `["prompts/"]` | Directories containing prompt JSON files |
| `repeat` | `1` | Number of times to repeat each prompt |
| `max_tokens` | `16384` | Max tokens per response (overridden by per-prompt `max_tokens`) |
| `temperature` | `0.0` | Sampling temperature (`0.0` = deterministic) |

## Testing a config

Before running a full benchmark, use the smoke test:

```bash
python3 bench.py test -c configs/my-config.json
```

This boots each model, sends one short prompt, and reports pass/fail in seconds.

## Tips

- **Labels must be unique** across configs when using `--skip-existing`
- **Thinking models** (Gemma 4, QwQ) need `max_tokens: 4096+` or responses will be empty
- **Multi-file GGUF repos**: use `hf_file` to select a specific quantization
- **`mmproj` files** are automatically excluded from model resolution
- Models are **downloaded automatically** — no need to pre-download
