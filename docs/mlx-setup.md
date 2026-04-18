# MLX Setup (Apple Silicon)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- `mlx-vlm` package

## Installation

```bash
pip install mlx-vlm
```

Or use unsloth's dedicated installer for Gemma 4:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/scripts/install_gemma4_mlx.sh | sh
```

## Creating a config

Set `"backend": "mlx"` in your config:

```json
{
  "configs": [
    {
      "label": "Gemma4-E4B-MLX-4bit",
      "hf_repo": "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
      "quantization": "MLX-4bit",
      "param_count": "E4B",
      "backend": "mlx",
      "port": 8999
    }
  ],
  "max_tokens": 16384
}
```

MLX configs ignore llama-specific fields (`ctx_size`, `n_gpu_layers`, `batch_size`, `flash_attn`, `threads`).

## Available models

Look for MLX-format repos on HuggingFace from `unsloth` or `mlx-community`. For Gemma 4:

| Model | Repo | Quantization |
|---|---|---|
| Gemma 4 E2B | `unsloth/gemma-4-E2B-it-UD-MLX-4bit` | 4-bit |
| Gemma 4 E4B | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` | 4-bit |
| Gemma 4 E4B | `unsloth/gemma-4-E4B-it-MLX-8bit` | 8-bit |
| Gemma 4 26B-A4B | `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` | 4-bit |
| Gemma 4 26B-A4B | `unsloth/gemma-4-26b-a4b-it-MLX-8bit` | 8-bit |
| Gemma 4 31B | `unsloth/gemma-4-31b-it-UD-MLX-4bit` | 4-bit |

`UD` = unsloth dynamic quantization. 8-bit models may require newer `mlx-vlm` versions.

## Testing

Always smoke test before running a full benchmark:

```bash
python3 bench.py test -c configs/gemma4-mlx.json
```

## Running benchmarks

```bash
python3 bench.py run -c configs/gemma4-mlx.json
```

Models are downloaded automatically on first use (~3-15 GB depending on size). The health check timeout is 600s for MLX to allow for large downloads.

## Using MLX as judge

```bash
python3 bench.py judge --backend mlx --hf-repo unsloth/gemma-4-E4B-it-UD-MLX-4bit
```

## Troubleshooting

### "Server failed to start"

Check the log in `results/logs/`:

```bash
cat results/logs/<run_id>_<label>.log
```

Common causes:
- **"Missing N parameters"**: Model architecture not supported by your `mlx-vlm` version. Upgrade with `pip install -U mlx-vlm`.
- **"No safetensors found"**: Incomplete download. Delete the cached model and re-run:
  ```bash
  python3 bench.py models --clean
  ```
- **Out of memory**: The model is too large for your unified memory. Try a smaller quantization or model.

### Model swaps to nanoLLaVA

If results look wrong (identical scores, generic responses), the server may have fallen back to its default model. This was caused by missing the `model` field in request payloads — fixed in current code. Upgrade to the latest version.

### Metrics show 0 TPS

`mlx_vlm` doesn't always populate the `usage` field in responses. The runner estimates tokens from response length (~4 chars/token). This gives approximate but usable TPS numbers.

### Server uses `mlx_vlm` not `mlx_lm`

Gemma 4 is a multimodal architecture. Even for text-only inference, it requires `mlx_vlm` (the vision-language model package), not `mlx_lm`.
