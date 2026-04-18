# Server Settings Tuning

Use `extra_args` in your config to pass additional flags to `llama-server`. This lets you benchmark the effect of different server configurations on speed, memory, and quality.

## Flags to experiment with

### KV Cache Quantization

Reduces memory usage by quantizing the key-value cache:

```json
{
  "label": "kv-cache-q8",
  "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
  "extra_args": {"cache-type-k": "q8_0", "cache-type-v": "q8_0"}
}
```

Options: `f16` (default), `q8_0`, `q4_0`. Lower precision = less memory, potentially lower quality.

### Context Size

Controls the maximum context window:

```json
{"ctx_size": 4096}   // fast, less memory, may truncate long prompts
{"ctx_size": 8192}   // good default
{"ctx_size": 16384}  // more context, more memory
{"ctx_size": 32768}  // large context, significantly more memory
```

### Flash Attention

Hardware-accelerated attention computation:

```json
{"flash_attn": true}    // default, faster on supported hardware
{"flash_attn": false}   // disable if causing issues
```

Note: `llama-server` requires `-fa on` or `-fa off` (not bare `-fa`). The config handles this automatically.

### Batch Size

Processing batch size for prompt evaluation:

```json
{"batch_size": 512}    // lower memory, slower prompt eval
{"batch_size": 2048}   // default
{"batch_size": 4096}   // faster prompt eval, more memory
```

### Thread Count

CPU threads for computation:

```json
{"threads": 0}   // default (server decides)
{"threads": 4}   // limit to 4 threads
{"threads": 8}   // use 8 threads
```

### Other Flags

Any `llama-server` flag can be passed via `extra_args`:

```json
{
  "extra_args": {
    "cache-type-k": "q4_0",
    "cache-type-v": "q4_0",
    "mlock": true
  }
}
```

Boolean `true` produces a bare flag (`--mlock`). Boolean `false` or `null` omits the flag. String/number values produce `--flag value`.

## Example: server settings comparison

See `configs/server-settings.json` for a comprehensive config that tests:
- Context sizes: 4K, 8K (baseline), 16K, 32K
- Flash attention: on vs off
- KV cache: f16 (default), q8_0, q4_0
- Batch sizes: 512, 2048 (default), 4096
- Thread counts: 4, 8, default

Run with:

```bash
python3 bench.py run -c configs/server-settings.json
python3 bench.py report
```

## MLX models

MLX models don't use any of these settings — context, GPU layers, batch size, and attention are managed automatically by the MLX framework. The only configurable settings are `port` and `extra_args` for mlx_vlm-specific flags.
