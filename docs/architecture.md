# Architecture

## Overview

The benchmark runs models sequentially: **start server -> run all prompts -> collect metrics -> stop server -> next model**.

```
bench.py run -c config.json
    |
    v
load_plan()          # Parse JSON config into ServerConfig + BenchmarkPlan
    |
    v
run_benchmark()      # Main loop in runner.py
    |
    +-- for each config:
        |
        +-- kill_orphans()       # Clean up stale servers on the port
        +-- start_server()       # Launch llama-server or mlx_vlm.server
        +-- wait_for_health()    # Poll /health until ready
        |
        +-- for each prompt:
        |   +-- run_single_prompt()   # POST /v1/chat/completions
        |   +-- evaluate()            # Auto-score the response
        |   +-- save_result()         # Write to SQLite
        |
        +-- stop_server()        # SIGTERM -> SIGKILL if needed
```

## Backends

### llama-server (GGUF)

Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) server binary. Models are GGUF files from HuggingFace.

- Binary: `llama-server` (PATH or `/opt/homebrew/bin/llama-server`)
- CLI args: `-m path`, `-c ctx_size`, `-ngl layers`, `-b batch`, `-fa on/off`, `--port`, `--metrics`
- Health endpoint: `GET /health` returns `{"status": "ok"}`
- Response format: OpenAI-compatible `/v1/chat/completions` with `usage` and `timings` fields

### mlx_vlm (MLX)

Uses [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) server for Apple Silicon. Models are safetensors from HuggingFace.

- Binary: `python -m mlx_vlm.server`
- CLI args: `--model repo_id`, `--port`
- Health endpoint: `GET /health` returns `{"status": "healthy", "loaded_model": "..."}`
- Response format: OpenAI-compatible `/v1/chat/completions` (no `timings` field, `usage` may be empty)

Key differences from llama-server:
- No `ctx_size`, `n_gpu_layers`, `batch_size`, or `flash_attn` — MLX manages these automatically
- Always pass the HF repo ID (not local paths) to avoid incomplete download issues
- The `model` field is **required** in request payloads — without it, mlx_vlm falls back to a default model
- Health check verifies `loaded_model` is not None to catch load failures

## Modules

| Module | Purpose |
|---|---|
| `bench.py` | CLI entry point, argparse subcommands |
| `config.py` | `ServerConfig` and `BenchmarkPlan` dataclasses, model path resolution, CLI arg generation |
| `server.py` | Server lifecycle: start, health check, stop, orphan cleanup |
| `runner.py` | Benchmark orchestration, prompt loading, model prefetching, smoke testing |
| `metrics.py` | Background RSS sampling via psutil |
| `storage.py` | SQLite persistence with auto-migrations |
| `evaluate.py` | Auto-scoring (keyword matching, code execution, pattern checks) |
| `judge.py` | LLM-as-judge evaluation with rubric |
| `report.py` | Rich terminal tables and markdown reports |
| `models.py` | HF cache scanning, disk usage, cleanup |

## Database

SQLite at `results/benchmark.db` with two tables:

**runs** — one row per model configuration benchmarked:
- `run_id`, `timestamp`, `model_path`, `hf_repo`, `model_label`
- `quantization`, `param_count`, `ctx_size`, `n_gpu_layers`, `threads`, `batch_size`
- `flash_attn`, `extra_args` (JSON), `backend`

**results** — one row per prompt execution:
- `run_id`, `category`, `prompt_name`, `prompt_text`, `response_text`
- `prompt_tokens`, `completion_tokens`, `generation_tps`, `prompt_eval_tps`
- `time_to_first_token_ms`, `total_time_sec`, `peak_rss_mb`
- `quality_score`, `quality_notes` (auto-eval)
- `judge_score`, `judge_reason` (LLM-as-judge)
- `iteration`, `finish_reason`, `truncated`

## Model Prefetching

The runner downloads upcoming models in background threads while the current benchmark runs. For GGUF models, it downloads specific files via `hf_hub_download`. For MLX models, it downloads the full repo via `snapshot_download`. The prefetcher looks ahead up to 2 models from the current position.
