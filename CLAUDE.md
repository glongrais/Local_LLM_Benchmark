# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python benchmarking framework for comparing local LLMs via `llama-server` (llama.cpp) and `mlx_vlm.server` (Apple MLX). Designed for an M4 Max Mac with 64GB unified memory. Models come from HuggingFace as GGUF (llama) or safetensors (MLX).

## Commands

```bash
pip install -r requirements.txt          # Install deps: requests, psutil, rich, numpy

python3 bench.py run -c configs/example.json   # Run benchmarks
python3 bench.py report                        # Show comparison table
python3 bench.py report --run-ids ID1 ID2      # Compare specific runs
python3 bench.py list                          # List all runs
python3 bench.py score <run_id>                # Interactive quality scoring (0-10)
python3 bench.py show <run_id>                 # Show detailed results
python3 bench.py export --output results.json  # Export as JSON
python3 bench.py models                        # List GGUF files in HF cache with sizes
python3 bench.py models --clean                # Interactive cleanup (protects benchmarked models)
python3 bench.py models --clean --all          # Include benchmarked models in cleanup
python3 bench.py models ~/custom/dir           # Scan custom directory instead of HF cache
python3 bench.py autoscore                       # Re-run auto-evaluation on all existing results
python3 bench.py autoscore --run-ids ID1 ID2     # Re-score specific runs
python3 bench.py run -c config.json --skip-existing  # Skip configs already in the DB
python3 bench.py test -c config.json               # Quick smoke test (boot + one prompt per model)
python3 bench.py judge                         # Score results using running server on port 8080
python3 bench.py judge --hf-repo repo:file     # Start a judge model and score all results
python3 bench.py judge --overwrite             # Re-judge already scored results
python3 bench.py judge --backend mlx --hf-repo repo  # Judge with MLX model
python3 bench.py purge <run_id> [run_id...]    # Delete runs from the DB
python3 bench.py autoscore --workers 8         # Parallel auto-scoring
.venv/bin/python -m pytest project_tests/ -v   # Run project tests (78 tests)
```

## Architecture

The benchmark flow is sequential per config: **start llama-server → run all prompts → collect metrics → stop server → next config**.

- `config.py` — `ServerConfig` and `BenchmarkPlan` dataclasses. `backend` field selects `"llama"` or `"mlx"`. Models specified via `hf_repo`/`hf_file` or `model_path`. `resolve_model_path()` checks local HF cache (GGUF files for llama, `config.json` dirs for MLX). `to_cli_args()` delegates to `_llama_cli_args()` or `_mlx_cli_args()`. The `extra_args` dict passes arbitrary flags.
- `server.py` — Starts/stops `llama-server` or `mlx_vlm.server` based on `config.backend`. Polls `/health` endpoint until ready (up to 180s). Health check accepts `"ok"` (llama) and `"healthy"` (mlx_vlm). Logs to `results/logs/`.
- `runner.py` — Main orchestration loop. Calls `/v1/chat/completions` (OpenAI-compatible endpoint) with `stream=false` and `temperature=0` for deterministic output. Extracts token counts from `usage` and TPS from `timings` if available.
- `metrics.py` — Background thread samples RSS of the llama-server process via psutil during each request. Returns peak RSS.
- `storage.py` — SQLite at `results/benchmark.db`. Two tables: `runs` (one per server config) and `results` (one per prompt execution). Quality scores are nullable and filled via `bench.py score`.
- `report.py` — Generates Rich terminal tables and markdown files in `results/`. Best values highlighted in green.
- `models.py` — GGUF file discovery, disk usage display, and interactive cleanup. Scans HF cache (`~/.cache/huggingface/hub/`) by default. Cross-references with benchmark DB to identify unbenchmarked models safe to delete.
- `tests/` — Test harness files for executable and ml prompts. Each file `from solution import ...` and runs assertions. Convention: print `PASS test_name` or `FAIL test_name (detail)` per test, end with `RESULT passed/total`.
- `evaluate.py` — Hybrid auto-evaluator. Scoring varies by category: coding/math/reasoning/general use keyword matching and pattern checks. executable/ml use file-based testing: writes response as `solution.py` in a tmpdir, runs `tests/test_*.py` which imports from it, parses `RESULT x/y` output. agentic_coding uses code substance + feature checks. `_clean_response_to_python()` handles both raw Python and markdown-wrapped responses.
- `judge.py` — LLM-as-judge evaluation. Sends each response + rubric to a judge model, parses JSON score from response (handles markdown fences, partial JSON, thinking model output). Stores `judge_score` and `judge_reason` in the results table.
- `project_tests/` — pytest test suite (78 tests) covering config, storage, models, evaluate, judge, and runner. Uses tmp DBs and mock HF cache. NOT the same as `tests/` (benchmark prompt harnesses).

## Prompt Test Suite

JSON files in `prompts/` with 33 prompts across 7 categories: coding (4), reasoning (4), math (4), general (3), agentic_coding (4), executable (8), ml (6). Each prompt has `name`, `category`, `system`, `prompt`, `max_tokens`, and `reference_keywords`. The executable and ml categories use `test_file` pointing to `tests/test_*.py` — the model's response is saved as `solution.py` and the test file imports from it. Test files print `PASS`/`FAIL` per test and `RESULT x/y` at the end. ML prompts have `eval_timeout` (up to 120s for RL training). System prompts for these categories ask for raw Python output (no markdown). The agentic_coding prompts need `max_tokens: 16384`.

## Key Design Decisions

- Uses `/v1/chat/completions` (not `/completion`) so chat templates are applied automatically by llama-server.
- RSS captures both CPU and GPU memory on Apple Silicon (unified memory).
- `n_gpu_layers: -1` offloads all layers to Metal by default.
- Models are primarily loaded via `-hf` (HuggingFace repo ID), not local file paths. The HF cache at `~/.cache/huggingface/hub/` stores downloaded GGUF files.
- Results DB is gitignored; reports in `results/` are also gitignored.
- Two quality scores exist: `judge_score` (LLM-as-judge, preferred) and `quality_score` (auto-evaluator, fallback). Use `COALESCE(judge_score, quality_score)` for combined analysis.

## Gotchas

- llama-server `-fa` flag requires a value: `-fa on`, not bare `-fa`. The `to_cli_args()` method handles this.
- Thinking models (Gemma 4, QwQ, etc.) consume tokens on internal reasoning before producing visible output. Use `max_tokens: 4096+` or responses will be empty/truncated. The runner also checks `reasoning_content`/`thinking` fields as fallback.
- The `--skip-existing` flag matches on `ServerConfig.label`. Labels must be unique across configs for this to work correctly.
- Runs are append-only: `bench.py run` always creates new `run_id` entries, never overwrites. Use `--skip-existing` to avoid duplicates.
- HF cache files under `snapshots/` are symlinks to `blobs/<hash>`. `models.py` uses symlink paths for display but deduplicates by resolved blob target.
- Default ports: benchmark server=8999, judge=8090. User has other services on 8080.
- `wait_for_health()` checks `proc.poll()` to detect early crashes — prevents benchmarking against a stale server from a previous run that's still listening on the same port.
- Models may append EOS tokens (`<eos>`, `<|im_end|>`) that break code compilation. `_clean_response_to_python()` strips these and trims trailing lines until code compiles.
- `storage.py` has a `MIGRATIONS` list that auto-adds new columns (judge_score, finish_reason, truncated) to existing DBs.
- Judge needs large context (default 32768) to evaluate long responses. Response truncation is at 20K chars.
- `bench.py purge` prompts for confirmation interactively. Pipe `echo "y"` for non-interactive use.
- MLX backend uses `mlx_vlm` (not `mlx_lm`) because Gemma 4 is a multimodal architecture. Even text-only inference needs the VLM package.
- `sqlite3.Row` does NOT have `.get()` — use `row["key"]` with bracket access. `.get()` raises `AttributeError` which broad `except Exception` blocks silently swallow.
- MLX models don't use ctx_size, n_gpu_layers, batch_size, or flash_attn — these are llama-server concepts. MLX manages context and GPU automatically.
- `mlx_vlm.server` requires `"model"` field in the request payload — without it, it silently falls back to the default `nanoLLaVA-1.5-8bit` model. The runner passes `config.hf_repo` as the model value.
- MLX CLI args always use the HF repo ID (never local cache paths) — incomplete downloads (metadata without weights) cause silent failures.
- Storage migrations now span both `runs` and `results` tables. Each migration entry is `(col_name, table, sql)`.
