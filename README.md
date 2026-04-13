# Local LLM Benchmark

A benchmarking framework for comparing local LLMs running via [llama.cpp](https://github.com/ggerganov/llama.cpp) server. Test different models, sizes, quantizations, and server settings to find the best configuration for your hardware.

Built for Apple Silicon Macs but works on any machine running `llama-server`.

## What it measures

- **Speed**: generation tokens/s, prompt eval tokens/s, time to complete
- **Memory**: peak RSS (captures unified memory on Apple Silicon)
- **Quality**: auto-scoring via code execution + test suites, keyword/pattern matching, and LLM-as-judge evaluation

## Prompt test suite

33 prompts across 7 categories:

| Category | Prompts | Scoring method |
|---|---|---|
| **Coding** | FizzBuzz, binary search, REST API, debugging | Keyword matching + code execution |
| **Reasoning** | River crossing, knights & knaves, lateral thinking, causal reasoning | Pattern matching for correct conclusions |
| **Math** | Arithmetic, algebra, probability, optimization | Exact answer checking (handles LaTeX) |
| **General** | Summarization, comparison, instruction following | Structural checks (word count, format) |
| **Agentic coding** | Full project from spec, refactoring, async debugging, architecture decisions | Code substance + feature coverage |
| **Executable** | Sorting, dict flattening, cron parsing, matrix ops, graphs, LRU cache, expression eval | Code saved as file, run against hidden test suite |
| **ML** | Neural net, K-means, linear regression, CartPole RL, decision tree, genetic algorithm | Code run against performance thresholds (accuracy, reward) |

The **executable** and **ML** categories test agentic capabilities: the model's response is saved directly as a `.py` file and executed against test harnesses in `tests/`.

## Quick start

```bash
# Install
pip install -r requirements.txt

# Run benchmarks (edit config with your models first)
python3 bench.py run -c configs/quick.json

# View results
python3 bench.py report

# Score with a judge model
python3 bench.py judge --hf-repo unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

## Configuration

Benchmark configs are JSON files listing models and settings to test. Models are loaded from HuggingFace via the `-hf` flag (downloaded automatically) or from local GGUF files.

```json
{
  "configs": [
    {
      "label": "Qwen3-8B-Q4_K_M",
      "hf_repo": "unsloth/Qwen3-8B-GGUF",
      "hf_file": "Qwen3-8B-Q4_K_M.gguf",
      "quantization": "Q4_K_M",
      "param_count": "8B",
      "ctx_size": 8192,
      "flash_attn": true
    }
  ],
  "prompt_dirs": ["prompts/"],
  "max_tokens": 8192,
  "temperature": 0.0
}
```

### Included configs

| Config | Purpose |
|---|---|
| `configs/quick.json` | Fast test with small models (Gemma 4 E2B/E4B) |
| `configs/gemma4-full.json` | All Gemma 4 sizes and quantizations |
| `configs/multi-model.json` | Cross-model comparison (Gemma 4, Qwen 3, Qwen 3.5, DeepSeek-R1) |
| `configs/server-settings.json` | Test llama-server flags (context size, KV cache quant, flash attention, batch size, threads) |

### Server settings to benchmark

Use `extra_args` to test llama-server flags:

```json
{
  "label": "kv-cache-q4",
  "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
  "hf_file": "gemma-4-E4B-it-Q4_K_M.gguf",
  "extra_args": {"cache-type-k": "q4_0", "cache-type-v": "q4_0"}
}
```

## Commands

```bash
# Benchmarking
python3 bench.py run -c config.json              # Run all configs
python3 bench.py run -c config.json --skip-existing  # Skip already-benchmarked configs

# Results
python3 bench.py report                           # Comparison table
python3 bench.py report --run-ids ID1 ID2         # Compare specific runs
python3 bench.py list                             # List all runs
python3 bench.py show <run_id>                    # Detailed results

# Scoring
python3 bench.py autoscore                        # Re-run auto-evaluation
python3 bench.py autoscore --workers 8            # Parallel scoring
python3 bench.py judge --hf-repo repo:file        # LLM-as-judge scoring
python3 bench.py score <run_id>                   # Manual interactive scoring

# Model management
python3 bench.py models                           # List downloaded GGUF files
python3 bench.py models --clean                   # Delete unused models

# Maintenance
python3 bench.py purge <run_id>                   # Delete a run
python3 bench.py export -o results.json           # Export data
```

## How scoring works

Three layers of evaluation:

1. **Auto-score** (immediate, during benchmark run): category-specific checks. Executable/ML prompts run the code against test suites. Coding/math/reasoning use pattern matching and answer verification.

2. **LLM-as-judge** (separate step): sends each response to a judge model with a scoring rubric. Returns correctness (0-4), completeness (0-3), and quality (0-3) scores.

3. **Manual scoring** (optional): interactive 0-10 scoring via `bench.py score`.

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) server (`llama-server` on PATH or via Homebrew)
- GGUF models from [HuggingFace](https://huggingface.co) (downloaded automatically)
