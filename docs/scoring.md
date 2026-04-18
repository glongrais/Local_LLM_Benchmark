# Scoring

Three layers of evaluation, each producing a 0-10 score:

1. **Auto-score** — immediate, runs during benchmark
2. **LLM-as-judge** — separate step, uses another model to evaluate
3. **Manual score** — optional, interactive human scoring

When querying results, use `COALESCE(judge_score, quality_score)` to prefer judge scores with auto-score as fallback.

## Auto-scoring by category

### Coding (50% keywords + 50% code execution)

- Extracts code blocks from the response (fenced, indented, or heuristic)
- Runs code with built-in test cases (e.g. fizzbuzz assertions)
- Code passes tests: 5 pts. Runs but unclear: 3 pts. Syntax error: 1.5 pts. No code: 0 pts.
- Keywords (e.g. `def fizzbuzz`, `range`, `FizzBuzz`): up to 5 pts based on match ratio

### Math (60% final answer + 40% intermediate steps)

- Checks the final numeric answer against expected values with tolerance
- Handles LaTeX formatting (`$`, `\frac{}{}`, `\times`), currency (`$1,234.56`), and text forms (`1/4`, `25%`)
- Checks for intermediate calculation steps (e.g. `63.0`, `5.355`, `68.355`)
- Tolerances vary by prompt (0.02 to 1.0)

### Reasoning (60% correctness + 40% keywords)

- Pattern-matches for correct conclusions (e.g. "A is a knight", "B is a knave")
- Checks for sufficient detail (e.g. min 5 steps for river crossing)
- Verifies key concepts are mentioned (e.g. "correlation", "causation", "confound")

### General (50% keywords + 50% structure)

- **Summarization**: word count within limit (<=250 words = full marks)
- **Comparison**: covers requested aspects (architecture, efficiency, learning curve, tooling, when to use)
- **Instruction following**: correct sentence count (5), LEARN acrostic ordering, numbers in each sentence

### Agentic Coding (30% keywords + 30% substance + 20% architecture + 20% prompt-specific)

- **Substance**: code lines (100+ ideal), function count (5+ ideal), class count (2+ ideal)
- **Architecture**: error handling, type hints, docstrings, dependency injection, separation of concerns
- **Prompt-specific**: checks for required features (e.g. argparse, topological sort, dry-run support)

### Executable & ML (file-based testing)

- Model's response is saved as `solution.py`
- External test file (`tests/test_*.py`) imports from it and runs assertions
- Test files print `PASS test_name` or `FAIL test_name (detail)` and end with `RESULT x/y`
- Score: `(passed / total_tests) * 10`
- ML prompts have extended timeouts (30-120s) for training

## LLM-as-judge

Run separately after benchmarks complete:

```bash
python3 bench.py judge --hf-repo unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
python3 bench.py judge --backend mlx --hf-repo unsloth/gemma-4-E4B-it-UD-MLX-4bit
```

The judge model scores each response on a rubric:

| Component | Range | What it measures |
|---|---|---|
| Correctness | 0-4 | Factual accuracy, code correctness, calculation accuracy |
| Completeness | 0-3 | All parts of the prompt addressed |
| Quality | 0-3 | Structure, clarity, code cleanliness |
| **Total** | **0-10** | Sum of components |

The judge also provides a one-sentence reason for the score.

### Judge settings

| Flag | Default | Description |
|---|---|---|
| `--hf-repo` | — | HuggingFace repo for judge model (starts a server) |
| `--hf-file` | `""` | Specific GGUF file. Also supports `repo:file` shorthand |
| `--port` | `8090` | Port for judge server |
| `--ctx-size` | `32768` | Context size (needs to be large for long responses) |
| `--backend` | `llama` | Server backend (`llama` or `mlx`) |
| `--overwrite` | off | Re-judge already scored results |
| `--run-ids` | all | Specific run IDs to judge |

Response text is truncated to 20K chars and prompts to 4K chars to fit the judge's context window.

## Manual scoring

```bash
python3 bench.py score <run_id>
```

Shows each response and prompts for a 0-10 score. Scores are saved to `quality_score` in the database.

## Re-scoring

```bash
python3 bench.py autoscore                    # Re-run auto-eval on all results
python3 bench.py autoscore --run-ids ID1 ID2  # Specific runs
python3 bench.py autoscore --workers 8        # Parallel scoring
```

This re-evaluates responses using the current scoring logic — useful after updating evaluation code.
