"""Test execution engine."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

from config import BenchmarkPlan, ServerConfig
from evaluate import evaluate
from metrics import MemorySampler, check_available_memory
from server import kill_orphans, start_server, stop_server, wait_for_health
from storage import get_db, save_result, save_run

console = Console()


def load_prompts(prompt_dirs: list[str]) -> list[dict]:
    """Load all prompt JSON files from the given directories."""
    prompts = []
    for d in prompt_dirs:
        path = Path(d)
        if path.is_file() and path.suffix == ".json":
            files = [path]
        elif path.is_dir():
            files = sorted(path.glob("*.json"))
        else:
            console.print(f"  [yellow]Warning:[/yellow] prompt path not found: {d}")
            continue
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    prompts.extend(data)
                else:
                    prompts.append(data)
    return prompts


def run_single_prompt(
    prompt: dict,
    port: int,
    max_tokens: int,
    temperature: float,
    server_pid: int,
) -> dict:
    """Run a single prompt against the server and collect metrics."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    messages = []
    if prompt.get("system"):
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["prompt"]})

    payload = {
        "messages": messages,
        "max_tokens": prompt.get("max_tokens", max_tokens),
        "temperature": temperature,
        "stream": False,
    }

    sampler = MemorySampler(server_pid)
    sampler.start()

    t_start = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        sampler.stop()
        return {
            "response_text": f"ERROR: {e}",
            "finish_reason": "error",
            "truncated": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generation_tps": 0,
            "prompt_eval_tps": 0,
            "time_to_first_token_ms": 0,
            "total_time_sec": time.monotonic() - t_start,
            "peak_rss_mb": 0,
        }
    t_end = time.monotonic()
    peak_rss = sampler.stop()
    total_time = t_end - t_start

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Extract response text (handle thinking models that split content)
    choices = data.get("choices", [])
    response_text = ""
    finish_reason = ""
    if choices:
        msg = choices[0].get("message", {})
        response_text = msg.get("content", "") or ""
        finish_reason = choices[0].get("finish_reason", "")
        # Some thinking models return empty content but have reasoning in other fields
        if not response_text.strip():
            # Check for thinking/reasoning content as fallback
            for key in ("reasoning_content", "thinking", "reasoning"):
                if msg.get(key):
                    response_text = msg[key]
                    break

    # Compute TPS from timings if available, otherwise from wall clock
    timings = data.get("timings", {})
    generation_tps = timings.get("predicted_per_second", 0)
    prompt_eval_tps = timings.get("prompt_per_second", 0)

    if not generation_tps and completion_tokens and total_time > 0:
        generation_tps = completion_tokens / total_time
    if not prompt_eval_tps and prompt_tokens and total_time > 0:
        prompt_eval_tps = prompt_tokens / total_time  # rough estimate

    # Time to first token: if we had streaming we'd know exactly;
    # estimate from prompt eval time if available
    ttft_ms = timings.get("prompt_ms", 0)

    # Warn if response was truncated
    truncated = finish_reason == "length"

    return {
        "response_text": response_text,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_tps": round(generation_tps, 2),
        "prompt_eval_tps": round(prompt_eval_tps, 2),
        "time_to_first_token_ms": round(ttft_ms, 2),
        "total_time_sec": round(total_time, 3),
        "peak_rss_mb": round(peak_rss, 1),
    }


def _get_existing_labels(conn) -> set[str]:
    """Return set of model labels that already have runs in the DB."""
    rows = conn.execute("SELECT DISTINCT model_label FROM runs").fetchall()
    return {r["model_label"] for r in rows}


def run_benchmark(plan: BenchmarkPlan, db_path=None, skip_existing: bool = False) -> list[str]:
    """Run the full benchmark plan. Returns list of run_ids."""
    conn = get_db(db_path)
    prompts = load_prompts(plan.prompt_dirs)
    if not prompts:
        console.print("[red]No prompts found![/red]")
        return []

    categories = sorted(set(p.get("category", "unknown") for p in prompts))
    console.print(f"Loaded [bold]{len(prompts)}[/bold] prompts: {', '.join(categories)}")

    existing_labels = _get_existing_labels(conn) if skip_existing else set()

    run_ids = []
    total_configs = len(plan.configs)

    for ci, config in enumerate(plan.configs, 1):
        console.rule(f"[bold]{config.label}[/bold]  ({ci}/{total_configs})")

        if skip_existing and config.label in existing_labels:
            console.print("  [dim]Skipped (already in DB)[/dim]")
            continue

        # Check memory
        avail_mb = check_available_memory()
        console.print(f"  Memory: [cyan]{avail_mb:.0f} MB[/cyan] available")

        # Kill any orphaned servers
        kill_orphans(config.port)
        time.sleep(1)

        # Start server
        run_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()
        proc = start_server(config, run_id)

        # Wait for server with spinner
        with console.status(f"  Loading model (PID: {proc.pid})...", spinner="dots"):
            ready = wait_for_health(config.port, timeout=180)

        if not ready:
            console.print("  [red]Server failed to start. Skipping.[/red]")
            stop_server(proc)
            continue

        console.print(f"  [green]Server ready[/green] (PID: {proc.pid})")
        save_run(conn, run_id, timestamp, config)
        run_ids.append(run_id)

        # Run prompts with progress bar
        total_prompts = len(prompts) * plan.repeat
        scores = []
        truncated_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• {task.fields[status]}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("  Prompts", total=total_prompts, status="starting...")

            prompt_num = 0
            for iteration in range(1, plan.repeat + 1):
                for prompt in prompts:
                    prompt_num += 1
                    name = prompt.get("name", "unnamed")
                    category = prompt.get("category", "unknown")

                    progress.update(task, status=f"{category}/{name}")

                    result = run_single_prompt(
                        prompt=prompt,
                        port=config.port,
                        max_tokens=plan.max_tokens,
                        temperature=plan.temperature,
                        server_pid=proc.pid,
                    )

                    # Auto-evaluate quality
                    quality_score, quality_notes = evaluate(prompt, result["response_text"])
                    scores.append(quality_score)

                    if result.get("truncated"):
                        truncated_count += 1

                    status = (
                        f"{category}/{name} "
                        f"[cyan]{result['generation_tps']}[/cyan] tok/s  "
                        f"score [{'green' if quality_score >= 7 else 'yellow' if quality_score >= 4 else 'red'}]"
                        f"{quality_score}/10[/]"
                    )
                    if result.get("truncated"):
                        status += " [red]TRUNC[/red]"
                    progress.update(task, advance=1, status=status)

                    save_result(conn, {
                        "run_id": run_id,
                        "category": category,
                        "prompt_name": name,
                        "prompt_text": prompt["prompt"],
                        "response_text": result["response_text"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "generation_tps": result["generation_tps"],
                        "prompt_eval_tps": result["prompt_eval_tps"],
                        "time_to_first_token_ms": result["time_to_first_token_ms"],
                        "total_time_sec": result["total_time_sec"],
                        "peak_rss_mb": result["peak_rss_mb"],
                        "quality_score": quality_score,
                        "quality_notes": quality_notes,
                        "iteration": iteration,
                        "finish_reason": result.get("finish_reason", ""),
                        "truncated": int(result.get("truncated", False)),
                    })

        # Summary line after progress bar completes
        avg_score = sum(scores) / len(scores) if scores else 0
        trunc_str = f"  [red]{truncated_count} truncated[/red]" if truncated_count else ""
        console.print(
            f"  [bold green]Done[/bold green] {run_id} • "
            f"{total_prompts} prompts • "
            f"avg score [bold]{avg_score:.1f}[/bold]/10"
            f"{trunc_str}"
        )

        # Stop server
        stop_server(proc)
        time.sleep(3)  # cooldown before next config

    conn.close()
    return run_ids
