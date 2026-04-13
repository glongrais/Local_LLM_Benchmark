"""LLM-as-judge evaluation — uses a local model to score benchmark responses."""

from __future__ import annotations

import json
import re
import time

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from config import ServerConfig
from server import kill_orphans, start_server, stop_server, wait_for_health
from storage import get_db, get_results, get_runs

console = Console()

JUDGE_RUBRIC = """\
You are an expert evaluator. Score an AI model's response. Be concise — respond with ONLY the JSON below, no explanation.

## Task
{prompt}

## Response
{response}

## Score (0-10)
- Correctness (0-4): Is it factually correct? Does code work? Calculations right?
- Completeness (0-3): All parts of the prompt addressed?
- Quality (0-3): Well-structured, clear, clean code?

Reply with ONLY this JSON:
{{"score": <0-10>, "correctness": <0-4>, "completeness": <0-3>, "quality": <0-3>, "reason": "<one sentence>"}}"""


def judge_results(
    run_ids: list[str] | None = None,
    judge_config: ServerConfig | None = None,
    judge_port: int = 8090,
    max_tokens: int = 4096,
    db_path=None,
    overwrite: bool = False,
) -> None:
    """Score existing benchmark results using a judge LLM."""
    conn = get_db(db_path)
    results = get_results(conn, run_ids=run_ids)

    if not results:
        console.print("[yellow]No results to judge.[/yellow]")
        return

    if not overwrite:
        results = [r for r in results if r.get("judge_score") is None]

    # Skip empty responses
    skipped = 0
    judgeable = []
    for r in results:
        text = r.get("response_text") or ""
        if not text.strip():
            _save_judge_score(conn, r["id"], 0, "empty response")
            skipped += 1
        else:
            judgeable.append(r)
    results = judgeable

    if not results and skipped == 0:
        console.print("[dim]All results already judged. Use --overwrite to re-judge.[/dim]")
        return

    if skipped:
        console.print(f"[dim]Skipped {skipped} empty responses (scored 0).[/dim]")

    if not results:
        return

    # Start judge server if config provided
    proc = None
    if judge_config:
        judge_config.port = judge_port
        kill_orphans(judge_port)
        time.sleep(1)
        proc = start_server(judge_config, "judge")

        with console.status("Loading judge model...", spinner="dots"):
            ready = wait_for_health(judge_port, timeout=180)

        if not ready:
            console.print("[red]Judge model failed to start.[/red]")
            stop_server(proc)
            return
        console.print(f"[green]Judge ready[/green] (PID: {proc.pid})")

    url = f"http://127.0.0.1:{judge_port}/v1/chat/completions"
    judged = 0
    errors = 0
    scores: list[float] = []

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
        task = progress.add_task("Judging", total=len(results), status="starting...")

        for i, r in enumerate(results):
            prompt_text = r["prompt_text"]
            response_text = r["response_text"] or ""
            name = r["prompt_name"]

            progress.update(task, status=f"{name}")

            # Truncate to fit judge context window
            max_response_chars = 20000
            max_prompt_chars = 4000
            if len(response_text) > max_response_chars:
                response_text = response_text[:max_response_chars] + "\n... [truncated]"

            judge_prompt = JUDGE_RUBRIC.format(
                prompt=prompt_text[:max_prompt_chars],
                response=response_text,
            )

            try:
                resp = requests.post(url, json={
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                }, timeout=300)
                resp.raise_for_status()
                data = resp.json()

                msg = data["choices"][0]["message"]
                all_text = ""
                for key in ("content", "reasoning_content", "thinking", "reasoning"):
                    if msg.get(key):
                        all_text += msg[key] + "\n"
                score, reason = _parse_judge_response(all_text)

                _save_judge_score(conn, r["id"], score, reason)
                judged += 1
                scores.append(score)

                color = "green" if score >= 7 else "yellow" if score >= 4 else "red"
                progress.update(
                    task, advance=1,
                    status=f"{name} [{color}]{score}/10[/{color}]"
                )

            except Exception as e:
                errors += 1
                progress.update(task, advance=1, status=f"{name} [red]ERROR[/red]")

    if proc:
        stop_server(proc)

    conn.close()

    # Summary
    avg = sum(scores) / len(scores) if scores else 0
    err_str = f"  [red]{errors} errors[/red]" if errors else ""
    console.print(
        f"[bold green]Done[/bold green] • "
        f"{judged} judged • "
        f"avg score [bold]{avg:.1f}[/bold]/10"
        f"{err_str}"
    )


def _parse_judge_response(content: str) -> tuple[float, str]:
    """Parse the judge's JSON response. Returns (score, reason)."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()

    # Try to find any JSON object with a "score" key
    for pattern in [
        r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}',
        r'\{.*?"score"\s*:\s*\d+.*?\}',
    ]:
        json_match = re.search(pattern, cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reason = data.get("reason", "")
                return min(max(score, 0), 10), reason
            except (json.JSONDecodeError, ValueError):
                continue

    # Fallback: find "score": N anywhere
    score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
    if score_match:
        score = float(score_match.group(1))
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
        reason = reason_match.group(1) if reason_match else "parsed from partial JSON"
        return min(max(score, 0), 10), reason

    # Fallback: try to find X/10 pattern
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*/\s*10', cleaned)
    if numbers:
        return min(float(numbers[0]), 10), cleaned[:100]

    return 0, f"unparseable: {cleaned[:150]}"


def _save_judge_score(conn, result_id: int, score: float, reason: str) -> None:
    """Save judge score to the database."""
    conn.execute(
        "UPDATE results SET judge_score = ?, judge_reason = ? WHERE id = ?",
        (score, reason, result_id),
    )
    conn.commit()
