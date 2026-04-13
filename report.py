"""Report generation for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.table import Table

from storage import get_db, get_results, get_runs


def _avg(values: list[float]) -> float:
    valid = [v for v in values if v and v > 0]
    return sum(valid) / len(valid) if valid else 0.0


def generate_report(
    run_ids: list[str] | None = None,
    categories: list[str] | None = None,
    db_path=None,
    save_markdown: bool = True,
) -> None:
    """Generate comparison report for the given runs."""
    conn = get_db(db_path)
    runs = get_runs(conn)

    if not run_ids:
        # Default: show all runs
        run_ids = [r["run_id"] for r in runs]

    if not run_ids:
        print("No benchmark runs found.")
        return

    results = get_results(conn, run_ids=run_ids, categories=categories)
    if not results:
        print("No results found for the selected runs.")
        return

    # Build run lookup
    run_map = {r["run_id"]: r for r in runs}

    # Aggregate per run
    stats: dict[str, dict] = {}
    for rid in run_ids:
        if rid not in run_map:
            continue
        run = run_map[rid]
        run_results = [r for r in results if r["run_id"] == rid]
        if not run_results:
            continue

        gen_tps = _avg([r["generation_tps"] for r in run_results])
        prompt_tps = _avg([r["prompt_eval_tps"] for r in run_results])
        peak_rss = max((r["peak_rss_mb"] or 0) for r in run_results)
        avg_time = _avg([r["total_time_sec"] for r in run_results])
        scores = [r["quality_score"] for r in run_results if r["quality_score"] is not None]
        avg_score = _avg(scores) if scores else None
        judge_scores = [r["judge_score"] for r in run_results if r.get("judge_score") is not None]
        avg_judge = _avg(judge_scores) if judge_scores else None
        total_prompts = len(run_results)

        # Per-category stats
        cat_stats = defaultdict(list)
        for r in run_results:
            cat_stats[r["category"]].append(r)

        stats[rid] = {
            "run": run,
            "gen_tps": gen_tps,
            "prompt_tps": prompt_tps,
            "peak_rss": peak_rss,
            "avg_time": avg_time,
            "avg_score": avg_score,
            "avg_judge": avg_judge,
            "total_prompts": total_prompts,
            "cat_stats": {
                cat: {
                    "gen_tps": _avg([r["generation_tps"] for r in rs]),
                    "avg_time": _avg([r["total_time_sec"] for r in rs]),
                    "count": len(rs),
                }
                for cat, rs in cat_stats.items()
            },
        }

    # Find best values for highlighting
    best_gen_tps = max(s["gen_tps"] for s in stats.values()) if stats else 0
    best_prompt_tps = max(s["prompt_tps"] for s in stats.values()) if stats else 0
    lowest_rss = min(s["peak_rss"] for s in stats.values()) if stats else 0
    best_time = min(s["avg_time"] for s in stats.values()) if stats else 0

    # Terminal output with rich
    console = Console()
    console.print("\n[bold]Benchmark Comparison[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model", min_width=20)
    table.add_column("Quant", min_width=8)
    table.add_column("Size", min_width=6)
    table.add_column("Ctx", min_width=6, justify="right")
    table.add_column("Gen TPS", justify="right")
    table.add_column("Prompt TPS", justify="right")
    table.add_column("Peak RSS (MB)", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Auto", justify="right")
    has_judge = any(s["avg_judge"] is not None for s in stats.values())
    if has_judge:
        table.add_column("Judge", justify="right")
    table.add_column("Prompts", justify="right")

    best_score = max((s["avg_score"] or 0) for s in stats.values()) if stats else 0
    best_judge = max((s["avg_judge"] or 0) for s in stats.values()) if has_judge else 0

    for rid, s in stats.items():
        run = s["run"]

        gen_style = "bold green" if s["gen_tps"] == best_gen_tps else ""
        prompt_style = "bold green" if s["prompt_tps"] == best_prompt_tps else ""
        rss_style = "bold green" if s["peak_rss"] == lowest_rss else ""
        time_style = "bold green" if s["avg_time"] == best_time else ""
        score_style = "bold green" if s["avg_score"] == best_score and best_score > 0 else ""
        judge_style = "bold green" if has_judge and s["avg_judge"] == best_judge and best_judge > 0 else ""

        score_str = f"{s['avg_score']:.1f}" if s["avg_score"] is not None else "-"
        judge_str = f"{s['avg_judge']:.1f}" if s["avg_judge"] is not None else "-"

        row = [
            run["model_label"],
            run["quantization"],
            run["param_count"],
            str(run["ctx_size"]),
            f"[{gen_style}]{s['gen_tps']:.1f}[/{gen_style}]" if gen_style else f"{s['gen_tps']:.1f}",
            f"[{prompt_style}]{s['prompt_tps']:.1f}[/{prompt_style}]" if prompt_style else f"{s['prompt_tps']:.1f}",
            f"[{rss_style}]{s['peak_rss']:.0f}[/{rss_style}]" if rss_style else f"{s['peak_rss']:.0f}",
            f"[{time_style}]{s['avg_time']:.1f}[/{time_style}]" if time_style else f"{s['avg_time']:.1f}",
            f"[{score_style}]{score_str}[/{score_style}]" if score_style else score_str,
        ]
        if has_judge:
            row.append(f"[{judge_style}]{judge_str}[/{judge_style}]" if judge_style else judge_str)
        row.append(str(s["total_prompts"]))

        table.add_row(*row)

    console.print(table)

    # Per-category breakdown
    all_cats = sorted(set(cat for s in stats.values() for cat in s["cat_stats"]))
    if all_cats:
        console.print("\n[bold]Per-Category Generation TPS[/bold]\n")
        cat_table = Table(show_header=True, header_style="bold cyan")
        cat_table.add_column("Model", min_width=20)
        for cat in all_cats:
            cat_table.add_column(cat.capitalize(), justify="right")

        for rid, s in stats.items():
            row = [s["run"]["model_label"]]
            for cat in all_cats:
                cs = s["cat_stats"].get(cat)
                row.append(f"{cs['gen_tps']:.1f}" if cs else "-")
            cat_table.add_row(*row)

        console.print(cat_table)

    # Save markdown
    if save_markdown:
        md = _to_markdown(stats, all_cats)
        from pathlib import Path
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"report_{ts}.md"
        out_path.write_text(md)
        console.print(f"\nReport saved to {out_path}")

    conn.close()


def _to_markdown(stats: dict, categories: list[str]) -> str:
    lines = ["# Benchmark Report\n"]
    lines.append("| Model | Quant | Size | Ctx | Gen TPS | Prompt TPS | Peak RSS (MB) | Avg Time (s) | Score | Prompts |")
    lines.append("|-------|-------|------|-----|---------|------------|---------------|--------------|-------|---------|")

    for rid, s in stats.items():
        run = s["run"]
        score_str = f"{s['avg_score']:.1f}" if s["avg_score"] is not None else "-"
        lines.append(
            f"| {run['model_label']} | {run['quantization']} | {run['param_count']} "
            f"| {run['ctx_size']} | {s['gen_tps']:.1f} | {s['prompt_tps']:.1f} "
            f"| {s['peak_rss']:.0f} | {s['avg_time']:.1f} | {score_str} | {s['total_prompts']} |"
        )

    if categories:
        lines.append("\n## Per-Category Generation TPS\n")
        header = "| Model | " + " | ".join(c.capitalize() for c in categories) + " |"
        sep = "|-------|" + "|".join("------" for _ in categories) + "|"
        lines.append(header)
        lines.append(sep)
        for rid, s in stats.items():
            row = f"| {s['run']['model_label']} |"
            for cat in categories:
                cs = s["cat_stats"].get(cat)
                row += f" {cs['gen_tps']:.1f} |" if cs else " - |"
            lines.append(row)

    return "\n".join(lines) + "\n"


def list_runs(db_path=None) -> None:
    """List all benchmark runs."""
    conn = get_db(db_path)
    runs = get_runs(conn)
    conn.close()

    if not runs:
        print("No runs found.")
        return

    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run ID")
    table.add_column("Timestamp")
    table.add_column("Model")
    table.add_column("Quant")
    table.add_column("Ctx")

    for r in runs:
        table.add_row(
            r["run_id"],
            r["timestamp"][:19],
            r["model_label"],
            r["quantization"],
            str(r["ctx_size"]),
        )
    console.print(table)
