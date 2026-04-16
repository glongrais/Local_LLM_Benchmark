#!/usr/bin/env python3
"""CLI entry point for the LLM benchmark framework."""

from __future__ import annotations

import argparse
import sys

from config import load_plan
from report import generate_report, list_runs
from storage import get_db, get_results, update_quality


def cmd_run(args):
    from runner import run_benchmark
    plan = load_plan(args.config)

    if args.category:
        # Filter prompts by loading and checking categories in runner
        pass  # runner loads all, we filter in plan

    run_ids = run_benchmark(plan, skip_existing=args.skip_existing)
    if run_ids:
        print(f"\nCompleted {len(run_ids)} runs: {', '.join(run_ids)}")
        print("Run 'python bench.py report' to see results.")


def cmd_report(args):
    run_ids = args.run_ids if args.run_ids else None
    categories = args.category.split(",") if args.category else None
    generate_report(
        run_ids=run_ids,
        categories=categories,
        save_markdown=not args.no_save,
    )


def cmd_list(args):
    list_runs()


def cmd_score(args):
    """Interactive quality scoring for a run's results."""
    conn = get_db()
    results = get_results(conn, run_ids=[args.run_id])
    if not results:
        print(f"No results found for run {args.run_id}")
        return

    unscored = [r for r in results if r["quality_score"] is None]
    if not unscored:
        print("All results already scored.")
        return

    print(f"\nScoring {len(unscored)} results for run {args.run_id}")
    print("Enter score (0-10), 's' to skip, 'q' to quit\n")

    for r in unscored:
        print(f"{'='*60}")
        print(f"Category: {r['category']} | Prompt: {r['prompt_name']}")
        print(f"Prompt: {r['prompt_text'][:200]}...")
        print(f"\nResponse ({r['completion_tokens']} tokens):")
        print(f"{r['response_text'][:500]}")
        if len(r['response_text']) > 500:
            print(f"... ({len(r['response_text'])} chars total)")
        print()

        while True:
            inp = input("Score (0-10/s/q): ").strip().lower()
            if inp == "q":
                conn.close()
                return
            if inp == "s":
                break
            try:
                score = float(inp)
                if 0 <= score <= 10:
                    notes = input("Notes (optional): ").strip()
                    update_quality(conn, r["id"], score, notes)
                    print(f"  Saved: {score}/10")
                    break
                else:
                    print("  Score must be 0-10")
            except ValueError:
                print("  Invalid input")

    conn.close()


def cmd_show(args):
    """Show detailed results for a specific run."""
    conn = get_db()
    results = get_results(conn, run_ids=[args.run_id])
    conn.close()

    if not results:
        print(f"No results for run {args.run_id}")
        return

    for r in results:
        print(f"\n{'='*60}")
        print(f"[{r['category']}/{r['prompt_name']}] "
              f"Gen: {r['generation_tps']} tok/s | "
              f"Time: {r['total_time_sec']}s | "
              f"RSS: {r['peak_rss_mb']} MB")
        score = r["quality_score"]
        if score is not None:
            print(f"Score: {score}/10 {r.get('quality_notes', '')}")
        print(f"\nResponse:\n{r['response_text'][:1000]}")


def cmd_export(args):
    """Export results as JSON."""
    import json
    conn = get_db()
    run_ids = args.run_ids if args.run_ids else None
    results = get_results(conn, run_ids=run_ids)
    runs = {r["run_id"]: dict(r) for r in (get_db().execute("SELECT * FROM runs").fetchall())}
    conn.close()

    output = {
        "runs": runs,
        "results": results,
    }
    out_path = args.output or "results/export.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Exported to {out_path}")


def cmd_autoscore(args):
    """Re-run auto-evaluation on existing results (parallel)."""
    import json
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from evaluate import evaluate
    from runner import load_prompts
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

    console = Console()
    conn = get_db()
    run_ids = args.run_ids if args.run_ids else [r["run_id"] for r in conn.execute("SELECT run_id FROM runs ORDER BY timestamp DESC").fetchall()]
    results = get_results(conn, run_ids=run_ids)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    prompt_dirs = args.prompts.split(",") if args.prompts else ["prompts/"]
    prompts = load_prompts(prompt_dirs)
    prompt_map = {(p["category"], p["name"]): p for p in prompts}

    # Build work items: (result_id, prompt, response_text)
    work = []
    for r in results:
        prompt = prompt_map.get((r["category"], r["prompt_name"]))
        if not prompt:
            console.print(f"  [yellow]Warning:[/yellow] no prompt for {r['category']}/{r['prompt_name']}")
            continue
        work.append((r["id"], prompt, r["response_text"], r["prompt_name"]))

    workers = min(args.workers or os.cpu_count() or 4, len(work))
    console.print(f"Scoring [bold]{len(work)}[/bold] results with [bold]{workers}[/bold] workers")

    updated = 0
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
        task = progress.add_task("Scoring", total=len(work), status="starting...")

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate, prompt, response): (rid, name)
                for rid, prompt, response, name in work
            }
            for future in as_completed(futures):
                rid, name = futures[future]
                try:
                    score, notes = future.result()
                    update_quality(conn, rid, score, notes)
                    updated += 1
                    color = "green" if score >= 7 else "yellow" if score >= 4 else "red"
                    progress.update(task, advance=1, status=f"{name} [{color}]{score}/10[/{color}]")
                except Exception as e:
                    progress.update(task, advance=1, status=f"{name} [red]ERROR[/red]")

    conn.close()
    console.print(f"[bold green]Done[/bold green] • {updated} results scored")


def cmd_judge(args):
    """Score results using a judge LLM."""
    from config import ServerConfig
    from judge import judge_results

    judge_config = None
    if args.hf_repo:
        hf_repo = args.hf_repo
        hf_file = args.hf_file or ""
        # Support repo:file shorthand, e.g. "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL"
        if ":" in hf_repo and not hf_file:
            hf_repo, hf_file = hf_repo.rsplit(":", 1)
        judge_config = ServerConfig(
            label="judge",
            hf_repo=hf_repo,
            hf_file=hf_file,
            port=args.port,
            ctx_size=args.ctx_size,
            flash_attn=True,
            backend=args.backend,
        )

    run_ids = args.run_ids if args.run_ids else None
    judge_results(
        run_ids=run_ids,
        judge_config=judge_config,
        judge_port=args.port,
        overwrite=args.overwrite,
    )


def cmd_purge(args):
    """Delete runs from the database."""
    conn = get_db()
    run_ids = args.run_ids

    # Show what will be deleted
    for rid in run_ids:
        rows = conn.execute("SELECT model_label, timestamp FROM runs WHERE run_id = ?", (rid,)).fetchall()
        if rows:
            r = rows[0]
            count = conn.execute("SELECT COUNT(*) FROM results WHERE run_id = ?", (rid,)).fetchone()[0]
            print(f"  {rid}: {r['model_label']} ({r['timestamp'][:19]}) — {count} results")
        else:
            print(f"  {rid}: not found")

    confirm = input("\nDelete these runs? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    for rid in run_ids:
        conn.execute("DELETE FROM results WHERE run_id = ?", (rid,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (rid,))
    conn.commit()
    conn.close()
    print(f"Deleted {len(run_ids)} run(s).")


def cmd_models(args):
    """List or clean up GGUF model files."""
    from config import HF_CACHE_DIR
    from models import clean_models, print_models
    dirs = args.dirs if args.dirs else [str(HF_CACHE_DIR)]
    if args.clean:
        clean_models(dirs, keep_benchmarked=not args.all)
    else:
        print_models(dirs)


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Framework")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run benchmarks")
    p_run.add_argument("--config", "-c", required=True, help="Path to benchmark config JSON")
    p_run.add_argument("--category", help="Run only this category")
    p_run.add_argument("--skip-existing", action="store_true", help="Skip configs that already have results in the DB")
    p_run.set_defaults(func=cmd_run)

    # report
    p_report = sub.add_parser("report", help="Show comparison report")
    p_report.add_argument("--run-ids", nargs="+", help="Specific run IDs to compare")
    p_report.add_argument("--category", help="Filter by category (comma-separated)")
    p_report.add_argument("--no-save", action="store_true", help="Don't save markdown report")
    p_report.set_defaults(func=cmd_report)

    # list
    p_list = sub.add_parser("list", help="List all runs")
    p_list.set_defaults(func=cmd_list)

    # score
    p_score = sub.add_parser("score", help="Interactively score a run")
    p_score.add_argument("run_id", help="Run ID to score")
    p_score.set_defaults(func=cmd_score)

    # show
    p_show = sub.add_parser("show", help="Show detailed results for a run")
    p_show.add_argument("run_id", help="Run ID")
    p_show.set_defaults(func=cmd_show)

    # export
    p_export = sub.add_parser("export", help="Export results as JSON")
    p_export.add_argument("--run-ids", nargs="+", help="Specific run IDs")
    p_export.add_argument("--output", "-o", help="Output path")
    p_export.set_defaults(func=cmd_export)

    # autoscore
    p_autoscore = sub.add_parser("autoscore", help="Re-run auto-evaluation on existing results")
    p_autoscore.add_argument("--run-ids", nargs="+", help="Specific run IDs (default: all)")
    p_autoscore.add_argument("--prompts", help="Prompt dirs, comma-separated (default: prompts/)")
    p_autoscore.add_argument("--workers", type=int, help="Number of parallel workers (default: CPU count)")
    p_autoscore.set_defaults(func=cmd_autoscore)

    # purge
    p_purge = sub.add_parser("purge", help="Delete runs from the database")
    p_purge.add_argument("run_ids", nargs="+", help="Run IDs to delete")
    p_purge.set_defaults(func=cmd_purge)

    # judge
    p_judge = sub.add_parser("judge", help="Score results using a judge LLM")
    p_judge.add_argument("--run-ids", nargs="+", help="Specific run IDs to judge (default: all)")
    p_judge.add_argument("--hf-repo", help="HuggingFace repo for judge model (starts a server)")
    p_judge.add_argument("--hf-file", help="Specific GGUF file in the repo")
    p_judge.add_argument("--port", type=int, default=8090, help="Port for judge server (default: 8090)")
    p_judge.add_argument("--ctx-size", type=int, default=32768, help="Context size for judge model")
    p_judge.add_argument("--overwrite", action="store_true", help="Re-judge already scored results")
    p_judge.add_argument("--backend", default="llama", choices=["llama", "mlx"], help="Server backend (default: llama)")
    p_judge.set_defaults(func=cmd_judge)

    # models
    p_models = sub.add_parser("models", help="List or clean up GGUF model files")
    p_models.add_argument("dirs", nargs="*", help="Directories to scan (default: HuggingFace cache)")
    p_models.add_argument("--clean", action="store_true", help="Interactive cleanup mode")
    p_models.add_argument("--all", action="store_true", help="Include benchmarked models in cleanup")
    p_models.set_defaults(func=cmd_models)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
