"""Model file management — list, analyze disk usage, and clean up GGUF files."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import HF_CACHE_DIR
from storage import get_db


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _parse_hf_repo(path: Path) -> str:
    """Extract HuggingFace repo name from cache path, e.g. 'unsloth/gemma-4-31B-it-GGUF'."""
    for parent in path.parents:
        if parent.name.startswith("models--"):
            parts = parent.name.removeprefix("models--").split("--", 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"
            return parts[0]
    return ""


def _find_gguf_files(directories: list[str]) -> list[Path]:
    """Recursively find all .gguf files in the given directories.

    In HF cache, files under snapshots/ are symlinks to blobs/<hash>.
    We keep the symlink path (which has the real filename) but deduplicate
    by resolve target so the same blob isn't listed twice.
    """
    files = []
    seen_targets: set[Path] = set()
    for d in directories:
        p = Path(d).expanduser()
        if not p.exists():
            continue
        if p.is_file() and p.suffix == ".gguf":
            target = p.resolve()
            if target not in seen_targets:
                files.append(p)
                seen_targets.add(target)
        else:
            for f in sorted(p.rglob("*.gguf")):
                target = f.resolve()
                if target not in seen_targets:
                    files.append(f)  # keep symlink path for display name
                    seen_targets.add(target)
    return files


def _get_benchmarked_models(db_path=None) -> set[str]:
    """Return set of model paths, labels, and hf_repos that have benchmark runs."""
    try:
        conn = get_db(db_path)
        rows = conn.execute("SELECT model_path, model_label, hf_repo FROM runs").fetchall()
        conn.close()
        result = set()
        for r in rows:
            if r["model_path"]:
                result.add(str(Path(r["model_path"]).resolve()))
            if r["model_label"]:
                result.add(r["model_label"])
            if r["hf_repo"]:
                result.add(r["hf_repo"])
        return result
    except Exception:
        return set()


def _is_benchmarked(path: Path, repo: str, benchmarked: set[str]) -> bool:
    """Check if a model file has been benchmarked (by path, repo, or filename match)."""
    resolved = str(path.resolve())
    if resolved in benchmarked:
        return True
    if repo and repo in benchmarked:
        return True
    # Also check if the filename appears in any benchmarked label
    name = path.stem  # e.g. "gemma-4-31B-it-UD-Q4_K_XL"
    return any(name in b for b in benchmarked)


def list_models(directories: list[str], db_path=None) -> list[dict]:
    """List GGUF files with size, repo info, and benchmark status."""
    files = _find_gguf_files(directories)
    benchmarked = _get_benchmarked_models(db_path)

    models = []
    for f in files:
        repo = _parse_hf_repo(f)
        models.append({
            "path": f,
            "name": f.name,
            "repo": repo,
            "size": f.stat().st_size,
            "benchmarked": _is_benchmarked(f, repo, benchmarked),
        })
    return models


def print_models(directories: list[str], db_path=None) -> None:
    """Print a table of GGUF files with sizes and benchmark status."""
    models = list_models(directories, db_path)
    console = Console()

    if not models:
        console.print("No .gguf files found in the specified directories.")
        return

    total_size = sum(m["size"] for m in models)
    benchmarked_count = sum(1 for m in models if m["benchmarked"])

    # Group by repo for cleaner display
    table = Table(show_header=True, header_style="bold cyan", show_lines=False)
    table.add_column("#", justify="right", width=4)
    table.add_column("Repo", min_width=20)
    table.add_column("File", min_width=30)
    table.add_column("Size", justify="right", min_width=10)
    table.add_column("Benchmarked", justify="center", min_width=12)

    for i, m in enumerate(models, 1):
        bench_str = "[green]yes[/green]" if m["benchmarked"] else "[dim]no[/dim]"
        repo = m["repo"] or "[dim]local[/dim]"
        table.add_row(
            str(i),
            repo,
            m["name"],
            _format_size(m["size"]),
            bench_str,
        )

    console.print(f"\n[bold]Model Files[/bold] ({len(models)} files, "
                  f"{_format_size(total_size)} total, "
                  f"{benchmarked_count} benchmarked)\n")
    console.print(table)


def clean_models(directories: list[str], db_path=None, keep_benchmarked: bool = True) -> None:
    """Interactive cleanup of GGUF files."""
    models = list_models(directories, db_path)
    console = Console()

    if not models:
        console.print("No .gguf files found.")
        return

    print_models(directories, db_path)

    if keep_benchmarked:
        candidates = [m for m in models if not m["benchmarked"]]
        if not candidates:
            console.print("\nAll models have been benchmarked. Use --all to include them.")
            return
        console.print(f"\n[bold]{len(candidates)}[/bold] unbenchmarked model(s) can be deleted.")
    else:
        candidates = models

    console.print("\nOptions:")
    console.print("  Enter file numbers to delete (comma-separated, e.g. '1,3,5')")
    console.print("  'unbenchmarked' or 'u' — delete all unbenchmarked models")
    console.print("  'q' — quit\n")

    while True:
        inp = input("Delete> ").strip().lower()
        if inp in ("q", "quit", ""):
            return

        to_delete: list[dict] = []

        if inp in ("u", "unbenchmarked"):
            to_delete = [m for m in models if not m["benchmarked"]]
        else:
            try:
                indices = [int(x.strip()) for x in inp.split(",")]
                for idx in indices:
                    if 1 <= idx <= len(models):
                        to_delete.append(models[idx - 1])
                    else:
                        console.print(f"  [red]Invalid number: {idx}[/red]")
                        to_delete = []
                        break
            except ValueError:
                console.print("  [red]Invalid input[/red]")
                continue

        if not to_delete:
            continue

        total_free = sum(m["size"] for m in to_delete)
        console.print(f"\nWill delete {len(to_delete)} file(s), freeing {_format_size(total_free)}:")
        for m in to_delete:
            bench_tag = " [yellow](benchmarked)[/yellow]" if m["benchmarked"] else ""
            console.print(f"  {m['name']} ({_format_size(m['size'])}){bench_tag}")

        confirm = input("\nConfirm? [y/N] ").strip().lower()
        if confirm == "y":
            deleted = 0
            freed = 0
            for m in to_delete:
                try:
                    m["path"].unlink()
                    console.print(f"  [red]Deleted[/red] {m['path']}")
                    deleted += 1
                    freed += m["size"]
                except OSError as e:
                    console.print(f"  [red]Error[/red] deleting {m['path']}: {e}")
            console.print(f"\nDeleted {deleted} file(s), freed {_format_size(freed)}.")
            return
        else:
            console.print("Cancelled.")
