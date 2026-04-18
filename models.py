"""Model management — list, analyze disk usage, and clean up HF cache repos."""

from __future__ import annotations

import shutil
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


def _parse_hf_repo(repo_dir: Path) -> str:
    """Extract HuggingFace repo name from cache dir name, e.g. 'unsloth/gemma-4-31B-it-GGUF'."""
    name = repo_dir.name
    if name.startswith("models--"):
        parts = name.removeprefix("models--").split("--", 1)
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]
    return name


def _dir_size(path: Path) -> int:
    """Calculate total size of a directory, following symlinks to blobs."""
    total = 0
    seen: set[Path] = set()
    for f in path.rglob("*"):
        if f.is_file():
            target = f.resolve()
            if target not in seen:
                total += target.stat().st_size
                seen.add(target)
    return total


def _detect_model_type(repo_dir: Path) -> str:
    """Detect model type from repo contents (checks snapshots and blobs)."""
    has_gguf = False
    has_safetensors = False
    # Check snapshots (symlinks with real filenames) and blobs
    for subdir in ("snapshots", "blobs"):
        d = repo_dir / subdir
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.name.endswith(".gguf"):
                has_gguf = True
            elif f.name.endswith(".safetensors"):
                has_safetensors = True
            if has_gguf and has_safetensors:
                return "gguf+safetensors"
    # Blobs have hash names — infer from repo name if no file extensions found
    if not has_gguf and not has_safetensors:
        repo_name = repo_dir.name.lower()
        if "gguf" in repo_name:
            return "gguf"
        elif "mlx" in repo_name:
            return "safetensors"
    if has_gguf and has_safetensors:
        return "gguf+safetensors"
    if has_gguf:
        return "gguf"
    if has_safetensors:
        return "safetensors"
    return "other"


def _get_gguf_files(repo_dir: Path) -> list[str]:
    """List GGUF filenames in a repo (for display), excluding mmproj."""
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return []
    seen: set[Path] = set()
    names = []
    for f in sorted(snapshots.rglob("*.gguf")):
        if "mmproj" in f.name:
            continue
        target = f.resolve()
        if target not in seen:
            names.append(f.name)
            seen.add(target)
    return names


def _get_benchmarked_repos(db_path=None) -> set[str]:
    """Return set of hf_repo values that have benchmark runs."""
    try:
        conn = get_db(db_path)
        rows = conn.execute("SELECT DISTINCT hf_repo FROM runs WHERE hf_repo IS NOT NULL").fetchall()
        conn.close()
        return {r["hf_repo"] for r in rows if r["hf_repo"]}
    except Exception:
        return set()


def list_models(directories: list[str], db_path=None) -> list[dict]:
    """List all HF cache repos with size, type, and benchmark status."""
    benchmarked_repos = _get_benchmarked_repos(db_path)
    models = []

    for d in directories:
        p = Path(d).expanduser()
        if not p.exists():
            continue
        for repo_dir in sorted(p.iterdir()):
            if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
                continue
            repo = _parse_hf_repo(repo_dir)
            model_type = _detect_model_type(repo_dir)
            size = _dir_size(repo_dir)
            gguf_files = _get_gguf_files(repo_dir) if "gguf" in model_type else []
            models.append({
                "path": repo_dir,
                "repo": repo,
                "type": model_type,
                "size": size,
                "gguf_files": gguf_files,
                "benchmarked": repo in benchmarked_repos,
            })

    models.sort(key=lambda m: m["size"], reverse=True)
    return models


def print_models(directories: list[str], db_path=None) -> None:
    """Print a table of HF cache repos with sizes and benchmark status."""
    models = list_models(directories, db_path)
    console = Console()

    if not models:
        console.print("No models found in the HuggingFace cache.")
        return

    total_size = sum(m["size"] for m in models)
    benchmarked_count = sum(1 for m in models if m["benchmarked"])

    table = Table(show_header=True, header_style="bold cyan", show_lines=False)
    table.add_column("#", justify="right", width=4)
    table.add_column("Repo", min_width=30)
    table.add_column("Type", min_width=12)
    table.add_column("Size", justify="right", min_width=10)
    table.add_column("Benchmarked", justify="center", min_width=12)
    table.add_column("Files", min_width=20)

    for i, m in enumerate(models, 1):
        bench_str = "[green]yes[/green]" if m["benchmarked"] else "[dim]no[/dim]"
        repo = m["repo"] or "[dim]unknown[/dim]"
        # Show GGUF file names or safetensors indicator
        if m["gguf_files"]:
            files_str = ", ".join(m["gguf_files"][:3])
            if len(m["gguf_files"]) > 3:
                files_str += f" +{len(m['gguf_files']) - 3}"
        elif m["type"] == "safetensors":
            files_str = "[dim]safetensors (MLX)[/dim]"
        else:
            files_str = f"[dim]{m['type']}[/dim]"
        table.add_row(
            str(i),
            repo,
            m["type"],
            _format_size(m["size"]),
            bench_str,
            files_str,
        )

    console.print(f"\n[bold]HuggingFace Cache[/bold] ({len(models)} repos, "
                  f"{_format_size(total_size)} total, "
                  f"{benchmarked_count} benchmarked)\n")
    console.print(table)


def clean_models(directories: list[str], db_path=None, keep_benchmarked: bool = True) -> None:
    """Interactive cleanup of HF cache repos."""
    models = list_models(directories, db_path)
    console = Console()

    if not models:
        console.print("No models found.")
        return

    print_models(directories, db_path)

    if keep_benchmarked:
        candidates = [m for m in models if not m["benchmarked"]]
        if not candidates:
            console.print("\nAll models have been benchmarked. Use --all to include them.")
            return
        console.print(f"\n[bold]{len(candidates)}[/bold] unbenchmarked repo(s) can be deleted.")
    else:
        candidates = models

    console.print("\nOptions:")
    console.print("  Enter repo numbers to delete (comma-separated, e.g. '1,3,5')")
    console.print("  'unbenchmarked' or 'u' — delete all unbenchmarked repos")
    console.print("  'all' or 'a' — delete all listed repos")
    console.print("  'q' — quit\n")

    while True:
        inp = input("Delete> ").strip().lower()
        if inp in ("q", "quit", ""):
            return

        to_delete: list[dict] = []

        if inp in ("a", "all"):
            to_delete = list(candidates)
        elif inp in ("u", "unbenchmarked"):
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
        console.print(f"\nWill delete {len(to_delete)} repo(s), freeing {_format_size(total_free)}:")
        for m in to_delete:
            bench_tag = " [yellow](benchmarked)[/yellow]" if m["benchmarked"] else ""
            console.print(f"  {m['repo']} ({_format_size(m['size'])}){bench_tag}")

        confirm = input("\nConfirm? [y/N] ").strip().lower()
        if confirm == "y":
            deleted = 0
            freed = 0
            for m in to_delete:
                try:
                    shutil.rmtree(m["path"])
                    console.print(f"  [red]Deleted[/red] {m['repo']}")
                    deleted += 1
                    freed += m["size"]
                except OSError as e:
                    console.print(f"  [red]Error[/red] deleting {m['repo']}: {e}")
            console.print(f"\nDeleted {deleted} repo(s), freed {_format_size(freed)}.")
            return
        else:
            console.print("Cancelled.")
