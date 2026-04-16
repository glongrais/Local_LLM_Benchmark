"""llama-server lifecycle management."""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path

import requests

LLAMA_SERVER_BIN = "llama-server"  # assumes it's on PATH; override via env or config

LOG_DIR = Path("results/logs")


def find_llama_server() -> str:
    """Return the llama-server binary path."""
    import shutil
    path = shutil.which(LLAMA_SERVER_BIN)
    if path:
        return path
    # Common homebrew location on Apple Silicon
    brew_path = "/opt/homebrew/bin/llama-server"
    if Path(brew_path).exists():
        return brew_path
    return LLAMA_SERVER_BIN


def start_server(config, run_id: str = "") -> subprocess.Popen:
    """Start llama-server or mlx_lm.server with the given config. Returns the Popen handle."""
    if config.backend == "mlx":
        import sys
        args = [sys.executable, "-m", "mlx_lm.server"] + config.to_cli_args()
    else:
        binary = find_llama_server()
        args = [binary] + config.to_cli_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"{run_id or 'server'}_{config.label}.log"
    log_file = open(LOG_DIR / log_name, "w")

    proc = subprocess.Popen(
        args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=None,
    )
    return proc


def wait_for_health(port: int = 8999, timeout: int = 180, interval: float = 2.0) -> bool:
    """Poll the health endpoint until the server is ready."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(interval)
    return False


def stop_server(proc: subprocess.Popen, timeout: int = 15) -> None:
    """Stop the server gracefully, then force-kill if needed."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    pass  # caller handles status display


def kill_orphans(port: int = 8999) -> None:
    """Kill any llama-server or mlx_lm.server process listening on the given port."""
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            cmdline = p.info.get("cmdline") or []
            name = p.info.get("name") or ""
            is_llama = "llama-server" in name
            is_mlx = "mlx_lm.server" in " ".join(cmdline)
            if not (is_llama or is_mlx):
                continue
            if "--port" in cmdline:
                idx = cmdline.index("--port")
                if idx + 1 < len(cmdline) and cmdline[idx + 1] == str(port):
                    p.kill()
            elif port == 8999:
                p.kill()
    except Exception:
        pass
