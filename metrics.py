"""Memory and performance sampling."""

from __future__ import annotations

import threading
import time

import psutil


class MemorySampler:
    """Samples RSS of a process in a background thread."""

    def __init__(self, pid: int, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            proc = psutil.Process(self.pid)
            while not self._stop.is_set():
                try:
                    mem = proc.memory_info()
                    self._samples.append(mem.rss / (1024 * 1024))  # MB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                self._stop.wait(self.interval)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def stop(self) -> float:
        """Stop sampling and return peak RSS in MB."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return max(self._samples) if self._samples else 0.0

    @property
    def current_rss(self) -> float:
        return self._samples[-1] if self._samples else 0.0


def get_process_rss(pid: int) -> float:
    """Get current RSS of a process in MB."""
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def check_available_memory() -> float:
    """Return available system memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)
