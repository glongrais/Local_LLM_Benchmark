"""Shared fixtures for project tests."""

import os
import pytest
from pathlib import Path

from config import ServerConfig


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path for a temporary SQLite database."""
    return tmp_path / "test.db"


@pytest.fixture
def db_conn(tmp_db):
    """Return an open DB connection with schema applied."""
    from storage import get_db
    conn = get_db(tmp_db)
    yield conn
    conn.close()


@pytest.fixture
def mock_hf_cache(tmp_path, monkeypatch):
    """Create a mock HuggingFace cache directory and patch HF_CACHE_DIR."""
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir()
    monkeypatch.setattr("config.HF_CACHE_DIR", cache_dir)
    monkeypatch.setattr("models.HF_CACHE_DIR", cache_dir)
    return cache_dir


def make_gguf_repo(cache_dir: Path, org: str, repo: str, files: list[str], snapshot: str = "abc123") -> Path:
    """Create a mock HF cache repo with .gguf files. Returns the snapshot dir."""
    repo_dir = cache_dir / f"models--{org}--{repo}" / "snapshots" / snapshot
    repo_dir.mkdir(parents=True)
    for fname in files:
        (repo_dir / fname).write_bytes(b"\x00" * 1024)
    return repo_dir


def make_mlx_repo(cache_dir: Path, org: str, repo: str, snapshot: str = "abc123") -> Path:
    """Create a mock MLX model repo with config.json. Returns the snapshot dir."""
    repo_dir = cache_dir / f"models--{org}--{repo}" / "snapshots" / snapshot
    repo_dir.mkdir(parents=True)
    (repo_dir / "config.json").write_text('{"model_type": "test"}')
    (repo_dir / "model.safetensors").write_bytes(b"\x00" * 2048)
    return repo_dir


@pytest.fixture
def sample_config():
    """Return a factory for ServerConfig with sensible defaults."""
    def _make(**overrides):
        defaults = dict(
            label="test-model",
            hf_repo="org/test-repo-GGUF",
            quantization="Q4_K_M",
            param_count="7B",
            ctx_size=8192,
            port=8999,
        )
        defaults.update(overrides)
        return ServerConfig(**defaults)
    return _make
