"""Tests for models.py — repo discovery, benchmarked detection, parsing."""

from pathlib import Path

from config import ServerConfig
from models import _format_size, _parse_hf_repo, _dir_size, _detect_model_type, _get_benchmarked_repos, list_models
from storage import save_run
from conftest import make_gguf_repo, make_mlx_repo


def test_format_size():
    assert _format_size(0) == "0.0 B"
    assert _format_size(1024) == "1.0 KB"
    assert _format_size(1024 * 1024 * 3) == "3.0 MB"
    assert "GB" in _format_size(1024 ** 3 * 5)


def test_parse_hf_repo():
    path = Path("/cache/models--unsloth--gemma-4-31B-it-GGUF")
    assert _parse_hf_repo(path) == "unsloth/gemma-4-31B-it-GGUF"


def test_parse_hf_repo_no_models_prefix():
    # Without models-- prefix, returns the dir name as-is
    assert _parse_hf_repo(Path("/some/random/path")) == "path"


def test_dir_size(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\x00" * 100)
    (tmp_path / "b.bin").write_bytes(b"\x00" * 200)
    assert _dir_size(tmp_path) == 300


def test_dir_size_deduplicates_symlinks(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"\x00" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "link1").symlink_to(blob)
    (sub / "link2").symlink_to(blob)
    # blob counted once (100), not three times
    assert _dir_size(tmp_path) == 100


def test_detect_model_type_gguf(mock_hf_cache):
    make_gguf_repo(mock_hf_cache, "org", "repo-GGUF", ["model.gguf"])
    repo_dir = mock_hf_cache / "models--org--repo-GGUF"
    assert _detect_model_type(repo_dir) == "gguf"


def test_detect_model_type_safetensors(mock_hf_cache):
    make_mlx_repo(mock_hf_cache, "org", "repo-MLX")
    repo_dir = mock_hf_cache / "models--org--repo-MLX"
    assert _detect_model_type(repo_dir) == "safetensors"


def test_detect_model_type_other(tmp_path):
    repo_dir = tmp_path / "models--org--empty"
    repo_dir.mkdir()
    assert _detect_model_type(repo_dir) == "other"


def test_get_benchmarked_repos(tmp_db):
    from storage import get_db, save_run
    conn = get_db(tmp_db)
    config = ServerConfig(label="TestLabel", hf_repo="org/test-repo")
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    repos = _get_benchmarked_repos(tmp_db)
    assert "org/test-repo" in repos


def test_get_benchmarked_repos_excludes_empty(tmp_db):
    from storage import get_db, save_run
    conn = get_db(tmp_db)
    config = ServerConfig(label="TestLabel", model_path="/local/model.gguf")
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    repos = _get_benchmarked_repos(tmp_db)
    assert "" not in repos


def test_list_models_shows_all_repo_types(mock_hf_cache, tmp_db):
    from storage import get_db, save_run

    make_gguf_repo(mock_hf_cache, "org", "gguf-model", ["model.gguf"])
    make_mlx_repo(mock_hf_cache, "org", "mlx-model")

    conn = get_db(tmp_db)
    config = ServerConfig(label="test", hf_repo="org/gguf-model")
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    models = list_models([str(mock_hf_cache)], db_path=tmp_db)
    assert len(models) == 2

    by_repo = {m["repo"]: m for m in models}
    assert by_repo["org/gguf-model"]["type"] == "gguf"
    assert by_repo["org/gguf-model"]["benchmarked"] is True
    assert by_repo["org/mlx-model"]["type"] == "safetensors"
    assert by_repo["org/mlx-model"]["benchmarked"] is False


def test_list_models_sorted_by_size(mock_hf_cache):
    # Create two repos with different sizes
    snap1 = make_gguf_repo(mock_hf_cache, "org", "small", ["tiny.gguf"])
    snap2 = make_gguf_repo(mock_hf_cache, "org", "big", ["huge.gguf"])
    (snap2 / "huge.gguf").write_bytes(b"\x00" * 10000)

    models = list_models([str(mock_hf_cache)])
    assert models[0]["repo"] == "org/big"
    assert models[1]["repo"] == "org/small"


def test_list_models_gguf_files_listed(mock_hf_cache):
    make_gguf_repo(mock_hf_cache, "org", "multi-GGUF", ["q4.gguf", "q8.gguf", "mmproj-BF16.gguf"])
    models = list_models([str(mock_hf_cache)])
    assert len(models) == 1
    # mmproj should be excluded from gguf_files
    assert "mmproj-BF16.gguf" not in models[0]["gguf_files"]
    assert "q4.gguf" in models[0]["gguf_files"]
    assert "q8.gguf" in models[0]["gguf_files"]
