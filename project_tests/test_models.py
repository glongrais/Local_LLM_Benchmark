"""Tests for models.py — file discovery, benchmarked detection, parsing."""

from pathlib import Path

from config import ServerConfig
from models import _format_size, _parse_hf_repo, _find_gguf_files, _is_benchmarked, _get_benchmarked_models, list_models
from storage import save_run
from conftest import make_gguf_repo


def test_format_size():
    assert _format_size(0) == "0.0 B"
    assert _format_size(1024) == "1.0 KB"
    assert _format_size(1024 * 1024 * 3) == "3.0 MB"
    assert "GB" in _format_size(1024 ** 3 * 5)


def test_parse_hf_repo():
    path = Path("/cache/models--unsloth--gemma-4-31B-it-GGUF/snapshots/abc/model.gguf")
    assert _parse_hf_repo(path) == "unsloth/gemma-4-31B-it-GGUF"


def test_parse_hf_repo_no_match():
    assert _parse_hf_repo(Path("/some/random/path/model.gguf")) == ""


def test_find_gguf_files_basic(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"\x00" * 100)
    (tmp_path / "other.txt").write_text("not a model")
    files = _find_gguf_files([str(tmp_path)])
    assert len(files) == 1
    assert files[0].name == "model.gguf"


def test_find_gguf_deduplicates_symlinks(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"\x00" * 100)
    link1 = tmp_path / "link1.gguf"
    link2 = tmp_path / "link2.gguf"
    link1.symlink_to(blob)
    link2.symlink_to(blob)
    files = _find_gguf_files([str(tmp_path)])
    # blob itself doesn't end in .gguf, so only the symlinks are found
    # but they resolve to the same target, so only one should be returned
    assert len(files) == 1


def test_find_gguf_skips_nonexistent():
    files = _find_gguf_files(["/nonexistent/path/that/does/not/exist"])
    assert files == []


def test_is_benchmarked_by_repo():
    benchmarked = {"unsloth/gemma-4-E4B-it-GGUF", "Gemma4-E4B-Q4_K_M"}
    path = Path("/cache/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/abc/model.gguf")
    assert _is_benchmarked(path, "unsloth/gemma-4-E4B-it-GGUF", benchmarked)


def test_is_benchmarked_by_label_substring():
    benchmarked = {"Gemma4-E4B-Q4_K_M"}
    path = Path("/cache/snapshots/gemma-4-E4B-it-UD-Q4_K_M.gguf")
    # filename stem is "gemma-4-E4B-it-UD-Q4_K_M", label is "Gemma4-E4B-Q4_K_M"
    # substring match: "gemma-4-E4B-it-UD-Q4_K_M" in "Gemma4-E4B-Q4_K_M" => False
    # But: "Gemma4-E4B-Q4_K_M" doesn't contain "gemma-4-E4B-it-UD-Q4_K_M"
    # The check is: name in b (for b in benchmarked)
    # So: "gemma-4-E4B-it-UD-Q4_K_M" in "Gemma4-E4B-Q4_K_M" => False
    assert not _is_benchmarked(path, "", benchmarked)


def test_is_benchmarked_not_matched():
    benchmarked = {"other-model"}
    path = Path("/cache/my-model.gguf")
    assert not _is_benchmarked(path, "org/different-repo", benchmarked)


def test_get_benchmarked_models_from_db(tmp_db):
    """Test _get_benchmarked_models with a real db_path (not a connection)."""
    from storage import get_db, save_run
    conn = get_db(tmp_db)
    config = ServerConfig(
        label="My-Model",
        hf_repo="org/my-model-GGUF",
        model_path="/path/to/model.gguf",
    )
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    benchmarked = _get_benchmarked_models(tmp_db)
    assert "My-Model" in benchmarked
    assert "org/my-model-GGUF" in benchmarked


def test_get_benchmarked_models_integration(tmp_db):
    """Full round-trip: save a run, then check _get_benchmarked_models finds it."""
    from storage import get_db, save_run
    conn = get_db(tmp_db)
    config = ServerConfig(label="TestLabel", hf_repo="org/test-repo")
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    benchmarked = _get_benchmarked_models(tmp_db)
    assert "TestLabel" in benchmarked
    assert "org/test-repo" in benchmarked


def test_get_benchmarked_models_null_model_path(tmp_db):
    """Ensure NULL model_path doesn't add 'None' to the set."""
    from storage import get_db, save_run
    conn = get_db(tmp_db)
    config = ServerConfig(label="TestLabel", hf_repo="org/repo")
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    benchmarked = _get_benchmarked_models(tmp_db)
    assert "None" not in benchmarked
    assert "" not in benchmarked


def test_list_models_with_benchmark_status(mock_hf_cache, tmp_db):
    from storage import get_db, save_run

    make_gguf_repo(mock_hf_cache, "org", "benchmarked-GGUF", ["model.gguf"])
    make_gguf_repo(mock_hf_cache, "org", "fresh-GGUF", ["model.gguf"])

    conn = get_db(tmp_db)
    config = ServerConfig(label="test", hf_repo="org/benchmarked-GGUF")
    save_run(conn, "r1", "2025-01-01T00:00:00Z", config)
    conn.close()

    models = list_models([str(mock_hf_cache)], db_path=tmp_db)
    assert len(models) == 2

    by_repo = {m["repo"]: m for m in models}
    assert by_repo["org/benchmarked-GGUF"]["benchmarked"] is True
    assert by_repo["org/fresh-GGUF"]["benchmarked"] is False
