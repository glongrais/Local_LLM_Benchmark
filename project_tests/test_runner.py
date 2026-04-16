"""Tests for runner.py — prompt loading and prefetcher logic."""

import json

from config import ServerConfig
from runner import load_prompts, ModelPrefetcher


# --- load_prompts ---

def test_load_prompts_from_directory(tmp_path):
    prompts1 = [{"name": "a", "prompt": "hi"}, {"name": "b", "prompt": "bye"}]
    prompts2 = [{"name": "c", "prompt": "hello"}]
    (tmp_path / "p1.json").write_text(json.dumps(prompts1))
    (tmp_path / "p2.json").write_text(json.dumps(prompts2))

    loaded = load_prompts([str(tmp_path)])
    assert len(loaded) == 3
    names = {p["name"] for p in loaded}
    assert names == {"a", "b", "c"}


def test_load_prompts_from_single_file(tmp_path):
    prompts = [{"name": "x", "prompt": "test"}]
    f = tmp_path / "single.json"
    f.write_text(json.dumps(prompts))

    loaded = load_prompts([str(f)])
    assert len(loaded) == 1
    assert loaded[0]["name"] == "x"


def test_load_prompts_nonexistent_path():
    loaded = load_prompts(["/nonexistent/path/nowhere"])
    assert loaded == []


def test_load_prompts_single_object(tmp_path):
    """A JSON file with a single prompt dict (not a list) should still work."""
    prompt = {"name": "solo", "prompt": "just one"}
    (tmp_path / "solo.json").write_text(json.dumps(prompt))

    loaded = load_prompts([str(tmp_path)])
    assert len(loaded) == 1
    assert loaded[0]["name"] == "solo"


# --- ModelPrefetcher ---

def test_prefetcher_skips_cached_model(monkeypatch):
    config = ServerConfig(label="cached", hf_repo="org/repo")
    monkeypatch.setattr(config, "resolve_model_path", lambda: "/some/cached/path")

    pf = ModelPrefetcher()
    pf.prefetch(config)
    assert len(pf._threads) == 0


def test_prefetcher_skips_no_hf_repo():
    config = ServerConfig(label="local", model_path="/local/model.gguf")
    pf = ModelPrefetcher()
    pf.prefetch(config)
    assert len(pf._threads) == 0


def test_prefetcher_deduplication():
    config = ServerConfig(label="test", hf_repo="org/repo", hf_file="model.gguf")
    pf = ModelPrefetcher()

    # Mock _download to avoid real downloads
    pf._download = lambda c, k: pf._done.add(k)

    pf.prefetch(config)
    pf.prefetch(config)  # second call should be a no-op
    # Only one thread should have been created
    assert len(pf._threads) == 1
