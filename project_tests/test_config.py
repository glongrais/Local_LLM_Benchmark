"""Tests for config.py — model resolution, CLI args, plan loading."""

import json
from pathlib import Path

from config import ServerConfig, load_plan
from conftest import make_gguf_repo, make_mlx_repo


def test_resolve_gguf_from_cache(mock_hf_cache):
    make_gguf_repo(mock_hf_cache, "org", "model-GGUF", ["model-Q4_K_M.gguf"])
    config = ServerConfig(label="test", hf_repo="org/model-GGUF")
    path = config.resolve_model_path()
    assert path is not None
    assert path.endswith("model-Q4_K_M.gguf")


def test_resolve_gguf_exact_match(mock_hf_cache):
    make_gguf_repo(mock_hf_cache, "org", "repo-GGUF", ["model-Q4.gguf", "model-Q8.gguf"])
    config = ServerConfig(label="test", hf_repo="org/repo-GGUF", hf_file="model-Q8.gguf")
    path = config.resolve_model_path()
    assert path is not None
    assert "model-Q8.gguf" in path


def test_resolve_gguf_partial_match(mock_hf_cache):
    make_gguf_repo(mock_hf_cache, "org", "repo-GGUF", ["big-model-UD-Q4_K_XL.gguf"])
    config = ServerConfig(label="test", hf_repo="org/repo-GGUF", hf_file="Q4_K_XL")
    path = config.resolve_model_path()
    assert path is not None
    assert "Q4_K_XL" in path


def test_resolve_skips_mmproj(mock_hf_cache):
    make_gguf_repo(mock_hf_cache, "org", "repo-GGUF", ["mmproj-BF16.gguf", "model-Q4.gguf"])
    config = ServerConfig(label="test", hf_repo="org/repo-GGUF")
    path = config.resolve_model_path()
    assert path is not None
    assert path.endswith("model-Q4.gguf")


def test_resolve_returns_none_when_missing(mock_hf_cache):
    config = ServerConfig(label="test", hf_repo="org/nonexistent-GGUF")
    assert config.resolve_model_path() is None


def test_resolve_mlx_model(mock_hf_cache):
    make_mlx_repo(mock_hf_cache, "org", "model-MLX-4bit")
    config = ServerConfig(label="test", hf_repo="org/model-MLX-4bit", backend="mlx")
    path = config.resolve_model_path()
    assert path is not None
    assert "config.json" not in path  # returns the snapshot dir, not the file
    assert Path(path).is_dir()


def test_resolve_mlx_returns_none_without_config_json(mock_hf_cache):
    # Create repo dir but without config.json
    repo_dir = mock_hf_cache / "models--org--repo" / "snapshots" / "abc"
    repo_dir.mkdir(parents=True)
    (repo_dir / "model.safetensors").write_bytes(b"\x00")
    config = ServerConfig(label="test", hf_repo="org/repo", backend="mlx")
    assert config.resolve_model_path() is None


def test_resolve_model_path_takes_priority(mock_hf_cache):
    config = ServerConfig(label="test", model_path="/some/local/model.gguf")
    assert config.resolve_model_path() == "/some/local/model.gguf"


def test_llama_cli_args_basic():
    config = ServerConfig(
        label="test", hf_repo="org/repo",
        ctx_size=4096, n_gpu_layers=-1, batch_size=2048,
        flash_attn=True, port=8999,
    )
    args = config.to_cli_args()
    assert "-hf" in args
    assert "org/repo" in args
    assert "-c" in args and "4096" in args
    assert "-ngl" in args and "-1" in args
    assert "-b" in args and "2048" in args
    assert "--port" in args and "8999" in args
    assert "--metrics" in args
    assert args[args.index("-fa") + 1] == "on"


def test_llama_cli_args_flash_attn_off():
    config = ServerConfig(label="test", hf_repo="org/repo", flash_attn=False)
    args = config.to_cli_args()
    assert args[args.index("-fa") + 1] == "off"


def test_llama_cli_args_extra_args():
    config = ServerConfig(
        label="test", hf_repo="org/repo",
        extra_args={"cache-type-k": "q8_0", "mlock": True, "disabled": False},
    )
    args = config.to_cli_args()
    assert "--cache-type-k" in args
    assert "q8_0" in args
    assert "--mlock" in args
    assert "--disabled" not in args


def test_mlx_cli_args():
    config = ServerConfig(label="test", hf_repo="org/mlx-model", backend="mlx", port=9000)
    args = config.to_cli_args()
    assert args == ["--model", "org/mlx-model", "--port", "9000"]


def test_mlx_cli_args_with_local_path(mock_hf_cache):
    snap = make_mlx_repo(mock_hf_cache, "org", "mlx-model")
    config = ServerConfig(label="test", hf_repo="org/mlx-model", backend="mlx", port=9000)
    args = config.to_cli_args()
    assert args[0] == "--model"
    assert args[1] == str(snap)


def test_cli_args_raises_without_model():
    config = ServerConfig(label="test")
    try:
        config.to_cli_args()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_load_plan(tmp_path):
    plan_data = {
        "configs": [
            {"label": "m1", "hf_repo": "org/repo", "quantization": "Q4"},
            {"label": "m2", "hf_repo": "org/repo2", "backend": "mlx"},
        ],
        "prompt_dirs": ["prompts/"],
        "max_tokens": 8192,
        "_comment": "this should be ignored",
    }
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan_data))

    plan = load_plan(plan_file)
    assert len(plan.configs) == 2
    assert plan.configs[0].label == "m1"
    assert plan.configs[1].backend == "mlx"
    assert plan.max_tokens == 8192
