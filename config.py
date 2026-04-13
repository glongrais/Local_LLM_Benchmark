"""Configuration dataclasses and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class ServerConfig:
    label: str  # human-readable, e.g. "Qwen2.5-14B-Q5_K_M"
    # Model source: use hf_repo for -hf flag, or model_path for local file
    hf_repo: str = ""  # e.g. "unsloth/gemma-4-31B-it-GGUF"
    hf_file: str = ""  # e.g. "gemma-4-31B-it-UD-Q4_K_XL.gguf" (optional, for multi-file repos)
    model_path: str = ""  # local path to .gguf file (alternative to hf_repo)
    quantization: str = ""  # e.g. "Q4_K_XL"
    param_count: str = ""  # e.g. "31B"
    ctx_size: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all
    threads: int = 0  # 0 = llama-server default
    batch_size: int = 2048
    flash_attn: bool = True
    port: int = 8999
    extra_args: dict = field(default_factory=dict)
    # extra_args can include: cache_type_k, cache_type_v, mlock, etc.

    def resolve_model_path(self) -> str | None:
        """Try to find the model in the local HF cache. Returns local path or None."""
        if self.model_path:
            return self.model_path
        if not self.hf_repo:
            return None
        repo_dir_name = "models--" + self.hf_repo.replace("/", "--")
        repo_dir = HF_CACHE_DIR / repo_dir_name
        if not repo_dir.exists():
            return None
        # Search snapshots for the matching gguf file
        candidates = []
        for gguf in repo_dir.rglob("*.gguf"):
            if "/snapshots/" not in str(gguf):
                continue
            if "mmproj" in gguf.name:
                continue
            # Exact match
            if self.hf_file and gguf.name == self.hf_file:
                return str(gguf)
            # Partial match (e.g. "UD-Q4_K_XL" matches "gemma-4-31B-it-UD-Q4_K_XL.gguf")
            if self.hf_file and self.hf_file in gguf.name:
                return str(gguf)
            candidates.append(gguf)
        # No hf_file specified or no match — return first non-mmproj gguf
        if not self.hf_file and candidates:
            return str(candidates[0])
        return None

    def to_cli_args(self) -> list[str]:
        args = []
        # Prefer local cache path to avoid network checks
        local_path = self.resolve_model_path()
        if local_path and Path(local_path).exists():
            args += ["-m", local_path]
        elif self.hf_repo:
            args += ["-hf", self.hf_repo]
            if self.hf_file:
                args += ["-hff", self.hf_file]
        elif self.model_path:
            args += ["-m", self.model_path]
        else:
            raise ValueError(f"Config '{self.label}': must specify hf_repo or model_path")

        args += [
            "-c", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers),
            "-b", str(self.batch_size),
            "--port", str(self.port),
            "--metrics",
        ]
        if self.threads > 0:
            args += ["-t", str(self.threads)]
        if self.flash_attn:
            args += ["-fa", "on"]
        else:
            args += ["-fa", "off"]
        for k, v in self.extra_args.items():
            flag = f"--{k}" if len(k) > 1 else f"-{k}"
            if v is True:
                args.append(flag)
            elif v is not False and v is not None:
                args += [flag, str(v)]
        return args


@dataclass
class BenchmarkPlan:
    configs: list[ServerConfig]
    prompt_dirs: list[str] = field(default_factory=lambda: ["prompts/"])
    repeat: int = 1
    max_tokens: int = 2048
    temperature: float = 0.0


def load_plan(path: str | Path) -> BenchmarkPlan:
    with open(path) as f:
        data = json.load(f)
    configs = [ServerConfig(**c) for c in data.get("configs", [])]
    plan_kwargs = {k: v for k, v in data.items() if k != "configs"}
    return BenchmarkPlan(configs=configs, **plan_kwargs)
