"""Chargement de la configuration TOML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:9931/v1"
    model: str = "mistral-small-3.3-24b-instruct"
    api_key: str = "local"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: float = 300.0


@dataclass
class AgentConfig:
    system_prompt: str = "Tu es Jarvis, l'assistant local de Bastien."
    max_iterations: int = 8


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # name -> argv du processus MCP server (stdio)
    mcp_servers: dict[str, list[str]] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    candidates = [
        Path(path) if path else None,
        Path(os.environ.get("JARVIS_CONFIG", "")) if os.environ.get("JARVIS_CONFIG") else None,
        Path("config.toml"),
    ]
    cfg_path = next((p for p in candidates if p and p.is_file()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            "config.toml introuvable (ni JARVIS_CONFIG, ni ./config.toml). "
            "Copie config.example.toml en config.toml."
        )

    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = Config()
    cfg.llm = LLMConfig(**{k: v for k, v in raw.get("llm", {}).items()
                           if k in LLMConfig.__dataclass_fields__})
    cfg.agent = AgentConfig(**{k: v for k, v in raw.get("agent", {}).items()
                              if k in AgentConfig.__dataclass_fields__})
    for name, spec in raw.get("mcp", {}).get("servers", {}).items():
        cmd = spec.get("command", [])
        if isinstance(cmd, str):
            cmd = [cmd]
        cfg.mcp_servers[name] = list(cmd)
    return cfg
