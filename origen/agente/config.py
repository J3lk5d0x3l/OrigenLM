"""Configuración local de herramientas. No conecta modelos externos."""
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.0"


@dataclass
class Config:
    checkpoint: Path | None = None
    temperature: float = 0.7
    max_tokens: int = 128
    workspace: Path = ROOT
    mcp_archivo: Path = ROOT / "origen.mcp.json"


def cargar_config(**overrides) -> Config:
    cfg = Config()
    ckpt = os.environ.get("ORIGEN_CHECKPOINT")
    if ckpt:
        cfg.checkpoint = Path(ckpt)
    for key, val in overrides.items():
        if val is not None:
            if key == "checkpoint":
                val = Path(val)
            setattr(cfg, key, val)
    if cfg.max_tokens < 1 or cfg.temperature < 0:
        raise ValueError("max_tokens debe ser positivo y temperatura no negativa")
    return cfg
