"""Inferencia exclusivamente con pesos originales de OrigenLM.

No hay URLs de Ollama, proveedores externos ni descargas de modelos.
"""
from pathlib import Path

import torch

from origen.model import ModelConfig, OrigenLM
from origen.v2 import OrigenV2Config, OrigenV2
from origen.resources import configure_cpu_threads
from origen.tokenizer import DEFAULT_SYSTEM, conversation_prefix_limited, decode
from .config import Config, ROOT


class BackendError(RuntimeError):
    """Error al leer o generar desde checkpoint local."""


def checkpoint_predeterminado():
    candidates = [
        ROOT / "checkpoints/origen_ajustado.pt",
        ROOT / "checkpoints/origen_masivo.pt",
        ROOT / "checkpoints/origen_v2_latest.pt",
        ROOT / "checkpoints/origen_latest.pt",
    ]
    return next((p for p in candidates if p.is_file()), None)


class OrigenBackend:
    nombre = "origen-local"

    def __init__(self, cfg: Config):
        configure_cpu_threads("chat")
        path = cfg.checkpoint or checkpoint_predeterminado()
        if path is None or not Path(path).is_file():
            raise BackendError("Falta checkpoint propio; revisa 'origen estado' y entrena OrigenLM.")
        self.path = Path(path)
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        config = saved["config"]
        if saved.get("architecture") == "origen-v2-gqa-v1":
            model = OrigenV2(OrigenV2Config(**config))
        elif saved.get("architecture") in (None, "origen-v1"):
            model = OrigenLM(ModelConfig(**config))
        else:
            raise BackendError(f"Arquitectura desconocida: {saved.get('architecture')}")
        model.load_state_dict(saved["model"])
        self.model = model.eval()
        self.step = int(saved.get("step", 0))
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

    def chat(self, messages: list[dict], tools=None, on_token=None) -> dict:
        # Una red pequeña aún no está entrenada para llamadas a herramientas.
        # Estas se invocan SOLO explícitamente desde la CLI con aprobación.
        users = [x["content"] for x in messages if x.get("role") == "user"]
        if not users:
            raise BackendError("No hay mensaje del usuario.")
        hist = []
        pending = None
        for item in messages:
            if item.get("role") == "user":
                pending = item.get("content", "")
            elif item.get("role") == "assistant" and pending is not None:
                hist.append((pending, item.get("content", "")))
                pending = None
        try:
            prefix = conversation_prefix_limited(
                DEFAULT_SYSTEM, hist, users[-1], self.model.config.max_seq_len
            )
        except ValueError as exc:
            raise BackendError(str(exc)) from exc
        try:
            if self.temperature == 0 and isinstance(self.model, OrigenLM):
                # Los pesos v1 solo admiten temperatura positiva.
                temperature = 0.01
            else:
                temperature = self.temperature
            tokens = self.model.generate(
                prefix, max_new_tokens=self.max_tokens, temperature=temperature
            )
            response = decode(tokens).strip() or "[OrigenLM no produjo una respuesta útil.]"
        except (RuntimeError, ValueError) as exc:
            raise BackendError(str(exc)) from exc
        if on_token:
            on_token(response)
        return {"content": response, "tool_calls": []}


def crear_backend(cfg: Config):
    return OrigenBackend(cfg)
