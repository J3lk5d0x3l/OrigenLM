"""Configuración del agente: valores por defecto, origen.json, .env y variables de entorno."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"


def cargar_env(ruta: Path | None = None) -> None:
    """Lee un archivo .env (CLAVE=valor) sin pisar variables ya definidas."""
    ruta = ruta or ROOT / ".env"
    try:
        texto = ruta.read_text(encoding="utf-8")
    except OSError:
        return
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


@dataclass
class Config:
    backend: str = "ollama"  # "ollama" o "openai" (API compatible con OpenAI)
    host: str = ""  # vacío = dirección típica del backend
    modelo: str = "qwen2.5:3b"
    api_key: str = ""  # solo desde variable de entorno ORIGEN_API_KEY
    contexto: int = 8192  # ventana de contexto que se pide a Ollama
    max_pasos: int = 8  # llamadas a herramientas por pregunta
    timeout: int = 600
    aprobar_todo: bool = False
    workspace: Path = ROOT
    mcp_archivo: Path = ROOT / "origen.mcp.json"

    def host_efectivo(self) -> str:
        if self.host:
            return self.host.rstrip("/")
        return "http://127.0.0.1:11434" if self.backend == "ollama" else "http://127.0.0.1:8080/v1"


def _aplicar(cfg: Config, datos: dict) -> None:
    for clave, valor in datos.items():
        if not hasattr(cfg, clave):
            continue
        actual = getattr(cfg, clave)
        if isinstance(actual, bool):
            valor = valor if isinstance(valor, bool) else str(valor).lower() in ("1", "true", "si", "sí", "yes")
        elif isinstance(actual, int):
            valor = int(valor)
        elif isinstance(actual, Path):
            valor = Path(valor)
        else:
            valor = str(valor)
        setattr(cfg, clave, valor)


def cargar_config(**sobrescribir) -> Config:
    """Prioridad: argumentos > variables ORIGEN_* (.env) > origen.json > valores por defecto."""
    cargar_env()
    cfg = Config()
    archivo = ROOT / "origen.json"
    if archivo.is_file():
        try:
            datos = json.loads(archivo.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"origen.json no es válido: {exc}") from exc
        datos.pop("api_key", None)  # las claves nunca van en un archivo versionado
        _aplicar(cfg, datos)
    nombres = {"backend": "ORIGEN_BACKEND", "host": "ORIGEN_HOST", "modelo": "ORIGEN_MODELO",
               "api_key": "ORIGEN_API_KEY", "contexto": "ORIGEN_CONTEXTO", "max_pasos": "ORIGEN_MAX_PASOS"}
    _aplicar(cfg, {k: os.environ[v] for k, v in nombres.items() if os.environ.get(v)})
    _aplicar(cfg, {k: v for k, v in sobrescribir.items() if v is not None})
    if cfg.backend not in ("ollama", "openai"):
        raise ValueError("backend debe ser 'ollama' o 'openai'")
    return cfg
