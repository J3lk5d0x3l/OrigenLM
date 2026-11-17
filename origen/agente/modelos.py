"""Catálogo de modelos sugeridos y estimación de si caben en la memoria disponible."""
from __future__ import annotations

import json
from pathlib import Path

from ..resources import memory_limit_bytes

CATALOGO = Path(__file__).with_name("modelos.json")


def cargar_catalogo(ruta: Path = CATALOGO) -> list[dict]:
    return json.loads(ruta.read_text(encoding="utf-8"))["modelos"]


def ram_gib() -> float | None:
    """RAM utilizable: límite del contenedor (cgroup) o, si no hay, la total del sistema."""
    limite = memory_limit_bytes()
    if limite:
        return limite / 1024 ** 3
    try:
        for linea in Path("/proc/meminfo").read_text().splitlines():
            if linea.startswith("MemTotal:"):
                return int(linea.split()[1]) / 1024 ** 2
    except (OSError, ValueError, IndexError):
        pass
    return None


def cabe(peso_gb: float, ram: float | None) -> bool | None:
    """Estimación prudente: pesos + margen para contexto y sistema."""
    if ram is None:
        return None
    return peso_gb * 1.25 + 1.5 <= ram
