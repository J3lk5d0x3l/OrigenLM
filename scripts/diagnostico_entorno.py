#!/usr/bin/env python3
"""Read-only environment checks, deliberately never print API key values."""
import importlib.util
import os
from pathlib import Path
import platform
import sys

root = Path(__file__).resolve().parents[1]
osrelease = Path("/etc/os-release")
distro = "no disponible"
if osrelease.is_file():
    for line in osrelease.read_text(errors="replace").splitlines():
        if line.startswith("PRETTY_NAME="):
            distro = line.partition("=")[2].strip().strip('"')
            break

print("OrigenLM - diagnóstico del entorno")
print("SO:", distro)
print("Arquitectura:", platform.machine())
print("Python:", sys.version.split()[0], sys.executable)
print("torch:", "disponible" if importlib.util.find_spec("torch") else "NO instalado en este entorno")
print("numpy:", "disponible" if importlib.util.find_spec("numpy") else "NO instalado en este entorno")
print("Checkpoint:", "presente" if (root / "checkpoints/origen_latest.pt").is_file() else "ausente")
print("CAMBER_API_KEY:", "definida (valor oculto)" if os.environ.get("CAMBER_API_KEY") else "no definida")
if "alpine" in distro.lower():
    print("AVISO: Alpine/musl no es la imagen Debian esperada; revisar logs de Codespaces.")
