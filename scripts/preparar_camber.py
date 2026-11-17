#!/usr/bin/env python3
"""Create a small, explicit allowlist ZIP for a Camber Cloud upload.

Offline; never reads credentials, .env, .git, or arbitrary project folders.
"""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "ARQUITECTURA.md",
    "docs/CAMBER.md",
    "requirements.txt",
    "train.py",
    "chat.py",
    "origen/__init__.py",
    "origen/model.py",
    "origen/tokenizer.py",
    "dataset/chat.jsonl",
    "tests/test_model.py",
)
CHECKPOINT = "checkpoints/origen_latest.pt"


def prepare(include_checkpoint: bool = False) -> Path:
    contents = list(FILES)
    if include_checkpoint:
        contents.append(CHECKPOINT)
    missing = [name for name in contents if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Faltan archivos del paquete: " + ", ".join(missing))
    target = ROOT / "dist" / "OrigenLM-Camber.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as package:
        for name in contents:
            package.write(ROOT / name, name)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara un ZIP local de OrigenLM para Camber (sin subir nada)")
    parser.add_argument("--include-checkpoint", action="store_true", help="Incluir los pesos guardados en la copia")
    args = parser.parse_args()
    file = prepare(args.include_checkpoint)
    print(f"Paquete listo: {file}")
    print("No se ha enviado nada a Camber ni se ha usado ninguna API key.")
