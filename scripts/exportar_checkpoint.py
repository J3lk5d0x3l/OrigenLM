"""Preparar el paquete de transferencia (corpus + checkpoint) sin subirlo.

Copia los artefactos a `dist/`, calcula SHA-256, escribe un manifiesto y deja
listos los `dataset-metadata.json` de Kaggle. **No ejecuta ninguna subida**:
solo imprime los comandos para que los autorice el dueño de los datos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

LICENCIA_CORPUS = "CC-BY-SA-4.0"


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as flujo:
        for bloque in iter(lambda: flujo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exportar corpus y checkpoint a dist/")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/origen_v2_es.pt"))
    p.add_argument("--datos", type=Path, default=Path("dataset/preparado_v2_es"))
    p.add_argument("--salida", type=Path, default=Path("dist"))
    p.add_argument("--usuario", default="TU_USUARIO_KAGGLE")
    args = p.parse_args(argv)

    for ruta in (args.checkpoint, args.datos / "train.bin", args.datos / "val.bin",
                 args.datos / "tokenizer.json", args.datos / "meta.json"):
        if not ruta.is_file():
            p.error(f"Falta {ruta}")

    carga = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta = json.loads((args.datos / "meta.json").read_text(encoding="utf-8"))
    tokenizador = args.datos / "tokenizer.json"
    sha_tokenizador = sha256(tokenizador)
    if carga.get("tokenizer_sha256") != sha_tokenizador:
        p.error("El checkpoint no corresponde a este tokenizer.json: no lo exportes")
    if meta.get("tokenizer_sha256") != sha_tokenizador:
        p.error("meta.json y tokenizer.json no coinciden: vuelve a preparar el corpus")

    corpus = args.salida / "corpus"
    checkpoint = args.salida / "checkpoint"
    corpus.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    for nombre in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
        shutil.copy2(args.datos / nombre, corpus / nombre)
    shutil.copy2(args.checkpoint, checkpoint / "origen_v2_es.pt")

    hashes = {
        "train.bin": sha256(corpus / "train.bin"),
        "val.bin": sha256(corpus / "val.bin"),
        "tokenizer.json": sha_tokenizador,
        "checkpoint": sha256(checkpoint / "origen_v2_es.pt"),
    }
    (corpus / "hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    (corpus / "dataset-metadata.json").write_text(json.dumps({
        "title": "OrigenLM V2 corpus ES",
        "id": f"{args.usuario}/origenlm-v2-es",
        "licenses": [{"name": LICENCIA_CORPUS}],
        "description": ("Bins uint16 y tokenizador BPE propios de OrigenLM V2, "
                        "derivados de Wikipedia en español (CC BY-SA 4.0)."),
    }, indent=2) + "\n", encoding="utf-8")
    (checkpoint / "dataset-metadata.json").write_text(json.dumps({
        "title": "OrigenLM V2 checkpoint ES",
        "id": f"{args.usuario}/origenlm-v2-checkpoint",
        "licenses": [{"name": "other"}],
        "description": "Checkpoint y estado del optimizador propios de OrigenLM V2.",
    }, indent=2) + "\n", encoding="utf-8")

    lineas = [
        "# Paquete de transferencia OrigenLM V2",
        "",
        "Nada de esto se ha subido a ningún servicio. Súbelo solo si lo autorizas.",
        "",
        f"- paso: {carga.get('step')}",
        f"- tokens vistos: {carga.get('tokens_vistos'):,}",
        f"- NLL validación: {carga.get('val_loss')}",
        f"- vocabulario: {meta.get('vocab_size')}",
        f"- tokens de entrenamiento: {meta['splits']['train']['tokens']:,}",
        "",
        "| archivo | bytes | sha256 |",
        "|---|---|---|",
    ]
    for nombre in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
        ruta = corpus / nombre
        lineas.append(f"| corpus/{nombre} | {ruta.stat().st_size:,} | {hashes[nombre]} |")
    lineas.append(f"| checkpoint/origen_v2_es.pt | "
                  f"{(checkpoint / 'origen_v2_es.pt').stat().st_size:,} | "
                  f"{hashes['checkpoint']} |")
    lineas += [
        "",
        "## Subida (tú, con tu `kaggle.json`)",
        "",
        "```bash",
        f"kaggle datasets create -p {corpus.as_posix()} --dir-mode zip",
        f"kaggle datasets create -p {checkpoint.as_posix()} --dir-mode zip",
        "```",
        "",
        "Ambos quedan privados por defecto. El notebook verifica `hashes.json`.",
    ]
    (args.salida / "MANIFIESTO.md").write_text("\n".join(lineas) + "\n",
                                               encoding="utf-8")
    print("\n".join(lineas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
