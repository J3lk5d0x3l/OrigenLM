"""Convierte los diálogos del repo al formato `messages` que usan los modelos abiertos.

Salida por defecto: `dataset/sft/train.jsonl` y `dataset/sft/val.jsonl`, con una
línea por ejemplo:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Sin dependencias externas. Es el paso previo a `train_qlora.py`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SISTEMA = "Eres Origen, un asistente claro y útil. Respondés en español."
ENTRADAS = ("dataset/chat_v2.jsonl", "dataset/evaluacion_v2.jsonl")


def leer_dialogos(ruta: Path, max_caracteres: int) -> list[tuple[str, str]]:
    pares: list[tuple[str, str]] = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        linea = linea.strip()
        if not linea:
            continue
        try:
            dato = json.loads(linea)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{ruta}:{numero}: JSON inválido ({exc})")
        usuario = str(dato.get("user", "")).strip()
        asistente = str(dato.get("assistant", "")).strip()
        if not usuario or not asistente:
            continue
        if len(usuario) + len(asistente) > max_caracteres:
            continue
        pares.append((usuario, asistente))
    return pares


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", action="append", type=Path, default=None,
                   help="JSONL con claves user/assistant (se puede repetir)")
    p.add_argument("--salida", type=Path, default=Path("dataset/sft"))
    p.add_argument("--sistema", default=SISTEMA)
    p.add_argument("--val-fraccion", type=float, default=0.1)
    p.add_argument("--semilla", type=int, default=13)
    p.add_argument("--max-caracteres", type=int, default=4000)
    args = p.parse_args()

    entradas = args.entrada or [Path(r) for r in ENTRADAS]
    pares: list[tuple[str, str]] = []
    vistos: set[tuple[str, str]] = set()
    descartados = 0
    for ruta in entradas:
        if not ruta.is_file():
            print(f"[aviso] no existe {ruta}, se omite")
            continue
        for par in leer_dialogos(ruta, args.max_caracteres):
            if par in vistos:
                descartados += 1
                continue
            vistos.add(par)
            pares.append(par)

    if not pares:
        raise SystemExit("No hay ejemplos válidos: revisá --entrada")

    random.Random(args.semilla).shuffle(pares)
    corte = max(1, int(len(pares) * args.val_fraccion))
    validacion, entrenamiento = pares[:corte], pares[corte:]
    if not entrenamiento:
        entrenamiento, validacion = pares, pares[:1]

    args.salida.mkdir(parents=True, exist_ok=True)

    def escribir(nombre: str, datos: list[tuple[str, str]]) -> int:
        destino = args.salida / nombre
        with destino.open("w", encoding="utf-8") as flujo:
            for usuario, asistente in datos:
                flujo.write(json.dumps({"messages": [
                    {"role": "system", "content": args.sistema},
                    {"role": "user", "content": usuario},
                    {"role": "assistant", "content": asistente},
                ]}, ensure_ascii=False) + "\n")
        return destino.stat().st_size

    tam_train = escribir("train.jsonl", entrenamiento)
    tam_val = escribir("val.jsonl", validacion)

    caracteres = sum(len(u) + len(a) for u, a in pares)
    print(f"ejemplos   : {len(pares)} (duplicados descartados: {descartados})")
    print(f"train/val  : {len(entrenamiento)} / {len(validacion)}")
    print(f"caracteres : {caracteres:,} (~{caracteres // 4:,} tokens aproximados)")
    print(f"salida     : {args.salida}/train.jsonl ({tam_train / 1024:.1f} KiB), "
          f"val.jsonl ({tam_val / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
