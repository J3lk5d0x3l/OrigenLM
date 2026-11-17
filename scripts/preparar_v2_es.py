"""Preparar V2 BPE sobre el corpus español, sin tocar V1.

Reutiliza `origen.tokenizer_v2` (mismos tokens especiales e IDs) y produce
bins uint16 con el formato `origen-bpe-v2`, igual que dataset/preparado_v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from origen.tokenizer_v2 import encode_document, load_tokenizer, train_tokenizer


def documentos(fuentes: Path):
    for path in sorted(fuentes.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for linea in stream:
                if linea.strip():
                    yield json.loads(linea)["text"]


def particion(texto: str, val_percent: int) -> str:
    digest = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    return "val" if int(digest[:12], 16) % 100 < val_percent else "train"


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 español: BPE propio + bins")
    parser.add_argument("--fuentes", type=Path, default=Path("dataset/fuentes_es"))
    parser.add_argument("--salida", type=Path, default=Path("dataset/preparado_v2_es"))
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--val-percent", type=int, default=4)
    parser.add_argument("--min-documentos", type=int, default=1000)
    args = parser.parse_args()
    if not args.fuentes.is_dir():
        parser.error(f"No existe {args.fuentes}: descarga el corpus primero")
    if not 1 <= args.val_percent <= 20:
        parser.error("val-percent debe estar en 1..20")

    total = sum(1 for _ in documentos(args.fuentes))
    if total < args.min_documentos:
        parser.error(f"Solo {total} documentos < mínimo {args.min_documentos}")

    print(f"[V2 ES] {total:,} documentos fuente | entrenando BPE vocab={args.vocab}...",
          flush=True)

    def textos_train():
        for texto in documentos(args.fuentes):
            if particion(texto, args.val_percent) == "train":
                yield texto

    tok = train_tokenizer(textos_train(), vocab_size=args.vocab)
    args.salida.mkdir(parents=True, exist_ok=True)
    temporal = {s: args.salida / f"{s}.bin.tmp" for s in ("train", "val")}
    token_tmp = args.salida / "tokenizer.json.tmp"
    meta_tmp = args.salida / "meta.json.tmp"
    tamanos = {"train": 0, "val": 0}
    registros: list[dict] = []
    caracteres = 0
    try:
        tok.save(str(token_tmp))
        verificado = load_tokenizer(token_tmp)
        with temporal["train"].open("wb") as train, temporal["val"].open("wb") as val:
            flujos = {"train": train, "val": val}
            for texto in documentos(args.fuentes):
                split = particion(texto, args.val_percent)
                ids = encode_document(verificado, texto)
                if max(ids) >= 65535:
                    raise ValueError("ID de token fuera de uint16")
                flujos[split].write(np.asarray(ids, dtype="<u2").tobytes())
                registros.append({"split": split, "offset": tamanos[split],
                                  "tokens": len(ids),
                                  "sha256": hashlib.sha256(
                                      texto.encode("utf-8")).hexdigest()})
                tamanos[split] += len(ids)
                caracteres += len(texto)
            for flujo in flujos.values():
                flujo.flush()
                os.fsync(flujo.fileno())
        if min(tamanos.values()) < 200_000:
            raise ValueError("Validación pequeña: baja --val-percent o añade datos")
        reporte = {
            "format": "origen-bpe-v2", "version": 1,
            "tokenizer": "tokenizer.json",
            "vocab_size": verificado.get_vocab_size(),
            "tokenizer_sha256": hashlib.sha256(token_tmp.read_bytes()).hexdigest(),
            "total_documents": len(registros),
            "total_tokens": sum(tamanos.values()),
            "total_chars": caracteres,
            "chars_per_token": caracteres / max(1, sum(tamanos.values())),
            "splits": {s: {"path": s + ".bin", "tokens": tamanos[s],
                           "bytes": tamanos[s] * 2} for s in tamanos},
            "docs": registros,
        }
        meta_tmp.write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        for split in temporal:
            os.replace(temporal[split], args.salida / f"{split}.bin")
        os.replace(token_tmp, args.salida / "tokenizer.json")
        os.replace(meta_tmp, args.salida / "meta.json")
    finally:
        for path in (*temporal.values(), token_tmp, meta_tmp):
            path.unlink(missing_ok=True)

    print(f"V2 ES LISTO: vocab={reporte['vocab_size']} | "
          f"docs={reporte['total_documents']:,} | "
          f"tokens={reporte['total_tokens']:,} | "
          f"{reporte['chars_per_token']:.2f} caracteres/token")
    for split, data in reporte["splits"].items():
        print(f"{split}: {data['tokens']:,} tokens | {data['bytes'] / 2 ** 20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
