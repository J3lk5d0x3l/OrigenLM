"""Preparar BPE V2 con corpus autorizado de V1, sin modificar pesos anteriores."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from origen.tokenizer_v2 import train_tokenizer, load_tokenizer, encode_document

ROOT = Path(__file__).resolve().parents[1]


def read_source(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Ruta fuera del corpus autorizado")
    if not target.is_file() or target.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"Documento no válido: {relative}")
    raw = target.read_bytes()
    if b"\x00" in raw[:1024]:
        raise ValueError(f"Documento binario: {relative}")
    return re.sub(r"\n{4,}", "\n\n\n", raw.decode("utf-8", errors="replace")
                  .replace("\r\n", "\n").replace("\r", "\n")).strip()


def prepare(index: Path, source: Path, output: Path, vocab=8192):
    if not source.is_dir() or not index.is_file():
        raise ValueError("Falta dataset/fuentes o dataset/preparado/meta.json")
    if output.resolve() == index.parent.resolve():
        raise ValueError("V2 no puede sobrescribir el corpus V1")
    meta = json.loads(index.read_text(encoding="utf-8"))
    if meta.get("format") != "origen-byte-v1":
        raise ValueError("Índice de entrenamiento V1 no reconocido")
    docs = meta.get("docs", [])
    counts = Counter(d.get("split") for d in docs)
    if counts["train"] == 0 or counts["val"] == 0:
        raise ValueError("Es necesario un split train/val con documentos separados")
    if len({d["path"] for d in docs}) != len(docs):
        raise ValueError("El índice contiene documentos repetidos")
    if shutil.disk_usage(ROOT).free < 120 * 1024 * 1024:
        raise ValueError("Espacio insuficiente para preparar V2")

    def training_texts():
        for doc in docs:
            if doc["split"] == "train":
                yield read_source(source, doc["path"])

    print("[OrigenLM V2] Entrenando vocabulario BPE propio SOLO con train...", flush=True)
    tok = train_tokenizer(training_texts(), vocab_size=vocab)
    output.mkdir(parents=True, exist_ok=True)
    temp = {s: output / (s + ".bin.tmp") for s in ("train", "val")}
    token_temp = output / "tokenizer.json.tmp"
    meta_temp = output / "meta.json.tmp"
    sizes, records = {"train": 0, "val": 0}, []
    try:
        tok.save(str(token_temp))
        checked = load_tokenizer(token_temp)
        for split in ("train", "val"):
            with temp[split].open("wb") as stream:
                for doc in docs:
                    if doc["split"] != split:
                        continue
                    encoded = encode_document(checked, read_source(source, doc["path"]))
                    if max(encoded) >= 65535:
                        raise ValueError("ID token fuera de uint16")
                    stream.write(np.asarray(encoded, dtype="<u2").tobytes())
                    records.append({"path": doc["path"], "split": split,
                                    "offset": sizes[split], "tokens": len(encoded),
                                    "sha256": doc["sha256"]})
                    sizes[split] += len(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        if min(sizes.values()) < 100:
            raise ValueError("Split demasiado pequeño para evaluar o entrenar")
        report = {
            "format": "origen-bpe-v2", "version": 1,
            "tokenizer": "tokenizer.json",
            "vocab_size": checked.get_vocab_size(),
            "tokenizer_sha256": sha256(token_temp.read_bytes()).hexdigest(),
            "total_documents": len(docs), "total_tokens": sum(sizes.values()),
            "splits": {s: {"path": s + ".bin", "tokens": sizes[s],
                           "bytes": sizes[s] * 2} for s in sizes},
            "docs": records}
        meta_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        for split in temp:
            os.replace(temp[split], output / (split + ".bin"))
        os.replace(token_temp, output / "tokenizer.json")
        os.replace(meta_temp, output / "meta.json")
        return report
    finally:
        for path in (*temp.values(), token_temp, meta_temp):
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="V2: BPE original + corpus sin tocar V1")
    parser.add_argument("--indice", type=Path, default=ROOT / "dataset/preparado/meta.json")
    parser.add_argument("--fuentes", type=Path, default=ROOT / "dataset/fuentes")
    parser.add_argument("--salida", type=Path, default=ROOT / "dataset/preparado_v2")
    parser.add_argument("--vocab", type=int, default=8192)
    args = parser.parse_args(argv)
    report = prepare(args.indice, args.fuentes, args.salida, vocab=args.vocab)
    print(f"V2 PREPARADO: vocab={report['vocab_size']} | "
          f"docs={report['total_documents']} | tokens={report['total_tokens']:,}")
    for split, data in report["splits"].items():
        print(f"{split}: {data['tokens']:,} tokens")
    print("Checkpoint V1 intacto. Siguiente: entrenador propio V2 BPE.")


if __name__ == "__main__":
    main()
