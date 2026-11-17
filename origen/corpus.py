"""Corpus grande de OrigenLM: streaming a uint16, división por documento.

No requiere modelos, tokenizadores ni bibliotecas externas. Usa bytes UTF-8
y tokens BOS/EOS propios. Los documentos privados nunca se versionan.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct

EXTENSIONS = {".txt", ".md", ".rst", ".py", ".js", ".jsx", ".ts", ".tsx",
              ".rs", ".go", ".java", ".cpp", ".c", ".h", ".cs", ".sh",
              ".sql", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "target",
             "dist", "build", ".next", "vendor"}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 350 * 1024 * 1024
HEADER = struct.Struct("<H")
BOS, EOS = 256, 257


def documentos(carpeta: Path):
    """Iterar archivos sin cargar árboles completos en RAM."""
    for current, dirs, files in os.walk(carpeta):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix.lower() not in EXTENSIONS or name.startswith("."):
                continue
            try:
                if 64 <= path.stat().st_size <= MAX_FILE_BYTES:
                    yield path
            except OSError:
                continue


def split_document(digest: str, val_percent: int = 5) -> str:
    """La misma fuente siempre cae en el mismo subconjunto."""
    return "val" if int(digest[:12], 16) % 100 < val_percent else "train"


def preparar_corpus(source: Path, output: Path, max_source_bytes: int = MAX_SOURCE_BYTES) -> dict:
    """Construir shards uint16 atómicos para sampling sobre mmap."""
    if not source.is_dir():
        raise ValueError(f"No existe carpeta de corpus: {source}")
    if max_source_bytes <= 0:
        raise ValueError("max_source_bytes debe ser positivo")
    output.mkdir(parents=True, exist_ok=True)
    data_tmp = {s: output / (s + ".bin.tmp") for s in ("train", "val")}
    data_out = {s: output / (s + ".bin") for s in ("train", "val")}
    metadata = {"format": "origen-byte-v1", "version": 1, "splits": {}, "docs": [],
                "source": str(source.resolve())}
    seen = set()
    sizes = {"train": 0, "val": 0}
    consumed = 0
    try:
        with data_tmp["train"].open("wb") as train, data_tmp["val"].open("wb") as val:
            writers = {"train": train, "val": val}
            for path in documentos(source):
                raw = path.read_bytes()
                consumed += len(raw)
                if consumed > max_source_bytes:
                    raise ValueError(f"Corpus supera límite de {max_source_bytes:,} bytes")
                # Los binarios y archivos vacíos no son corpus de texto.
                if b"\0" in raw[:1024]:
                    continue
                content = raw.decode("utf-8", errors="replace")
                content = content.replace("\r\n", "\n").replace("\r", "\n")
                content = re.sub(r"\n{4,}", "\n\n\n", content).strip()
                encoded = content.encode("utf-8")
                if len(encoded) < 64:
                    continue
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                split = split_document(digest)
                start = sizes[split]
                writer = writers[split]
                # Escribir secuencias incrementales; no construir lista de tokens.
                writer.write(HEADER.pack(BOS))
                writer.write(encoded.translate(bytes.maketrans(bytes(range(256)),bytes(range(256))))) if False else None
                # uint16 le; por bloques para no duplicar el documento en RAM.
                for i in range(0, len(encoded), 16384):
                    piece = encoded[i:i + 16384]
                    writer.write(struct.pack("<" + "H" * len(piece), *piece))
                writer.write(HEADER.pack(EOS))
                length = len(encoded) + 2
                sizes[split] += length
                metadata["docs"].append({"path": str(path.relative_to(source)),
                                         "sha256": digest, "split": split,
                                         "offset": start, "tokens": length})
            for file in writers.values():
                file.flush()
                os.fsync(file.fileno())
        if sizes["train"] < 256 or sizes["val"] < 128:
            raise ValueError("Corpus insuficiente o sin validación: añade más documentos.")
        for split in ("train", "val"):
            os.replace(data_tmp[split], data_out[split])
            metadata["splits"][split] = {"path": data_out[split].name,
                                         "tokens": sizes[split],
                                         "bytes": sizes[split] * 2}
        metadata["total_documents"] = len(metadata["docs"])
        metadata["total_tokens"] = sum(sizes.values())
        meta_tmp = output / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(meta_tmp, output / "meta.json")
        return metadata
    except Exception:
        for path in data_tmp.values():
            path.unlink(missing_ok=True)
        raise
