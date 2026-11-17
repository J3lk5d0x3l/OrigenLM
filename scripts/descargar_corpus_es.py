"""Corpus español acotado y con licencia identificada (Wikipedia es).

Descarga los shards parquet uno por uno, extrae documentos, aplica filtros de
calidad y deduplicación por SHA-256, y escribe JSONL de tamaño fijo.
El parquet se borra inmediatamente después de extraerlo para no gastar disco.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = "wikimedia/wikipedia"
CONFIG = "20231101.es"
LICENCIA = "CC BY-SA 4.0 / GFDL"
ATRIBUCION = "https://dumps.wikimedia.org/eswiki/"

FUNCIONALES = (" de ", " la ", " que ", " el ", " en ", " y ", " los ",
               " del ", " se ", " con ")
MIN_CARACTERES = 400
MAX_CARACTERES = 40000


def shards_parquet(ruta: str) -> list[str]:
    from huggingface_hub import HfApi

    entradas = HfApi().list_repo_tree(
        REPO, path_in_repo=ruta, repo_type="dataset", recursive=False
    )
    nombres = sorted(x.path for x in entradas
                     if getattr(x, "path", "").endswith(".parquet"))
    if not nombres:
        raise RuntimeError(f"{REPO}: no hay parquet en {ruta}")
    return nombres


def parece_espanol(texto: str) -> bool:
    """Filtro barato: sin palabras funcionales no es prosa española."""
    muestra = texto[:2500].lower()
    return sum(palabra in muestra for palabra in FUNCIONALES) >= 4


def limpiar(texto: str) -> str:
    """Normaliza saltos y descarta páginas dominadas por tablas."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    lineas: list[str] = []
    tablas = 0
    for linea in texto.split("\n"):
        linea = linea.rstrip()
        if linea.lstrip().startswith("|"):
            tablas += 1
            continue
        lineas.append(linea)
    if tablas > max(3, len(lineas) // 4):
        return ""
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Descargar corpus español acotado")
    parser.add_argument("--salida", type=Path, default=Path("dataset/fuentes_es"))
    parser.add_argument("--cache", type=Path, default=Path("dataset/_cache_hf"))
    parser.add_argument("--objetivo-mb", type=int, default=420,
                        help="Bytes de texto a conservar (no de descarga)")
    parser.add_argument("--shard-mb", type=int, default=40)
    parser.add_argument("--minimo-mb", type=int, default=200,
                        help="Falla si se consigue menos texto que esto")
    args = parser.parse_args()
    if args.objetivo_mb < args.minimo_mb or args.shard_mb < 1:
        parser.error("objetivo-mb debe ser >= minimo-mb y shard-mb positivo")

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    args.salida.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    libre = shutil.disk_usage(args.salida).free
    if libre < (args.objetivo_mb + 4096) * 1024 * 1024:
        print("[error] Se necesitan 4 GiB libres además del texto a conservar",
              file=sys.stderr)
        return 1

    objetivo = args.objetivo_mb * 1024 * 1024
    shard_limite = args.shard_mb * 1024 * 1024
    vistos: set[str] = set()
    documentos = 0
    descartados = 0
    conservados = 0
    buffer: list[str] = []
    buffer_bytes = 0
    shard_idx = 0
    usados: list[dict] = []

    def volcar() -> None:
        nonlocal buffer, buffer_bytes, shard_idx
        if not buffer:
            return
        destino = args.salida / f"es-{shard_idx:03d}.jsonl"
        destino.write_text("\n".join(buffer) + "\n", encoding="utf-8")
        shard_idx += 1
        buffer = []
        buffer_bytes = 0

    try:
        for nombre in shards_parquet(CONFIG):
            if conservados >= objetivo:
                break
            ruta = hf_hub_download(REPO, nombre, repo_type="dataset",
                                   local_dir=str(args.cache))
            del_shard = 0
            try:
                parquet = pq.ParquetFile(ruta)
                for lote in parquet.iter_batches(batch_size=1024, columns=["text"]):
                    for texto in lote.column("text").to_pylist():
                        if not texto:
                            continue
                        texto = limpiar(texto)
                        if (not MIN_CARACTERES <= len(texto) <= MAX_CARACTERES
                                or not parece_espanol(texto)):
                            descartados += 1
                            continue
                        digest = hashlib.sha256(texto.encode("utf-8")).hexdigest()
                        if digest in vistos:
                            descartados += 1
                            continue
                        vistos.add(digest)
                        linea = json.dumps(
                            {"text": texto, "sha256": digest,
                             "source": f"{REPO}:{nombre}"},
                            ensure_ascii=False)
                        tam = len(linea.encode("utf-8")) + 1
                        buffer.append(linea)
                        buffer_bytes += tam
                        conservados += tam
                        documentos += 1
                        del_shard += 1
                        if buffer_bytes >= shard_limite:
                            volcar()
                            print(f"[corpus] {documentos:,} docs | "
                                  f"{conservados / 2 ** 20:.0f} MiB", flush=True)
                    if conservados >= objetivo:
                        break
                usados.append({"shard": nombre, "documentos": del_shard})
            finally:
                Path(ruta).unlink(missing_ok=True)
        volcar()
    finally:
        shutil.rmtree(args.cache, ignore_errors=True)

    if conservados < args.minimo_mb * 1024 * 1024:
        print(f"[error] Solo {conservados / 2 ** 20:.0f} MiB < mínimo "
              f"{args.minimo_mb} MiB", file=sys.stderr)
        return 1

    procedencia = {
        "format": "origen-es-wikipedia-v1",
        "repo": REPO,
        "config": CONFIG,
        "license": LICENCIA,
        "attribution": ATRIBUCION,
        "notice": "Obra derivada de Wikipedia en español; conserva esta atribución "
                  "y comparte bajo CC BY-SA 4.0 cualquier redistribución del corpus.",
        "documentos": documentos,
        "descartados": descartados,
        "bytes": conservados,
        "shards": sorted(p.name for p in args.salida.glob("*.jsonl")),
        "parquet_usados": usados,
    }
    (args.salida / "procedencia.json").write_text(
        json.dumps(procedencia, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CORPUS ES: {documentos:,} docs | {conservados / 2 ** 20:.1f} MiB | "
          f"{descartados:,} descartados | {args.salida / 'procedencia.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
