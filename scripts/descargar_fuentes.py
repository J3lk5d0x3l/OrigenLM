"""Descarga explícita y acotada de documentación con licencias identificadas.

Fuentes: libro y referencia de Rust (Apache-2.0/MIT) y documentación oficial
de Python (PSF-2.0). Se preservan notices de licencia y procedencia.
No usa modelos preentrenados ni datasets propietarios.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_DOWNLOAD = 30 * 1024 * 1024
MAX_EACH = 2 * 1024 * 1024
USER_AGENT = "OrigenLM-research-corpus/1.0"
SOURCES = [
    ("rust-book", "https://github.com/rust-lang/book/archive/refs/heads/main.tar.gz",
     "Apache-2.0 OR MIT"),
    ("rust-reference", "https://github.com/rust-lang/reference/archive/refs/heads/master.tar.gz",
     "Apache-2.0 OR MIT"),
    ("rust-by-example", "https://github.com/rust-lang/rust-by-example/archive/refs/heads/master.tar.gz",
     "Apache-2.0 OR MIT"),
]
PYTHON_DIRS = ["tutorial", "reference", "howto", "faq", "library"]


def descargar(url, max_bytes=MAX_DOWNLOAD):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=35) as response:
        total = 0
        chunks = []
        while True:
            part = response.read(65536)
            if not part:
                break
            total += len(part)
            if total > max_bytes:
                raise ValueError("Descarga supera el límite establecido")
            chunks.append(part)
    return b"".join(chunks)


def descargar_tar(source, url, license_id, dest, limit_files):
    if dest.exists() and any(dest.rglob("*.md")):
        return {"name": source, "url": url, "license": license_id,
                "files": sum(1 for _ in dest.rglob("*.md")), "cached": True}
    payload = descargar(url)
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    extracted = 0
    import io
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            # Ignorar ejecutables y enlaces; jamás extraer rutas de tar directamente.
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                continue
            if relative.suffix.lower() not in (".md", ".rst", ".txt") and relative.name not in ("LICENSE-MIT", "LICENSE-APACHE", "COPYRIGHT"):
                continue
            if member.size > MAX_EACH or member.size < 32:
                continue
            if count >= limit_files:
                break
            # No usar código no publicado o archivos de GitHub Actions como datos.
            if relative.parts[0] in (".github", "target", "book"):
                continue
            extracted += member.size
            if extracted > 45 * 1024 * 1024:
                raise ValueError("Archivo comprimido excede el presupuesto de extracción")
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source_file:
                target.write_bytes(source_file.read(MAX_EACH))
            count += 1
    if count < 5:
        raise RuntimeError(f"{source}: no se obtuvieron suficientes archivos")
    return {"name": source, "url": url, "license": license_id, "files": count,
            "download_bytes": len(payload), "cached": False}


def descargar_python(dest: Path, limit_files: int):
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    repositorio = "https://api.github.com/repos/python/cpython/contents/Doc"
    for carpeta in PYTHON_DIRS:
        try:
            contents = json.loads(descargar(repositorio + "/" + carpeta + "?ref=3.14", MAX_EACH).decode("utf-8"))
        except (HTTPError, URLError, ValueError) as exc:
            print(f"[aviso] Python/{carpeta}: {exc}", flush=True)
            continue
        for item in contents:
            if count >= limit_files:
                break
            if item.get("type") != "file" or not item.get("name", "").endswith(".rst"):
                continue
            destination = dest / carpeta / item["name"]
            if destination.exists() and destination.stat().st_size > 60:
                count += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = descargar(item["download_url"], MAX_EACH)
            destination.write_bytes(content)
            count += 1
        if count >= limit_files:
            break
    return {"name": "python-docs", "url": "https://github.com/python/cpython/tree/3.14/Doc",
            "license": "PSF-2.0", "files": count}


def main():
    parser = argparse.ArgumentParser(description="Corpus autorizado de programación (sin modelos de terceros)")
    parser.add_argument("--dest", type=Path, default=ROOT / "dataset/fuentes")
    parser.add_argument("--max-files", type=int, default=650)
    parser.add_argument("--max-mb", type=int, default=150)
    args = parser.parse_args()
    if args.max_files < 10 or args.max_mb < 10:
        parser.error("Usa al menos 10 documentos y 10 MiB de cuota máxima")
    remaining = shutil.disk_usage(args.dest.parent if args.dest.parent.exists() else ROOT).free
    if remaining < (args.max_mb + 2048) * 1024 * 1024:
        parser.error("Se requieren al menos 2 GiB libres además del presupuesto de descargas")
    results = []
    for name, url, license_id in SOURCES:
        try:
            result = descargar_tar(name, url, license_id, args.dest / name, args.max_files)
            results.append(result)
            print(f"{name}: {result['files']} archivos", flush=True)
        except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
            print(f"[aviso] {name}: {exc}", flush=True)
    try:
        result = descargar_python(args.dest / "python-docs", args.max_files)
        results.append(result)
        print(f"Python: {result['files']} archivos", flush=True)
    except (OSError, ValueError) as exc:
        print(f"[aviso] Python: {exc}", flush=True)
    files = [p for p in args.dest.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    if total_bytes > args.max_mb * 1024 * 1024:
        parser.error("Se excedió el límite de corpus indicado; revisa los archivos")
    metadata = {"version": 1, "sources": results, "files_total": len(files),
                "bytes_total": total_bytes,
                "licenses": ["https://github.com/rust-lang/book/blob/main/LICENSE-MIT",
                             "https://github.com/rust-lang/book/blob/main/LICENSE-APACHE",
                             "https://github.com/python/cpython/blob/3.14/LICENSE"],
                "notice": "Conserva atribuciones y verifica licencias antes de redistribuir."}
    report = args.dest.parent / "fuentes_procedencia.json"
    report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CORPUS DESCARGADO: {len(files)} archivos, {total_bytes / 1024 ** 2:.2f} MiB; {report}", flush=True)
    if len(files) < 10:
        raise RuntimeError("No se consiguió un corpus suficiente; comprueba Internet o los URLs")


if __name__ == "__main__":
    main()
