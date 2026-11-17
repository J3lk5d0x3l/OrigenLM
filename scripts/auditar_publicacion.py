"""Auditoría conservadora del repositorio antes de publicarlo.

Inspecciona rutas y firmas de secretos evidentes en archivos versionados y,
opcionalmente, en los parches de TODO el historial Git local.
NO imprime valores encontrados y NO reescribe ni borra ningún commit.
No reemplaza una auditoría humana o un detector especializado.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

PROJECT = Path(__file__).resolve().parents[1]

# Dividir firmas en patrones evita usar tokens auténticos de ejemplo.
SIGNATURES = {
    "credencial GitHub": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})"),
    "clave tipo OpenAI": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{25,}"),
    "identificador de acceso AWS": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "clave privada": re.compile(r"^\s*-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "token Slack": re.compile(r"xox[baprs]-[A-Za-z0-9-]{25,}"),
    "URI MongoDB con credenciales": re.compile(
        r"mongodb(?:\+srv)?://[^/\s:@]{1,150}:[^/\s@]{1,150}@", re.IGNORECASE),
}
SECRET_FILENAMES = {".env", ".npmrc", ".pypirc", "id_rsa", "id_ed25519",
                    "credentials.json", "service-account.json"}
SECRET_SUFFIXES = {".pem", ".p12", ".pfx", ".key", ".safetensors", ".ckpt", ".pt", ".pth"}
PRIVATE_DIRS = ("checkpoints/", "dist/", "conversaciones/",
                "dataset/fuentes/", "dataset/preparado/", "dataset/preparado_v2/")


def sensitive_filename(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1].lower()
    if name == ".env.example":
        return False
    return (name in SECRET_FILENAMES or name.startswith(".env.")
            or any(name.endswith(suffix) for suffix in SECRET_SUFFIXES)
            or any(normalized.startswith(prefix) for prefix in PRIVATE_DIRS))


def secret_signatures(line: str) -> list[str]:
    return [label for label, pattern in SIGNATURES.items() if pattern.search(line)]


def git_text(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return done.stdout.decode("utf-8", errors="replace")


def scan_current(root: Path) -> list[tuple[str, str, str]]:
    tracked = git_text(root, "ls-files", "-z").split("\x00")
    results = []
    for name in (item for item in tracked if item):
        if sensitive_filename(name):
            results.append(("actual", name, "ruta privada versionada"))
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            results.append(("actual", name, "archivo demasiado grande para auditar"))
            continue
        blob = path.read_bytes()
        if b"\x00" in blob[:1024]:
            continue  # Git también puede guardar binarios; revisar por separado
        for number, line in enumerate(blob.decode("utf-8", errors="replace").splitlines(), 1):
            for label in secret_signatures(line):
                results.append(("actual", f"{name}:{number}", label))
    return results


def scan_history(root: Path) -> list[tuple[str, str, str]]:
    """Busca en los cambios históricos y en nombres de archivos de cada commit."""
    text = git_text(root, "log", "--all", "--no-ext-diff", "--no-color",
                    "--format=ORIGEN_COMMIT:%H", "--patch", "--no-renames")
    commit = "desconocido"
    file = "(desconocido)"
    findings = []
    for line in text.splitlines():
        if line.startswith("ORIGEN_COMMIT:"):
            commit = line.partition(":")[2][:12]
        elif line.startswith("diff --git "):
            if " b/" in line:
                file = line.rsplit(" b/", 1)[1]
                if sensitive_filename(file):
                    findings.append((commit, file, "ruta privada en historial"))
        elif line.startswith("+") and not line.startswith("+++"):
            for label in secret_signatures(line[1:]):
                findings.append((commit, file, label))
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auditar archivos e historial sin revelar secretos")
    parser.add_argument("--history", action="store_true", help="Inspeccionar los commits anteriores")
    parser.add_argument("--root", type=Path, default=PROJECT)
    args = parser.parse_args(argv)
    try:
        findings = scan_current(args.root)
        if args.history:
            findings.extend(scan_history(args.root))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[AUDITORÍA] Falló el análisis: {type(exc).__name__}")
        return 2
    unique = list(dict.fromkeys(findings))
    for scope, path, issue in unique[:40]:
        print(f"[REVISAR] {scope}: {path} | {issue}")
    if unique:
        print(f"[AUDITORÍA] {len(unique)} hallazgos posibles. "
              "Revisa en privado; revoca cualquier credencial real.")
        return 1
    print("[AUDITORÍA] Sin firmas evidentes en " +
          ("archivos e historial Git." if args.history else "archivos actuales."))
    print("[AVISO] No garantiza ausencia de secretos, PII o licencias incompatibles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
