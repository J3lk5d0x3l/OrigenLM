"""Punto de entrada del comando origen en Codespaces."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

from .resources import effective_cpu_count, memory_limit_bytes

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_V2 = ROOT / "checkpoints/origen_v2_latest.pt"
CHECKPOINT_V1 = ROOT / "checkpoints/origen_latest.pt"
CHECKPOINT_MASIVO = ROOT / "checkpoints/origen_masivo.pt"
# Agente con PESOS PROPIOS, herramientas de Internet y MCP.
COMANDOS_AGENTE = ("agente", "preguntar", "herramientas", "doctor")


def chosen_chat(v1: Path = CHECKPOINT_V1, v2: Path = CHECKPOINT_V2) -> str | None:
    if v2.is_file():
        return "chat_origen_v2.py"
    if v1.is_file():
        return "chat.py"
    return None


def format_bytes(value: int | None) -> str:
    return f"{value / (1024 ** 3):.1f} GiB" if value is not None else "desconocida"


def show_status() -> int:
    print("OrigenLM | recursos y checkpoints")
    print(f"Repositorio: {ROOT}")
    print(f"CPU efectivas detectadas: {effective_cpu_count()}")
    print(f"Límite RAM cgroup: {format_bytes(memory_limit_bytes())}")
    disk = shutil.disk_usage(ROOT)
    print(f"Espacio libre: {format_bytes(disk.free)} / total {format_bytes(disk.total)}")
    for name, path in (("masivo", CHECKPOINT_MASIVO), ("v2", CHECKPOINT_V2), ("v1", CHECKPOINT_V1)):
        print(f"Checkpoint {name}: {path.relative_to(ROOT)}" if path.is_file()
              else f"Checkpoint {name}: todavía no existe")
    data = ROOT / "dataset/chat_v2.jsonl"
    if data.is_file():
        with data.open(encoding="utf-8") as source:
            count = sum(bool(row.strip()) for row in source)
        print(f"Datos conversacionales: {count} ejemplos (demostración)")
    print("Agente con pesos propios y MCP: origen doctor  |  origen agente")
    print("Iniciar conversación con checkpoint propio: origen")
    print("Continuar pesos propios: origen entrenar-masivo --steps 20000")
    return 0


def exec_project_script(script: str, arguments: list[str]) -> None:
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, str(ROOT / script), *arguments])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help", "ayuda", "help"):
        print("""OrigenLM — sin modelos externos
  origen                           Abrir chat con checkpoint propio
  origen agente                    Chat con herramientas Internet/MCP y pesos propios
  origen preguntar "texto"         Pregunta al checkpoint propio
  origen herramientas              Ver herramientas y conexiones MCP
  origen doctor                    Diagnosticar recursos y checkpoint
  origen entrenar-masivo           Descargar corpus autorizado y entrenar pesos ORIGINALES
  origen entrenar-masivo --dry-run  Ver el plan sin ejecutarlo
  origen entrenar --steps 1000     Crear modelo v2 desde cero (opcional)
  origen estado                    Estado de recursos, corpus y pesos
  origen v1 | origen v2            Elegir un checkpoint histórico
  origen pruebas                   Ejecutar pruebas
  origen --help                    Mostrar esta ayuda
""")
        return 0
    if args and args[0] in COMANDOS_AGENTE:
        from .agente.cli import main as agente_main
        return agente_main(args)
    if args and args[0] == "estado":
        return show_status()
    if args and args[0] == "pruebas":
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT)
    if args and args[0] == "entrenar-masivo":
        exec_project_script("scripts/entrenamiento_masivo.py", args[1:])
    if args and args[0] == "entrenar":
        exec_project_script("train_origen_v2.py", args[1:])
    if args and args[0] in ("v1", "v2"):
        script = "chat.py" if args[0] == "v1" else "chat_origen_v2.py"
        path = CHECKPOINT_V1 if args[0] == "v1" else CHECKPOINT_V2
        if not path.is_file():
            print(f"No existe el checkpoint: {path}", file=sys.stderr)
            return 2
        exec_project_script(script, args[1:])
    if args and args[0] == "chat":
        args = args[1:]
    elif args and not args[0].startswith("-"):
        print(f"Subcomando desconocido: {args[0]}; usa origen --help", file=sys.stderr)
        return 2
    # El checkpoint masivo guarda los pesos históricos actualizados de v1.
    if CHECKPOINT_MASIVO.is_file():
        exec_project_script("chat.py", ["--checkpoint", str(CHECKPOINT_MASIVO), *args])
    script = chosen_chat()
    if script is None:
        print("No existe un checkpoint entrenado propio; ejecuta origen doctor", file=sys.stderr)
        return 2
    if script == "chat.py":
        print("OrigenLM: no hay checkpoint v2; abriendo la versión v1.", flush=True)
    exec_project_script(script, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
