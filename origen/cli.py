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
# Comandos del agente (modelos abiertos + herramientas + MCP); no necesitan PyTorch.
COMANDOS_AGENTE = ("agente", "preguntar", "modelos", "herramientas", "doctor")


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
    for name, path in (("v2", CHECKPOINT_V2), ("v1", CHECKPOINT_V1)):
        print(f"Checkpoint {name}: {path.relative_to(ROOT)}" if path.is_file()
              else f"Checkpoint {name}: todavía no existe")
    data = ROOT / "dataset/chat_v2.jsonl"
    if data.is_file():
        with data.open(encoding="utf-8") as source:
            count = sum(bool(row.strip()) for row in source)
        print(f"Datos conversacionales: {count} ejemplos (demostración)")
    print("Agente con modelos abiertos: origen doctor  |  origen agente")
    print("Iniciar conversación con checkpoint propio: origen")
    print("Entrenamiento explícito: origen entrenar --steps 1000")
    return 0


def exec_project_script(script: str, arguments: list[str]) -> None:
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, str(ROOT / script), *arguments])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help", "ayuda", "help"):
        print("""Comando OrigenLM
Agente (modelos abiertos + herramientas + internet + MCP):
  origen doctor              Diagnosticar el entorno (Ollama, modelo, MCP)
  origen modelos             Ver modelos instalados y sugeridos (--instalar ETIQUETA)
  origen agente [-m MODELO]  Chat interactivo con herramientas (/ayuda dentro del chat)
  origen preguntar "texto"   Una pregunta y respuesta
  origen herramientas        Listar herramientas integradas y de servidores MCP

Modelo propio entrenado desde cero:
  origen                     Abrir chat con mejor checkpoint disponible (v2 o v1)
  origen estado              Inspeccionar recursos del Codespace y checkpoints
  origen v1                  Abrir chat del modelo clásico
  origen v2                  Abrir chat del modelo nuevo
  origen entrenar [opciones] Entrenar v2 explícitamente (NO se ejecuta al abrir chat)
  origen pruebas             Ejecutar los tests
  origen --seed 42           Pasar opciones directamente al chat
  origen --help              Mostrar ayuda
""")
        return 0
    if args and args[0] in COMANDOS_AGENTE:
        from .agente.cli import main as agente_main
        return agente_main(args)
    if args and args[0] == "estado":
        return show_status()
    if args and args[0] == "pruebas":
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT)
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
    script = chosen_chat()
    if script is None:
        print("No existe un checkpoint entrenado propio. Para conversar con un modelo abierto usa: "
              "origen doctor  y  origen agente", file=sys.stderr)
        return 2
    if script == "chat.py":
        print("OrigenLM: no hay checkpoint v2; abriendo la versión v1.", flush=True)
    exec_project_script(script, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
