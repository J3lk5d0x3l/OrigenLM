"""Herramientas de Internet y MCP junto a checkpoints PROPIOS de OrigenLM.

Los modelos locales todavía no saben llamar herramientas por sí mismos:
el usuario puede invocar /usar NOMBRE {JSON}, /buscar o /pagina.
"""
import argparse
import json
import shutil
import sys

from .backends import BackendError, crear_backend, checkpoint_predeterminado
from .config import ROOT, VERSION, cargar_config
from .mcp import GestorMcp
from .tools import RegistroHerramientas, herramientas_integradas
from ..resources import effective_cpu_count

COMANDOS = ("agente", "preguntar", "herramientas", "doctor")


def preparar(cfg, sin_mcp=False):
    registry = RegistroHerramientas(herramientas_integradas(cfg))
    gestor = None
    if not sin_mcp and cfg.mcp_archivo.is_file():
        gestor = GestorMcp(cfg.mcp_archivo, cfg.workspace)
        for tool in gestor.iniciar():
            registry.agregar(tool)
        for name, error in gestor.errores.items():
            print(f"[MCP] {name}: {error}", file=sys.stderr)
    return registry, gestor


def confirmar_y_ejecutar(registry, name, arguments, allow=False):
    tool = registry.obtener(name)
    if tool is None:
        return "Herramienta desconocida. Usa /herramientas."
    if tool.confirmar and not allow:
        if not sys.stdin.isatty():
            return "Necesita aprobación interactiva."
        print(f"Se solicita ejecutar {name} {json.dumps(arguments, ensure_ascii=False)[:300]}")
        if input("¿Autorizar? [s/N] ").strip().lower() not in ("s", "si", "sí"):
            return "Operación rechazada por el usuario."
    return registry.ejecutar(name, arguments)


def mostrar(registry):
    for tool in registry.todas():
        print(f"- {tool.nombre} ({tool.origen})" + (" [requiere aprobación]" if tool.confirmar else ""))


def herramienta_desde_linea(line, registry):
    if line.startswith("/buscar "):
        return confirmar_y_ejecutar(registry, "buscar_web", {"consulta": line[8:]})
    if line.startswith("/pagina "):
        return confirmar_y_ejecutar(registry, "leer_web", {"url": line[8:]})
    if not line.startswith("/usar "):
        return None
    parts = line[6:].strip().split(maxsplit=1)
    if not parts:
        return "Uso: /usar nombre {\"argumento\":\"valor\"}"
    try:
        arguments = json.loads(parts[1]) if len(parts) > 1 else {}
        if not isinstance(arguments, dict):
            raise ValueError("Los parámetros deben ser un objeto JSON")
    except ValueError as exc:
        return f"Argumentos JSON inválidos: {exc}"
    return confirmar_y_ejecutar(registry, parts[0], arguments)


def comando_agente(args):
    cfg = cargar_config(checkpoint=args.checkpoint, temperature=args.temperature,
                        max_tokens=args.max_tokens)
    backend = crear_backend(cfg)
    registry, gestor = preparar(cfg, sin_mcp=args.sin_mcp)
    print(f"OrigenLM PROPIO | {backend.path.name} | paso {backend.step} | CPU")
    print("Herramientas: /herramientas /buscar texto /pagina URL /usar NOMBRE {JSON}")
    print("Escribe /salir, /limpiar o /ayuda. Ningún modelo externo.")
    hist = []
    try:
        while True:
            try:
                question = input("\nTú > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question in ("/salir", "/exit"):
                break
            if question == "/ayuda":
                print("Comandos: /herramientas, /buscar TEXTO, /pagina URL, /usar NOMBRE {JSON}, /limpiar, /salir")
                continue
            if question == "/herramientas":
                mostrar(registry)
                continue
            if question == "/limpiar":
                hist.clear()
                print("Historial reiniciado.")
                continue
            if question.startswith(("/buscar ", "/pagina ", "/usar ")):
                print("Herramienta >", herramienta_desde_linea(question, registry))
                continue
            try:
                messages = [{"role": "user" if idx % 2 == 0 else "assistant", "content": value}
                            for idx, value in enumerate(hist)]
                messages.append({"role": "user", "content": question})
                reply = backend.chat(messages)["content"]
                print("Origen >", reply)
                hist.extend([question, reply])
                hist = hist[-8:]
            except BackendError as exc:
                print(f"[OrigenLM] {exc}")
    finally:
        if gestor:
            gestor.cerrar()
    return 0


def comando_herramientas(args):
    cfg = cargar_config()
    registry, gestor = preparar(cfg, sin_mcp=args.sin_mcp)
    try:
        mostrar(registry)
    finally:
        if gestor:
            gestor.cerrar()
    return 0


def comando_doctor(args):
    checkpoint = checkpoint_predeterminado()
    print(f"OrigenLM local v{VERSION} | CPU efectivos: {effective_cpu_count()}")
    print(f"Checkpoint propio: {checkpoint if checkpoint else 'no disponible'}")
    disk = shutil.disk_usage(ROOT)
    print(f"Disco libre: {disk.free / 1024 ** 3:.1f} GiB")
    cfg = cargar_config()
    print(f"MCP: {cfg.mcp_archivo if cfg.mcp_archivo.is_file() else 'sin servidores registrados'}")
    print("Ollama y modelos externos: no utilizados")
    return 0 if checkpoint else 1


def comando_preguntar(args):
    cfg = cargar_config(checkpoint=args.checkpoint,
                        temperature=args.temperature, max_tokens=args.max_tokens)
    backend = crear_backend(cfg)
    question = " ".join(args.texto).strip()
    if not question:
        print("Uso: origen preguntar '¿Qué es Python?'", file=sys.stderr)
        return 2
    print(backend.chat([{"role": "user", "content": question}])["content"])
    return 0


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", help="Pesos propios .pt, nunca modelo externo")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--max-tokens", type=int, default=128)
    common.add_argument("--sin-mcp", action="store_true")
    parser = argparse.ArgumentParser(prog="origen")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name in COMANDOS:
        cmd = sub.add_parser(name, parents=[common])
        if name == "preguntar":
            cmd.add_argument("texto", nargs="*")
    args = parser.parse_args(argv)
    try:
        return {"agente": comando_agente, "preguntar": comando_preguntar,
                "herramientas": comando_herramientas, "doctor": comando_doctor}[args.subcommand](args)
    except (BackendError, ValueError) as exc:
        print(f"[OrigenLM] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
