"""Comandos del agente: agente, preguntar, modelos, herramientas, doctor."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime

from ..resources import effective_cpu_count
from .agent import Agente
from .backends import BackendError, crear_backend
from .config import ROOT, VERSION, cargar_config
from .mcp import GestorMcp
from .modelos import cabe, cargar_catalogo, ram_gib
from .tools import RegistroHerramientas, herramientas_integradas

COMANDOS = ("agente", "preguntar", "modelos", "herramientas", "doctor")
AYUDA_REPL = """Comandos del chat:
  /ayuda              Esta ayuda
  /salir              Terminar
  /limpiar            Olvidar la conversación
  /modelo [etiqueta]  Ver o cambiar el modelo
  /herramientas       Listar herramientas disponibles
  /si                 Activar/desactivar la aprobación automática de herramientas
  /guardar [nombre]   Guardar la conversación en conversaciones/"""


def _preparar(args, con_herramientas: bool = True):
    cfg = cargar_config(backend=args.backend, host=args.host, modelo=args.modelo,
                        aprobar_todo=True if getattr(args, "si", False) else None)
    registro, gestor = RegistroHerramientas(), None
    if con_herramientas and not getattr(args, "sin_herramientas", False):
        for herramienta in herramientas_integradas(cfg):
            registro.agregar(herramienta)
        if not getattr(args, "sin_mcp", False) and cfg.mcp_archivo.is_file():
            gestor = GestorMcp(cfg.mcp_archivo, cfg.workspace)
            for herramienta in gestor.iniciar():
                registro.agregar(herramienta)
            for nombre, error in gestor.errores.items():
                print(f"[MCP] {nombre}: {error}", file=sys.stderr)
    return cfg, crear_backend(cfg), registro, gestor


def _confirmador(cfg):
    def confirmar(nombre: str, argumentos: dict) -> bool:
        if cfg.aprobar_todo:
            return True
        if not sys.stdin.isatty():
            return False
        resumen = json.dumps(argumentos, ensure_ascii=False)[:300]
        respuesta = input(f"\n⚠ El modelo quiere ejecutar {nombre} {resumen}\n  ¿Permitir? [s/N/t=todas] ").strip().lower()
        if respuesta == "t":
            cfg.aprobar_todo = True
            return True
        return respuesta in ("s", "si", "sí", "y", "yes")
    return confirmar


def _eventos(tipo: str, nombre: str, datos) -> None:
    if tipo == "herramienta":
        print(f"\n  ⚙ {nombre} {json.dumps(datos, ensure_ascii=False)[:200]}", file=sys.stderr)
    elif tipo == "modo_texto":
        print("\n  (el modelo no admite herramientas nativas: usando modo de respaldo por texto)", file=sys.stderr)


def _escribir(texto: str) -> None:
    sys.stdout.write(texto)
    sys.stdout.flush()


def _construir_agente(cfg, backend, registro) -> Agente:
    return Agente(backend, registro, cfg.max_pasos, confirmar=_confirmador(cfg), on_token=_escribir,
                  on_evento=_eventos, limite_chars=cfg.contexto * 3)


def _listar_herramientas(registro: RegistroHerramientas) -> None:
    for h in registro.todas():
        marca = " (pide permiso)" if h.confirmar else ""
        print(f"  {h.nombre:<28} [{h.origen}]{marca}\n      {h.descripcion[:110]}")


def comando_preguntar(args) -> int:
    texto = " ".join(args.texto).strip() or ("" if sys.stdin.isatty() else sys.stdin.read().strip())
    if not texto:
        print("Escribe una pregunta: origen preguntar \"¿qué hora es?\"", file=sys.stderr)
        return 2
    cfg, backend, registro, gestor = _preparar(args)
    try:
        _construir_agente(cfg, backend, registro).responder(texto)
        print()
        return 0
    except BackendError as exc:
        print(f"\nError: {exc}\nRevisa tu entorno con: origen doctor", file=sys.stderr)
        return 1
    finally:
        if gestor:
            gestor.cerrar()


def _guardar(agente: Agente, nombre: str) -> str:
    carpeta = ROOT / "conversaciones"
    carpeta.mkdir(exist_ok=True)
    seguro = "".join(c for c in nombre if c.isalnum() or c in "-_") or datetime.now().strftime("%Y%m%d-%H%M%S")
    destino = carpeta / f"{seguro}.md"
    lineas = []
    for m in agente.mensajes[1:]:
        if m["role"] in ("user", "assistant") and m.get("content"):
            lineas.append(f"**{'Tú' if m['role'] == 'user' else 'Origen'}:** {m['content']}\n")
    destino.write_text("\n".join(lineas), encoding="utf-8")
    return str(destino.relative_to(ROOT))


def comando_agente(args) -> int:
    try:
        import readline  # noqa: F401  (historial con flechas)
    except ImportError:
        pass
    cfg, backend, registro, gestor = _preparar(args)
    agente = _construir_agente(cfg, backend, registro)
    print(f"OrigenLM {VERSION} | modelo {cfg.modelo} ({cfg.backend}) | {len(registro.todas())} herramientas")
    print("Escribe /ayuda para ver los comandos. Ctrl+C interrumpe una respuesta.")
    try:
        while True:
            try:
                entrada = input("\nTú> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not entrada:
                continue
            if entrada.startswith("/"):
                orden, _, resto = entrada.partition(" ")
                resto = resto.strip()
                if orden in ("/salir", "/exit", "/quit"):
                    break
                elif orden == "/ayuda":
                    print(AYUDA_REPL)
                elif orden == "/limpiar":
                    agente.reiniciar()
                    print("Conversación reiniciada.")
                elif orden == "/modelo":
                    if resto:
                        cfg.modelo = backend.modelo = resto
                        print(f"Modelo: {resto}")
                    else:
                        print(f"Modelo actual: {cfg.modelo}")
                elif orden == "/herramientas":
                    _listar_herramientas(registro)
                elif orden == "/si":
                    cfg.aprobar_todo = not cfg.aprobar_todo
                    print("Aprobación automática " + ("ACTIVADA (peligroso)" if cfg.aprobar_todo else "desactivada"))
                elif orden == "/guardar":
                    print("Guardado en", _guardar(agente, resto))
                else:
                    print("Comando desconocido. Usa /ayuda")
                continue
            _escribir("Origen> ")
            try:
                agente.responder(entrada)
                print()
            except BackendError as exc:
                print(f"\nError: {exc}\nRevisa tu entorno con: origen doctor")
            except KeyboardInterrupt:
                print("\n(interrumpido)")
    finally:
        if gestor:
            gestor.cerrar()
    return 0


def comando_modelos(args) -> int:
    if args.instalar:
        if backend_ollama_ausente():
            print("No se encontró 'ollama'. Instálalo con: bash scripts/instalar_ollama.sh", file=sys.stderr)
            return 1
        return subprocess.call(["ollama", "pull", args.instalar])
    cfg, backend, _, _ = _preparar(args, con_herramientas=False)
    instalados: set[str] = set()
    try:
        modelos = backend.listar_modelos()
        instalados = {m["nombre"] for m in modelos}
        print(f"Instalados en {cfg.host_efectivo()}:")
        for m in modelos:
            tam = f" ({m['tam'] / 1024 ** 3:.1f} GB)" if m.get("tam") else ""
            print(f"  {m['nombre']}{tam}")
        if not modelos:
            print("  (ninguno)")
    except BackendError as exc:
        print(f"No se pudo consultar el servidor de modelos: {exc}")
    ram = ram_gib()
    print(f"\nSugeridos (RAM utilizable: {f'{ram:.1f} GiB' if ram else 'desconocida'}):")
    for m in cargar_catalogo():
        ajuste = cabe(m["peso_gb"], ram)
        simbolo = "?" if ajuste is None else ("✓" if ajuste else "✗")
        estado = " [instalado]" if m["id"] in instalados else ""
        print(f"  {simbolo} {m['id']:<14} ~{m['peso_gb']:>4.1f} GB  {m['notas']}{estado}")
    print("\nInstalar uno: origen modelos --instalar qwen2.5:3b   |   Usarlo: origen agente -m qwen2.5:3b")
    return 0


def backend_ollama_ausente() -> bool:
    return shutil.which("ollama") is None


def comando_herramientas(args) -> int:
    _, _, registro, gestor = _preparar(args)
    try:
        _listar_herramientas(registro)
    finally:
        if gestor:
            gestor.cerrar()
    return 0


def comando_doctor(args) -> int:
    problemas = 0

    def linea(ok: bool | None, texto: str) -> None:
        print(f"  {'✓' if ok else ('·' if ok is None else '✗')} {texto}")

    cfg, backend, _, _ = _preparar(args, con_herramientas=False)
    ram = ram_gib()
    print(f"OrigenLM {VERSION} | diagnóstico")
    linea(True, f"Python {sys.version.split()[0]} | {effective_cpu_count()} CPU | RAM utilizable "
                f"{f'{ram:.1f} GiB' if ram else 'desconocida'}")
    uso = shutil.disk_usage(ROOT)
    linea(uso.free > 5 * 1024 ** 3, f"Disco libre {uso.free / 1024 ** 3:.1f} GiB")
    if cfg.backend == "ollama":
        linea(not backend_ollama_ausente(), "Comando 'ollama' instalado (bash scripts/instalar_ollama.sh)")
    try:
        nombres = {m["nombre"] for m in backend.listar_modelos()}
        linea(True, f"Servidor de modelos responde en {cfg.host_efectivo()} ({len(nombres)} modelos)")
        presente = cfg.modelo in nombres or f"{cfg.modelo}:latest" in nombres
        linea(presente, f"Modelo '{cfg.modelo}' disponible" + ("" if presente else f" → origen modelos --instalar {cfg.modelo}"))
        problemas += 0 if presente else 1
    except BackendError as exc:
        linea(False, f"Servidor de modelos no responde: {exc}")
        problemas += 1
    mcp = cfg.mcp_archivo
    if mcp.is_file():
        gestor = GestorMcp(mcp, cfg.workspace)
        n = len(gestor.definiciones())
        linea(not gestor.errores, f"{mcp.name}: {n} servidor(es) MCP definidos"
                                  + ("".join(f" | {k}: {v}" for k, v in gestor.errores.items())))
    else:
        linea(None, "Sin origen.mcp.json (opcional; copia origen.mcp.example.json)")
    linea(shutil.which("npx") is not None or None, "Node/npx (necesario para muchos servidores MCP)")
    print("\n" + ("Todo listo: escribe `origen agente`." if not problemas else "Hay problemas por resolver (✗)."))
    return 1 if problemas else 0


def main(argv: list[str] | None = None) -> int:
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("-m", "--modelo", help="Etiqueta del modelo (ej. qwen2.5:3b)")
    comun.add_argument("--backend", choices=["ollama", "openai"], help="Tipo de servidor de modelos")
    comun.add_argument("--host", help="URL del servidor de modelos")
    comun.add_argument("--si", action="store_true", help="Aprobar herramientas sensibles sin preguntar")
    comun.add_argument("--sin-herramientas", action="store_true", help="Solo conversar")
    comun.add_argument("--sin-mcp", action="store_true", help="No iniciar servidores MCP")
    parser = argparse.ArgumentParser(prog="origen", description="Agente OrigenLM")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("agente", parents=[comun], help="Chat interactivo con herramientas").set_defaults(func=comando_agente)
    p = sub.add_parser("preguntar", parents=[comun], help="Una pregunta y respuesta")
    p.add_argument("texto", nargs="*")
    p.set_defaults(func=comando_preguntar)
    p = sub.add_parser("modelos", parents=[comun], help="Ver modelos instalados y sugeridos")
    p.add_argument("--instalar", metavar="ETIQUETA", help="Descargar un modelo con ollama pull")
    p.set_defaults(func=comando_modelos)
    sub.add_parser("herramientas", parents=[comun], help="Listar herramientas (integradas y MCP)").set_defaults(func=comando_herramientas)
    sub.add_parser("doctor", parents=[comun], help="Diagnosticar el entorno").set_defaults(func=comando_doctor)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:  # configuración inválida
        print(f"Configuración inválida: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
