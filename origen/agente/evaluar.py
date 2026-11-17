"""Prueba rápida de un modelo: ¿conversa? ¿sabe usar herramientas? Genera un resumen en Markdown.

Uso: python -m origen.agente.evaluar --modelo qwen2.5:3b [--salida resumen.md]
Código de salida: 0 = se pudo evaluar (aunque algún caso falle), 2 = el servidor de modelos no responde.
"""
from __future__ import annotations

import argparse
import re
import time

from .agent import Agente
from .backends import BackendError, crear_backend
from .config import cargar_config
from .tools import RegistroHerramientas, herramientas_integradas


def _caso(nombre: str, pregunta: str, herramienta: str | None, esperado: str | None):
    return {"nombre": nombre, "pregunta": pregunta, "herramienta": herramienta, "esperado": esperado}


CASOS = [
    _caso("Conversación", "Saluda en una sola frase en español.", None, None),
    _caso("Cálculo con herramienta", "Usa la herramienta calcular para obtener 1234*5678 y dime solo el resultado.",
          "calcular", "7006652"),
    _caso("Hora con herramienta", "Usa la herramienta hora_actual y dime qué hora es.", "hora_actual", None),
]


def evaluar(cfg) -> list[dict]:
    resultados = []
    for caso in CASOS:
        usadas: list[str] = []
        registro = RegistroHerramientas(herramientas_integradas(cfg))
        agente = Agente(crear_backend(cfg), registro, max_pasos=4,
                        on_evento=lambda tipo, nombre, _d, usadas=usadas: usadas.append(nombre) if tipo == "herramienta" else None)
        inicio = time.monotonic()
        try:
            respuesta = agente.responder(caso["pregunta"])
            error = ""
        except BackendError as exc:
            respuesta, error = "", str(exc)
        segundos = time.monotonic() - inicio
        digitos = re.sub(r"[^0-9]", "", respuesta)
        ok = bool(respuesta) and not error
        if caso["herramienta"]:
            ok = ok and caso["herramienta"] in usadas
        if caso["esperado"]:
            ok = ok and caso["esperado"] in digitos
        resultados.append({"nombre": caso["nombre"], "ok": ok, "segundos": segundos, "usadas": usadas,
                           "respuesta": respuesta, "error": error, "modo_texto": agente.modo_texto})
    return resultados


def a_markdown(modelo: str, resultados: list[dict]) -> str:
    filas = ["| Caso | Resultado | Tiempo | Herramientas | Respuesta |", "|---|---|---|---|---|"]
    for r in resultados:
        texto = (r["error"] or r["respuesta"]).replace("\n", " ").replace("|", "\\|")[:120]
        modo = " (modo texto)" if r["modo_texto"] else ""
        filas.append(f"| {r['nombre']} | {'✅' if r['ok'] else '❌'} | {r['segundos']:.1f} s | "
                     f"{', '.join(r['usadas']) or '—'}{modo} | {texto} |")
    aprobados = sum(r["ok"] for r in resultados)
    return f"## Evaluación de `{modelo}`\n\n{aprobados}/{len(resultados)} casos superados\n\n" + "\n".join(filas) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prueba rápida de un modelo con el agente OrigenLM")
    parser.add_argument("--modelo")
    parser.add_argument("--backend", choices=["ollama", "openai"])
    parser.add_argument("--host")
    parser.add_argument("--salida", help="Archivo donde añadir el resumen Markdown")
    args = parser.parse_args(argv)
    cfg = cargar_config(modelo=args.modelo, backend=args.backend, host=args.host)
    try:
        crear_backend(cfg).listar_modelos()
    except BackendError as exc:
        print(f"El servidor de modelos no responde: {exc}")
        return 2
    resultados = evaluar(cfg)
    resumen = a_markdown(cfg.modelo, resultados)
    print(resumen)
    if args.salida:
        with open(args.salida, "a", encoding="utf-8") as destino:
            destino.write(resumen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
