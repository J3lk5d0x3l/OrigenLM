"""Chat real para OrigenLM usando pesos abiertos servidos por llama.cpp.

El modelo propio de `checkpoints/` sigue siendo el experimento del repo; esto
es la vía práctica: un modelo abierto (por defecto Qwen2.5-Coder-1.5B-Instruct,
licencia Apache 2.0) corre en este codespace y responde de verdad.

Requiere `llama-server` escuchando en http://127.0.0.1:8080. Arrancalo con:
    bash scripts/chat_local.sh

Sin dependencias externas: solo la biblioteca estándar.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_SYSTEM = "Eres Origen, un asistente claro y útil. Respondés en español."


def stream_chat(url: str, messages: list[dict], temperature: float,
                max_tokens: int) -> str:
    """Envía la conversación y va escribiendo la respuesta en streaming."""
    cuerpo = json.dumps({
        "model": "local",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode("utf-8")
    peticion = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=cuerpo,
        headers={"Content-Type": "application/json"},
    )
    partes: list[str] = []
    with urllib.request.urlopen(peticion, timeout=600) as respuesta:
        for linea in respuesta:
            texto = linea.decode("utf-8", errors="replace").strip()
            if not texto.startswith("data:"):
                continue
            carga = texto[5:].strip()
            if carga == "[DONE]":
                break
            try:
                evento = json.loads(carga)
            except json.JSONDecodeError:
                continue
            trozo = evento.get("choices", [{}])[0].get("delta", {}).get("content")
            if trozo:
                partes.append(trozo)
                print(trozo, end="", flush=True)
    print()
    return "".join(partes)


def servidor_vivo(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as respuesta:
            return respuesta.status == 200
    except (urllib.error.URLError, OSError):
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=400)
    args = p.parse_args()

    if not servidor_vivo(args.url):
        print(f"No hay servidor en {args.url}.\n"
              "Arrancalo con: bash scripts/chat_local.sh\n"
              "(o: tools/llama/llama-server -m models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf "
              "-c 4096 -t 4 --port 8080 --jinja)")
        return 1

    print("OrigenLM | modelo abierto local (Qwen2.5-Coder) | escribe /salir para terminar")
    historia: list[dict] = [{"role": "system", "content": args.system}]
    while True:
        try:
            usuario = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break
        if usuario.lower() in {"/salir", "/exit", "/quit"}:
            break
        if usuario.lower() == "/limpiar":
            historia = [{"role": "system", "content": args.system}]
            print("Historial borrado.")
            continue
        if not usuario:
            continue
        historia.append({"role": "user", "content": usuario})
        print("Origen > ", end="", flush=True)
        try:
            respuesta = stream_chat(args.url, historia, args.temperature,
                                    args.max_new_tokens)
        except urllib.error.URLError as exc:
            print(f"\n[error] {exc}")
            historia.pop()
            continue
        historia.append({"role": "assistant", "content": respuesta})
        # Se conservan los últimos turnos para no crecer sin límite.
        historia = [historia[0], *historia[-8:]]
    return 0


if __name__ == "__main__":
    sys.exit(main())
