"""Conexión con modelos locales (Ollama) o servidores compatibles con OpenAI. Solo biblioteca estándar.

Formato interno de mensajes (los backends lo convierten al formato de cada API):
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from .config import Config


class BackendError(RuntimeError):
    """No se pudo hablar con el modelo."""


class HerramientasNoSoportadas(BackendError):
    """El modelo elegido no admite llamadas a herramientas nativas."""


def _abrir(url: str, payload: dict | None = None, api_key: str = "", timeout: int = 600):
    cabeceras = {"Content-Type": "application/json"}
    if api_key:
        cabeceras["Authorization"] = f"Bearer {api_key}"
    datos = None if payload is None else json.dumps(payload).encode("utf-8")
    peticion = urllib.request.Request(url, data=datos, headers=cabeceras)
    try:
        return urllib.request.urlopen(peticion, timeout=timeout)
    except urllib.error.HTTPError as exc:
        cuerpo = exc.read().decode("utf-8", "replace")[:500]
        if exc.code == 400 and "tool" in cuerpo.lower():
            raise HerramientasNoSoportadas(cuerpo) from exc
        raise BackendError(f"HTTP {exc.code} en {url}: {cuerpo}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise BackendError(f"No se pudo conectar con {url}: {exc}") from exc


def _argumentos(valor) -> dict:
    if isinstance(valor, dict):
        return valor
    if isinstance(valor, str):
        try:
            datos = json.loads(valor or "{}")
        except ValueError:
            return {}
        return datos if isinstance(datos, dict) else {}
    return {}


def _a_ollama(mensajes: list[dict]) -> list[dict]:
    salida = []
    for m in mensajes:
        if m["role"] == "assistant" and m.get("tool_calls"):
            llamadas = [{"function": {"name": t["name"], "arguments": t["arguments"]}} for t in m["tool_calls"]]
            salida.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": llamadas})
        elif m["role"] == "tool":
            salida.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
        else:
            salida.append({"role": m["role"], "content": m["content"]})
    return salida


def _a_openai(mensajes: list[dict]) -> list[dict]:
    salida = []
    for m in mensajes:
        if m["role"] == "assistant" and m.get("tool_calls"):
            llamadas = [{"id": t["id"], "type": "function",
                         "function": {"name": t["name"], "arguments": json.dumps(t["arguments"], ensure_ascii=False)}}
                        for t in m["tool_calls"]]
            salida.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": llamadas})
        elif m["role"] == "tool":
            salida.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            salida.append({"role": m["role"], "content": m["content"]})
    return salida


class OllamaBackend:
    nombre = "ollama"

    def __init__(self, cfg: Config):
        self.base = cfg.host_efectivo()
        self.modelo = cfg.modelo
        self.contexto = cfg.contexto
        self.timeout = cfg.timeout

    def listar_modelos(self) -> list[dict]:
        with _abrir(self.base + "/api/tags", timeout=15) as resp:
            datos = json.load(resp)
        return [{"nombre": m.get("name", ""), "tam": m.get("size")} for m in datos.get("models", [])]

    def chat(self, mensajes: list[dict], herramientas: list[dict] | None = None,
             on_token: Callable[[str], None] | None = None) -> dict:
        payload = {"model": self.modelo, "messages": _a_ollama(mensajes), "stream": True,
                   "options": {"num_ctx": self.contexto}}
        if herramientas:
            payload["tools"] = herramientas
        contenido: list[str] = []
        llamadas: list[dict] = []
        with _abrir(self.base + "/api/chat", payload, timeout=self.timeout) as resp:
            for linea in resp:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    trozo = json.loads(linea)
                except ValueError:
                    continue
                if trozo.get("error"):
                    texto = str(trozo["error"])
                    raise (HerramientasNoSoportadas if "tool" in texto.lower() else BackendError)(texto)
                msg = trozo.get("message") or {}
                if msg.get("content"):
                    contenido.append(msg["content"])
                    if on_token:
                        on_token(msg["content"])
                for tc in msg.get("tool_calls") or []:
                    f = tc.get("function", {})
                    llamadas.append({"id": f"call_{len(llamadas)}", "name": f.get("name", ""),
                                     "arguments": _argumentos(f.get("arguments"))})
        return {"content": "".join(contenido), "tool_calls": llamadas}


class OpenAIBackend:
    """Cualquier servidor con /chat/completions: llama.cpp, vLLM, LM Studio, etc."""
    nombre = "openai"

    def __init__(self, cfg: Config):
        self.base = cfg.host_efectivo()  # debe incluir /v1 si el servidor lo necesita
        self.modelo = cfg.modelo
        self.api_key = cfg.api_key
        self.timeout = cfg.timeout

    def listar_modelos(self) -> list[dict]:
        with _abrir(self.base + "/models", api_key=self.api_key, timeout=15) as resp:
            datos = json.load(resp)
        return [{"nombre": m.get("id", ""), "tam": None} for m in datos.get("data", [])]

    def chat(self, mensajes: list[dict], herramientas: list[dict] | None = None,
             on_token: Callable[[str], None] | None = None) -> dict:
        payload = {"model": self.modelo, "messages": _a_openai(mensajes)}
        if herramientas:
            payload["tools"] = herramientas
        with _abrir(self.base + "/chat/completions", payload, self.api_key, self.timeout) as resp:
            datos = json.load(resp)
        try:
            msg = datos["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"Respuesta inesperada del servidor: {str(datos)[:300]}") from exc
        contenido = msg.get("content") or ""
        llamadas = []
        for tc in msg.get("tool_calls") or []:
            f = tc.get("function", {})
            llamadas.append({"id": tc.get("id") or f"call_{len(llamadas)}", "name": f.get("name", ""),
                             "arguments": _argumentos(f.get("arguments"))})
        if on_token and contenido:
            on_token(contenido)
        return {"content": contenido, "tool_calls": llamadas}


def crear_backend(cfg: Config):
    return OllamaBackend(cfg) if cfg.backend == "ollama" else OpenAIBackend(cfg)
