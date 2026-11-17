"""Cliente MCP mínimo (transporte stdio, JSON-RPC por líneas). Solo biblioteca estándar.

Formato de origen.mcp.json (compatible con el de otros clientes MCP):
  {"mcpServers": {"nombre": {"command": "npx", "args": ["-y", "paquete"], "env": {"TOKEN": "${TOKEN}"}}}}
Cada servidor admite además "confirmar": false (no pedir permiso) y "desactivado": true.
"""
from __future__ import annotations

import collections
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .tools import Herramienta

PROTOCOLO = "2024-11-05"


class McpError(RuntimeError):
    """Fallo al hablar con un servidor MCP."""


class ServidorMcp:
    def __init__(self, nombre: str, comando: str, args=(), env: dict | None = None,
                 cwd: Path | None = None, timeout: float = 60):
        self.nombre, self.comando, self.args = nombre, comando, [str(a) for a in args]
        self.env, self.cwd, self.timeout = dict(env or {}), cwd, timeout
        self.proc: subprocess.Popen | None = None
        self._cola: queue.Queue = queue.Queue()
        self._errores: collections.deque = collections.deque(maxlen=15)
        self._id = 0
        self._cerrojo = threading.Lock()

    def iniciar(self) -> None:
        ejecutable = shutil.which(self.comando)
        if ejecutable is None:
            raise McpError(f"No se encontró el comando '{self.comando}' (¿falta instalarlo?)")
        entorno = dict(os.environ)
        entorno.update({k: os.path.expandvars(str(v)) for k, v in self.env.items()})
        self.proc = subprocess.Popen([ejecutable, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, cwd=self.cwd, env=entorno, text=True,
                                     encoding="utf-8", bufsize=1)
        threading.Thread(target=self._leer_stdout, daemon=True).start()
        threading.Thread(target=self._leer_stderr, daemon=True).start()
        try:
            self._pedir("initialize", {"protocolVersion": PROTOCOLO, "capabilities": {},
                                       "clientInfo": {"name": "origenlm", "version": "0.1"}})
            self._enviar({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self.cerrar()
            raise

    def _leer_stdout(self) -> None:
        try:
            for linea in self.proc.stdout:
                try:
                    self._cola.put(json.loads(linea))
                except ValueError:
                    continue  # algunos servidores imprimen registros que no son JSON
        except (ValueError, OSError):
            pass  # el flujo se cerró al terminar
        self._cola.put(None)

    def _leer_stderr(self) -> None:
        try:
            for linea in self.proc.stderr:
                self._errores.append(linea.rstrip())
        except (ValueError, OSError):
            pass

    def _enviar(self, mensaje: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(mensaje) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise McpError(f"El servidor '{self.nombre}' no responde: {exc}") from exc

    def _pedir(self, metodo: str, params: dict | None = None, timeout: float | None = None):
        with self._cerrojo:
            self._id += 1
            identificador = self._id
            self._enviar({"jsonrpc": "2.0", "id": identificador, "method": metodo, "params": params or {}})
            limite = time.monotonic() + (timeout or self.timeout)
            while True:
                restante = limite - time.monotonic()
                if restante <= 0:
                    raise McpError(f"Tiempo de espera agotado en {metodo} ({self.nombre})")
                try:
                    mensaje = self._cola.get(timeout=restante)
                except queue.Empty:
                    raise McpError(f"Tiempo de espera agotado en {metodo} ({self.nombre})") from None
                if mensaje is None:
                    cola = " | ".join(self._errores)[-400:]
                    raise McpError(f"El servidor '{self.nombre}' terminó. {cola}".strip())
                if mensaje.get("id") == identificador and ("result" in mensaje or "error" in mensaje):
                    if "error" in mensaje:
                        raise McpError(str(mensaje["error"].get("message", mensaje["error"])))
                    return mensaje.get("result") or {}
                if "method" in mensaje and "id" in mensaje:  # petición del servidor: no soportada salvo ping
                    if mensaje["method"] == "ping":
                        self._enviar({"jsonrpc": "2.0", "id": mensaje["id"], "result": {}})
                    else:
                        self._enviar({"jsonrpc": "2.0", "id": mensaje["id"],
                                      "error": {"code": -32601, "message": "No soportado"}})

    def listar_herramientas(self) -> list[dict]:
        herramientas, cursor = [], None
        while True:
            resultado = self._pedir("tools/list", {"cursor": cursor} if cursor else {})
            herramientas += resultado.get("tools", [])
            cursor = resultado.get("nextCursor")
            if not cursor:
                return herramientas

    def llamar(self, nombre: str, argumentos: dict) -> str:
        resultado = self._pedir("tools/call", {"name": nombre, "arguments": argumentos}, timeout=self.timeout * 2)
        partes = []
        for parte in resultado.get("content", []):
            if parte.get("type") == "text":
                partes.append(parte.get("text", ""))
            else:
                partes.append(f"[contenido {parte.get('type', 'desconocido')} omitido]")
        texto = "\n".join(partes) or json.dumps(resultado.get("structuredContent", ""), ensure_ascii=False)
        return ("Error del servidor MCP: " + texto) if resultado.get("isError") else texto

    def cerrar(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for flujo in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                flujo.close()
            except (OSError, ValueError):
                pass


def _a_herramienta(servidor: ServidorMcp, definicion: dict, confirmar: bool) -> Herramienta:
    original = definicion["name"]
    nombre = re.sub(r"[^A-Za-z0-9_-]", "_", f"{servidor.nombre}__{original}")[:64]

    def funcion(**argumentos) -> str:
        return servidor.llamar(original, argumentos)

    return Herramienta(nombre, definicion.get("description") or original,
                       definicion.get("inputSchema") or {"type": "object", "properties": {}},
                       funcion, confirmar=confirmar, origen=f"mcp:{servidor.nombre}")


class GestorMcp:
    def __init__(self, archivo: Path, cwd: Path | None = None):
        self.archivo, self.cwd = Path(archivo), cwd
        self.servidores: dict[str, ServidorMcp] = {}
        self.errores: dict[str, str] = {}

    def definiciones(self) -> dict:
        try:
            datos = json.loads(self.archivo.read_text(encoding="utf-8"))
        except OSError:
            return {}
        except ValueError as exc:
            self.errores["(archivo)"] = f"{self.archivo.name} no es JSON válido: {exc}"
            return {}
        return datos.get("mcpServers") or datos.get("servidores") or {}

    def iniciar(self) -> list[Herramienta]:
        herramientas: list[Herramienta] = []
        for nombre, d in self.definiciones().items():
            if not isinstance(d, dict) or d.get("desactivado") or d.get("disabled"):
                continue
            servidor = None
            try:
                servidor = ServidorMcp(nombre, d["command"], d.get("args", []), d.get("env", {}), self.cwd,
                                       float(d.get("timeout", 60)))
                servidor.iniciar()
                definiciones = servidor.listar_herramientas()
            except (McpError, KeyError, OSError, TypeError, ValueError) as exc:
                self.errores[nombre] = f"falta {exc}" if isinstance(exc, KeyError) else str(exc)
                if servidor:
                    servidor.cerrar()
                continue
            self.servidores[nombre] = servidor
            herramientas += [_a_herramienta(servidor, t, bool(d.get("confirmar", True))) for t in definiciones]
        return herramientas

    def cerrar(self) -> None:
        for servidor in self.servidores.values():
            servidor.cerrar()
