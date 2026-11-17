"""Herramientas integradas del agente (calcular, hora, web, archivos) y registro de herramientas."""
from __future__ import annotations

import ast
import html
import ipaddress
import json
import math
import operator
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Config

MAX_SALIDA = 6000
UA = "Mozilla/5.0 (compatible; OrigenLM/0.1)"


def recortar(texto: str, limite: int = MAX_SALIDA) -> str:
    if len(texto) <= limite:
        return texto
    return texto[:limite] + f"\n[... recortado: {len(texto) - limite} caracteres más]"


@dataclass
class Herramienta:
    nombre: str
    descripcion: str
    parametros: dict
    funcion: Callable[..., str]
    confirmar: bool = False  # pide permiso al usuario antes de ejecutarse
    origen: str = "integrada"

    def esquema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.nombre, "description": self.descripcion, "parameters": self.parametros}}


class RegistroHerramientas:
    def __init__(self, herramientas=()):
        self._h: dict[str, Herramienta] = {}
        for h in herramientas:
            self.agregar(h)

    def agregar(self, herramienta: Herramienta) -> None:
        self._h[herramienta.nombre] = herramienta

    def obtener(self, nombre: str) -> Herramienta | None:
        return self._h.get(nombre)

    def todas(self) -> list[Herramienta]:
        return list(self._h.values())

    def esquemas(self) -> list[dict]:
        return [h.esquema() for h in self._h.values()]

    def ejecutar(self, nombre: str, argumentos) -> str:
        herramienta = self._h.get(nombre)
        if herramienta is None:
            return f"Error: la herramienta '{nombre}' no existe."
        if not isinstance(argumentos, dict):
            argumentos = {}
        try:
            salida = herramienta.funcion(**argumentos)
        except TypeError as exc:
            return f"Error: argumentos no válidos para {nombre}: {exc}"
        except Exception as exc:  # una herramienta nunca debe tumbar el agente
            return f"Error en {nombre}: {exc}"
        return recortar(str(salida))


# --- calcular -------------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
          "log10": math.log10, "exp": math.exp, "abs": abs, "round": round, "min": min, "max": max,
          "floor": math.floor, "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e}


def _evaluar(nodo):
    if isinstance(nodo, ast.Expression):
        return _evaluar(nodo.body)
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
        return nodo.value
    if isinstance(nodo, ast.Name) and nodo.id in _CONST:
        return _CONST[nodo.id]
    if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, (ast.UAdd, ast.USub)):
        valor = _evaluar(nodo.operand)
        return valor if isinstance(nodo.op, ast.UAdd) else -valor
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPS:
        a, b = _evaluar(nodo.left), _evaluar(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("Exponente demasiado grande")
        return _OPS[type(nodo.op)](a, b)
    if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) and nodo.func.id in _FUNCS and not nodo.keywords:
        return _FUNCS[nodo.func.id](*[_evaluar(a) for a in nodo.args])
    raise ValueError("Expresión no permitida")


def calcular(expresion: str) -> str:
    expresion = str(expresion).strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expresion or len(expresion) > 200:
        raise ValueError("Expresión vacía o demasiado larga")
    return str(_evaluar(ast.parse(expresion, mode="eval")))


def hora_actual(zona: str = "") -> str:
    ahora = datetime.now(timezone.utc)
    if zona:
        try:
            from zoneinfo import ZoneInfo
            ahora = ahora.astimezone(ZoneInfo(zona))
        except Exception:
            return f"Zona horaria desconocida: {zona}"
    else:
        ahora = ahora.astimezone()
    return ahora.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


# --- web ------------------------------------------------------------------------------------
def _es_privada(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"No se pudo resolver {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def validar_url(url: str) -> str:
    partes = urllib.parse.urlparse(url)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ValueError("Solo se admiten URL http(s) completas")
    if _es_privada(partes.hostname):
        raise ValueError("Direcciones privadas o locales bloqueadas por seguridad")
    return url


class _RedireccionSegura(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validar_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def html_a_texto(contenido: str) -> str:
    contenido = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", contenido)
    contenido = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", contenido)
    contenido = html.unescape(re.sub(r"(?s)<[^>]+>", " ", contenido))
    lineas = (re.sub(r"[ \t\r\f\v]+", " ", linea).strip() for linea in contenido.split("\n"))
    return "\n".join(linea for linea in lineas if linea)


def leer_web(url: str, max_caracteres: int = 4000) -> str:
    validar_url(url)
    opener = urllib.request.build_opener(_RedireccionSegura)
    peticion = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "es,en;q=0.8"})
    with opener.open(peticion, timeout=20) as resp:
        tipo = resp.headers.get_content_type()
        datos = resp.read(1_000_000)
        codificacion = resp.headers.get_content_charset() or "utf-8"
    texto = datos.decode(codificacion, "replace")
    if tipo in ("text/html", "application/xhtml+xml"):
        texto = html_a_texto(texto)
    elif not tipo.startswith("text/") and tipo not in ("application/json", "application/xml"):
        return f"Contenido no textual ({tipo}); no se puede leer."
    return recortar(texto, max(500, min(int(max_caracteres), MAX_SALIDA)))


def parsear_ddg(contenido: str, maximo: int) -> list[dict]:
    enlaces = re.findall(r'(?is)<a\b[^>]*class="[^"]*result__a[^"]*"[^>]*>.*?</a>', contenido)
    resumenes = re.findall(r'(?is)class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', contenido)
    resultados = []
    for i, etiqueta in enumerate(enlaces[:maximo]):
        href = re.search(r'href="([^"]+)"', etiqueta)
        if not href:
            continue
        url = html.unescape(href.group(1))
        destino = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("uddg")
        if destino:
            url = destino[0]
        elif url.startswith("//"):
            url = "https:" + url
        titulo = html_a_texto(re.sub(r"(?is)^<a\b[^>]*>|</a>$", "", etiqueta))
        resumen = html_a_texto(resumenes[i]) if i < len(resumenes) else ""
        resultados.append({"titulo": titulo, "url": url, "resumen": resumen})
    return resultados


def _buscar_ddg(consulta: str, maximo: int) -> list[dict]:
    datos = urllib.parse.urlencode({"q": consulta}).encode()
    peticion = urllib.request.Request("https://html.duckduckgo.com/html/", data=datos, headers={"User-Agent": UA})
    with urllib.request.urlopen(peticion, timeout=20) as resp:
        return parsear_ddg(resp.read(1_000_000).decode("utf-8", "replace"), maximo)


def _buscar_searx(base: str, consulta: str, maximo: int) -> list[dict]:
    url = f"{base}/search?" + urllib.parse.urlencode({"q": consulta, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as resp:
        datos = json.load(resp)
    return [{"titulo": r.get("title", ""), "url": r.get("url", ""), "resumen": r.get("content", "")}
            for r in datos.get("results", [])[:maximo]]


def buscar_web(consulta: str, max_resultados: int = 5) -> str:
    consulta = str(consulta).strip()
    if not consulta:
        raise ValueError("La consulta está vacía")
    maximo = max(1, min(int(max_resultados), 10))
    searx = os.environ.get("ORIGEN_SEARX_URL", "").rstrip("/")
    resultados = _buscar_searx(searx, consulta, maximo) if searx else _buscar_ddg(consulta, maximo)
    if not resultados:
        return "Sin resultados (el buscador puede haber bloqueado la consulta; prueba con un servidor MCP de búsqueda)."
    return "\n\n".join(f"{i}. {r['titulo']}\n   {r['url']}\n   {r['resumen']}" for i, r in enumerate(resultados, 1))


# --- archivos (limitados al espacio de trabajo) ---------------------------------------------
def ruta_segura(workspace: Path, ruta: str) -> Path:
    base = workspace.resolve()
    destino = (base / str(ruta)).resolve()
    try:
        relativa = destino.relative_to(base)
    except ValueError:
        raise ValueError("Ruta fuera del espacio de trabajo") from None
    if destino.name == "origen.mcp.json" or any(p == ".git" or p.startswith(".env") for p in relativa.parts):
        raise ValueError("Ruta protegida (secretos o configuración)")
    return destino


def herramientas_integradas(cfg: Config) -> list[Herramienta]:
    base = cfg.workspace

    def listar_archivos(ruta: str = ".") -> str:
        carpeta = ruta_segura(base, ruta)
        if not carpeta.is_dir():
            raise ValueError("No es una carpeta")
        nombres = sorted(p.name + ("/" if p.is_dir() else "") for p in carpeta.iterdir()
                         if p.name != ".git" and not p.name.startswith(".env"))
        return "\n".join(nombres[:200]) or "(vacía)"

    def leer_archivo(ruta: str, max_caracteres: int = 4000) -> str:
        archivo = ruta_segura(base, ruta)
        if not archivo.is_file():
            raise ValueError("El archivo no existe")
        limite = max(100, min(int(max_caracteres), MAX_SALIDA))
        with archivo.open("r", encoding="utf-8", errors="replace") as fuente:
            return fuente.read(limite + 1)[:limite]

    def escribir_archivo(ruta: str, contenido: str) -> str:
        archivo = ruta_segura(base, ruta)
        if len(contenido) > 200_000:
            raise ValueError("Contenido demasiado grande")
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_text(contenido, encoding="utf-8")
        return f"Escrito {archivo.relative_to(base.resolve())} ({len(contenido)} caracteres)"

    def obj(propiedades: dict, requeridos: list[str]) -> dict:
        return {"type": "object", "properties": propiedades, "required": requeridos}

    txt, num = {"type": "string"}, {"type": "integer"}
    return [
        Herramienta("calcular", "Evalúa una expresión matemática exacta (+ - * / ** sqrt sin cos log pi...).",
                    obj({"expresion": {**txt, "description": "Ej.: 1234*5678"}}, ["expresion"]), calcular),
        Herramienta("hora_actual", "Devuelve la fecha y hora actuales, opcionalmente de una zona horaria.",
                    obj({"zona": {**txt, "description": "Zona IANA, ej. America/Mexico_City"}}, []), hora_actual),
        Herramienta("buscar_web", "Busca en internet y devuelve títulos, enlaces y resúmenes.",
                    obj({"consulta": txt, "max_resultados": num}, ["consulta"]), buscar_web),
        Herramienta("leer_web", "Descarga una página web pública y devuelve su texto.",
                    obj({"url": txt, "max_caracteres": num}, ["url"]), leer_web),
        Herramienta("listar_archivos", "Lista archivos de una carpeta del proyecto.",
                    obj({"ruta": txt}, []), listar_archivos),
        Herramienta("leer_archivo", "Lee un archivo de texto del proyecto.",
                    obj({"ruta": txt, "max_caracteres": num}, ["ruta"]), leer_archivo),
        Herramienta("escribir_archivo", "Crea o reemplaza un archivo de texto del proyecto.",
                    obj({"ruta": txt, "contenido": txt}, ["ruta", "contenido"]), escribir_archivo, confirmar=True),
    ]
