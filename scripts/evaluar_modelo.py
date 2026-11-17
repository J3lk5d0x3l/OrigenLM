"""Evaluación reproducible de OrigenLM V2: NLL, perplejidad y pruebas puntuales.

Mide tres cosas separadas, para no confundir "bajó la pérdida" con "mejoró":
1. NLL y perplejidad en el split de validación reservado (no visto).
2. Respuestas a un conjunto de indicaciones escritas a mano (no salen del
   corpus) con comprobaciones declaradas y deterministas (greedy).
3. El mismo informe en JSON, para comparar antes/después sin depender de la
   memoria de nadie.

No usa pesos externos: solo el checkpoint y el tokenizador indicados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from origen.resources import configure_cpu_threads
from origen.tokenizer_v2 import ASSISTANT, BOS, END, EOS, SYSTEM, USER, load_tokenizer
from origen.v2 import OrigenV2, OrigenV2Config

SISTEMA = "Eres un asistente claro y útil. Respondes en español."
NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?")


def huella_sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as flujo:
        for bloque in iter(lambda: flujo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_tareas(ruta: Path) -> list[dict]:
    tareas = []
    with Path(ruta).open(encoding="utf-8") as flujo:
        for numero, linea in enumerate(flujo, 1):
            if not linea.strip():
                continue
            tarea = json.loads(linea)
            if not {"id", "categoria", "prompt", "comprobar"} <= tarea.keys():
                raise ValueError(f"Línea {numero} de {ruta} incompleta")
            tareas.append(tarea)
    if not tareas:
        raise ValueError(f"{ruta} no contiene indicaciones")
    return tareas


def numero_final(texto: str) -> float | None:
    encontrados = NUMERO.findall(texto.replace("−", "-"))
    if not encontrados:
        return None
    return float(encontrados[-1].replace(",", "."))


def comprobar(texto: str, regla: dict) -> tuple[bool, str]:
    """Comprobación explícita y determinista de una respuesta."""
    tipo = regla.get("tipo")
    minusculas = texto.casefold()
    if tipo == "no_vacio":
        ok = bool(texto.strip())
        return ok, "responde algo" if ok else "respuesta vacía"
    if tipo == "min_palabras":
        palabras = len(texto.split())
        minimo = int(regla.get("valor", 5))
        return palabras >= minimo, f"{palabras} palabras (mínimo {minimo})"
    if tipo == "contiene":
        valores = [v.casefold() for v in regla.get("valores", [])]
        presentes = [v for v in valores if v in minusculas]
        if regla.get("todos", True):
            ok = len(presentes) == len(valores)
        else:
            ok = bool(presentes)
        return ok, f"presentes {presentes} de {valores}"
    if tipo == "numero":
        obtenido = numero_final(texto)
        esperado = float(regla["valor"])
        ok = obtenido is not None and abs(obtenido - esperado) < 1e-6
        return ok, f"número {obtenido} (esperado {esperado:g})"
    raise ValueError(f"Regla desconocida: {tipo}")


def prefijo(tok, pregunta: str) -> list[int]:
    return [BOS, SYSTEM, *tok.encode(SISTEMA, add_special_tokens=False).ids, END,
            USER, *tok.encode(pregunta, add_special_tokens=False).ids, END, ASSISTANT]


def evaluar_validacion(modelo, datos: Path, ctx: int, batch: int,
                       dispositivo: torch.device, max_tokens: int):
    from train_v2_bpe import cargar_bins, evaluar

    meta, bins = cargar_bins(datos)
    nll, vistos = evaluar(modelo, bins["val"], ctx, batch, dispositivo, max_tokens)
    return nll, vistos, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluar OrigenLM V2 (NLL + pruebas)")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_v2_es.pt"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("dataset/preparado_v2_es/tokenizer.json"))
    p.add_argument("--datos", type=Path, default=Path("dataset/preparado_v2_es"))
    p.add_argument("--prompts", type=Path, default=Path("evals/prompts_es.jsonl"))
    p.add_argument("--salida", type=Path, default=Path("evals/informe_actual.json"))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limite-val-tokens", type=int, default=200_000)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--sin-validacion", action="store_true")
    args = p.parse_args(argv)
    if not args.checkpoint.is_file():
        p.error(f"No existe el checkpoint: {args.checkpoint}")
    if not args.tokenizer.is_file():
        p.error(f"Falta el tokenizador: {args.tokenizer}")

    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("Pediste --device cuda pero este torch no ve GPU")
    dispositivo = torch.device(
        "cuda" if (args.device == "cuda" or
                   (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if dispositivo.type == "cpu":
        configure_cpu_threads("chat")

    tok = load_tokenizer(args.tokenizer)
    guardado = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if guardado.get("architecture") != "origen-v2-gqa-v1":
        p.error("El checkpoint no es de OrigenLM v2")
    if guardado.get("tokenizer_sha256") not in (None, huella_sha256(args.tokenizer)):
        p.error("El tokenizador no coincide con el del entrenamiento")
    modelo = OrigenV2(OrigenV2Config(**guardado["config"]))
    modelo.load_state_dict(guardado["model"])
    modelo.to(dispositivo)
    modelo.eval()

    informe = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": huella_sha256(args.checkpoint),
        "paso": guardado.get("step"),
        "tokens_vistos": guardado.get("tokens_vistos"),
        "val_loss_guardado": guardado.get("val_loss"),
        "dispositivo": dispositivo.type,
        "tokenizer_sha256": guardado.get("tokenizer_sha256"),
    }
    print(f"OrigenLM V2 | paso {informe['paso']} | dispositivo {dispositivo.type}")

    if not args.sin_validacion and (args.datos / "meta.json").is_file():
        nll, vistos, meta = evaluar_validacion(
            modelo, args.datos, min(256, modelo.config.max_seq_len), 4,
            dispositivo, args.limite_val_tokens)
        informe.update(nll_validacion=nll, perplejidad=math.exp(min(20, nll)),
                       tokens_validacion=vistos,
                       val_tokenizer_sha256=meta.get("tokenizer_sha256"))
        print(f"validación: NLL={nll:.4f} | perplejidad={math.exp(min(20, nll)):.2f} | "
              f"{vistos:,} tokens")

    resultados = []
    for tarea in cargar_tareas(args.prompts):
        ids = prefijo(tok, tarea["prompt"])
        with torch.no_grad():
            nuevos = modelo.generate(ids, max_new_tokens=args.max_new_tokens,
                                     temperature=0.0, top_k=0, stop_ids=(EOS, END))
        texto = tok.decode(nuevos, skip_special_tokens=True).strip()
        ok, detalle = comprobar(texto, tarea["comprobar"])
        resultados.append({"id": tarea["id"], "categoria": tarea["categoria"],
                           "prompt": tarea["prompt"], "respuesta": texto,
                           "cumple": ok, "detalle": detalle})
        print(f"[{'OK ' if ok else 'NO '}] {tarea['id']}: {detalle}")

    categorias: dict[str, dict] = {}
    for fila in resultados:
        datos = categorias.setdefault(fila["categoria"], {"n": 0, "aciertos": 0})
        datos["n"] += 1
        datos["aciertos"] += int(fila["cumple"])
    for datos in categorias.values():
        datos["tasa"] = datos["aciertos"] / datos["n"]

    informe["categorias"] = categorias
    informe["resultados"] = resultados
    informe["aciertos"] = sum(r["cumple"] for r in resultados)
    informe["total"] = len(resultados)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    print("\n| categoría | aciertos | tasa |")
    print("|---|---|---|")
    for nombre, datos in sorted(categorias.items()):
        print(f"| {nombre} | {datos['aciertos']}/{datos['n']} | {datos['tasa']:.0%} |")
    print(f"\nInforme: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
