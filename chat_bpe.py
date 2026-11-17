"""Chat con el modelo V2 BPE propio (vocabulario español, pesos propios).

No usa modelos externos: solo el checkpoint indicado y su tokenizer.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from origen.resources import configure_cpu_threads
from origen.tokenizer_v2 import ASSISTANT, BOS, END, SYSTEM, USER, load_tokenizer
from origen.v2 import OrigenV2, OrigenV2Config

SISTEMA = "Eres Origen, un asistente claro y útil. Respondes en español."


def prefijo(tok, sistema: str, historial: list[tuple[str, str]], pregunta: str,
           limite: int) -> list[int]:
    def construir(turnos):
        ids = [BOS, SYSTEM, *tok.encode(sistema, add_special_tokens=False).ids, END]
        for pregunta_previa, respuesta in turnos:
            ids += [USER, *tok.encode(pregunta_previa, add_special_tokens=False).ids,
                    END, ASSISTANT,
                    *tok.encode(respuesta, add_special_tokens=False).ids, END]
        ids += [USER, *tok.encode(pregunta, add_special_tokens=False).ids, END, ASSISTANT]
        return ids

    seleccionados: list[tuple[str, str]] = []
    if len(construir(seleccionados)) > limite:
        raise ValueError("Tu mensaje no cabe en el contexto del modelo.")
    for turno in reversed(historial):
        candidato = construir([turno, *seleccionados])
        if len(candidato) > limite:
            break
        seleccionados.insert(0, turno)
    return construir(seleccionados)


def main() -> int:
    p = argparse.ArgumentParser(description="Chat con OrigenLM V2 (BPE)")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_v2_es.pt"))
    p.add_argument("--tokenizer", type=Path,
                   default=Path("dataset/preparado_v2_es/tokenizer.json"))
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f"No existe el checkpoint: {args.checkpoint}")
    if not args.tokenizer.is_file():
        p.error(f"Falta el tokenizador: {args.tokenizer}")

    configure_cpu_threads("chat")
    tok = load_tokenizer(args.tokenizer)
    guardado = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if guardado.get("architecture") != "origen-v2-gqa-v1":
        p.error("El checkpoint no es de OrigenLM v2")
    modelo = OrigenV2(OrigenV2Config(**guardado["config"]))
    modelo.load_state_dict(guardado["model"])
    modelo.eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    print(f"OrigenLM V2 ES | paso {guardado.get('step')} | "
          f"vocab {modelo.config.vocab_size} | ventana {modelo.config.max_seq_len}")
    print(f"Tokens vistos: {guardado.get('tokens_vistos', 0):,} | "
          f"NLL validación: {guardado.get('val_loss')}")
    print("Modelo pequeño entrenado desde cero: puede fallar en razonamiento.")
    print("Comandos: /limpiar, /salir")

    historial: list[tuple[str, str]] = []
    while True:
        try:
            pregunta = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break
        if pregunta.casefold() == "/salir":
            break
        if pregunta.casefold() == "/limpiar":
            historial.clear()
            print("Historial reiniciado.")
            continue
        if not pregunta:
            continue
        try:
            ids = prefijo(tok, SISTEMA, historial, pregunta, modelo.config.max_seq_len)
        except ValueError as exc:
            print(f"[Aviso] {exc}")
            continue
        nuevos = modelo.generate(ids, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 stop_ids=(1, 5))
        respuesta = tok.decode(nuevos, skip_special_tokens=True).strip()
        print("Origen >", respuesta or "[sin respuesta]")
        historial.append((pregunta, respuesta))
        historial = historial[-8:]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
