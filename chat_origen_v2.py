"""Chat interactivo con los pesos PROPIOS de OrigenLM v2."""
import argparse
from pathlib import Path
import os

import torch

from origen.v2 import OrigenV2Config, OrigenV2
from origen.resources import configure_cpu_threads
from origen.tokenizer import DEFAULT_SYSTEM, conversation_prefix_limited, decode


def main():
    parser = argparse.ArgumentParser(description="Chat de OrigenLM v2")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_v2_latest.pt"))
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 = muestrear todo el vocabulario")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Configurable, sin tope artificial")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"No existe el checkpoint: {args.checkpoint}. Entrena antes de conversar.")
    configure_cpu_threads("chat")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("architecture") != "origen-v2-gqa-v1":
        parser.error("El checkpoint indicado no es de OrigenLM v2")
    model = OrigenV2(OrigenV2Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    print(f"OrigenLM v2 | paso {saved['step']} | dispositivo={device} | ventana={model.config.max_seq_len}")
    print("Prototipo en entrenamiento; puede responder incoherencias.")
    print("Escribe /salir para cerrar, /limpiar para reiniciar, /actualizar para recargar.")
    history = []
    while True:
        try:
            question = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break
        if question.casefold() == "/salir":
            break
        if question.casefold() == "/actualizar":
            lanzador = Path(__file__).resolve().parent / "scripts/origen"
            os.execv("/bin/bash", ["/bin/bash", str(lanzador), "v2"])
        if question.casefold() == "/limpiar":
            history.clear()
            print("Historial reiniciado.")
            continue
        if not question:
            continue
        try:
            prefix = conversation_prefix_limited(
                args.system, history, question, model.config.max_seq_len
            )
            tokens = model.generate(
                prefix, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
            )
        except ValueError as exc:
            print(f"[Aviso] {exc}")
            continue
        reply = decode(tokens).strip() or "[El modelo no produjo texto útil.]"
        print("Origen >", reply)
        history.append((question, reply))
        history = history[-10:]


if __name__ == "__main__":
    main()
