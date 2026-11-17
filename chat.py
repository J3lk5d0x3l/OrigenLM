"""Run a fully local, conversational demo of your OWN model checkpoint."""
import argparse
from pathlib import Path
import os
import shlex
import sys
import torch
from origen.model import ModelConfig, OrigenLM
from origen.resources import configure_cpu_threads
from origen.herramientas_chat import SesionHerramientas, es_comando_herramienta
from origen.tokenizer import DEFAULT_SYSTEM, conversation_prefix_limited, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_latest.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--seed", type=int, default=None, help="Semilla opcional para repetir una generación")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if not args.checkpoint.exists():
        parser.error("No checkpoint yet. Run: python train.py --steps 100")
    configure_cpu_threads("chat")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = OrigenLM(ModelConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    print(f"OrigenLM | local device={device} | trained steps={saved['step']}")
    print("This example may produce nonsense: it is not pretrained on a large corpus.")
    print("Comandos: /entrenar-masivo, /buscar, /pagina, /herramientas, /usar, /actualizar, /salir")
    print(f"Contexto máximo: {model.config.max_seq_len} tokens de bytes UTF-8. Usa preguntas cortas.")
    herramientas = None
    history = []
    while True:
        try:
            user = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break
        if user.startswith("/entrenar-masivo"):
            # Entrena el checkpoint original v1 y lo guarda en otro archivo.
            try:
                opciones = shlex.split(user)[1:]
            except ValueError as exc:
                print(f"[OrigenLM] Opciones inválidas: {exc}")
                continue
            if herramientas is not None:
                herramientas.cerrar()
            print("[OrigenLM] Iniciando entrenamiento real del modelo propio...", flush=True)
            os.execv(sys.executable, [sys.executable, "-m", "scripts.entrenamiento_masivo", *opciones])
        if es_comando_herramienta(user):
            try:
                if herramientas is None:
                    herramientas = SesionHerramientas()
                herramientas.ejecutar(user)
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[OrigenLM] Error de herramienta: {exc}")
            continue
        if user.lower() == "/salir":
            break
        if user.lower() == "/actualizar":
            lanzador = Path(__file__).resolve().parent / "scripts/origen"
            os.execv("/bin/bash", ["/bin/bash", str(lanzador), "v1"])
        if user.lower() == "/limpiar":
            history.clear()
            print("History cleared.")
            continue
        if not user:
            continue
        try:
            prompt = conversation_prefix_limited(args.system, history, user, model.config.max_seq_len)
        except ValueError as exc:
            print(f"[Aviso] {exc}")
            continue
        tokens = model.generate(prompt, args.max_new_tokens, args.temperature)
        reply = decode(tokens).strip()
        print("Origen >", reply or "[Sin respuesta textual. Necesita más entrenamiento.]")
        history.append((user, reply))
        history = history[-3:]

    if herramientas is not None:
        herramientas.cerrar()


if __name__ == "__main__":
    main()
