"""Run a fully local, conversational demo of your OWN model checkpoint."""
import argparse
from pathlib import Path
import torch
from origen.model import ModelConfig, OrigenLM
from origen.tokenizer import conversation_prefix, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/origen_latest.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--system", default="Eres Origen, un asistente de IA útil, claro y respetuoso. Responde en español.")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error("No checkpoint yet. Run: python train.py --steps 100")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = OrigenLM(ModelConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    print(f"OrigenLM | local device={device} | trained steps={saved['step']}")
    print("This example may produce nonsense: it is not pretrained on a large corpus.")
    print("Type /salir to quit, /limpiar to clear conversation history.")
    history = []
    while True:
        try:
            user = input("\nTú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break
        if user.lower() == "/salir":
            break
        if user.lower() == "/limpiar":
            history.clear()
            print("History cleared.")
            continue
        if not user:
            continue
        prompt = conversation_prefix(args.system, history, user)
        tokens = model.generate(prompt, args.max_new_tokens, args.temperature)
        reply = decode(tokens).strip()
        print("Origen >", reply or "[Sin respuesta textual. Necesita más entrenamiento.]")
        history.append((user, reply))
        history = history[-3:]


if __name__ == "__main__":
    main()
