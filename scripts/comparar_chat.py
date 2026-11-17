"""Comparación cualitativa reproducible entre checkpoints de OrigenLM.

No entrena ni modifica los archivos. Cada checkpoint usa la misma semilla y
las mismas preguntas para facilitar la comparación de respuestas generadas.
"""
import argparse
from pathlib import Path

import torch
from origen.model import ModelConfig, OrigenLM
from origen.tokenizer import DEFAULT_SYSTEM, conversation_prefix_limited, decode


def load_checkpoint(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = OrigenLM(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar generaciones de OrigenLM")
    parser.add_argument(
        "--a", type=Path,
        default=Path("/workspaces/backup-origenlm/origen_200_pasos.pt"),
        help="Checkpoint A"
    )
    parser.add_argument(
        "--b", type=Path,
        default=Path("/workspaces/backup-origenlm/origen_300_pasos.pt"),
        help="Checkpoint B"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    prompts = ["Hola", "¿Cómo te llamas?", "Gracias"]
    for path in [args.a, args.b]:
        if not path.is_file():
            parser.error(f"No existe el checkpoint: {path}")
    if args.max_new_tokens <= 0 or args.temperature <= 0:
        parser.error("max-new-tokens y temperature deben ser positivos")
    for path in [args.a, args.b]:
        model, checkpoint = load_checkpoint(path)
        print(f"\\n=== {path.name} | paso {checkpoint['step']} ===")
        for question in prompts:
            prefix = conversation_prefix_limited(
                DEFAULT_SYSTEM, [], question, model.config.max_seq_len
            )
            torch.manual_seed(args.seed)
            response = decode(
                model.generate(prefix, args.max_new_tokens, args.temperature)
            ).strip()
            print(f"Tú: {question}")
            print(f"Origen: {response or '[Sin texto generado]'}")
    print("\\nComparación cualitativa; no mide comprensión ni garantiza mejoras.")


if __name__ == "__main__":
    main()
