"""Train original model weights on YOUR data; no model downloads or API calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import torch
from origen.model import ModelConfig, OrigenLM, PRESETS
from origen.tokenizer import training_example


def load_data(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    tokens = []
    if path.suffix.lower() == ".txt":
        for paragraph in path.read_text(encoding="utf-8").split("\n\n"):
            if paragraph.strip():
                tokens.extend(training_example({"text": paragraph.strip()}))
    else:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    tokens.extend(training_example(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as e:
                    raise ValueError(f"Invalid dataset line {line_number}: {e}") from e
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Train OrigenLM from random weights")
    parser.add_argument("--data", type=Path, default=Path("dataset/chat.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--preset", choices=PRESETS, default="tiny")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.seq_len, args.save_every, args.log_every) <= 0:
        parser.error("steps, batch-size, seq-len, save-every and log-every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume:
        stored = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = ModelConfig(**stored["config"])
    else:
        cfg = ModelConfig(**PRESETS[args.preset], max_seq_len=args.seq_len)
    if args.seq_len > cfg.max_seq_len:
        parser.error(f"seq-len must be <= checkpoint context limit {cfg.max_seq_len}")
    model = OrigenLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start = 0
    if args.resume:
        model.load_state_dict(stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        start = int(stored["step"])

    data = load_data(args.data)
    if len(data) <= args.seq_len + 1:
        raise ValueError(f"Need more than {args.seq_len+1} training tokens, found {len(data)}")
    data = torch.tensor(data, dtype=torch.long)
    print(f"Device: {device} | model parameters: {model.count_parameters():,} | training tokens: {len(data):,}")
    print("Warning: the example dataset is only a technical test, not enough for an intelligent assistant.")
    args.output.mkdir(parents=True, exist_ok=True)

    def checkpoint(step, loss):
        path = args.output / "origen_latest.pt"
        torch.save({"config": asdict(cfg), "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "loss": float(loss)}, path)
        print(f"Saved: {path} (step {step})")

    model.train()
    for step in range(start + 1, start + args.steps + 1):
        idx = torch.randint(0, len(data) - args.seq_len - 1, (args.batch_size,))
        positions = idx[:, None] + torch.arange(args.seq_len + 1)[None, :]
        sequences = data[positions].to(device)
        source, target = sequences[:, :-1], sequences[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(source, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == start + 1 or step % args.log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == start + args.steps:
            checkpoint(step, loss.item())


if __name__ == "__main__":
    main()
