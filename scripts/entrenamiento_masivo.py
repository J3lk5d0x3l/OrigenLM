"""Preparar corpus autorizado y continuar ENTRENAMIENTO REAL de OrigenLM v1.

Se ejecuta con 'origen entrenar-masivo' desde Codespaces. Los pasos actualizan
pesos en checkpoints/origen_masivo.pt, no en el checkpoint histórico.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def execute(command):
    print("\n[OrigenLM] Ejecutando:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Preentrenar pesos originales con documentación autorizada")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--max-minutes", type=float, default=180)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--no-download", action="store_true",
                        help="Usa únicamente archivos locales ya presentes")
    parser.add_argument("--sft-steps", type=int, default=500,
                        help="Ajuste conversacional final con datos propios (0=desactivar)")
    parser.add_argument("--no-chat", action="store_true",
                        help="No reabrir el chat después del entrenamiento")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.seq_len <= 0 or args.max_minutes <= 0 or args.sft_steps < 0:
        parser.error("Pasos, contexto y tiempo deben ser positivos")
    sources = ROOT / "dataset/fuentes"
    prepared = ROOT / "dataset/preparado/meta.json"
    commands = []
    if not args.no_download and not sources.exists():
        commands.append([sys.executable, "-m", "scripts.descargar_fuentes"])
    if not prepared.is_file():
        commands.append([sys.executable, "-m", "scripts.preparar_corpus", "--confirmar-derechos"])
    commands.append([
        sys.executable, "train_masivo.py",
        "--steps", str(args.steps),
        "--max-minutes", str(args.max_minutes),
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
    ])
    if args.sft_steps:
        commands.append([
            sys.executable, "train_respuestas.py",
            "--resume", "checkpoints/origen_masivo.pt",
            "--output", "checkpoints/origen_ajustado.pt",
            "--steps", str(args.sft_steps),
            "--save-every", str(min(100, args.sft_steps)),
            "--context", str(max(384, args.seq_len)),
        ])
    if args.dry_run:
        for command in commands:
            print("PLAN:", " ".join(command))
        return 0
    for command in commands:
        try:
            execute(command)
        except subprocess.CalledProcessError as exc:
            print(f"[OrigenLM] El proceso terminó con error ({exc.returncode}); se conservan los checkpoints.", file=sys.stderr)
            return exc.returncode or 1
    result = ROOT / ("checkpoints/origen_ajustado.pt" if args.sft_steps else "checkpoints/origen_masivo.pt")
    if not result.is_file():
        print("[OrigenLM] No se generó checkpoint de entrenamiento.", file=sys.stderr)
        return 2
    print(f"[OrigenLM] Pesos actualizados guardados en {result}", flush=True)
    if not args.no_chat:
        print("[OrigenLM] Regresando al chat con el modelo actualizado...", flush=True)
        os.chdir(ROOT)
        os.execv(sys.executable, [
            sys.executable, str(ROOT / "chat.py"), "--checkpoint", str(result)
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
