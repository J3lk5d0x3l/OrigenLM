#!/usr/bin/env bash
set -euo pipefail
STEPS="${1:-100}"
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Uso: bash scripts/entrenar_codespaces.sh [pasos_positivos]" >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
if [ -f checkpoints/origen_latest.pt ]; then
  echo "[OrigenLM] Continuando desde checkpoint existente (${STEPS} pasos adicionales)"
  python train.py --resume checkpoints/origen_latest.pt --steps "$STEPS" --batch-size 2 --seq-len 96 --save-every 50
else
  echo "[OrigenLM] Entrenando pesos aleatorios (${STEPS} pasos)"
  python train.py --preset tiny --steps "$STEPS" --batch-size 2 --seq-len 96 --save-every 50
fi
