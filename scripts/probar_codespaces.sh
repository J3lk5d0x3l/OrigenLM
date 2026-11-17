#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python -m unittest discover -s tests -v
if [ -f checkpoints/origen_latest.pt ]; then
  python chat.py
else
  echo 'No hay checkpoint: ejecuta bash scripts/entrenar_codespaces.sh 100'
fi
