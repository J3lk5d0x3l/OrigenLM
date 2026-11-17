#!/usr/bin/env bash
set -euo pipefail
printf '\n[OrigenLM] Instalando PyTorch CPU dentro del Codespace...\n'
python -m pip install --disable-pip-version-check --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
printf '\n[OrigenLM] Ejecutando pruebas...\n'
python -m unittest discover -s tests -v
printf '\n[OrigenLM] Listo: bash scripts/entrenar_codespaces.sh 100\n'
