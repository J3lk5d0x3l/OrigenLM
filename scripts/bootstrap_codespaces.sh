#!/usr/bin/env bash
set -euo pipefail
printf '\n[OrigenLM] Instalando PyTorch CPU dentro del Codespace...\n'
python -m pip install --disable-pip-version-check --no-cache-dir "numpy>=1.26,<2" "tokenizers>=0.20,<1"
python -m pip install --disable-pip-version-check --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
printf '\n[OrigenLM] Ejecutando pruebas...\n'
python -m unittest discover -s tests -v
printf '\n[OrigenLM] Dependencias y pruebas listas. Utiliza: origen estado\n'
