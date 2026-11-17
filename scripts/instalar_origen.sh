#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE="$ROOT/scripts/origen"
SYSTEM="/usr/local/bin/origen"
if [[ ! -f "$SOURCE" ]]; then
  echo "No existe $SOURCE" >&2
  exit 1
fi
if [[ -e "$SYSTEM" || -L "$SYSTEM" ]]; then
  existing="$(readlink -f "$SYSTEM" 2>/dev/null || true)"
  if [[ "$existing" != "$SOURCE" ]]; then
    echo "Existe otro ejecutable: $SYSTEM. No se modifica." >&2
    exit 1
  fi
fi
if [[ -w "/usr/local/bin" ]]; then
  ln -sfn "$SOURCE" "$SYSTEM"
  DEST="$SYSTEM"
elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  sudo ln -sfn "$SOURCE" "$SYSTEM"
  DEST="$SYSTEM"
else
  mkdir -p "$HOME/.local/bin"
  DEST="$HOME/.local/bin/origen"
  if [[ -e "$DEST" || -L "$DEST" ]]; then
    existing="$(readlink -f "$DEST" 2>/dev/null || true)"
    [[ "$existing" == "$SOURCE" ]] || { echo "Otro programa ocupa $DEST" >&2; exit 1; }
  fi
  ln -sfn "$SOURCE" "$DEST"
  if ! grep -Fq '.local/bin:$PATH' "$HOME/.bashrc" 2>/dev/null; then
    printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.bashrc"
  fi
fi
echo "[OrigenLM] Comando instalado: $DEST -> $SOURCE"
echo "[OrigenLM] Abre una terminal y escribe: origen"
