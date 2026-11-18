#!/usr/bin/env bash
# Arranca un modelo abierto local (llama.cpp) y abre el chat de OrigenLM.
#
#   bash scripts/chat_local.sh                     # Qwen2.5-Coder-7B (recomendado)
#   MODEL=models/Qwen3-8B-Q4_K_M.gguf bash scripts/chat_local.sh
#   MODEL=models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf bash scripts/chat_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-models/qwen2.5-coder-7b-instruct-q4_k_m.gguf}"
case "$(basename "$MODEL")" in
  Qwen3-8B*) MODEL_URL="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf" ;;
  *7b-instruct*) MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf" ;;
  *) MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" ;;
esac
BIN=tools/llama
LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/b12484/llama-b12484-bin-ubuntu-x64.tar.gz
PUERTO="${PUERTO:-8080}"
CTX="${CTX:-4096}"

mkdir -p "$BIN" models

if [ ! -x "$BIN/llama-server" ]; then
  echo "[origen] bajando llama.cpp..."
  curl -L --retry 3 -o /tmp/llama.tar.gz "$LLAMA_URL"
  tar -xzf /tmp/llama.tar.gz -C "$BIN" && rm -f /tmp/llama.tar.gz
fi

if [ ! -s "$MODEL" ]; then
  echo "[origen] bajando $(basename "$MODEL")..."
  curl -L --retry 3 -C - -o "$MODEL" "$MODEL_URL"
fi

# Si ya hay otro modelo cargado, se reinicia el servidor con el pedido.
if curl -sf --max-time 2 "http://127.0.0.1:$PUERTO/health" >/dev/null; then
  cargado=$(tr '\0' ' ' < "/proc/$(pgrep -f 'llama-server' | head -1)/cmdline" 2>/dev/null | grep -o 'models/[^ ]*' | head -1 || true)
  if [ -n "$cargado" ] && [ "$cargado" != "$MODEL" ]; then
    echo "[origen] cambiando de $(basename "$cargado") a $(basename "$MODEL")..."
    pkill -f llama-server 2>/dev/null || true
    sleep 2
  fi
fi

if ! curl -sf --max-time 2 "http://127.0.0.1:$PUERTO/health" >/dev/null; then
  echo "[origen] cargando $(basename "$MODEL") en :$PUERTO (4 hilos)..."
  setsid nohup "$BIN/llama-server" -m "$MODEL" -c "$CTX" -t "$(nproc)" \
    --host 127.0.0.1 --port "$PUERTO" --jinja -np 1 \
    > /tmp/llama-server.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    curl -sf --max-time 2 "http://127.0.0.1:$PUERTO/health" >/dev/null && break
    sleep 1
  done
fi

exec python3 chat_local.py --url "http://127.0.0.1:$PUERTO" "$@"
