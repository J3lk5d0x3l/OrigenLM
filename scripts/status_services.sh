#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
docker compose -f compose.yaml ps
echo
echo "Modelos Ollama:"
docker compose -f compose.yaml exec -T ollama ollama list || true
echo
echo "Aplicación local: http://localhost:3001"
