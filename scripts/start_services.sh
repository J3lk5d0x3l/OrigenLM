#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: falta Docker. Abre este repositorio dentro de GitHub Codespaces con Docker disponible." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: el daemon de Docker no está disponible en este Codespace (docker info falló)." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: falta Docker Compose v2." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: se requiere Python 3 para generar claves locales." >&2
  exit 1
fi

mkdir -p .runtime/anythingllm/storage
touch .runtime/anythingllm/.env
if [[ ! -f .runtime/secrets.env ]]; then
  (umask 077; python3 - <<'PY' > .runtime/secrets.env
import secrets
print("JWT_SECRET=" + secrets.token_urlsafe(48))
print("SIG_KEY=" + secrets.token_urlsafe(48))
PY
  )
  echo "Se generaron claves locales privadas (.runtime/secrets.env)."
fi

export ORIGEN_MODEL="${ORIGEN_MODEL:-qwen2.5-coder:1.5b}"
echo "Iniciando frontend/backend de OrigenLM y Ollama..."
docker compose -f compose.yaml up -d

if docker compose -f compose.yaml exec -T ollama ollama list | grep -Fq "$ORIGEN_MODEL"; then
  echo "Modelo $ORIGEN_MODEL ya disponible."
else
  echo "Descargando el modelo $ORIGEN_MODEL en Ollama (primera ejecución)..."
  docker compose -f compose.yaml exec -T ollama ollama pull "$ORIGEN_MODEL"
fi

echo
docker compose -f compose.yaml ps
echo
echo "OrigenLM: abre el puerto 3001 en la pestaña Ports de Codespaces."
echo "Mantén el puerto como PRIVATE y crea la cuenta administradora en el asistente inicial."
echo "En Settings > AI Providers comprueba: Ollama, http://ollama:11434 y $ORIGEN_MODEL."
echo "Marca y nombre: Settings > Customization > Branding & Whitelabeling > OrigenLM."
