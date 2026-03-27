#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

read_env_value() {
  local key="$1"
  local default="$2"
  local value=""
  if [[ -f "$ROOT_DIR/.env" ]]; then
    value="$(awk -F= -v k="$key" '$1==k {print substr($0, index($0, "=")+1); exit}' "$ROOT_DIR/.env")"
  fi
  if [[ -n "$value" ]]; then
    printf "%s" "$value"
  else
    printf "%s" "$default"
  fi
}

POSTGRES_PORT="$(read_env_value "POSTGRES_HOST_PORT" "55432")"
REDIS_PORT="$(read_env_value "REDIS_HOST_PORT" "56379")"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing virtual environment (.venv). Run ./scripts/setup_mps_env.sh first."
  exit 1
fi
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

# macOS native mode to use Apple Silicon GPU (MPS).
export DATABASE_URL="postgresql+asyncpg://rag:rag@127.0.0.1:${POSTGRES_PORT}/ragdb"
export REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/2"
export CELERY_BROKER_URL="redis://127.0.0.1:${REDIS_PORT}/0"
export CELERY_RESULT_BACKEND="redis://127.0.0.1:${REDIS_PORT}/1"
export UPLOAD_DIR="${ROOT_DIR}/data/uploads"
export MODEL_DEVICE="mps"

"$PYTHON_BIN" - <<'PY'
import torch
if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    raise SystemExit("MPS is not available. Use native macOS Python + Apple Silicon PyTorch build.")
print("MPS is available and enabled.")
PY

exec "$PYTHON_BIN" -m celery -A app.tasks.celery_app:celery_app worker -Q documents --pool=solo --concurrency=1 --loglevel=info
