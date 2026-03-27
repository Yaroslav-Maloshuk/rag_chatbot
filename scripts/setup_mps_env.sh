#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    echo "python3.11 is required for this project. Install it and rerun."
    exit 1
  fi
fi

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -r requirements.txt

"$ROOT_DIR/.venv/bin/python" - <<'PY'
import platform
import torch

is_macos = platform.system() == "Darwin"
has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print(f"macOS={is_macos}, mps_available={has_mps}")
if not (is_macos and has_mps):
    raise SystemExit("MPS is unavailable in current Python environment.")
PY

echo "MPS environment is ready in .venv"
