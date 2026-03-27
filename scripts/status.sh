#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/.runtime"

print_local_proc() {
  local pid_file="$1"
  local name="$2"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name: not running (no pid file)"
    return
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "$name: running (pid=$pid)"
  else
    echo "$name: not running (stale pid file)"
  fi
}

echo "== Local MPS processes =="
print_local_proc "$RUNTIME_DIR/api_mps.pid" "api_mps"
print_local_proc "$RUNTIME_DIR/worker_mps.pid" "worker_mps"

echo
echo "== Docker services =="
docker compose ps || true
