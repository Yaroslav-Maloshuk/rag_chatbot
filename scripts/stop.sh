#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/.runtime"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/stop.sh [--mode all|docker|mps]

Modes:
  all    - stop local MPS processes and stop/remove docker services
  docker - stop/remove docker services only
  mps    - stop local MPS processes and docker infra/frontend
EOF
}

MODE="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "all" && "$MODE" != "docker" && "$MODE" != "mps" ]]; then
  echo "Invalid mode: $MODE"
  usage
  exit 1
fi

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

stop_pid_file() {
  local pid_file="$1"
  local name="$2"
  local pid=""

  [[ -f "$pid_file" ]] || return
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && is_pid_running "$pid"; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if is_pid_running "$pid"; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    echo "Stopped $name (pid=$pid)"
  fi
  rm -f "$pid_file"
}

stop_local_mps() {
  stop_pid_file "$RUNTIME_DIR/api_mps.pid" "api_mps"
  stop_pid_file "$RUNTIME_DIR/worker_mps.pid" "worker_mps"
}

if [[ "$MODE" == "docker" ]]; then
  docker compose down --remove-orphans
  exit 0
fi

if [[ "$MODE" == "mps" ]]; then
  stop_local_mps
  docker compose stop frontend postgres redis >/dev/null 2>&1 || true
  echo "Stopped local MPS processes and docker infra/frontend."
  exit 0
fi

# MODE=all
stop_local_mps
docker compose down --remove-orphans
echo "Stopped all local and docker services."
