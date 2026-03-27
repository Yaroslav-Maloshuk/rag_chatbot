#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/.runtime"
mkdir -p "$RUNTIME_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start.sh [--mode auto|docker|mps]

Modes:
  auto   - docker on all systems except Apple Silicon macOS, where mps is selected
  docker - full docker-compose stack (api + worker + postgres + redis + frontend)
  mps    - hybrid mode for Apple Silicon macOS:
           docker: postgres + redis + frontend
           native: api + worker with MODEL_DEVICE=mps
EOF
}

MODE="auto"
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

if [[ "$MODE" != "auto" && "$MODE" != "docker" && "$MODE" != "mps" ]]; then
  echo "Invalid mode: $MODE"
  usage
  exit 1
fi

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

ensure_env_file() {
  if [[ ! -f "$ROOT_DIR/.env" ]]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "Created .env from .env.example"
  fi
}

ensure_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command is missing: $cmd"
    exit 1
  fi
}

is_apple_silicon_macos() {
  [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]
}

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

stop_local_mps_processes() {
  local pid_file=""
  local pid=""
  for pid_file in "$RUNTIME_DIR/api_mps.pid" "$RUNTIME_DIR/worker_mps.pid"; do
    [[ -f "$pid_file" ]] || continue
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && is_pid_running "$pid"; then
      kill "$pid" >/dev/null 2>&1 || true
      sleep 1
      if is_pid_running "$pid"; then
        kill -9 "$pid" >/dev/null 2>&1 || true
      fi
    fi
    rm -f "$pid_file"
  done
}

start_local_process() {
  local name="$1"
  local command="$2"
  local pid_file="$RUNTIME_DIR/${name}.pid"
  local log_file="$RUNTIME_DIR/${name}.log"
  local pid=""

  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && is_pid_running "$pid"; then
      echo "$name is already running (pid=$pid)"
      return
    fi
    rm -f "$pid_file"
  fi

  nohup bash -lc "cd \"$ROOT_DIR\" && $command" >"$log_file" 2>&1 &
  pid=$!
  echo "$pid" >"$pid_file"
  sleep 2

  if ! is_pid_running "$pid"; then
    echo "Failed to start $name. Last log lines:"
    tail -n 60 "$log_file" || true
    exit 1
  fi

  echo "Started $name (pid=$pid, log=$log_file)"
}

print_endpoints() {
  local api_port
  local frontend_port
  api_port="$(read_env_value "API_HOST_PORT" "18000")"
  frontend_port="$(read_env_value "FRONTEND_HOST_PORT" "15173")"
  echo "API:      http://localhost:${api_port}/docs"
  echo "Frontend: http://localhost:${frontend_port}"
}

start_docker_mode() {
  echo "Starting full Docker stack..."
  stop_local_mps_processes
  ensure_command docker
  ensure_env_file
  docker compose up -d --build
  echo "Docker stack is up."
  print_endpoints
}

start_mps_mode() {
  if ! is_apple_silicon_macos; then
    echo "MPS mode is supported only on Apple Silicon macOS."
    exit 1
  fi

  ensure_command docker
  ensure_command bash
  ensure_env_file

  mkdir -p "$ROOT_DIR/data/uploads"
  docker compose stop api worker >/dev/null 2>&1 || true
  ./scripts/start_infra.sh
  docker compose up -d --build frontend
  ./scripts/setup_mps_env.sh

  start_local_process "api_mps" "UVICORN_RELOAD=false ./scripts/run_api_mps.sh"
  start_local_process "worker_mps" "./scripts/run_worker_mps.sh"

  echo "Hybrid MPS mode is up."
  print_endpoints
  echo "Logs: $RUNTIME_DIR/api_mps.log and $RUNTIME_DIR/worker_mps.log"
}

if [[ "$MODE" == "auto" ]]; then
  if is_apple_silicon_macos; then
    MODE="mps"
  else
    MODE="docker"
  fi
fi

if [[ "$MODE" == "docker" ]]; then
  start_docker_mode
else
  start_mps_mode
fi
