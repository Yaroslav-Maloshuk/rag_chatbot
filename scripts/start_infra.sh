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

docker compose up -d postgres redis
echo "Infra is up: postgres on ${POSTGRES_PORT}, redis on ${REDIS_PORT}"
