param(
  [ValidateSet("all", "docker", "mps")]
  [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if ($Mode -eq "mps") {
  throw "MPS mode is managed by macOS script ./scripts/stop.sh --mode mps"
}

docker compose down --remove-orphans
Write-Host "Docker services stopped."
