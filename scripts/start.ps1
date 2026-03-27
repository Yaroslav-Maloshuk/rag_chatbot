param(
  [ValidateSet("auto", "docker", "mps")]
  [string]$Mode = "auto"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

function Ensure-EnvFile {
  if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
  }
}

function Read-EnvValue([string]$Key, [string]$DefaultValue) {
  if (-not (Test-Path ".env")) {
    return $DefaultValue
  }
  foreach ($line in Get-Content ".env") {
    if ($line.StartsWith("$Key=")) {
      return $line.Substring($Key.Length + 1)
    }
  }
  return $DefaultValue
}

if ($Mode -eq "auto") {
  $Mode = "docker"
}

if ($Mode -eq "mps") {
  throw "MPS mode is only available on Apple Silicon macOS via ./scripts/start.sh --mode mps"
}

Ensure-EnvFile
docker compose up -d --build

$apiPort = Read-EnvValue "API_HOST_PORT" "18000"
$frontendPort = Read-EnvValue "FRONTEND_HOST_PORT" "15173"

Write-Host "Docker stack is up."
Write-Host "API:      http://localhost:$apiPort/docs"
Write-Host "Frontend: http://localhost:$frontendPort"
