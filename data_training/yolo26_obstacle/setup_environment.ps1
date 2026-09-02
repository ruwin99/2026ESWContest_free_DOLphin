[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path -LiteralPath (Join-Path $Project "..\..")).Path
$Venv = Join-Path $Project ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Models = Join-Path $Project "models\base"
$Reports = Join-Path $Project "reports"
$YoloConfig = Join-Path $Project ".ultralytics"
$UvCache = Join-Path $Root "data_training\.cache\uv"
$Uv = Get-Command uv -ErrorAction SilentlyContinue

if (-not $Uv) { throw "uv is required. Install it from https://docs.astral.sh/uv/" }
New-Item -ItemType Directory -Force -Path $Models,$Reports,$YoloConfig,$UvCache | Out-Null
$env:UV_CACHE_DIR = $UvCache
$env:UV_LINK_MODE = "copy"
$env:YOLO_CONFIG_DIR = $YoloConfig

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "[1/4] Creating isolated Python 3.12 environment"
    & $Uv.Source venv --python 3.12 $Venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create YOLO26 environment" }
} else {
    Write-Host "[1/4] Reusing $Venv"
}

Write-Host "[2/4] Installing current YOLO26-compatible Ultralytics and Roboflow SDK"
& $Uv.Source pip install --python $Python --upgrade --requirements (Join-Path $Project "requirements-core.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install core dependencies" }

Write-Host "[3/4] Pinning the verified CUDA 12.8 PyTorch build"
& $Uv.Source pip install --python $Python --upgrade --requirements (Join-Path $Project "requirements-gpu.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch" }

Write-Host "[4/4] Downloading and checking official YOLO26n weights on CUDA"
Push-Location $Models
try {
    & $Python (Join-Path $Project "tools\preflight.py") `
        --model "yolo26n.pt" `
        --report (Join-Path $Reports "preflight.json")
    if ($LASTEXITCODE -ne 0) { throw "YOLO26 CUDA preflight failed" }
}
finally {
    Pop-Location
}

Write-Host "Environment ready: $Python"
