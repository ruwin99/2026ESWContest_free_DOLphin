param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$Work = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Work ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$UvCache = Join-Path $Work ".uv-cache-training"
$env:UV_CACHE_DIR = $UvCache

if (-not (Test-Path -LiteralPath $Python) -and -not $Install) {
    throw "Training venv does not exist. Re-run with -Install to create and populate it."
}

if (-not (Test-Path -LiteralPath $Python)) {
    uv venv --python 3.11 $Venv
}

if ($Install) {
    uv pip install --python $Python torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python $Python -r (Join-Path $Work "requirements.txt")
}

& $Python (Join-Path $Work "tools\record_environment.py") --output-dir (Join-Path $Work "environment")
if ($LASTEXITCODE -ne 0) { throw "Environment recording failed: $LASTEXITCODE" }

Write-Host "Training environment: $Python"
