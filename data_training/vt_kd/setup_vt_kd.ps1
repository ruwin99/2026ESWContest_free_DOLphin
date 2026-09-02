[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$TrainingRoot = $PSScriptRoot
$Venv = Join-Path $TrainingRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$GpuRequirements = Join-Path $TrainingRoot "requirements-gpu.txt"
$CommonRequirements = Join-Path $TrainingRoot "requirements-common.txt"

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    $resolvedTraining = (Resolve-Path -LiteralPath $TrainingRoot).Path
    $resolvedVenv = (Resolve-Path -LiteralPath $Venv).Path
    if (-not $resolvedVenv.StartsWith($resolvedTraining, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a venv outside the VT KD folder: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $Python)) {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        & $uv.Source venv --python 3.12 $Venv
    }
    else {
        & py -3.12 -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Python 3.12 virtual environment: $Venv"
    }
}

if (-not $SkipInstall) {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        # Install torch separately. Combining PyPI with the PyTorch index can
        # silently select the much smaller CPU wheel with the same public version.
        & $uv.Source pip install --python $Python --reinstall -r $GpuRequirements
        if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
        & $uv.Source pip install --python $Python -r $CommonRequirements
    }
    else {
        & $Python -m pip install --upgrade pip setuptools wheel
        & $Python -m pip install --force-reinstall -r $GpuRequirements
        if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }
        & $Python -m pip install -r $CommonRequirements
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}

& $Python (Join-Path $TrainingRoot "scripts\verify_gpu.py")
if ($LASTEXITCODE -ne 0) {
    throw "GPU verification failed."
}

Write-Host "VT KD environment ready: $Python"
