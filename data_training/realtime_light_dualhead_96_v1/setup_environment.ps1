param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Project ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Requirements = Join-Path $Project "environment\requirements-lock.txt"

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    $ResolvedProject = (Resolve-Path -LiteralPath $Project).Path
    $ResolvedVenv = (Resolve-Path -LiteralPath $Venv).Path
    if (-not $ResolvedVenv.StartsWith($ResolvedProject, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a venv outside the project: $ResolvedVenv"
    }
    Remove-Item -LiteralPath $ResolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $Python)) {
    & uv venv $Venv --python 3.11
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }
}
& uv pip install --python $Python -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed with exit code $LASTEXITCODE" }
& uv pip freeze --python $Python | Out-File -LiteralPath (Join-Path $Project "environment\successful-freeze.txt") -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "environment freeze failed with exit code $LASTEXITCODE" }
& $Python -c "import torch,onnx,onnxruntime,scipy; print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available()); print('onnx',onnx.__version__); print('providers',onnxruntime.get_available_providers()); print('scipy',scipy.__version__)"
