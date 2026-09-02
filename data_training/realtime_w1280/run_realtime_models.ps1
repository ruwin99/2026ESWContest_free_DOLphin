[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("verify", "export-rust", "export-crack", "export-all")]
    [string]$Task,
    [string]$Checkpoint = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$RealtimeRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $RealtimeRoot "..\..")).Path
$Exporter = Join-Path $RealtimeRoot "scripts\export_realtime_onnx.py"
$RustPython = Join-Path $ProjectRoot "data_training\vt_kd\.venv\Scripts\python.exe"
$CrackPython = Join-Path $ProjectRoot "data_training\steelcrack\.venv\Scripts\python.exe"
$DefaultRustCheckpoint = Join-Path $ProjectRoot "outputs\training\vt_kd\kd-seed42\weights\best.pt"
$DefaultCrackCheckpoint = Join-Path $ProjectRoot "outputs\training\steelcrack\bgcrack-seed42\weights\best.pth"
$OutputRoot = Join-Path $ProjectRoot "output\models\realtime_w1280"
$DefaultRustOutput = Join-Path $OutputRoot "realtime-rust-mnv2-os8-w1280-h240-fp32.onnx"
$DefaultCrackOutput = Join-Path $OutputRoot "realtime-crack-bgcrack-w1280-h128-fp32.onnx"

function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-Export([string]$Model, [string]$Python, [string]$ModelCheckpoint, [string]$ModelOutput) {
    Assert-File $Python "Python environment"
    Assert-File $ModelCheckpoint "Checkpoint"
    & $Python $Exporter --model $Model --checkpoint $ModelCheckpoint --output $ModelOutput --opset 18
    if ($LASTEXITCODE -ne 0) { throw "ONNX export failed with exit code $LASTEXITCODE" }
}

if ($Task -eq "verify") {
    Assert-File $RustPython "Rust Python environment"
    Assert-File $CrackPython "Crack Python environment"
    Assert-File $DefaultRustCheckpoint "Rust checkpoint"
    Assert-File $DefaultCrackCheckpoint "Crack checkpoint"
    Write-Host "Realtime ONNX prerequisites are ready."
    exit 0
}

if ($Task -eq "export-all" -and ($Checkpoint -or $Output)) {
    throw "-Checkpoint and -Output are only valid with export-rust or export-crack."
}

if ($Task -eq "export-rust" -or $Task -eq "export-all") {
    $RustCheckpoint = if ($Checkpoint -and $Task -eq "export-rust") { $Checkpoint } else { $DefaultRustCheckpoint }
    $RustOutput = if ($Output -and $Task -eq "export-rust") { $Output } else { $DefaultRustOutput }
    Invoke-Export "rust" $RustPython $RustCheckpoint $RustOutput
}

if ($Task -eq "export-crack" -or $Task -eq "export-all") {
    $CrackCheckpoint = if ($Checkpoint -and $Task -eq "export-crack") { $Checkpoint } else { $DefaultCrackCheckpoint }
    $CrackOutput = if ($Output -and $Task -eq "export-crack") { $Output } else { $DefaultCrackOutput }
    Invoke-Export "crack" $CrackPython $CrackCheckpoint $CrackOutput
}
