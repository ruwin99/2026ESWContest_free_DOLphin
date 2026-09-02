[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("verify", "audit", "smoke", "train", "evaluate-val", "evaluate-test", "export-onnx")]
    [string]$Task,

    [int]$Epochs = 70,
    [int]$Batch = 0,
    [int]$Workers = -1,
    [double]$LearningRate = 0.006,
    [int]$Seed = 42,
    [string]$RunName = "bgcrack-seed42",
    [string]$Resume = "",
    [string]$Checkpoint = "",
    [string]$Output = "",
    [switch]$Amp,
    [switch]$NoAmp,
    [switch]$AllowTestRerun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$TrainingRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $TrainingRoot "..\..")).Path
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$Scripts = Join-Path $TrainingRoot "scripts"
$DataRoot = Join-Path $TrainingRoot "data\Steelcrack"
$OfficialCode = Join-Path $TrainingRoot "official_bgcrack"
$RunsRoot = Join-Path $ProjectRoot "outputs\training\steelcrack"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Steelcrack virtual environment not found. Run setup_steelcrack.ps1 first."
}
if ($Amp -and $NoAmp) {
    throw "Choose only one precision option: -Amp or -NoAmp. FP32 is the default."
}

function Invoke-Python([string]$Script, [string[]]$Arguments) {
    & $Python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

switch ($Task) {
    "verify" {
        Invoke-Python (Join-Path $Scripts "verify_gpu.py") @()
    }
    "audit" {
        Invoke-Python (Join-Path $Scripts "audit_dataset.py") @(
            "--data-root", $DataRoot,
            "--output", (Join-Path $TrainingRoot "audit\steelcrack_audit.json")
        )
    }
    { $_ -in @("smoke", "train") } {
        $EffectiveBatch = if ($Batch -gt 0) { $Batch } elseif ($Task -eq "smoke") { 9 } else { 9 }
        $EffectiveWorkers = if ($Workers -ge 0) { $Workers } else { 4 }
        $Arguments = @(
            "--data-root", $DataRoot,
            "--official-code", $OfficialCode,
            "--runs-root", $RunsRoot,
            "--run-name", $RunName,
            "--epochs", "$Epochs",
            "--batch-size", "$EffectiveBatch",
            "--workers", "$EffectiveWorkers",
            "--lr", "$LearningRate",
            "--seed", "$Seed"
        )
        if ($Amp) { $Arguments += "--amp" } else { $Arguments += "--no-amp" }
        if ($Task -eq "smoke") { $Arguments += @("--smoke-batches", "1") }
        if ($Resume) {
            $ResolvedResume = (Resolve-Path -LiteralPath $Resume).Path
            $Arguments += @("--resume", $ResolvedResume)
        }
        Invoke-Python (Join-Path $Scripts "train_windows.py") $Arguments
    }
    { $_ -in @("evaluate-val", "evaluate-test") } {
        if (-not $Checkpoint) { throw "$Task requires -Checkpoint <best.pth>." }
        $ResolvedCheckpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        $Split = if ($Task -eq "evaluate-test") { "Test" } else { "Validation" }
        if (-not $Output) {
            $Output = Join-Path (Split-Path -Parent (Split-Path -Parent $ResolvedCheckpoint)) "evaluation-$($Split.ToLowerInvariant())"
        }
        $EffectiveWorkers = if ($Workers -ge 0) { $Workers } else { 4 }
        $Arguments = @(
            "--checkpoint", $ResolvedCheckpoint,
            "--split", $Split,
            "--data-root", $DataRoot,
            "--official-code", $OfficialCode,
            "--output", $Output,
            "--workers", "$EffectiveWorkers"
        )
        if ($Amp) { $Arguments += "--amp" } else { $Arguments += "--no-amp" }
        if ($AllowTestRerun) { $Arguments += "--allow-test-rerun" }
        Invoke-Python (Join-Path $Scripts "evaluate.py") $Arguments
    }
    "export-onnx" {
        if (-not $Checkpoint) { throw "export-onnx requires -Checkpoint <best.pth>." }
        $ResolvedCheckpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        if (-not $Output) {
            $Output = Join-Path $ProjectRoot "output\models\steelcrack\bgcrack-steelcrack-512-fp32.onnx"
        }
        Invoke-Python (Join-Path $Scripts "export_onnx.py") @(
            "--checkpoint", $ResolvedCheckpoint,
            "--output", $Output,
            "--official-code", $OfficialCode,
            "--data-root", $DataRoot,
            "--opset", "17",
            "--input-size", "512", "512"
        )
    }
}
