[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "verify",
        "audit",
        "convert-teacher",
        "smoke-supervised",
        "smoke-kd",
        "train-supervised",
        "train-kd",
        "evaluate-val",
        "evaluate-test",
        "export-onnx"
    )]
    [string]$Task,

    [int]$Epochs = 0,
    [int]$Batch = 0,
    [int]$Workers = -1,
    [int]$Seed = 42,
    [string]$RunName = "",
    [string]$Checkpoint = "",
    [string]$Resume = "",
    [string]$Output = "",
    [switch]$AcknowledgeLegacyPickleRisk,
    [switch]$AllowTestRerun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$TrainingRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $TrainingRoot "..\..")).Path
$Python = Join-Path $TrainingRoot ".venv\Scripts\python.exe"
$Scripts = Join-Path $TrainingRoot "scripts"
$Config = Join-Path $TrainingRoot "configs\student_mnv2_os8.yaml"
$Dataset = Join-Path $ProjectRoot "data_training\data\raw\virginia_tech_cssd\dataset\512x512"
$Split = Join-Path $TrainingRoot "splits\vt_train316_val80_seed42.csv"
$TeacherRaw = Join-Path $ProjectRoot "models\virginia_tech_cssd\teacher_raw\var_original_wbatch_2_plus\var_original_wbatch_2_plus_weights_40.pt"
$TeacherConverted = Join-Path $ProjectRoot "models\virginia_tech_cssd\teacher_converted\weighted_ce_r101_state_dict.pt"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "VT KD virtual environment not found. Run: & '$TrainingRoot\setup_vt_kd.ps1'"
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
        Invoke-Python (Join-Path $Scripts "audit_and_split.py") @(
            "--dataset-root", $Dataset,
            "--seed", "$Seed",
            "--val-ratio", "0.20",
            "--output", $Split
        )
    }
    "convert-teacher" {
        if (-not $AcknowledgeLegacyPickleRisk) {
            throw "The official legacy .pt is a Python pickle. Review the risk, then add -AcknowledgeLegacyPickleRisk."
        }
        Invoke-Python (Join-Path $Scripts "convert_teacher.py") @(
            "--config", $Config,
            "--input", $TeacherRaw,
            "--expected-sha256", "9110b28a7d027679076f831d2987d34be3be47883a9d724456810452a32e1dd5",
            "--architecture", "deeplabv3plus_resnet101",
            "--output", $TeacherConverted,
            "--allow-unsafe-official-pickle"
        )
    }
    { $_ -in @("smoke-supervised", "smoke-kd", "train-supervised", "train-kd") } {
        $Mode = if ($Task.EndsWith("-kd")) { "kd" } else { "supervised" }
        $Arguments = @("--config", $Config, "--mode", $Mode, "--seed", "$Seed")
        if ($Task.StartsWith("smoke-")) {
            $Arguments += @("--smoke-batches", "1")
        }
        if ($Epochs -gt 0) { $Arguments += @("--epochs", "$Epochs") }
        if ($Batch -gt 0) { $Arguments += @("--batch-size", "$Batch") }
        if ($Workers -ge 0) { $Arguments += @("--workers", "$Workers") }
        if ($RunName) { $Arguments += @("--run-name", $RunName) }
        if ($Resume) { $Arguments += @("--resume", (Resolve-Path -LiteralPath $Resume).Path) }
        if ($Mode -eq "kd") { $Arguments += @("--teacher", $TeacherConverted) }
        Invoke-Python (Join-Path $Scripts "train_student.py") $Arguments
    }
    { $_ -in @("evaluate-val", "evaluate-test") } {
        if (-not $Checkpoint) { throw "-$Task requires -Checkpoint <best.pt>." }
        $ResolvedCheckpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        $SplitName = if ($Task -eq "evaluate-test") { "test" } else { "val" }
        if (-not $Output) {
            $Output = Join-Path (Split-Path -Parent (Split-Path -Parent $ResolvedCheckpoint)) "evaluation-$SplitName"
        }
        $Arguments = @(
            "--config", $Config,
            "--checkpoint", $ResolvedCheckpoint,
            "--split", $SplitName,
            "--dataset-root", $Dataset,
            "--output", $Output
        )
        if ($Batch -gt 0) { $Arguments += @("--batch-size", "$Batch") }
        if ($Workers -ge 0) { $Arguments += @("--workers", "$Workers") }
        if ($AllowTestRerun) { $Arguments += "--allow-test-rerun" }
        Invoke-Python (Join-Path $Scripts "evaluate.py") $Arguments
    }
    "export-onnx" {
        if (-not $Checkpoint) { throw "export-onnx requires -Checkpoint <best.pt>." }
        $ResolvedCheckpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        if (-not $Output) {
            $Output = Join-Path $ProjectRoot "output\models\vt_kd\mnv2-deeplabv3plus-os8-fp32.onnx"
        }
        Invoke-Python (Join-Path $Scripts "export_onnx.py") @(
            "--config", $Config,
            "--checkpoint", $ResolvedCheckpoint,
            "--output", $Output,
            "--opset", "17",
            "--input-size", "512", "512"
        )
    }
}
