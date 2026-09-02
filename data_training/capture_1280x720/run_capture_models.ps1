[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "verify", "audit", "smoke-corrosion", "smoke-crack",
        "baseline-corrosion", "baseline-crack",
        "regression-corrosion-val", "regression-corrosion-test",
        "regression-crack-val", "regression-crack-test",
        "finetune-corrosion", "finetune-crack", "train-corrosion", "train-crack",
        "evaluate-corrosion-val", "evaluate-corrosion-test",
        "evaluate-crack-val", "evaluate-crack-test",
        "export-corrosion-onnx", "export-crack-onnx"
    )]
    [string]$Task,
    [string]$Checkpoint = "",
    [string]$Resume = "",
    [string]$Output = "",
    [string]$RunName = "",
    [int]$Epochs = 30,
    [int]$Batch = 1,
    [int]$Accumulate = 4,
    [int]$Workers = 2,
    [double]$LearningRate = 0.0,
    [int]$Seed = 42,
    [ValidateSet("zero", "reflect", "replicate")]
    [string]$PaddingPolicy = "reflect",
    [switch]$AllowTestRerun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$CaptureRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $CaptureRoot "..\..")).Path
$Scripts = Join-Path $CaptureRoot "scripts"
$CorrosionPython = Join-Path $ProjectRoot "data_training\vt_kd\.venv\Scripts\python.exe"
$CrackPython = Join-Path $ProjectRoot "data_training\steelcrack\.venv\Scripts\python.exe"
$RunsRoot = Join-Path $ProjectRoot "outputs\training\capture_1280x720"

function Invoke-Python([string]$Python, [string]$Script, [string[]]$Arguments) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Required Python environment not found: $Python"
    }
    & $Python $Script @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

function Get-DefaultCheckpoint([string]$Model) {
    if ($Model -eq "corrosion") {
        return Join-Path $ProjectRoot "models\virginia_tech_cssd\teacher_converted\weighted_ce_r101_state_dict.pt"
    }
    return Join-Path $ProjectRoot "outputs\training\steelcrack\bgcrack-seed42\weights\best.pth"
}

function Assert-CaptureData([string]$Model, [string[]]$Splits) {
    $TaskFolder = if ($Model -eq "corrosion") { "rust" } else { "crack" }
    $Kinds = if ($Model -eq "corrosion") { @("images", "masks") } else { @("images", "masks", "edges") }
    $Extensions = @(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    $Problems = @()
    foreach ($Split in $Splits) {
        $Counts = @{}
        foreach ($Kind in $Kinds) {
            $Folder = Join-Path $CaptureRoot "$TaskFolder\$Split\$Kind"
            $Count = @(
                Get-ChildItem -LiteralPath $Folder -File -ErrorAction SilentlyContinue |
                    Where-Object { $Extensions -contains $_.Extension.ToLowerInvariant() }
            ).Count
            $Counts[$Kind] = $Count
            if ($Count -eq 0) { $Problems += "$TaskFolder/$Split/$Kind has 0 files: $Folder" }
        }
        $UniqueCounts = @($Counts.Values | Sort-Object -Unique)
        if ($UniqueCounts.Count -gt 1) {
            $Summary = ($Kinds | ForEach-Object { "$_=$($Counts[$_])" }) -join ", "
            $Problems += "$TaskFolder/$Split file counts do not match: $Summary"
        }
    }
    if ($Problems.Count -gt 0) {
        $Details = $Problems -join [Environment]::NewLine
        throw "Capture dataset is not ready. Add real 1280x720 files before running $Task.$([Environment]::NewLine)$Details$([Environment]::NewLine)Then run: -Task audit"
    }
}

if ($Task -eq "verify") {
    Write-Host "Corrosion environment: $CorrosionPython"
    Invoke-Python $CorrosionPython (Join-Path $ProjectRoot "data_training\vt_kd\scripts\verify_gpu.py") @()
    Write-Host "Crack environment: $CrackPython"
    Invoke-Python $CrackPython (Join-Path $ProjectRoot "data_training\steelcrack\scripts\verify_gpu.py") @()
    exit 0
}

if ($Task -eq "audit") {
    Invoke-Python $CrackPython (Join-Path $Scripts "audit_dataset.py") @("--root", $CaptureRoot)
    exit 0
}

$Model = if ($Task -match "corrosion") { "corrosion" } else { "crack" }
$Python = if ($Model -eq "corrosion") { $CorrosionPython } else { $CrackPython }
$EffectiveCheckpoint = if ($Checkpoint) { (Resolve-Path -LiteralPath $Checkpoint).Path } else { Get-DefaultCheckpoint $Model }

function Get-OnnxOutput([string]$Model) {
    $FileName = if ($Model -eq "corrosion") {
        "corrosion-capture-r101-os8-w1280-h720-fp32.onnx"
    } else {
        "crack-capture-bgcrack-ext-w1280-h720-int-w1280-h736-fp32.onnx"
    }
    return Join-Path $ProjectRoot "output\models\capture_1280x720\$FileName"
}

function Invoke-PublicRegression([string]$Model, [string]$Split) {
    $RegressionOutput = Join-Path $CaptureRoot "manifests\public_regression_$Model`_$Split`_$PaddingPolicy.json"
    $Arguments = @(
        "--model", $Model,
        "--split", $Split,
        "--checkpoint", $EffectiveCheckpoint,
        "--output", $RegressionOutput,
        "--padding-policy", $PaddingPolicy
    )
    if ($AllowTestRerun) { $Arguments += "--allow-test-rerun" }
    Invoke-Python $Python (Join-Path $Scripts "regression_public.py") $Arguments
}

if ($Task -match "^baseline-") {
    Invoke-Python $Python (Join-Path $Scripts "smoke_models.py") @(
        "--model", $Model, "--checkpoint", $EffectiveCheckpoint
    )
    Invoke-Python $Python (Join-Path $Scripts "export_onnx.py") @(
        "--model", $Model,
        "--checkpoint", $EffectiveCheckpoint,
        "--output", (Get-OnnxOutput $Model),
        "--opset", "18"
    )
    Invoke-PublicRegression $Model "validation"
    exit 0
}

if ($Task -match "^regression-") {
    $Split = if ($Task -match "-test$") { "test" } else { "validation" }
    Invoke-PublicRegression $Model $Split
    exit 0
}

if ($Task -match "^smoke-") {
    Invoke-Python $Python (Join-Path $Scripts "smoke_models.py") @(
        "--model", $Model, "--checkpoint", $EffectiveCheckpoint
    )
    exit 0
}

if ($Task -match "^(train|finetune)-") {
    Assert-CaptureData $Model @("train", "validation")
    if (-not $RunName) { $RunName = "$Model-capture-seed$Seed" }
    $Arguments = @(
        "--model", $Model,
        "--data-root", $CaptureRoot,
        "--initial-checkpoint", $EffectiveCheckpoint,
        "--runs-root", $RunsRoot,
        "--run-name", $RunName,
        "--epochs", "$Epochs",
        "--batch-size", "$Batch",
        "--accumulate", "$Accumulate",
        "--workers", "$Workers",
        "--seed", "$Seed"
    )
    if ($LearningRate -gt 0) { $Arguments += @("--lr", "$LearningRate") }
    if ($Resume) { $Arguments += @("--resume", (Resolve-Path -LiteralPath $Resume).Path) }
    Invoke-Python $Python (Join-Path $Scripts "train_capture.py") $Arguments
    exit 0
}

if ($Task -match "^evaluate-") {
    if (-not $Checkpoint) { throw "$Task requires -Checkpoint <capture best.pt>." }
    $Split = if ($Task -match "-test$") { "test" } else { "validation" }
    Assert-CaptureData $Model @($Split)
    if (-not $Output) {
        $Output = Join-Path (Split-Path -Parent (Split-Path -Parent $EffectiveCheckpoint)) "evaluation-$Split.json"
    }
    $Arguments = @(
        "--model", $Model,
        "--checkpoint", $EffectiveCheckpoint,
        "--split", $Split,
        "--data-root", $CaptureRoot,
        "--output", $Output,
        "--workers", "$Workers"
    )
    if ($AllowTestRerun) { $Arguments += "--allow-test-rerun" }
    Invoke-Python $Python (Join-Path $Scripts "evaluate_capture.py") $Arguments
    exit 0
}

if ($Task -match "^export-") {
    if (-not $Output) {
        $Output = Get-OnnxOutput $Model
    }
    Invoke-Python $Python (Join-Path $Scripts "export_onnx.py") @(
        "--model", $Model,
        "--checkpoint", $EffectiveCheckpoint,
        "--output", $Output,
        "--opset", "18"
    )
    exit 0
}
