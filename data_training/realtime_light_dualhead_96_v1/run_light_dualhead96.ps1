param(
    [ValidateSet("setup", "prepare", "audit", "preflight", "test", "cache-smoke", "phase-a-smoke", "cache-train", "cache-validation", "train", "resume", "evaluate", "export", "verify", "complexity", "build-wrapper")]
    [string]$Task = "audit",
    [ValidateSet(17, 29, 43)][int]$Seed = 17,
    [ValidateSet("1", "2", "3", "4", "all")][string]$Stage = "all",
    [string]$Checkpoint = "",
    [string]$Onnx = "",
    [ValidateSet("development_calibration", "validation")][string]$Split = "validation",
    [ValidateRange(0, 128)][int]$Batch = 0,
    [ValidateRange(0, 64)][int]$Workers = 0,
    [ValidateRange(0, 128)][int]$Accumulate = 0,
    [switch]$AllowBlocked,
    [switch]$AllowCpuCache
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path -LiteralPath (Join-Path $Project "..\..")).Path
$Config = Join-Path $Project "configs\light_dualhead_96_w1280_h240.yaml"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$FallbackPython = Join-Path $Root "data_training\realtime_multitask_5ch_hrseg_v1\.venv\Scripts\python.exe"

if ($Task -eq "setup") {
    & (Join-Path $Project "setup_environment.ps1")
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $Python)) {
    if (Test-Path -LiteralPath $FallbackPython) {
        $Python = $FallbackPython
        Write-Warning "Using the verified existing GPU environment. Run -Task setup to create the isolated environment."
    } else {
        throw "Python environment missing. Run this script with -Task setup first."
    }
}

function Invoke-Checked([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Task failed with exit code $LASTEXITCODE" }
}

function Add-TrainingOverrides([string[]]$Arguments) {
    if ($Batch -gt 0) { $Arguments += @("--batch-size", "$Batch") }
    if ($Workers -gt 0) { $Arguments += @("--workers", "$Workers") }
    if ($Accumulate -gt 0) { $Arguments += @("--accumulation", "$Accumulate") }
    return $Arguments
}

switch ($Task) {
    "prepare" { Invoke-Checked @((Join-Path $Project "tools\prepare_phase_a.py"), "--config", $Config) }
    "audit" {
        $Arguments = @((Join-Path $Project "tools\audit_split.py"), "--config", $Config)
        if ($AllowBlocked) { $Arguments += "--allow-blocked" }
        Invoke-Checked $Arguments
    }
    "preflight" { Invoke-Checked @((Join-Path $Project "tools\preflight.py"), "--config", $Config) }
    "test" { Invoke-Checked @("-m", "pytest", (Join-Path $Project "tests"), "-q") }
    "cache-smoke" { Invoke-Checked @((Join-Path $Project "tools\cache_teachers.py"), "--config", $Config, "--split", "train", "--smoke-only") }
    "phase-a-smoke" { Invoke-Checked @((Join-Path $Project "tools\phase_a_smoke.py"), "--config", $Config) }
    "cache-train" {
        $Arguments = @((Join-Path $Project "tools\cache_teachers.py"), "--config", $Config, "--split", "train")
        if ($AllowCpuCache) { $Arguments += "--allow-cpu" }
        Invoke-Checked $Arguments
    }
    "cache-validation" {
        $Arguments = @((Join-Path $Project "tools\cache_teachers.py"), "--config", $Config, "--split", "validation")
        if ($AllowCpuCache) { $Arguments += "--allow-cpu" }
        Invoke-Checked $Arguments
    }
    "train" {
        $Arguments = @((Join-Path $Project "tools\train.py"), "--config", $Config, "--seed", "$Seed", "--stage", $Stage)
        Invoke-Checked (Add-TrainingOverrides $Arguments)
    }
    "resume" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for resume" }
        $Arguments = @((Join-Path $Project "tools\train.py"), "--config", $Config, "--seed", "$Seed", "--stage", $Stage, "--resume", $Checkpoint)
        Invoke-Checked (Add-TrainingOverrides $Arguments)
    }
    "evaluate" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for evaluate" }
        Invoke-Checked @((Join-Path $Project "tools\evaluate.py"), "--config", $Config, "--split", $Split, "--checkpoint", $Checkpoint)
    }
    "export" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for export" }
        if (-not $Onnx) { $Onnx = Join-Path $Project "exports\realtime-rust-crack-light-dualhead96-os8-w1280-h240-fp32.onnx" }
        Invoke-Checked @((Join-Path $Project "tools\export_onnx.py"), "--config", $Config, "--checkpoint", $Checkpoint, "--output", $Onnx)
    }
    "verify" {
        if (-not $Checkpoint -or -not $Onnx) { throw "-Checkpoint and -Onnx are required for verify" }
        Invoke-Checked @((Join-Path $Project "tools\verify_onnx.py"), "--config", $Config, "--checkpoint", $Checkpoint, "--onnx", $Onnx)
    }
    "complexity" {
        if (-not $Onnx) { throw "-Onnx is required for complexity" }
        Invoke-Checked @((Join-Path $Project "tools\verify_model_complexity.py"), "--onnx", $Onnx)
    }
    "build-wrapper" {
        if (-not $Onnx) { $Onnx = Join-Path $Project "exports\light-dualhead96-gpu-prefilter.onnx" }
        Invoke-Checked @((Join-Path $Project "tools\build_gpu_postprocess_wrapper.py"), "--output", $Onnx)
    }
}
