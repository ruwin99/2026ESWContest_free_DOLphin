[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("inventory", "approve-good-all", "approve-overlay", "prepare-sealed", "build-manifests", "audit", "lock", "smoke-structure", "smoke-data", "train-stage-a", "train-stage-b", "evaluate-validation", "search-postprocess", "evaluate-sealed", "evaluate-v8-confirmation", "export-onnx")]
    [string]$Task,
    [int]$Seed = 42,
    [string]$RunName = "",
    [string]$Checkpoint = "",
    [string]$Resume = "",
    [string]$Output = "",
    [string]$SealedSource = "",
    [ValidateSet("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7b")]
    [string]$Profile = "v1",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$WorkRoot = $PSScriptRoot
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $WorkRoot "..\..")).Path
$Python = Join-Path $ProjectRoot "data_training\.venv\Scripts\python.exe"
$Tools = Join-Path $WorkRoot "tools"
$ConfigName = switch ($Profile) {
    "v2" { "hardneg_r101_os8_stagea_v2.yaml" }
    "v3" { "hardneg_r101_os8_stagea_v3.yaml" }
    "v4" { "hardneg_r101_os8_stagea_v4.yaml" }
    "v5" { "hardneg_r101_os8_stagea_v5.yaml" }
    "v6" { "hardneg_r101_os8_stagea_v6.yaml" }
    "v7" { "hardneg_r101_os8_stagea_v7_refit.yaml" }
    "v7b" { "hardneg_r101_os8_stagea_v7b_canonical_lr1e6.yaml" }
    default { "hardneg_r101_os8.yaml" }
}
$Config = Join-Path $WorkRoot (Join-Path "configs" $ConfigName)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}

function Invoke-Tool([string]$Script, [string[]]$Arguments) {
    & $Python (Join-Path $Tools $Script) @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Task failed with exit code $LASTEXITCODE" }
}

switch ($Task) {
    "inventory" {
        $Args = @("--config", $Config, "--mode", "inventory")
        if ($Force) { $Args += "--force" }
        Invoke-Tool "prepare_manifests.py" $Args
    }
    "approve-good-all" {
        Invoke-Tool "approve_good.py" @("--config", $Config, "--approved-by", "team-reviewer")
    }
    "approve-overlay" {
        Invoke-Tool "record_overlay_approval.py" @("--config", $Config, "--approved-by", "team-reviewer")
    }
    "prepare-sealed" {
        if (-not $SealedSource) { $SealedSource = Join-Path $ProjectRoot "for model test v8" }
        Invoke-Tool "prepare_sealed.py" @("--config", $Config, "--source", $SealedSource, "--approved-by", "team-reviewer")
    }
    "build-manifests" {
        $Args = @("--config", $Config, "--mode", "build")
        if ($Force) { $Args += "--force" }
        Invoke-Tool "prepare_manifests.py" $Args
    }
    "audit" { Invoke-Tool "audit.py" @("--config", $Config) }
    "lock" { Invoke-Tool "audit.py" @("--config", $Config, "--lock") }
    "smoke-structure" { Invoke-Tool "smoke_test.py" @("--config", $Config, "--mode", "structure") }
    "smoke-data" { Invoke-Tool "smoke_test.py" @("--config", $Config, "--mode", "data", "--seed", "$Seed") }
    "train-stage-a" {
        if (-not $RunName) { $RunName = "r101-os8-hardneg-stage-a-seed$Seed" }
        $Args = @("--config", $Config, "--stage", "a", "--seed", "$Seed", "--run-name", $RunName)
        if ($Resume) { $Args += @("--resume", $Resume) }
        Invoke-Tool "train.py" $Args
    }
    "train-stage-b" {
        if (-not $Resume) { throw "train-stage-b requires -Resume <Stage A best.pt>." }
        if (-not $RunName) { $RunName = "r101-os8-hardneg-stage-b-seed$Seed" }
        Invoke-Tool "train.py" @("--config", $Config, "--stage", "b", "--seed", "$Seed", "--run-name", $RunName, "--resume", $Resume)
    }
    "evaluate-validation" {
        if (-not $Checkpoint) { throw "evaluate-validation requires -Checkpoint <best.pt>." }
        if (-not $Output) { $Output = [IO.Path]::ChangeExtension($Checkpoint, ".validation.json") }
        Invoke-Tool "evaluate.py" @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
    "search-postprocess" {
        if (-not $Checkpoint) { throw "search-postprocess requires -Checkpoint <validation-selected best.pt>." }
        if (-not $Output) { $Output = [IO.Path]::ChangeExtension($Checkpoint, ".postprocess-search.json") }
        Invoke-Tool "search_postprocess.py" @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
    "evaluate-sealed" {
        if (-not $Checkpoint) { throw "evaluate-sealed requires -Checkpoint <validation-selected best.pt>." }
        if (-not $Output) { $Output = [IO.Path]::ChangeExtension($Checkpoint, ".sealed-v8.json") }
        Invoke-Tool "evaluate_sealed.py" @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
    "evaluate-v8-confirmation" {
        if (-not $Checkpoint) { throw "evaluate-v8-confirmation requires -Checkpoint <v7-refit best.pt>." }
        if (-not $Output) { $Output = [IO.Path]::ChangeExtension($Checkpoint, ".v8-confirmation.json") }
        Invoke-Tool "evaluate_v8_confirmation.py" @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
    "export-onnx" {
        if (-not $Checkpoint) { throw "export-onnx requires -Checkpoint <validation-selected best.pt>." }
        if (-not $Output) {
            $Output = Join-Path $ProjectRoot "output\models\capture_rust_teacher_hardneg_v1\capture-rust-r101-os8-hardneg-candidate-NOT_FOR_UART-fp32.onnx"
        }
        Invoke-Tool "export_onnx.py" @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
}
