param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("prepare-crackseg", "build-crackseg-panels", "audit", "cache", "train", "evaluate", "search-postprocess", "evaluate-demo-release", "export", "structure-export", "verify")]
    [string]$Task,
    [string]$Config = "",
    [string]$Stage = "crack_bootstrap",
    [string]$RunName = "",
    [string]$Checkpoint = "",
    [string]$Initialize = "",
    [string]$Resume = "",
    [string]$Output = "",
    [string]$PostprocessReport = "",
    [string]$EvaluationReport = "",
    [string]$Image = "",
    [string]$DownloadDir = "",
    [string]$ExtractDir = "",
    [int]$Epochs = 0,
    [int]$Batch = 0,
    [int]$Accumulate = 0,
    [int]$Workers = -1,
    [int]$Seed = -1,
    [ValidateSet("cuda", "cpu")]
    [string]$Provider = "cuda",
    [switch]$OverwriteCache,
    [switch]$AllowBlocked,
    [switch]$RunInference
)

$ErrorActionPreference = "Stop"
$Work = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $Work "..\..")).Path
$Python = Join-Path $Work ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Fallback = Join-Path $Root "data_training\vt_kd\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $Fallback) { $Python = $Fallback }
    else { throw "Python environment missing. Run setup_environment.ps1 -Install first." }
}
if (-not $Config) { $Config = Join-Path $Work "configs\mnv2_os8_dualhead_w1280_h240.yaml" }

$Arguments = @()
switch ($Task) {
    "prepare-crackseg" {
        $Script = Join-Path $Work "tools\prepare_crackseg9k_v4.py"
        $Arguments = @("--extract")
        if ($DownloadDir) { $Arguments += @("--download-dir", $DownloadDir) }
        if ($ExtractDir) { $Arguments += @("--extract-dir", $ExtractDir) }
    }
    "build-crackseg-panels" {
        $Script = Join-Path $Work "tools\build_crackseg9k_panels.py"
        $Arguments = @()
    }
    "audit" {
        $Script = Join-Path $Work "tools\prepare_manifest.py"
        $Arguments = @("--config", $Config)
        if ($AllowBlocked) { $Arguments += "--allow-blocked" }
    }
    "cache" {
        $Script = Join-Path $Work "tools\cache_teachers.py"
        $Arguments = @("--config", $Config, "--provider", $Provider)
        if ($OverwriteCache) { $Arguments += "--overwrite" }
    }
    "train" {
        if (-not $RunName) { throw "-RunName is required for train" }
        $Script = Join-Path $Work "tools\train_multitask.py"
        $Arguments = @("--config", $Config, "--stage", $Stage, "--run-name", $RunName)
        if ($Initialize) { $Arguments += @("--initialize", $Initialize) }
        if ($Resume) { $Arguments += @("--resume", $Resume) }
        if ($Epochs -gt 0) { $Arguments += @("--epochs", "$Epochs") }
        if ($Batch -gt 0) { $Arguments += @("--batch", "$Batch") }
        if ($Accumulate -gt 0) { $Arguments += @("--accumulate", "$Accumulate") }
        if ($Workers -ge 0) { $Arguments += @("--workers", "$Workers") }
        if ($Seed -ge 0) { $Arguments += @("--seed", "$Seed") }
    }
    "evaluate" {
        if (-not $Checkpoint -or -not $Output) { throw "-Checkpoint and -Output are required" }
        $Script = Join-Path $Work "tools\evaluate_multitask.py"
        $Arguments = @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
    "search-postprocess" {
        if (-not $Checkpoint -or -not $Output) { throw "-Checkpoint and -Output are required" }
        $Script = Join-Path $Work "tools\search_postprocess.py"
        $Arguments = @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
    }
    "evaluate-demo-release" {
        if (-not $Checkpoint -or -not $PostprocessReport -or -not $Output) {
            throw "-Checkpoint, -PostprocessReport and -Output are required"
        }
        $Script = Join-Path $Work "tools\evaluate_demo_release.py"
        $Arguments = @(
            "--config", $Config,
            "--checkpoint", $Checkpoint,
            "--positive-report", $PostprocessReport,
            "--output", $Output
        )
    }
    "export" {
        if (-not $Checkpoint -or -not $Output) { throw "-Checkpoint and -Output are required" }
        $Script = Join-Path $Work "tools\export_onnx.py"
        $Arguments = @("--config", $Config, "--checkpoint", $Checkpoint, "--output", $Output)
        if ($EvaluationReport) { $Arguments += @("--evaluation-report", $EvaluationReport) }
    }
    "structure-export" {
        if (-not $Output) { $Output = Join-Path $Work "exports\STRUCTURE_ONLY-mnv2-os8-dualhead-w1280-h240.onnx" }
        $Script = Join-Path $Work "tools\export_onnx.py"
        $Arguments = @("--config", $Config, "--structure-only", "--output", $Output)
    }
    "verify" {
        if (-not $Output) { throw "Pass the ONNX path with -Output" }
        $Script = Join-Path $Work "tools\verify_onnx.py"
        $Arguments = @("--onnx", $Output)
        if ($RunInference) { $Arguments += "--run-inference" }
        if ($Checkpoint) { $Arguments += @("--config", $Config, "--checkpoint", $Checkpoint) }
        if ($Image) { $Arguments += @("--image", $Image) }
    }
}

& $Python $Script @Arguments
if ($LASTEXITCODE -ne 0) { throw "$Task failed with exit code $LASTEXITCODE" }
