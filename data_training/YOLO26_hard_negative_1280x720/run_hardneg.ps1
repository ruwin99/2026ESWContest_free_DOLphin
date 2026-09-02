[CmdletBinding()]
param(
    [ValidateSet("audit", "approve-reviewed", "prepare", "prepare-dedup", "prepare-all", "install-baseline", "preflight", "smoke", "train", "train-all-smoke", "train-all", "train-safe", "train-cls-smoke", "train-cls-safe")]
    [string]$Task = "audit",
    [ValidateRange(1, 128)][int]$Batch = 8,
    [ValidateRange(0, 32)][int]$Workers = 2,
    [switch]$AcceptExactDedupPolicy,
    [string]$BaselineSource = "",
    [string]$ProjectRoot = "",
    [string]$PythonExecutable = "",
    [string]$YoloExecutable = ""
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = $env:RAIL_ROBOT_ROOT
}
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Join-Path $Project "..\.."
}
$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Config = Join-Path $Project "configs\hardneg_1280x720_v1.yaml"
$Python = if ([string]::IsNullOrWhiteSpace($PythonExecutable)) { Join-Path $Root "data_training\yolo26_obstacle\.venv\Scripts\python.exe" } else { $PythonExecutable }
$Yolo = if ([string]::IsNullOrWhiteSpace($YoloExecutable)) { Join-Path $Root "data_training\yolo26_obstacle\.venv\Scripts\yolo.exe" } else { $YoloExecutable }
$Workspace = Join-Path $Project "workspace"
$DataYaml = Join-Path $Workspace "datasets\waste_detect_hn_v8_1280_seed42\data.yaml"
$DataYamlAll = Join-Path $Workspace "datasets\waste_detect_hn_all1410_v2_1280_seed42\data.yaml"
$Baseline = Join-Path $Project "models\baseline\best.pt"
$ExpectedBaselineSha256 = "333c49d129698e2ed781a6862b12224ea3eafe973b582322bffc6ed8573087bd"
$Outputs = Join-Path $Root "outputs\training\yolo26_obstacle_hardneg_v1"
$Reports = Join-Path $Project "reports"
$Tool = Join-Path $Project "tools\prepare_hardneg.py"
$YoloConfig = Join-Path $Project ".ultralytics"

if (-not (Test-Path -LiteralPath $Python) -or -not (Test-Path -LiteralPath $Yolo)) {
    throw "Existing pinned YOLO environment is missing under data_training\yolo26_obstacle\.venv"
}
New-Item -ItemType Directory -Force -Path $Reports,$Outputs,$YoloConfig | Out-Null
$env:YOLO_CONFIG_DIR = $YoloConfig
$env:WANDB_DISABLED = "true"
$env:RAIL_ROBOT_ROOT = $Root
Set-Location -LiteralPath $Project

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Task failed with exit code $LASTEXITCODE" }
}

switch ($Task) {
    "audit" {
        Invoke-Checked $Python @($Tool, "audit", "--config", $Config)
    }
    "approve-reviewed" {
        Invoke-Checked $Python @($Tool, "audit", "--config", $Config, "--approve-all-reviewed")
    }
    "prepare" {
        Invoke-Checked $Python @($Tool, "prepare", "--config", $Config)
    }
    "prepare-dedup" {
        if (-not $AcceptExactDedupPolicy) {
            throw "-AcceptExactDedupPolicy is required. Policy: keep one exact-SHA copy globally with test > valid > train precedence."
        }
        Invoke-Checked $Python @($Tool, "prepare", "--config", $Config, "--accept-exact-dedup-policy")
    }
    "prepare-all" {
        Invoke-Checked $Python @((Join-Path $Project "tools\prepare_all_hardneg.py"))
    }
    "install-baseline" {
        if ([string]::IsNullOrWhiteSpace($BaselineSource)) {
            throw "-BaselineSource with the full path to the original best.pt is required."
        }
        if (-not (Test-Path -LiteralPath $BaselineSource -PathType Leaf)) {
            throw "Baseline source missing: $BaselineSource"
        }
        $SourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $BaselineSource).Hash.ToLowerInvariant()
        if ($SourceHash -ne $ExpectedBaselineSha256) {
            throw "Baseline SHA mismatch. Expected $ExpectedBaselineSha256, actual $SourceHash"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Baseline) | Out-Null
        if (Test-Path -LiteralPath $Baseline) {
            $DestinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Baseline).Hash.ToLowerInvariant()
            if ($DestinationHash -ne $ExpectedBaselineSha256) {
                throw "Refusing to overwrite an unexpected destination model: $Baseline"
            }
            Write-Output "Baseline already installed and verified: $Baseline"
        }
        else {
            Copy-Item -LiteralPath $BaselineSource -Destination $Baseline
            $DestinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Baseline).Hash.ToLowerInvariant()
            if ($DestinationHash -ne $ExpectedBaselineSha256) {
                throw "Copied baseline failed SHA verification: $DestinationHash"
            }
            Write-Output "Baseline installed and verified: $Baseline"
        }
    }
    "preflight" {
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\preflight.py"),
            "--config", $Config,
            "--report", (Join-Path $Reports "preflight.json")
        )
    }
    "smoke" {
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\preflight.py"),
            "--config", $Config,
            "--report", (Join-Path $Reports "preflight.json")
        )
        $RunName = "yolo26n_hn_v8_1280rect_smoke_seed42_b4_v1"
        $RunDir = Join-Path $Outputs $RunName
        if (Test-Path -LiteralPath $RunDir) { throw "Refusing automatic suffix; run exists: $RunDir" }
        Invoke-Checked $Yolo @(
            "detect", "train", "model=$Baseline", "data=$DataYaml", "epochs=1", "fraction=0.02",
            "imgsz=1280", "rect=True", "batch=4", "workers=$Workers", "device=0",
            "amp=True", "cache=False", "seed=42", "deterministic=True", "plots=False",
            "project=$Outputs", "name=$RunName", "resume=False"
        )
    }
    "train" {
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\preflight.py"),
            "--config", $Config,
            "--report", (Join-Path $Reports "preflight.json")
        )
        if (-not (Test-Path -LiteralPath $DataYamlAll -PathType Leaf)) {
            throw "All-hard-negative dataset missing. Run -Task prepare-all first."
        }
        $RunName = "yolo26n_hn_all1410_1280rect_clsout_lr2e5_seed42_b${Batch}_v2"
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\train_cls_output_safe.py"),
            "--baseline", $Baseline, "--data", $DataYamlAll, "--project", $Outputs,
            "--name", $RunName, "--epochs", "3", "--batch", "$Batch", "--workers", "$Workers",
            "--lr0", "0.00002", "--audit-report", (Join-Path $Reports "all1410_train_frozen_audit.json")
        )
    }
    "train-safe" {
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\preflight.py"),
            "--config", $Config,
            "--report", (Join-Path $Reports "preflight.json")
        )
        $RunName = "yolo26n_hn_v8_1280rect_headonly_lr2e5_seed42_b${Batch}_v2"
        $RunDir = Join-Path $Outputs $RunName
        if (Test-Path -LiteralPath $RunDir) { throw "Refusing automatic suffix; run exists: $RunDir" }
        Invoke-Checked $Yolo @(
            "detect", "train", "model=$Baseline", "data=$DataYaml", "epochs=10", "imgsz=1280",
            "rect=True", "batch=$Batch", "workers=$Workers", "device=0", "amp=True", "cache=disk",
            "patience=5", "save_period=1", "optimizer=AdamW", "lr0=0.00002", "lrf=0.1",
            "warmup_epochs=0.0", "warmup_bias_lr=0.0", "freeze=23", "mosaic=0.0", "close_mosaic=0",
            "seed=42", "deterministic=True", "plots=True", "project=$Outputs", "name=$RunName", "resume=False"
        )
    }
    "train-cls-smoke" {
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\preflight.py"),
            "--config", $Config,
            "--report", (Join-Path $Reports "preflight.json")
        )
        $RunName = "yolo26n_hn_v8_1280rect_clsout_lr2e6_seed42_b${Batch}_smoke_v3"
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\train_cls_output_safe.py"),
            "--baseline", $Baseline, "--data", $DataYaml, "--project", $Outputs,
            "--name", $RunName, "--epochs", "1", "--batch", "$Batch", "--workers", "$Workers",
            "--lr0", "0.000002", "--audit-report", (Join-Path $Reports "safe_cls_smoke_audit.json")
        )
    }
    "train-cls-safe" {
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\preflight.py"),
            "--config", $Config,
            "--report", (Join-Path $Reports "preflight.json")
        )
        $RunName = "yolo26n_hn_v8_1280rect_clsout_lr2e5_seed42_b${Batch}_v4"
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\train_cls_output_safe.py"),
            "--baseline", $Baseline, "--data", $DataYaml, "--project", $Outputs,
            "--name", $RunName, "--epochs", "3", "--batch", "$Batch", "--workers", "$Workers",
            "--lr0", "0.00002", "--audit-report", (Join-Path $Reports "safe_cls_train_audit.json")
        )
    }
    "train-all-smoke" {
        if (-not (Test-Path -LiteralPath $DataYamlAll -PathType Leaf)) {
            throw "All-hard-negative dataset missing. Run -Task prepare-all first."
        }
        $RunName = "yolo26n_hn_all1410_1280rect_clsout_lr2e5_seed42_b${Batch}_smoke_v2"
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\train_cls_output_safe.py"),
            "--baseline", $Baseline, "--data", $DataYamlAll, "--project", $Outputs,
            "--name", $RunName, "--epochs", "1", "--batch", "$Batch", "--workers", "$Workers",
            "--lr0", "0.00002", "--audit-report", (Join-Path $Reports "all1410_smoke_frozen_audit.json")
        )
    }
    "train-all" {
        if (-not (Test-Path -LiteralPath $DataYamlAll -PathType Leaf)) {
            throw "All-hard-negative dataset missing. Run -Task prepare-all first."
        }
        $RunName = "yolo26n_hn_all1410_1280rect_clsout_lr2e5_seed42_b${Batch}_v2"
        Invoke-Checked $Python @(
            (Join-Path $Project "tools\train_cls_output_safe.py"),
            "--baseline", $Baseline, "--data", $DataYamlAll, "--project", $Outputs,
            "--name", $RunName, "--epochs", "5", "--batch", "$Batch", "--workers", "$Workers",
            "--lr0", "0.00002", "--audit-report", (Join-Path $Reports "all1410_train_frozen_audit.json")
        )
    }
}

