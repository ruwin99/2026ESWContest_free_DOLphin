[CmdletBinding()]
param(
    [ValidateSet("setup", "preflight", "download", "audit", "smoke8", "smoke16", "smoke32", "benchmark", "train", "resume", "val", "export-onnx")]
    [string]$Task = "audit",
    [ValidateRange(1, 500)][int]$Epochs = 50,
    [ValidateRange(1, 128)][int]$Batch = 16,
    [ValidateRange(0, 32)][int]$Workers = 8,
    [string]$RunName = "waste_yolo26n_640_b16_seed42",
    [string]$Checkpoint = "",
    [string]$DataRoot = "C:\rail_robot_cache\yolo26_obstacle\datasets\waste_detect"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path -LiteralPath (Join-Path $Project "..\..")).Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Yolo = Join-Path $Project ".venv\Scripts\yolo.exe"
$ModelRoot = Join-Path $Project "models\base"
$Model = Join-Path $ModelRoot "yolo26n.pt"
$Reports = Join-Path $Project "reports"
$YoloConfig = Join-Path $Project ".ultralytics"
$SourceDataYaml = Join-Path $DataRoot "data.yaml"
$DataYaml = Join-Path $DataRoot "data.local.yaml"
$Runs = Join-Path $Root "outputs\training\yolo26_obstacle"

if ($Task -eq "setup") {
    & (Join-Path $Project "setup_environment.ps1")
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $Python) -or -not (Test-Path -LiteralPath $Yolo)) {
    throw "YOLO26 environment missing. Run this script with -Task setup first."
}

New-Item -ItemType Directory -Force -Path $Reports,$Runs,$ModelRoot,$YoloConfig | Out-Null
$env:YOLO_CONFIG_DIR = $YoloConfig
$env:WANDB_DISABLED = "true"

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Task failed with exit code $LASTEXITCODE" }
}

function Invoke-Audit {
    Invoke-Checked $Python @(
        (Join-Path $Project "tools\prepare_local_yaml.py"),
        "--source", $SourceDataYaml,
        "--output", $DataYaml
    )
    Invoke-Checked $Python @(
        (Join-Path $Project "tools\audit_dataset.py"),
        "--data", $DataYaml,
        "--report", (Join-Path $Reports "dataset_audit.json")
    )
}

function Invoke-Smoke([int]$SmokeBatch) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Push-Location $ModelRoot
    try {
        Invoke-Checked $Yolo @(
            "detect", "train", "model=$Model", "data=$DataYaml", "epochs=1", "fraction=0.02",
            "imgsz=640", "batch=$SmokeBatch", "workers=$Workers", "device=0", "amp=True",
            "cache=False", "seed=42", "deterministic=False", "plots=False",
            "project=$Runs", "name=smoke_b${SmokeBatch}_$stamp"
        )
    }
    finally { Pop-Location }
}

switch ($Task) {
    "preflight" {
        Push-Location $ModelRoot
        try {
            Invoke-Checked $Python @(
                (Join-Path $Project "tools\preflight.py"), "--model", "yolo26n.pt",
                "--report", (Join-Path $Reports "preflight.json")
            )
        }
        finally { Pop-Location }
    }
    "download" {
        Invoke-Checked $Python @((Join-Path $Project "tools\download_dataset.py"), "--location", $DataRoot)
        Invoke-Audit
    }
    "audit" { Invoke-Audit }
    "smoke8" { Invoke-Audit; Invoke-Smoke 8 }
    "smoke16" { Invoke-Audit; Invoke-Smoke 16 }
    "smoke32" { Invoke-Audit; Invoke-Smoke 32 }
    "benchmark" { Invoke-Audit; Invoke-Smoke 8; Invoke-Smoke 16 }
    "train" {
        Invoke-Audit
        $RunDirectory = Join-Path $Runs $RunName
        if (Test-Path -LiteralPath $RunDirectory) {
            throw "Run already exists: $RunDirectory. Use -Task resume with its weights\last.pt or choose another -RunName."
        }
        Push-Location $ModelRoot
        try {
            Invoke-Checked $Yolo @(
                "detect", "train", "model=$Model", "data=$DataYaml", "epochs=$Epochs", "imgsz=640",
                "batch=$Batch", "workers=$Workers", "device=0", "amp=True", "cache=disk",
                "patience=15", "save_period=5", "seed=42", "deterministic=False", "plots=False",
                "project=$Runs", "name=$RunName"
            )
        }
        finally { Pop-Location }
    }
    "resume" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for resume" }
        if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint missing: $Checkpoint" }
        Push-Location $ModelRoot
        try {
            Invoke-Checked $Yolo @(
                "detect", "train", "resume=True", "model=$Checkpoint",
                "batch=$Batch", "workers=$Workers", "cache=disk"
            )
        }
        finally { Pop-Location }
    }
    "val" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for val" }
        Invoke-Audit
        Push-Location $ModelRoot
        try {
            Invoke-Checked $Yolo @(
                "detect", "val", "model=$Checkpoint", "data=$DataYaml", "imgsz=640",
                "batch=$Batch", "workers=$Workers", "device=0", "plots=True",
                "project=$Runs", "name=validation"
            )
        }
        finally { Pop-Location }
    }
    "export-onnx" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for export-onnx" }
        Push-Location $ModelRoot
        try {
            Invoke-Checked $Yolo @(
                "export", "model=$Checkpoint", "format=onnx", "imgsz=640", "batch=1",
                "half=False", "dynamic=False", "simplify=True", "opset=17", "device=0"
            )
        }
        finally { Pop-Location }
    }
}
