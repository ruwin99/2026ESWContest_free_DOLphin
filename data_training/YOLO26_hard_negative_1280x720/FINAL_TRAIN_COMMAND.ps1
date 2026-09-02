[CmdletBinding()]
param(
    [string]$ProjectRoot = $env:RAIL_ROBOT_ROOT,
    [string]$PythonExecutable = "",
    [string]$YoloExecutable = ""
)

$ErrorActionPreference = "Stop"
$Archive = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $Archive "run_hardneg.ps1"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    throw "Set -ProjectRoot or RAIL_ROBOT_ROOT to the rail_robot project root."
}

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Original training runner is missing: $Runner"
}

# Exact final training parameters used for the released candidate.
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File $Runner `
    -Task train `
    -Batch 8 `
    -Workers 2 `
    -ProjectRoot $ProjectRoot `
    -PythonExecutable $PythonExecutable `
    -YoloExecutable $YoloExecutable

if ($LASTEXITCODE -ne 0) {
    throw "Final training command failed with exit code $LASTEXITCODE"
}
