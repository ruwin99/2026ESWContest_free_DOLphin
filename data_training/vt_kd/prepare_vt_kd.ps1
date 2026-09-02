[CmdletBinding()]
param(
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$DataRoot = Join-Path $ProjectRoot "data_training\data\raw\virginia_tech_cssd"
$ModelRoot = Join-Path $ProjectRoot "models\virginia_tech_cssd"
$DataZip = Join-Path $DataRoot "_downloads\Corrosion Condition State Classification.zip"
$ModelZip = Join-Path $ModelRoot "_downloads\Corrosion Condition State Classification - Trained Model.zip"
$Dataset = Join-Path $DataRoot "dataset"
$TeacherRaw = Join-Path $ModelRoot "teacher_raw"
$OfficialCode = Join-Path $ModelRoot "official_code"
$MobileNetWeight = Join-Path $ModelRoot "torch_cache\hub\checkpoints\mobilenet_v2-b0353104.pth"

$Expected = @{
    DataZip = "45F0EC8B26F1C09D707F3010AF359A28E0985D385D2BF6D98B5D4DD308E9DBE5"
    ModelZip = "AD394E3728EFA239A9647C20AB607B4ED97FE963E5373D3D331225C4F355D208"
    MobileNet = "B03531047FFACF1E2488318DCD2ABA1126CDE36E3BFE1AA5CB07700AEEEE9889"
    TeacherL1 = "328392A3B86D2037BCD2EC5D9C11D473B170B57BFF638640E7D4A96971C0024D"
    TeacherL2 = "07278842FB12931FA5ACA2ED5573FE2D2D7529BE8150DDECF65BC31F9A1EA477"
    TeacherResNet50 = "59B294BDA1F8F475E14F30015C8BDC4BA81A08A837B37243E9A541AA3B5CA382"
    TeacherWeighted = "9110B28A7D027679076F831D2987D34BE3BE47883A9D724456810452A32E1DD5"
}

function Assert-Hash([string]$Path, [string]$ExpectedHash) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing required file: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
    if ($actual -ne $ExpectedHash.ToUpperInvariant()) {
        throw "SHA-256 mismatch: $Path`nexpected=$ExpectedHash`nactual=$actual"
    }
}

function Receive-VerifiedFile(
    [string]$Url,
    [string]$Destination,
    [string]$ExpectedHash
) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        Assert-Hash $Destination $ExpectedHash
        return
    }
    if ($SkipDownload) {
        throw "Required download is missing and -SkipDownload was set: $Destination"
    }
    $partial = "$Destination.part"
    if (Test-Path -LiteralPath $partial) {
        throw "Partial file already exists; inspect it before retrying: $partial"
    }
    & curl.exe -L --fail --retry 5 --retry-delay 5 --output $partial $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
    Assert-Hash $partial $ExpectedHash
    Move-Item -LiteralPath $partial -Destination $Destination
}

Receive-VerifiedFile `
    "https://ndownloader.figshare.com/files/31729733" `
    $DataZip `
    $Expected.DataZip
Receive-VerifiedFile `
    "https://ndownloader.figshare.com/files/30914629" `
    $ModelZip `
    $Expected.ModelZip
Receive-VerifiedFile `
    "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth" `
    $MobileNetWeight `
    $Expected.MobileNet

$ExpectedCommit = "f3d098b09784ac7e78b160906952e7bc79940fb1"
if (-not (Test-Path -LiteralPath (Join-Path $OfficialCode ".git"))) {
    if ($SkipDownload) {
        throw "Official code is missing and -SkipDownload was set: $OfficialCode"
    }
    New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
    & git clone https://github.com/beric7/corrosion_cs_classification.git $OfficialCode
    if ($LASTEXITCODE -ne 0) { throw "Official Git clone failed." }
    & git -C $OfficialCode checkout $ExpectedCommit
    if ($LASTEXITCODE -ne 0) { throw "Official Git checkout failed." }
}
$actualCommitOutput = & git -C $OfficialCode rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $null -eq $actualCommitOutput) {
    throw "Unable to verify the official code commit: $OfficialCode"
}
$actualCommit = $actualCommitOutput.Trim()
if ($actualCommit -ne $ExpectedCommit) {
    throw "Unexpected official code commit: expected=$ExpectedCommit actual=$actualCommit"
}

$Backbones = @(
    (Join-Path $OfficialCode "Training - Testing\network\backbone\mobilenetv2.py"),
    (Join-Path $OfficialCode "Training - Testing\network\backbone\resnet.py")
)
foreach ($file in $Backbones) {
    $text = Get-Content -LiteralPath $file -Raw -Encoding UTF8
    if ($text.Contains("from torchvision.models.utils import load_state_dict_from_url")) {
        $text = $text.Replace(
            "from torchvision.models.utils import load_state_dict_from_url",
            "from torch.hub import load_state_dict_from_url"
        )
        Set-Content -LiteralPath $file -Value $text -Encoding UTF8 -NoNewline
    }
    if (-not (Select-String -LiteralPath $file -SimpleMatch "from torch.hub import load_state_dict_from_url" -Quiet)) {
        throw "Compatibility import was not applied: $file"
    }
}

$DataReady = Test-Path -LiteralPath (Join-Path $Dataset "512x512\Train\images_512")
if (-not $DataReady) {
    if ((Test-Path -LiteralPath $Dataset) -and (Get-ChildItem -LiteralPath $Dataset -Force | Select-Object -First 1)) {
        throw "Dataset target is non-empty but incomplete: $Dataset"
    }
    New-Item -ItemType Directory -Force -Path $Dataset | Out-Null
    & tar.exe -xf $DataZip -C $Dataset --strip-components 1
    if ($LASTEXITCODE -ne 0) { throw "Dataset extraction failed." }
}

$TeacherReady = Test-Path -LiteralPath (
    Join-Path $TeacherRaw "var_original_wbatch_2_plus\var_original_wbatch_2_plus_weights_40.pt"
)
if (-not $TeacherReady) {
    if ((Test-Path -LiteralPath $TeacherRaw) -and (Get-ChildItem -LiteralPath $TeacherRaw -Force | Select-Object -First 1)) {
        throw "Teacher target is non-empty but incomplete: $TeacherRaw"
    }
    New-Item -ItemType Directory -Force -Path $TeacherRaw | Out-Null
    & tar.exe -xf $ModelZip -C $TeacherRaw --strip-components 1
    if ($LASTEXITCODE -ne 0) { throw "Teacher extraction failed." }
}

$Data512 = Join-Path $Dataset "512x512"
$TrainImages = @(Get-ChildItem -LiteralPath (Join-Path $Data512 "Train\images_512") -File)
$TrainMasks = @(Get-ChildItem -LiteralPath (Join-Path $Data512 "Train\mask_512") -File)
$TestImages = @(Get-ChildItem -LiteralPath (Join-Path $Data512 "Test\images_512") -File)
$TestMasks = @(Get-ChildItem -LiteralPath (Join-Path $Data512 "Test\mask_512") -File)
if ($TrainImages.Count -ne 396 -or $TrainMasks.Count -ne 396 -or $TestImages.Count -ne 44 -or $TestMasks.Count -ne 44) {
    throw "Unexpected VT dataset counts: train=$($TrainImages.Count)/$($TrainMasks.Count), test=$($TestImages.Count)/$($TestMasks.Count)"
}

$WeightedTeacher = Join-Path $TeacherRaw "var_original_wbatch_2_plus\var_original_wbatch_2_plus_weights_40.pt"
$TeacherFiles = [ordered]@{
    l1_resnet101 = [ordered]@{
        path = (Join-Path $TeacherRaw "l1_loss\weights_27.pt")
        sha256 = $Expected.TeacherL1
    }
    l2_resnet101 = [ordered]@{
        path = (Join-Path $TeacherRaw "l2_loss\weights_35.pt")
        sha256 = $Expected.TeacherL2
    }
    augmented_resnet50 = [ordered]@{
        path = (Join-Path $TeacherRaw "var_aug_batch_2_resnet50\var_aug_batch_2_resnet50_weights_18.pt")
        sha256 = $Expected.TeacherResNet50
    }
    weighted_ce_resnet101 = [ordered]@{
        path = $WeightedTeacher
        sha256 = $Expected.TeacherWeighted
    }
}
foreach ($entry in $TeacherFiles.Values) {
    Assert-Hash $entry.path $entry.sha256
}

$Manifest = [ordered]@{
    prepared_at = (Get-Date).ToString("o")
    dataset_root = $Data512
    train_pairs = 396
    test_pairs = 44
    data_zip_sha256 = $Expected.DataZip
    model_zip_sha256 = $Expected.ModelZip
    license = "CC0"
    sources = [ordered]@{
        dataset = "https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663"
        trained_models = "https://data.lib.vt.edu/articles/software/Trained_Model_for_the_Semantic_Segmentation_of_Corrosion_Condition_States/16628668"
        official_code = "https://github.com/beric7/corrosion_cs_classification"
        mobilenet_imagenet = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
    }
    teacher_checkpoints = $TeacherFiles
    mobilenet_imagenet_sha256 = $Expected.MobileNet
    official_code_commit = $ExpectedCommit
}
$ManifestPath = Join-Path $PSScriptRoot "asset_manifest.json"
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
Write-Host "VT CSSD assets ready. Manifest: $ManifestPath"
