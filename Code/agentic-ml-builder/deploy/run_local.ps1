
# ============================================================================
# run_local.ps1
# ============================================================================
<#
.SYNOPSIS
    Run ML Builder locally

.DESCRIPTION
    Runs the Agentic ML Builder on a local Windows machine

.EXAMPLE
    .\run_local.ps1 -DatasetType "scene"
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("scene", "lyrics", "birds", "custom")]
    [string]$DatasetType,
    
    [string]$CustomInput,
    [string]$OutputDir = "output",
    [switch]$Validate
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Agentic ML Builder - Local Execution" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "! Virtual environment not activated" -ForegroundColor Yellow
    Write-Host "  Activating..." -ForegroundColor Yellow
    
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & ".\venv\Scripts\Activate.ps1"
        Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Virtual environment not found. Run: python -m venv venv" -ForegroundColor Red
        exit 1
    }
}

# Check .env file
if (-not (Test-Path ".env")) {
    Write-Host "✗ .env file not found. Please create it with your API keys." -ForegroundColor Red
    exit 1
}

# Create input directory
if (-not (Test-Path "input")) {
    New-Item -ItemType Directory -Path "input" | Out-Null
}

# Select or create input file
$inputFile = ""
switch ($DatasetType) {
    "scene" {
        $inputFile = "input/scene_dataset.json"
        $spec = @{
            project_name = "scene_classifier"
            purpose = "Multi-label scene classification"
            data_type = "image"
            dataset_url = "https://www.openml.org/api/v1/json/data/40595"
            ml_task = "classification"
            target_environment = "local"
            models = @("resnet", "efficientnet")
        } | ConvertTo-Json
        $spec | Out-File -FilePath $inputFile -Encoding utf8
    }
    "lyrics" {
        $inputFile = "input/lyrics_dataset.json"
        $spec = @{
            project_name = "lyrics_sentiment"
            purpose = "Lyrics sentiment analysis"
            data_type = "text"
            dataset_url = "https://www.openml.org/api/v1/json/data/43597"
            ml_task = "regression"
            target_environment = "local"
            models = @("bert", "lstm")
        } | ConvertTo-Json
        $spec | Out-File -FilePath $inputFile -Encoding utf8
    }
    "birds" {
        $inputFile = "input/birds_dataset.json"
        $spec = @{
            project_name = "bird_classifier"
            purpose = "Bird species classification"
            data_type = "audio"
            dataset_url = "https://www.openml.org/api/v1/json/data/40588"
            ml_task = "classification"
            target_environment = "local"
            models = @("cnn", "transformer")
        } | ConvertTo-Json
        $spec | Out-File -FilePath $inputFile -Encoding utf8
    }
    "custom" {
        if (-not $CustomInput) {
            Write-Host "✗ Custom input file path required" -ForegroundColor Red
            exit 1
        }
        $inputFile = $CustomInput
    }
}

Write-Host "Input: $inputFile" -ForegroundColor Yellow
Write-Host "Output: $OutputDir" -ForegroundColor Yellow
Write-Host ""

# Run ML Builder
Write-Host "Starting ML Builder..." -ForegroundColor Yellow
$validateArg = if ($Validate) { "--validate" } else { "" }

python src/main.py --input $inputFile --output $OutputDir $validateArg

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ ML Builder completed successfully!" -ForegroundColor Green
    Write-Host "`nGenerated project available in: $OutputDir" -ForegroundColor Green
}
else {
    Write-Host "`n✗ ML Builder failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
