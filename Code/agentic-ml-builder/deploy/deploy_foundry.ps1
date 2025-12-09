
# ============================================================================
# deploy_foundry.ps1
# ============================================================================
<#
.SYNOPSIS
    Deploy ML project to Azure AI Foundry

.DESCRIPTION
    Deploys generated ML project to Azure AI Foundry for training and deployment

.EXAMPLE
    .\deploy_foundry.ps1 -ProjectPath "output/scene_classifier"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectPath,
    
    [string]$ResourceGroup = $env:AZURE_RESOURCE_GROUP,
    [string]$WorkspaceName = $env:AZURE_AI_PROJECT_NAME,
    [string]$ExperimentName = "ml-experiment",
    [string]$ComputeTarget = "gpu-cluster"
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Azure AI Foundry Deployment" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Validate inputs
if (-not (Test-Path $ProjectPath)) {
    Write-Host "✗ Project path not found: $ProjectPath" -ForegroundColor Red
    exit 1
}

if (-not $ResourceGroup -or -not $WorkspaceName) {
    Write-Host "✗ Azure configuration not found. Run setup_azure.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Project: $ProjectPath" -ForegroundColor Yellow
Write-Host "Workspace: $WorkspaceName" -ForegroundColor Yellow
Write-Host "Resource Group: $ResourceGroup" -ForegroundColor Yellow
Write-Host ""

# Login check
Write-Host "Checking Azure login..." -ForegroundColor Yellow
az account show > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "! Not logged in. Logging in..." -ForegroundColor Yellow
    az login
}
Write-Host "✓ Logged in" -ForegroundColor Green

# Create environment file
Write-Host "`nCreating conda environment file..." -ForegroundColor Yellow
$condaEnv = @"
name: ml-environment
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pip
  - pip:
    - torch==2.5.1
    - torchvision==0.20.1
    - scikit-learn==1.5.2
    - pandas==2.2.3
    - numpy==2.1.3
    - matplotlib==3.9.2
    - mlflow==2.18.0
    - azureml-mlflow==1.59.0
"@

$condaEnv | Out-File -FilePath "$ProjectPath/conda_env.yml" -Encoding utf8
Write-Host "✓ Environment file created" -ForegroundColor Green

# Create Azure ML job configuration
Write-Host "`nCreating job configuration..." -ForegroundColor Yellow
$jobConfig = @"
`$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json

type: command
experiment_name: $ExperimentName
display_name: training-job-$(Get-Date -Format 'yyyyMMdd-HHmmss')
description: ML training job deployed by Agentic ML Builder

compute: azureml:$ComputeTarget

environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04
  conda_file: conda_env.yml

code: .

command: >-
  python train.py
  --epochs 10
  --batch-size 32
  --learning-rate 0.001

outputs:
  model_output:
    type: uri_folder
    mode: rw_mount

resources:
  instance_count: 1
"@

$jobConfig | Out-File -FilePath "$ProjectPath/job.yml" -Encoding utf8
Write-Host "✓ Job configuration created" -ForegroundColor Green

# Submit job
Write-Host "`nSubmitting training job..." -ForegroundColor Yellow
Push-Location $ProjectPath
try {
    az ml job create `
        --file job.yml `
        --resource-group $ResourceGroup `
        --workspace-name $WorkspaceName `
        --stream
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Job submitted successfully" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Job submission failed" -ForegroundColor Red
    }
}
finally {
    Pop-Location
}

# Show job status
Write-Host "`nTo monitor job:" -ForegroundColor Yellow
Write-Host "  az ml job list --resource-group $ResourceGroup --workspace-name $WorkspaceName"
Write-Host "`nTo view in portal:" -ForegroundColor Yellow
Write-Host "  https://ml.azure.com"
Write-Host ""

