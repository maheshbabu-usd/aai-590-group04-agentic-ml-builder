# ============================================================================
# setup_azure.ps1
# ============================================================================
<#
.SYNOPSIS
    Setup Azure resources for Agentic ML Builder

.DESCRIPTION
    Creates necessary Azure resources including:
    - Resource Group
    - Azure ML Workspace
    - Storage Account
    - Container Registry
    - Key Vault
    - Application Insights

.EXAMPLE
    .\setup_azure.ps1
#>

param(
    [string]$ResourceGroup = "ml-builder-rg",
    [string]$Location = "eastus",
    [string]$WorkspaceName = "ml-builder-workspace",
    [string]$StorageAccountName = "mlbuilderstorage$(Get-Random -Maximum 9999)",
    [string]$KeyVaultName = "ml-builder-kv-$(Get-Random -Maximum 9999)",
    [string]$AppInsightsName = "ml-builder-insights"
)

Write-Host "======================================" 
Write-Host "Azure ML Builder Setup" 
Write-Host "======================================" 
Write-Host ""

# Check if Azure CLI is installed
Write-Host "Checking Azure CLI installation..."
try {
    $azVersion = az version | ConvertFrom-Json
    Write-Host "Azure CLI version: $($azVersion.'azure-cli')"
}
catch {
    Write-Host "Azure CLI not found. Please install from: https://aka.ms/installazurecliwindows"
    exit 1
}

# Login to Azure
Write-Host ""
Write-Host "Logging in to Azure..."
az login
if ($LASTEXITCODE -ne 0) {
    Write-Host "Azure login failed"
    exit 1
}
Write-Host "Logged in successfully"

# Set subscription
Write-Host ""
Write-Host "Setting subscription..."
$subscriptionId = $env:AZURE_SUBSCRIPTION_ID
if ($subscriptionId) {
    az account set --subscription $subscriptionId
    Write-Host "Subscription set to: $subscriptionId"
}
else {
    Write-Host "Using default subscription"
}

# Create Resource Group
Write-Host ""
Write-Host "Creating Resource Group: $ResourceGroup"

