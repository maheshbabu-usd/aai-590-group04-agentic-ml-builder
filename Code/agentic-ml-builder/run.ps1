#!/usr/bin/env pwsh
# Run the Agentic ML Builder with proper virtual environment activation

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate virtual environment
& "$scriptDir\venv\Scripts\Activate.ps1"

# Run the application
python src/main.py @args
