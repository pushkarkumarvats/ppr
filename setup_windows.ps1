#!/usr/bin/env pwsh
# PowerShell setup script for Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RAW Fusion Diffusion - Windows Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if running in virtual environment
$inVenv = $env:VIRTUAL_ENV -ne $null
if (-not $inVenv) {
    Write-Host ""
    Write-Host "⚠️  Not in a virtual environment. Creating one..." -ForegroundColor Yellow
    Write-Host ""
    
    # Create virtual environment
    python -m venv .venv
    
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        Write-Host "Virtual environment created. Activating..." -ForegroundColor Green
        .\.venv\Scripts\Activate.ps1
    } else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Ask user which requirements to install
Write-Host ""
Write-Host "Choose installation type:" -ForegroundColor Cyan
Write-Host "  1) Minimal (core dependencies only)"
Write-Host "  2) Full (all features including API, UI, etc.)"
Write-Host "  3) Dev (full + development tools)"
Write-Host ""

$choice = Read-Host "Enter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Installing minimal requirements..." -ForegroundColor Yellow
        pip install -r requirements-minimal.txt
    }
    "2" {
        Write-Host ""
        Write-Host "Installing full requirements..." -ForegroundColor Yellow
        pip install -r requirements-full.txt
    }
    "3" {
        Write-Host ""
        Write-Host "Installing full requirements + dev tools..." -ForegroundColor Yellow
        pip install -r requirements-full.txt
        pip install -e ".[dev]"
    }
    default {
        Write-Host "Invalid choice. Installing full requirements..." -ForegroundColor Yellow
        pip install -r requirements-full.txt
    }
}

# Check if installation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Installation complete!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Installation failed. Check errors above." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "Creating project directories..." -ForegroundColor Yellow

$directories = @(
    "data/train",
    "data/val",
    "data/test",
    "outputs",
    "checkpoints",
    "logs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✓ Created $dir" -ForegroundColor Green
    }
}

# Run validation
Write-Host ""
Write-Host "Running validation checks..." -ForegroundColor Yellow
Write-Host ""
python validate_setup.py

# Print next steps
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Prepare your data:" -ForegroundColor White
Write-Host "   - Place training RAW images in: .\data\train" -ForegroundColor Gray
Write-Host "   - Place validation RAW images in: .\data\val" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start training:" -ForegroundColor White
Write-Host "   python quick_train.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or run demo:" -ForegroundColor White
Write-Host "   python example.py --demo" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Start web UI:" -ForegroundColor White
Write-Host "   python web\gradio_interface.py" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Start API server:" -ForegroundColor White
Write-Host "   python quick_deploy.py" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see:" -ForegroundColor Yellow
Write-Host "  README.md" -ForegroundColor Gray
Write-Host "  INSTALLATION.md" -ForegroundColor Gray
Write-Host "  QUICKSTART.md" -ForegroundColor Gray
Write-Host ""
