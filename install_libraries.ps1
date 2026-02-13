# install_libraries.ps1
# Sets up the 'meeting' virtual environment with all dependencies for
# meeting detection on an NVIDIA RTX 5090 (Blackwell / CUDA 12.8).
#
# Prerequisites:
#   - Python 3.10+ on PATH
#   - NVIDIA CUDA Toolkit 12.8 installed (required for RTX 5090)
#
# Usage:
#   .\install_libraries.ps1

$ErrorActionPreference = "Stop"

$VENV_DIR = "meeting"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $SCRIPT_DIR

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  Meeting Detector - Library Installer (RTX 5090)   " -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Create virtual environment ──────────────────────────────────────────
if (-Not (Test-Path "$VENV_DIR\Scripts\activate")) {
    Write-Host "[1/5] Creating virtual environment '$VENV_DIR'..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
} else {
    Write-Host "[1/5] Virtual environment '$VENV_DIR' already exists." -ForegroundColor Green
}

# ── 2. Activate venv ──────────────────────────────────────────────────────
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& "$VENV_DIR\Scripts\Activate.ps1"

# Upgrade pip first
Write-Host "       Upgrading pip..." -ForegroundColor DarkGray
python -m pip install --upgrade pip --quiet

# ── 3. Install llama-cpp-python with CUDA 12.8 ────────────────────────────
#
#  RTX 5090 (Blackwell, SM 100) requires CUDA 12.8+.
#  The abetlen index provides pre-built wheels with CUDA support so you
#  do NOT need a C++ compiler or CMake.
#
Write-Host "[3/5] Installing llama-cpp-python (CUDA 12.8 wheel)..." -ForegroundColor Yellow
pip install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128 `
    --prefer-binary --quiet

# ── 4. Install HuggingFace Hub (for GGUF model downloads) ─────────────────
Write-Host "[4/5] Installing huggingface-hub..." -ForegroundColor Yellow
pip install huggingface-hub --quiet

# ── 5. Install cloud LLM SDKs (optional but included for convenience) ─────
Write-Host "[5/5] Installing cloud provider SDKs (openai, anthropic)..." -ForegroundColor Yellow
pip install openai anthropic --quiet

# ── Done ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "  Installation complete!                            " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:"
Write-Host "  .\$VENV_DIR\Scripts\activate" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick test:"
Write-Host "  python detect_meetings.py --help" -ForegroundColor Cyan
Write-Host ""

Pop-Location
