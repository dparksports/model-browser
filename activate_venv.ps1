<# 
  Meeting Detector - Quick Start
  Activates the virtual environment and shows usage instructions.
  Run:  .\activate_venv.ps1
#>

# Resolve venv path (matches install_libraries.py: ~/venvs/meetings)
$venvDir = Join-Path $HOME "venvs\meetings"
$venvPath = Join-Path $venvDir "Scripts\Activate.ps1"

if (-not (Test-Path $venvPath)) {
  Write-Host ""
  Write-Host "  [!] Virtual environment not found at: $venvDir" -ForegroundColor Yellow
  Write-Host ""
  $answer = Read-Host "      Create it now? (Y/n)"
  if ($answer -and $answer -notin @('Y', 'y', 'yes', '')) {
    Write-Host "  Aborted." -ForegroundColor Red
    exit 1
  }

  Write-Host ""
  Write-Host "  Creating virtual environment ..." -ForegroundColor Cyan
  python -m venv $venvDir
  if (-not (Test-Path $venvPath)) {
    Write-Host "  [!] Failed to create venv. Make sure Python is on PATH." -ForegroundColor Red
    exit 1
  }

  # Activate, then upgrade pip and install core deps
  & $venvPath
  Write-Host "  Installing core dependencies ..." -ForegroundColor Cyan
  pip install --upgrade pip --quiet
  pip install huggingface-hub openai anthropic --quiet
  Write-Host "  Done!" -ForegroundColor Green
  Write-Host ""
}
else {
  & $venvPath
}

Write-Host ""
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host "    Meeting Detector - Ready" -ForegroundColor Cyan
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host '  1. BROWSE and DOWNLOAD MODELS' -ForegroundColor Green
Write-Host "     python model_browser.py                  # top Unsloth + community models"
Write-Host "     python model_browser.py --search 'qwen3' # search for specific models"
Write-Host "     python model_browser.py --top 20         # show more results"
Write-Host ""
Write-Host "  2. FIND MEETINGS" -ForegroundColor Green
Write-Host "     python find_meetings.py --model pick   # pick from downloaded models"
Write-Host "     python find_meetings.py                # uses last picked model"
Write-Host "     python find_meetings.py --skip-checked # resume a previous scan"
Write-Host ""
Write-Host "  3. CLOUD PROVIDERS (optional)" -ForegroundColor Green
Write-Host "     python find_meetings.py --provider gemini --api-key KEY"
Write-Host "     python find_meetings.py --provider openai --api-key KEY"
Write-Host "     python find_meetings.py --provider claude --api-key KEY"
Write-Host ""
Write-Host "  Transcripts default dir: %APPDATA%\LongAudioApp\Transcripts\" -ForegroundColor DarkGray
Write-Host '  Override with: --dir "C:\MyFolder"' -ForegroundColor DarkGray
Write-Host ""
