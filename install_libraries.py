#!/usr/bin/env python3
"""
install_libraries.py
Sets up the 'meetings' virtual environment under ~/venvs/ with all
dependencies for meeting detection on an NVIDIA RTX 5090 (Blackwell).

Prerequisites:
  - Python 3.12 on PATH
  - NVIDIA driver 580+ (CUDA Toolkit NOT required)

Usage:
  python install_libraries.py
"""

import os
import sys
import subprocess
import tempfile
import venv
import urllib.request
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
VENV_NAME = "meetings"

# Community pre-built wheel for RTX 5090 (Blackwell SM 100, CUDA 13.0, Python 3.12)
# Source: https://github.com/dougeeai/llama-cpp-python-wheels/releases
LLAMA_WHEEL_NAME = "llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp312-cp312-win_amd64.whl"
LLAMA_WHEEL_TAG = "v0.3.16-cuda13.0-sm100-py312"
LLAMA_WHEEL_URL = (
    f"https://github.com/dougeeai/llama-cpp-python-wheels/releases/"
    f"download/{LLAMA_WHEEL_TAG}/{LLAMA_WHEEL_NAME}"
)


# ── Color helpers ──────────────────────────────────────────────────────────────
class C:
    CYAN    = '\033[96m'
    YELLOW  = '\033[93m'
    GREEN   = '\033[92m'
    GRAY    = '\033[90m'
    RED     = '\033[91m'
    RESET   = '\033[0m'


def cprint(text, color=''):
    print(f"{color}{text}{C.RESET}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def run_pip(venv_python, *args, quiet=False, allow_fail=False):
    """Run a pip command inside the venv. Returns True on success."""
    cmd = [str(venv_python), '-m', 'pip'] + list(args)
    if quiet:
        cmd.append('--quiet')
    result = subprocess.run(cmd, capture_output=quiet)
    if result.returncode != 0:
        if allow_fail:
            return False
        cprint(f"  ERROR: pip {' '.join(args)}", C.RED)
        if quiet and result.stderr:
            print(result.stderr.decode(errors='replace'))
        sys.exit(1)
    return True


def get_nvidia_driver_version():
    """Detect installed NVIDIA driver version via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def download_file(url, dest_path):
    """Download a file with progress indication."""
    try:
        cprint(f"       Downloading {dest_path.name} ...", C.GRAY)
        urllib.request.urlretrieve(url, str(dest_path))
        cprint(f"       Downloaded ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)", C.GRAY)
        return True
    except Exception as e:
        cprint(f"       Download failed: {e}", C.YELLOW)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    SCRIPT_DIR = Path(__file__).parent.resolve()

    HOME_DIR  = Path.home()
    VENVS_DIR = HOME_DIR / "venvs"
    VENV_PATH = VENVS_DIR / VENV_NAME

    os.chdir(SCRIPT_DIR)

    print()
    cprint("====================================================", C.CYAN)
    cprint("  Meeting Detector — Library Installer (RTX 5090)   ", C.CYAN)
    cprint("====================================================", C.CYAN)
    print()

    # ── Show NVIDIA driver version ────────────────────────────────────────────
    driver_ver = get_nvidia_driver_version()
    if driver_ver:
        cprint(f"  NVIDIA Driver: {driver_ver}", C.GREEN)
    else:
        cprint("  NVIDIA Driver: not detected (nvidia-smi not found)", C.RED)
        cprint("  GPU acceleration may not work.", C.YELLOW)
    print()

    # ── 1. Create virtual environment ─────────────────────────────────────────
    VENVS_DIR.mkdir(parents=True, exist_ok=True)
    venv_python  = VENV_PATH / "Scripts" / "python.exe"
    venv_activate = VENV_PATH / "Scripts" / "activate.bat"

    if not venv_activate.exists():
        cprint(f"[1/5] Creating virtual environment '{VENV_NAME}' at {VENV_PATH} ...", C.YELLOW)
        venv.create(VENV_PATH, with_pip=True)
    else:
        cprint(f"[1/5] Virtual environment '{VENV_NAME}' already exists at {VENV_PATH}.", C.GREEN)

    # ── 2. Upgrade pip ────────────────────────────────────────────────────────
    cprint("[2/5] Upgrading pip ...", C.YELLOW)
    run_pip(venv_python, 'install', '--upgrade', 'pip', quiet=True)

    # ── 3. Install llama-cpp-python (Blackwell CUDA wheel) ────────────────────
    #
    #  Official abetlen wheels only cover CUDA ≤ 12.5 — no Blackwell support.
    #  We use a community pre-built wheel from dougeeai/llama-cpp-python-wheels
    #  targeting CUDA 13.0 + SM 100 (Blackwell).  No C++ compiler needed.
    #
    cprint("[3/5] Installing llama-cpp-python (Blackwell CUDA wheel) ...", C.YELLOW)
    cprint("       Source: community pre-built wheel (not official PyPI)", C.YELLOW)
    cprint("       Repo:   https://github.com/dougeeai/llama-cpp-python-wheels", C.CYAN)
    cprint(f"       URL:    {LLAMA_WHEEL_URL}", C.CYAN)
    cprint("       Review the repo above before proceeding if concerned.", C.GRAY)

    with tempfile.TemporaryDirectory() as tmp:
        whl_path = Path(tmp) / LLAMA_WHEEL_NAME
        if download_file(LLAMA_WHEEL_URL, whl_path):
            run_pip(venv_python, 'install', str(whl_path))
        else:
            cprint("       Could not download Blackwell wheel.", C.RED)
            cprint("       Please download manually from:", C.YELLOW)
            cprint(f"       {LLAMA_WHEEL_URL}", C.CYAN)
            sys.exit(1)

    # ── 4. Install HuggingFace Hub (for GGUF model downloads) ─────────────────
    cprint("[4/5] Installing huggingface-hub ...", C.YELLOW)
    run_pip(venv_python, 'install', 'huggingface-hub', quiet=True)

    # ── 5. Install cloud LLM SDKs ────────────────────────────────────────────
    cprint("[5/5] Installing cloud provider SDKs (openai, anthropic) ...", C.YELLOW)
    run_pip(venv_python, 'install', 'openai', 'anthropic', quiet=True)

    # ── Done ──────────────────────────────────────────────────────────────────
    print()
    cprint("====================================================", C.GREEN)
    cprint("  Installation complete!                            ", C.GREEN)
    cprint("====================================================", C.GREEN)
    print()
    print("To activate the environment:")
    cprint(f"  {VENV_PATH}\\Scripts\\activate", C.CYAN)
    print()
    print("Quick test:")
    cprint("  python detect_meetings.py --help", C.CYAN)
    print()


if __name__ == "__main__":
    main()
