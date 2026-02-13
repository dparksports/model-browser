#!/usr/bin/env python3
"""
install_libraries.py
Sets up the 'meetings' virtual environment under ~/venvs/ with all
dependencies for meeting detection.

Auto-detects environment or allows manual selection:
  --cpu         Force CPU-only installation (works everywhere)
  --gpu         Force GPU installation (requires CUDA 13.x on host)
  --install-cuda  Download & install CUDA 13.1 Toolkit

Usage:
  python install_libraries.py
  python install_libraries.py --gpu --install-cuda
"""

import os
import sys
import subprocess
import tempfile
import re
import venv
import urllib.request
import argparse
import ctypes
import shutil
import time
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
VENV_NAME = "meetings"

# Community pre-built wheel for RTX 5090 (Blackwell SM 100, CUDA 13.0, Python 3.12)
# Source: https://github.com/dougeeai/llama-cpp-python-wheels/releases
# Note: This wheel builds against CUDA 12.x compatible APIs but requires 12.x+ runtime
LLAMA_WHEEL_NAME = "llama_cpp_python-0.3.16+cuda13.0.sm100.blackwell-cp312-cp312-win_amd64.whl"
LLAMA_WHEEL_TAG = "v0.3.16-cuda13.0-sm100-py312"
LLAMA_WHEEL_URL = (
    f"https://github.com/dougeeai/llama-cpp-python-wheels/releases/"
    f"download/{LLAMA_WHEEL_TAG}/{LLAMA_WHEEL_NAME}"
)

# Core CUDA 13.1.0 Installer (required for Blackwell/sm100 wheel)
CUDA_INSTALLER_URL = "https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_windows.exe"
CUDA_INSTALLER_NAME = "cuda_13.1.0_windows.exe"


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
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


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
            return result.stdout.strip().split('\\n')[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def check_cuda_compiler():
    """Check if nvcc is on PATH."""
    try:
        subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_cuda_toolkit_version():
    """
    Detect the installed CUDA Toolkit version by parsing nvcc --version output.
    Returns a tuple (major, minor) or None if not found.
    """
    try:
        result = subprocess.run(
            ['nvcc', '--version'], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # nvcc output contains a line like: "Cuda compilation tools, release 13.1, V13.1.105"
            match = re.search(r'release\s+(\d+)\.(\d+)', result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def is_cuda_version_sufficient(min_major=13, min_minor=0):
    """
    Check if the installed CUDA Toolkit version meets the minimum requirement.
    Returns (is_sufficient: bool, version_str: str or None).
    """
    ver = get_cuda_toolkit_version()
    if ver is None:
        return False, None
    ver_str = f"{ver[0]}.{ver[1]}"
    if ver[0] > min_major or (ver[0] == min_major and ver[1] >= min_minor):
        return True, ver_str
    return False, ver_str


def check_and_add_cuda_path():
    """
    Check for default CUDA installation path and add to PATH if present.
    Helpful for users who just installed CUDA without restarting terminal.
    """
    # Check for CUDA 13.1 first (needed for Blackwell wheel), then fall back to 12.6
    for ver in ["v13.1", "v12.6"]:
        default_cuda_bin = Path(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}\bin")
        if default_cuda_bin.exists() and (default_cuda_bin / "nvcc.exe").exists():
            break
    
    if default_cuda_bin.exists() and (default_cuda_bin / "nvcc.exe").exists():
        # Add to PATH for this process
        os.environ["PATH"] += os.pathsep + str(default_cuda_bin)
        return True
    return False


def download_file(url, dest_path):
    """Download a file with a pretty progress bar."""
    try:
        cprint(f"       Source:      {url}", C.GRAY)
        cprint(f"       Destination: {dest_path.name}", C.GRAY)
        cprint(f"       Downloading...", C.CYAN)

        def progress_bar(count, block_size, total_size):
            if total_size <= 0:
                return
            percent = int(count * block_size * 100 / total_size)
            percent = min(100, max(0, percent))
            
            bar_length = 30
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # \r returns to start of line, end='' prevents newline
            sys.stdout.write(f'\r       [{bar}] {percent}% ')
            sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_bar)
        print() # Newline after progress finishes
        
        file_size_mb = dest_path.stat().st_size / 1024 / 1024
        cprint(f"       ✓ Download complete ({file_size_mb:.1f} MB)", C.GREEN)
        return True
    except Exception as e:
        print()
        cprint(f"       [ERROR] Download failed: {e}", C.RED)
        return False


def install_cuda_toolkit():
    """Download CUDA Toolkit 13.1 and prompt user to install."""
    cprint("[CUDA] Checking for CUDA Toolkit 13.0+ ...", C.CYAN)
    if check_cuda_compiler():
        sufficient, ver = is_cuda_version_sufficient(min_major=13, min_minor=0)
        if sufficient:
            cprint(f"       CUDA Toolkit {ver} found — meets 13.0+ requirement. Skipping download.", C.GREEN)
            return True
        elif ver:
            cprint(f"       CUDA Toolkit {ver} found, but 13.0+ is required. Proceeding with upgrade...", C.YELLOW)
        # else: nvcc found but couldn't parse version — continue with download

    cprint("       CUDA Toolkit not found. Preparing to download...", C.YELLOW)
    
    installer_path = Path.cwd() / CUDA_INSTALLER_NAME
    
    # Check if already downloaded
    if installer_path.exists() and installer_path.stat().st_size > 2 * 1024 * 1024 * 1024:
        cprint(f"       Found existing installer: {installer_path}", C.GREEN)
    else:
        if not download_file(CUDA_INSTALLER_URL, installer_path):
            return False

    cprint("       -----------------------------------------------------", C.CYAN)
    cprint("       DOWNLOAD COMPLETE", C.GREEN)
    cprint("       -----------------------------------------------------", C.CYAN)
    print()
    cprint(f"       Please manually install CUDA Toolkit 13.1:", C.YELLOW)
    cprint(f"       1. Double-click: {installer_path.name}", C.CYAN)
    cprint(f"       2. Follow the on-screen prompts (Express Install is fine).", C.CYAN)
    cprint(f"       3. Once finished, run this script again.", C.CYAN)
    print()
    
    # Open Explorer with file selected
    try:
        subprocess.run(f'explorer.exe /select,"{installer_path}"')
    except Exception:
        pass
        
    cprint("       Exiting so you can install it...", C.GRAY)
    sys.exit(0)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Install meeting detector dependencies.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only installation")
    parser.add_argument("--gpu", action="store_true", help="Force GPU installation (requires CUDA)")
    # Renamed/Aliased for clarity, but keeping old flag for compat
    parser.add_argument("--install-cuda", action="store_true", help="Download CUDA 13.1 Toolkit installer")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm prompts")
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).parent.resolve()
    # Ensure we are in the script directory so local downloads are found
    os.chdir(SCRIPT_DIR)

    HOME_DIR  = Path.home()
    VENVS_DIR = HOME_DIR / "venvs"
    VENV_PATH = VENVS_DIR / VENV_NAME

    print()
    cprint("====================================================", C.CYAN)
    cprint("  Meeting Detector — Library Installer              ", C.CYAN)
    cprint("====================================================", C.CYAN)
    print()

    # Disable GPU flag if CPU requested
    if args.cpu:
        args.gpu = False

    # ── Auto-Install (Download) CUDA if requested via flag ───────────────────
    if args.install_cuda:
        cprint("  [Auto-Download] CUDA Toolkit requested...", C.CYAN)
        install_cuda_toolkit() # This will exit if it downloads
        # If it returns, it means CUDA is already there or failed
        
    # ── Check GPU Env & Interactive Prompt ───────────────────────────────────
    # Try to add default CUDA path to environment if not found naturally
    if not check_cuda_compiler():
        check_and_add_cuda_path()

    driver_ver = get_nvidia_driver_version()
    has_nvidia = bool(driver_ver)
    has_cuda_compiler = check_cuda_compiler()
    cuda_sufficient, cuda_ver = is_cuda_version_sufficient(min_major=13, min_minor=0)

    if has_nvidia:
        cprint(f"  NVIDIA Driver: {driver_ver}", C.GREEN)
    else:
        cprint("  NVIDIA Driver: not detected", C.GRAY)

    if has_cuda_compiler and cuda_ver:
        if cuda_sufficient:
            cprint(f"  CUDA Toolkit:  {cuda_ver} ✓ (meets 13.0+ requirement)", C.GREEN)
        else:
            cprint(f"  CUDA Toolkit:  {cuda_ver} ✗ (need 13.0+, upgrade required)", C.RED)
    elif has_cuda_compiler:
        cprint("  CUDA Toolkit:  found (version unknown)", C.YELLOW)
    else:
        cprint("  CUDA Toolkit:  not detected", C.GRAY)

    # Logic: If NVIDIA GPU present but CUDA Toolkit missing or too old, ask user.
    needs_cuda = has_nvidia and (not has_cuda_compiler or not cuda_sufficient)
    if needs_cuda and not args.cpu and not args.gpu:
        print()
        if has_cuda_compiler and not cuda_sufficient:
            cprint(f"  [!] CUDA Toolkit {cuda_ver} is too old. Version 13.0+ is required.", C.YELLOW)
            cprint("      An upgrade to CUDA Toolkit 13.1 is needed for GPU acceleration.", C.YELLOW)
        else:
            cprint("  [!] NVIDIA GPU detected, but CUDA Toolkit is missing.", C.YELLOW)
            cprint("      To use GPU acceleration, you need to install CUDA Toolkit 13.1.", C.YELLOW)
        cprint(f"      Installer Source: {CUDA_INSTALLER_URL}", C.CYAN)
        
        if args.yes:
             choice = 'y'
        else:
             try:
                 choice = input("      Download installer now? (Y/n) ").strip().lower()
             except (EOFError, KeyboardInterrupt):
                 choice = 'n'

        if choice in ('y', 'yes', ''):
            install_cuda_toolkit() # Will download and exit
        else:
            cprint("      Skipping CUDA download. Using CPU mode.", C.GRAY)
            args.cpu = True

    # If no flags provided, set defaults based on environment state
    if not args.cpu and not args.gpu:
        if has_cuda_compiler and cuda_sufficient and has_nvidia:
            cprint(f"  Environment:   CUDA {cuda_ver} detected. Defaulting to GPU mode.", C.GREEN)
            args.gpu = True
        else:
            cprint("  Environment:   CUDA 13.0+ not confirmed. Defaulting to CPU mode.", C.YELLOW)
            args.cpu = True

    print()
    mode_str = "GPU (Blackwell/CUDA 13.x)" if args.gpu else "CPU (Universal)"
    cprint(f"  Target Mode:   {mode_str}", C.CYAN)
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

    # ── 3. Install llama-cpp-python ───────────────────────────────────────────
    if args.gpu:
        cprint("[3/5] Installing llama-cpp-python (GPU/CUDA Wheel) ...", C.YELLOW)
        cprint(f"       URL: {LLAMA_WHEEL_URL}", C.CYAN)
        
        with tempfile.TemporaryDirectory() as tmp:
            whl_path = Path(tmp) / LLAMA_WHEEL_NAME
            if download_file(LLAMA_WHEEL_URL, whl_path):
                # Force reinstall to ensure we don't keep the CPU version
                run_pip(venv_python, 'install', '--force-reinstall', '--no-cache-dir', str(whl_path))
            else:
                cprint("       Download failed. Falling back to CPU version.", C.RED)
                run_pip(venv_python, 'install', 'llama-cpp-python')
    else:
        cprint("[3/5] Installing llama-cpp-python (CPU Standard) ...", C.YELLOW)
        # Use upgrade to ensure we replace any broken GPU versions
        run_pip(venv_python, 'install', '--upgrade', 'llama-cpp-python')

    # ── 4. Install HuggingFace Hub ────────────────────────────────────────────
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
    
    # Warn if GPU mode was selected but might fail
    if args.gpu:
        cuda_ok, cv = is_cuda_version_sufficient(min_major=13, min_minor=0)
        if not check_cuda_compiler():
            cprint("  [WARNING] You installed the GPU version but 'nvcc' was not found.", C.YELLOW)
            cprint("            It might be installed but not on PATH yet.", C.GRAY)
            cprint("            Try restarting your terminal.", C.GRAY)
            print()
        elif not cuda_ok:
            cprint(f"  [WARNING] CUDA Toolkit {cv} detected but 13.0+ is required.", C.YELLOW)
            cprint("            Run: python install_libraries.py --install-cuda", C.GRAY)
            print()

    print("Quick test:")
    cprint("  python detect_meetings.py --help", C.CYAN)
    print()


if __name__ == "__main__":
    main()
