#!/usr/bin/env python3
"""
Install Timestamp Models — Auto-Downloader
───────────────────────────────────────────
Download the Qwen2.5-VL-7B-Instruct model used by timestamp_engine.py
for extracting burned-in timestamps from video files.

Usage::

    python install_timestamp_models.py              # interactive install
    python install_timestamp_models.py --download   # download model directly
    python install_timestamp_models.py --status     # check if model is cached
"""

import argparse
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colors
# ──────────────────────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_WHITE = "\033[97m"


def _enable_ansi_windows():
    """Enable ANSI escape sequences on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Dependency management — auto-install missing packages
# ──────────────────────────────────────────────────────────────────────────────
_REQUIRED_PACKAGES = {
    # import_name: pip_install_name
    "transformers":    "transformers",
    "accelerate":      "accelerate",
    "torch":           "torch",
    "torchvision":     "torchvision",
    "PIL":             "Pillow",
    "qwen_vl_utils":   "qwen-vl-utils",
}


def _ensure_deps():
    """Check for missing dependencies and offer to install them."""
    missing = []
    for import_name, pip_name in _REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    # Also check if torch is CPU-only (needs CUDA reinstall)
    torch_needs_cuda = False
    if "torch" not in [p for p in missing]:
        try:
            import torch
            if not torch.cuda.is_available() and "+cpu" in torch.__version__:
                torch_needs_cuda = True
        except Exception:
            pass

    if not missing and not torch_needs_cuda:
        return True

    if missing:
        print(f"{_YELLOW}⚠  Missing packages: {', '.join(missing)}{_RESET}")
        print(f"{_DIM}   These are needed for the timestamp engine.{_RESET}")
    if torch_needs_cuda:
        print(f"{_YELLOW}⚠  PyTorch is CPU-only — GPU acceleration is disabled.{_RESET}")
        print(f"{_DIM}   Will reinstall PyTorch with CUDA 12.8 support for GPU usage.{_RESET}")
    print()

    try:
        answer = input(f"{_BOLD}Install them now? (Y/n): {_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if answer not in ("y", "yes", ""):
        print(f"{_DIM}   Skipped. Some features may not work.{_RESET}")
        return False

    # Separate torch packages (need special index URL) from regular packages
    _TORCH_PKGS = {"torch", "torchvision"}
    torch_pkgs = [p for p in missing if p in _TORCH_PKGS]
    other_pkgs = [p for p in missing if p not in _TORCH_PKGS]

    # If torch is CPU-only, also reinstall it with CUDA
    if torch_needs_cuda and not torch_pkgs:
        torch_pkgs = ["torch", "torchvision"]

    total = len(torch_pkgs) + len(other_pkgs)
    print(f"\n{_CYAN}📦 Installing {total} package(s)...{_RESET}")

    # Install torch packages with CUDA 12.8 (required for RTX 50-series / Blackwell)
    if torch_pkgs:
        print(f"   {_DIM}pip install {' '.join(torch_pkgs)} (with CUDA 12.8)...{_RESET}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + torch_pkgs +
            ["--index-url", "https://download.pytorch.org/whl/cu128", "--force-reinstall", "--no-deps"],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"   {_GREEN}✅ {' '.join(torch_pkgs)} (CUDA 12.8){_RESET}")
        else:
            print(f"   {_RED}❌ {' '.join(torch_pkgs)} — {result.stderr.decode(errors='replace').strip()[-200:]}{_RESET}")

    # Install other packages normally
    for pkg in other_pkgs:
        print(f"   {_DIM}pip install {pkg}...{_RESET}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"   {_GREEN}✅ {pkg}{_RESET}")
        else:
            print(f"   {_RED}❌ {pkg} — {result.stderr.decode(errors='replace').strip()}{_RESET}")

    print(f"{_GREEN}Done!{_RESET}\n")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_gpu():
    """Try nvidia-smi to get GPU name and VRAM. Returns (name, vram_gb) or (None, 0)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if "," in out:
            name, mem = out.split(",", 1)
            return name.strip(), round(float(mem.strip()) / 1024, 1)
    except Exception:
        pass
    return None, 0


# ──────────────────────────────────────────────────────────────────────────────
# Model info
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
MODEL_PARAMS = "7.6B"
MODEL_SIZE_GB = 15.5   # approximate VRAM when loaded in bf16
MODEL_DISK_GB = 15.2   # approximate download size


def _check_model_cached():
    """Check if the Qwen VLM model is already cached locally."""
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.exists():
            cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if not cache_dir.exists():
            return False, None
        safe_name = "models--" + MODEL_REPO.replace("/", "--")
        repo_cache = cache_dir / safe_name
        if repo_cache.exists():
            for p in repo_cache.rglob("*.safetensors"):
                return True, str(repo_cache)
            for p in repo_cache.rglob("*.bin"):
                return True, str(repo_cache)
    except Exception:
        pass
    return False, None


def _download_model():
    """Download the Qwen VLM model using transformers from_pretrained (caches automatically)."""
    print(f"\n{_CYAN}⬇  Downloading {MODEL_REPO}...{_RESET}")
    print(f"{_DIM}   This is ~{MODEL_DISK_GB} GB and may take a while on first run.{_RESET}")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print(f"{_RED}[ERROR] transformers not installed. Run: pip install transformers{_RESET}")
        return None

    try:
        import torch
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32

        print(f"   Downloading model weights...")
        t0 = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_REPO,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else "cpu",
        )
        print(f"   Downloading processor/tokenizer...")
        processor = AutoProcessor.from_pretrained(MODEL_REPO)

        elapsed = time.perf_counter() - t0
        device = next(model.parameters()).device

        print(f"\n{_GREEN}✅ Model ready!{_RESET}")
        print(f"   {_DIM}Device: {device} | Dtype: {dtype} | Load time: {elapsed:.1f}s{_RESET}")

        if use_cuda:
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"   {_DIM}GPU VRAM used: {vram_used:.1f} GB{_RESET}")

        # Free memory
        del model, processor
        if use_cuda:
            torch.cuda.empty_cache()

        return MODEL_REPO

    except Exception as e:
        print(f"{_RED}[ERROR] Download failed: {e}{_RESET}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────

def _print_status(gpu_name, gpu_vram):
    """Print model status and system info."""
    W = 70

    print()
    print(f"{_BOLD}{_CYAN}{'═' * W}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'  📸  Timestamp Engine — Model Installer':^{W}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * W}{_RESET}")

    # GPU info
    if gpu_name:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_name}{_RESET} ({gpu_vram} GB)")
    else:
        print(f"  {_YELLOW}⚠  No GPU detected — model will run on CPU (very slow).{_RESET}")
    print()

    # Model info
    cached, cache_path = _check_model_cached()
    status = f"{_GREEN}Downloaded ✅{_RESET}" if cached else f"{_YELLOW}Not yet downloaded{_RESET}"
    fit = ""
    if gpu_vram > 0:
        if MODEL_SIZE_GB + 0.5 <= gpu_vram:
            fit = f"{_GREEN}✅ Fits on GPU{_RESET}"
        elif MODEL_SIZE_GB <= gpu_vram:
            fit = f"{_YELLOW}⚠️  Tight fit{_RESET}"
        else:
            fit = f"{_RED}❌ May not fit{_RESET}"

    print(f"  {_BOLD}Model:{_RESET}      {_CYAN}{MODEL_NAME}{_RESET}")
    print(f"  {_BOLD}Repo:{_RESET}       {_DIM}{MODEL_REPO}{_RESET}")
    print(f"  {_BOLD}Parameters:{_RESET} {MODEL_PARAMS}")
    print(f"  {_BOLD}VRAM needed:{_RESET} ~{MODEL_SIZE_GB} GB (bfloat16)")
    print(f"  {_BOLD}Disk size:{_RESET}  ~{MODEL_DISK_GB} GB")
    print(f"  {_BOLD}Status:{_RESET}     {status}")
    if fit:
        print(f"  {_BOLD}GPU fit:{_RESET}    {fit}")
    if cached and cache_path:
        print(f"  {_BOLD}Cache:{_RESET}      {_DIM}{cache_path}{_RESET}")
    print()

    # Dependencies
    print(f"  {_BOLD}Dependencies:{_RESET}")
    for import_name, display_name in _REQUIRED_PACKAGES.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    {display_name:<18s} {_GREEN}{ver}{_RESET}")
        except ImportError:
            print(f"    {display_name:<18s} {_RED}Not installed{_RESET}")

    print()
    print(f"{_BOLD}{_CYAN}{'═' * W}{_RESET}")
    print()

    return cached


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    _enable_ansi_windows()
    _ensure_deps()

    parser = argparse.ArgumentParser(
        description="Download the Qwen2.5-VL-7B model for timestamp extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python install_timestamp_models.py              # interactive install
              python install_timestamp_models.py --download   # download model directly
              python install_timestamp_models.py --status     # check model status
        """),
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the model immediately (non-interactive)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show model and dependency status, then exit",
    )

    args = parser.parse_args()

    # Resolve GPU
    gpu_name, gpu_vram = _detect_gpu()

    # Show status
    cached = _print_status(gpu_name, gpu_vram)

    if args.status:
        return

    # Direct download
    if args.download:
        if cached:
            print(f"{_GREEN}Model already downloaded.{_RESET}")
            try:
                raw = input(f"Re-download? (y/N): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if raw.lower() != "y":
                return
        _download_model()
        return

    # Interactive
    if cached:
        print(f"  {_GREEN}Model is already downloaded and ready!{_RESET}")
        print(f"  {_DIM}Run timestamp_engine.py to extract timestamps from videos.{_RESET}")
        print()
        return

    try:
        answer = input(f"{_BOLD}Download model now? (Y/n): {_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if answer not in ("y", "yes", ""):
        print(f"{_DIM}Skipped. Run again when ready.{_RESET}")
        return

    _download_model()


if __name__ == "__main__":
    main()
