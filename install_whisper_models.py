#!/usr/bin/env python3
"""
Install Whisper Models — Auto-Downloader
─────────────────────────────────────────
Browse and download Whisper speech recognition models from HuggingFace.
Supports OpenAI Whisper, Distil-Whisper, and faster-whisper (CTranslate2) variants.

Usage::

    python install_whisper_models.py                      # show all models
    python install_whisper_models.py --list                # print table and exit
    python install_whisper_models.py --gpu 5090            # show GPU fit info
    python install_whisper_models.py --format ct2          # show faster-whisper models
    python install_whisper_models.py --download large-v3   # download by alias
"""

import argparse
import json
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
    # package_name: pip_install_name
    "huggingface_hub": "huggingface-hub",
    "faster_whisper":  "faster-whisper",
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

    if not missing:
        return True

    print(f"{_YELLOW}⚠  Missing packages: {', '.join(missing)}{_RESET}")
    print(f"{_DIM}   These are needed for downloading and running Whisper models.{_RESET}")
    print()

    try:
        answer = input(f"{_BOLD}Install them now? (Y/n): {_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if answer not in ("y", "yes", ""):
        print(f"{_DIM}   Skipped. Some features may not work.{_RESET}")
        return False

    print(f"\n{_CYAN}📦 Installing {len(missing)} package(s)...{_RESET}")
    for pkg in missing:
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
# GPU detection (same pattern as install_models.py)
# ──────────────────────────────────────────────────────────────────────────────
_KNOWN_GPUS = {
    "3060": 12, "3060 ti": 8,
    "3070": 8,  "3070 ti": 8,
    "3080": 10, "3080 ti": 12,
    "3090": 24, "3090 ti": 24,
    "4060": 8,  "4060 ti": 8,
    "4070": 12, "4070 ti": 12, "4070 ti super": 16,
    "4080": 16, "4080 super": 16,
    "4090": 24,
    "5070": 12, "5070 ti": 16,
    "5080": 16,
    "5090": 32,
    "a100": 80, "a6000": 48, "h100": 80,
}


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


def _vram_for_gpu_flag(flag):
    """Resolve --gpu shorthand like '4090' into VRAM GB."""
    flag_lower = flag.strip().lower()
    if flag_lower in _KNOWN_GPUS:
        return _KNOWN_GPUS[flag_lower]
    try:
        return float(flag_lower)
    except ValueError:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Registry — track installed models
# ──────────────────────────────────────────────────────────────────────────────
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "installed_whisper_models.json")


def _load_registry():
    """Load installed whisper models from JSON registry."""
    if not os.path.exists(_REGISTRY_FILE):
        return {}
    try:
        with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_registry(registry):
    """Save registry to JSON."""
    try:
        with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"{_RED}[ERROR] Failed to save registry: {e}{_RESET}")


def _update_registry(repo_id, alias, model_format="hf"):
    """Register a newly downloaded model."""
    registry = _load_registry()
    registry[repo_id] = {
        "repo_id": repo_id,
        "alias": alias,
        "format": model_format,
        "downloaded_at": time.time(),
    }
    _save_registry(registry)
    print(f"{_GREEN}📝 Registered in installed_whisper_models.json (alias: {alias}){_RESET}")


def _is_installed(repo_id):
    """Check if this model is in our registry."""
    registry = _load_registry()
    return repo_id in registry


def _check_hf_cache(repo_id):
    """Check if model exists in HuggingFace cache."""
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.exists():
            cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if not cache_dir.exists():
            return False
        safe_name = "models--" + repo_id.replace("/", "--")
        repo_cache = cache_dir / safe_name
        if repo_cache.exists():
            # Check for any model files in snapshots
            for p in repo_cache.rglob("*.bin"):
                return True
            for p in repo_cache.rglob("*.safetensors"):
                return True
            for p in repo_cache.rglob("model.bin"):
                return True
    except Exception:
        pass
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Whisper Model Catalog
# ──────────────────────────────────────────────────────────────────────────────

# WER = Word Error Rate on LibriSpeech test-clean (lower is better)
# Speed is relative to large-v3 (1x baseline)

_WHISPER_MODELS = [
    # ── OpenAI Whisper (original HuggingFace transformers format) ──
    {"name": "tiny",          "repo": "openai/whisper-tiny",           "alias": "tiny",
     "params": "39M",  "size_gb": 0.15, "vram_gb": 1,  "format": "hf",
     "wer": 7.6,  "speed": "32x", "langs": 99,
     "desc": "Smallest; fast prototyping and edge deployment"},

    {"name": "tiny.en",       "repo": "openai/whisper-tiny.en",        "alias": "tiny.en",
     "params": "39M",  "size_gb": 0.15, "vram_gb": 1,  "format": "hf",
     "wer": 5.6,  "speed": "32x", "langs": 1,
     "desc": "English-only tiny; better WER than multilingual"},

    {"name": "base",          "repo": "openai/whisper-base",           "alias": "base",
     "params": "74M",  "size_gb": 0.29, "vram_gb": 1,  "format": "hf",
     "wer": 5.0,  "speed": "16x", "langs": 99,
     "desc": "Good balance of speed and accuracy for quick tasks"},

    {"name": "base.en",       "repo": "openai/whisper-base.en",        "alias": "base.en",
     "params": "74M",  "size_gb": 0.29, "vram_gb": 1,  "format": "hf",
     "wer": 4.2,  "speed": "16x", "langs": 1,
     "desc": "English-only base; slight accuracy improvement"},

    {"name": "small",         "repo": "openai/whisper-small",          "alias": "small",
     "params": "244M", "size_gb": 0.93, "vram_gb": 2,  "format": "hf",
     "wer": 3.4,  "speed": "6x",  "langs": 99,
     "desc": "Sweet spot for many use cases; multilingual"},

    {"name": "small.en",      "repo": "openai/whisper-small.en",       "alias": "small.en",
     "params": "244M", "size_gb": 0.93, "vram_gb": 2,  "format": "hf",
     "wer": 3.0,  "speed": "6x",  "langs": 1,
     "desc": "English-only small; popular for English transcription"},

    {"name": "medium",        "repo": "openai/whisper-medium",         "alias": "medium",
     "params": "769M", "size_gb": 2.9,  "vram_gb": 5,  "format": "hf",
     "wer": 2.9,  "speed": "2x",  "langs": 99,
     "desc": "High accuracy multilingual; good for production"},

    {"name": "medium.en",     "repo": "openai/whisper-medium.en",      "alias": "medium.en",
     "params": "769M", "size_gb": 2.9,  "vram_gb": 5,  "format": "hf",
     "wer": 2.7,  "speed": "2x",  "langs": 1,
     "desc": "English-only medium; strong general-purpose model"},

    {"name": "large-v2",      "repo": "openai/whisper-large-v2",       "alias": "large-v2",
     "params": "1.5B", "size_gb": 5.8,  "vram_gb": 10, "format": "hf",
     "wer": 2.7,  "speed": "1x",  "langs": 99,
     "desc": "Previous best; excellent multilingual quality"},

    {"name": "large-v3",      "repo": "openai/whisper-large-v3",       "alias": "large-v3",
     "params": "1.5B", "size_gb": 5.8,  "vram_gb": 10, "format": "hf",
     "wer": 2.4,  "speed": "1x",  "langs": 99,
     "desc": "Best overall accuracy; 99 languages supported"},

    {"name": "large-v3-turbo","repo": "openai/whisper-large-v3-turbo", "alias": "large-v3-turbo",
     "params": "809M", "size_gb": 3.1,  "vram_gb": 6,  "format": "hf",
     "wer": 2.5,  "speed": "4x",  "langs": 99,
     "desc": "Near large-v3 quality at 4x speed — best value"},
]

_DISTIL_MODELS = [
    # ── Distil-Whisper (distilled for English, much faster) ──
    {"name": "distil-small.en",  "repo": "distil-whisper/distil-small.en",  "alias": "distil-small.en",
     "params": "166M", "size_gb": 0.63, "vram_gb": 2,  "format": "hf",
     "wer": 3.3,  "speed": "12x", "langs": 1,
     "desc": "Distilled small; 6x faster than whisper-small"},

    {"name": "distil-medium.en", "repo": "distil-whisper/distil-medium.en", "alias": "distil-medium.en",
     "params": "394M", "size_gb": 1.5,  "vram_gb": 3,  "format": "hf",
     "wer": 2.9,  "speed": "6x",  "langs": 1,
     "desc": "Distilled medium; great speed-accuracy tradeoff"},

    {"name": "distil-large-v2",  "repo": "distil-whisper/distil-large-v2",  "alias": "distil-large-v2",
     "params": "756M", "size_gb": 2.9,  "vram_gb": 5,  "format": "hf",
     "wer": 2.6,  "speed": "5x",  "langs": 1,
     "desc": "Near large-v2 quality at 5x speed"},

    {"name": "distil-large-v3",  "repo": "distil-whisper/distil-large-v3",  "alias": "distil-large-v3",
     "params": "756M", "size_gb": 2.9,  "vram_gb": 5,  "format": "hf",
     "wer": 2.5,  "speed": "5x",  "langs": 1,
     "desc": "Best distilled model; near large-v3 quality"},
]

_CT2_MODELS = [
    # ── faster-whisper CTranslate2 format (Systran) ──
    {"name": "faster-tiny",        "repo": "Systran/faster-whisper-tiny",           "alias": "ct2-tiny",
     "params": "39M",  "size_gb": 0.08, "vram_gb": 1,  "format": "ct2",
     "wer": 7.6,  "speed": "50x+", "langs": 99,
     "desc": "CTranslate2 tiny; fastest possible inference"},

    {"name": "faster-base",        "repo": "Systran/faster-whisper-base",           "alias": "ct2-base",
     "params": "74M",  "size_gb": 0.15, "vram_gb": 1,  "format": "ct2",
     "wer": 5.0,  "speed": "25x",  "langs": 99,
     "desc": "CTranslate2 base; very fast and lightweight"},

    {"name": "faster-small",       "repo": "Systran/faster-whisper-small",          "alias": "ct2-small",
     "params": "244M", "size_gb": 0.47, "vram_gb": 2,  "format": "ct2",
     "wer": 3.4,  "speed": "10x",  "langs": 99,
     "desc": "CTranslate2 small; great for real-time use"},

    {"name": "faster-medium",      "repo": "Systran/faster-whisper-medium",         "alias": "ct2-medium",
     "params": "769M", "size_gb": 1.5,  "vram_gb": 4,  "format": "ct2",
     "wer": 2.9,  "speed": "4x",   "langs": 99,
     "desc": "CTranslate2 medium; production quality"},

    {"name": "faster-large-v2",    "repo": "Systran/faster-whisper-large-v2",       "alias": "ct2-large-v2",
     "params": "1.5B", "size_gb": 2.9,  "vram_gb": 8,  "format": "ct2",
     "wer": 2.7,  "speed": "2x",   "langs": 99,
     "desc": "CTranslate2 large-v2; high accuracy with speed"},

    {"name": "faster-large-v3",    "repo": "Systran/faster-whisper-large-v3",       "alias": "ct2-large-v3",
     "params": "1.5B", "size_gb": 2.9,  "vram_gb": 8,  "format": "ct2",
     "wer": 2.4,  "speed": "2x",   "langs": 99,
     "desc": "CTranslate2 large-v3; best quality + fast"},

    {"name": "faster-large-v3-turbo","repo": "Systran/faster-whisper-large-v3-turbo","alias": "ct2-large-v3-turbo",
     "params": "809M", "size_gb": 1.6,  "vram_gb": 5,  "format": "ct2",
     "wer": 2.5,  "speed": "8x",   "langs": 99,
     "desc": "CTranslate2 turbo; best speed-accuracy balance"},
]

_ALL_MODELS = _WHISPER_MODELS + _DISTIL_MODELS + _CT2_MODELS

_SIZE_TIERS = [
    ("TINY (< 100M params)",   lambda m: _parse_params(m["params"]) < 100),
    ("SMALL (100M–500M)",      lambda m: 100 <= _parse_params(m["params"]) < 500),
    ("MEDIUM (500M–1B)",       lambda m: 500 <= _parse_params(m["params"]) < 1000),
    ("LARGE (1B+)",            lambda m: _parse_params(m["params"]) >= 1000),
]


def _parse_params(s):
    """Parse '39M' or '1.5B' to millions."""
    s = s.strip().upper()
    if s.endswith("B"):
        return float(s[:-1]) * 1000
    if s.endswith("M"):
        return float(s[:-1])
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download_model(repo_id, alias, model_format="hf"):
    """Download a Whisper model from HuggingFace."""
    print(f"\n{_CYAN}⬇  Downloading {repo_id}...{_RESET}")

    if model_format == "ct2":
        # CTranslate2 models: download everything using snapshot_download
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print(f"{_RED}[ERROR] huggingface-hub not installed. Run: pip install huggingface-hub{_RESET}")
            return None

        try:
            path = snapshot_download(repo_id=repo_id)
            print(f"{_GREEN}✅ Downloaded to: {path}{_RESET}")
            _update_registry(repo_id, alias, model_format="ct2")
            return path
        except Exception as e:
            print(f"{_RED}[ERROR] Download failed: {e}{_RESET}")
            return None

    else:
        # HuggingFace transformers models: use from_pretrained (downloads + caches)
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError:
            print(f"{_RED}[ERROR] transformers not installed. Run: pip install transformers{_RESET}")
            return None

        try:
            print(f"   Downloading model weights...")
            model = WhisperForConditionalGeneration.from_pretrained(repo_id)
            print(f"   Downloading processor/tokenizer...")
            processor = WhisperProcessor.from_pretrained(repo_id)
            # Free from memory — we just wanted to cache the files
            del model, processor

            # Find where it was cached
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo_info in cache_info.repos:
                if repo_info.repo_id == repo_id:
                    print(f"{_GREEN}✅ Cached at: {repo_info.repo_path}{_RESET}")
                    _update_registry(repo_id, alias, model_format="hf")
                    return str(repo_info.repo_path)

            print(f"{_GREEN}✅ Downloaded successfully (cached in HuggingFace hub){_RESET}")
            _update_registry(repo_id, alias, model_format="hf")
            return repo_id

        except Exception as e:
            print(f"{_RED}[ERROR] Download failed: {e}{_RESET}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Table renderer
# ──────────────────────────────────────────────────────────────────────────────

def _print_catalog(models, gpu_name, gpu_vram, title="Whisper Models"):
    """Render the model catalog table."""
    w = 130

    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'🎙️  ' + title:^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")

    if gpu_name:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_name}{_RESET} ({gpu_vram} GB)")
    elif gpu_vram > 0:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_vram} GB VRAM{_RESET}")
    else:
        print(f"  {_YELLOW}⚠  No GPU detected — models will run on CPU.{_RESET}")

    print(f"  {_DIM}WER = Word Error Rate on LibriSpeech test-clean (lower is better){_RESET}")
    print()

    # Header
    hdr = (
        f"  {_BOLD}{_WHITE}"
        f"{'#':>3} │ {'Model':<22} │ {'Params':<7} │ {'Size':>8} │ {'VRAM':>6} │ "
        f"{'WER':>5} │ {'Speed':>7} │ {'Langs':>5} │ {'Status':<12} │ {'Fit':<6}"
        f"{_RESET}"
    )

    catalog = []
    idx = 0

    # Group by format family
    families = [
        ("⚡ OpenAI Whisper (HuggingFace Transformers)", [m for m in models if m["format"] == "hf" and "distil" not in m["repo"]]),
        ("🏎️  Distil-Whisper (Distilled, English-focused)", [m for m in models if "distil" in m["repo"]]),
        ("🚀 faster-whisper (CTranslate2, optimized)", [m for m in models if m["format"] == "ct2"]),
    ]

    for family_name, family_models in families:
        if not family_models:
            continue

        print(f"{_BOLD}{_MAGENTA}  ── {family_name} {'─' * max(1, w - len(family_name) - 7)}{_RESET}")
        print(hdr)
        print(f"  {'─' * (w - 4)}")

        for m in family_models:
            idx += 1
            catalog.append(m)

            # Download status
            downloaded = _is_installed(m["repo"]) or _check_hf_cache(m["repo"])
            status = f"{_GREEN}Downloaded{_RESET}" if downloaded else f"{_DIM}Not yet{_RESET}"

            # GPU fit
            if gpu_vram > 0:
                if m["vram_gb"] + 0.5 <= gpu_vram:
                    fit = f"{_GREEN}✅{_RESET}"
                elif m["vram_gb"] <= gpu_vram:
                    fit = f"{_YELLOW}⚠️{_RESET}"
                else:
                    fit = f"{_RED}❌{_RESET}"
            else:
                fit = f"{_DIM}?{_RESET}"

            # WER coloring (lower is better)
            if m["wer"] <= 2.5:
                wer_c = _GREEN
            elif m["wer"] <= 3.5:
                wer_c = _YELLOW
            else:
                wer_c = _DIM

            print(
                f"  {_BOLD}{idx:>3}{_RESET} │ "
                f"{_CYAN}{m['name']:<22}{_RESET} │ "
                f"{m['params']:<7} │ "
                f"{m['size_gb']:>7.2f}G │ "
                f"{m['vram_gb']:>4} GB │ "
                f"{wer_c}{m['wer']:>5.1f}{_RESET} │ "
                f"{m['speed']:>7} │ "
                f"{m['langs']:>5} │ "
                f"{status:<21} │ "
                f"{fit}"
            )
            print(f"  {'':>3}   {_DIM}{m['desc']}{_RESET}")

        print()

    # Recommended model
    if gpu_vram > 0:
        fitting = [m for m in models if m["vram_gb"] + 0.5 <= gpu_vram]
        if fitting:
            best = min(fitting, key=lambda m: m["wer"])
            print(f"  {_GREEN}💡 Recommended for your GPU: {_BOLD}{best['name']}{_RESET} "
                  f"{_GREEN}(WER: {best['wer']}, Speed: {best['speed']}){_RESET}")
            print()

    print(f"  {_DIM}Select a # to download, 'a' download all recommended, or 'q' quit.{_RESET}")
    print()

    return catalog


# ──────────────────────────────────────────────────────────────────────────────
# Interactive loop
# ──────────────────────────────────────────────────────────────────────────────

def _interactive_loop(catalog, gpu_vram=0):
    """Main interactive loop: pick a model and download."""
    total = len(catalog)
    if total == 0:
        print(f"{_YELLOW}  No models to show.{_RESET}")
        return

    while True:
        try:
            raw = input(
                f"{_BOLD}Enter # (1-{total}), 'a' all recommended, or 'q' quit: {_RESET}"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if raw.lower() in ("q", "quit", "exit"):
            print("Bye!")
            return

        if raw.lower() in ("a", "all"):
            # Download all models that fit on this GPU (or all if no GPU)
            fitting = [m for m in catalog if gpu_vram <= 0 or m["vram_gb"] + 0.5 <= gpu_vram]
            if not fitting:
                print(f"{_YELLOW}  No models fit your GPU.{_RESET}")
                continue
            print(f"\n{_BOLD}Downloading {len(fitting)} model(s)...{_RESET}")
            for m in fitting:
                if _is_installed(m["repo"]) or _check_hf_cache(m["repo"]):
                    print(f"  {_DIM}Skipping {m['name']} — already downloaded{_RESET}")
                    continue
                _download_model(m["repo"], m["alias"], m["format"])
            print(f"\n{_GREEN}✅ All downloads complete!{_RESET}")
            continue

        try:
            choice = int(raw)
        except ValueError:
            print(f"{_YELLOW}  Enter a number 1-{total}, 'a', or 'q'.{_RESET}")
            continue

        if choice < 1 or choice > total:
            print(f"{_YELLOW}  Enter a number 1-{total}, 'a', or 'q'.{_RESET}")
            continue

        m = catalog[choice - 1]
        if _is_installed(m["repo"]) or _check_hf_cache(m["repo"]):
            print(f"{_GREEN}  Already downloaded: {m['name']}{_RESET}")
            try:
                raw2 = input(f"  Re-download? (y/N): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if raw2.lower() != "y":
                continue

        _download_model(m["repo"], m["alias"], m["format"])
        print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    _enable_ansi_windows()
    _ensure_deps()

    parser = argparse.ArgumentParser(
        description="Browse and download Whisper speech recognition models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python install_whisper_models.py                      # interactive browser
              python install_whisper_models.py --list               # print table and exit
              python install_whisper_models.py --gpu 5090           # show GPU fit
              python install_whisper_models.py --format ct2         # show faster-whisper only
              python install_whisper_models.py --format hf          # show HF transformers only
              python install_whisper_models.py --download large-v3  # download by alias

            Model formats:
              hf   - HuggingFace Transformers (for transformers/whisper pipeline)
              ct2  - CTranslate2 (for faster-whisper, optimized inference)
        """),
    )
    parser.add_argument(
        "--gpu", default=None,
        help="Override GPU type (e.g. 3090, 4090, 5090) or raw VRAM in GB",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print model table and exit (no interactive download)",
    )
    parser.add_argument(
        "--format", choices=["hf", "ct2", "all"], default="all",
        help="Filter by model format: hf (transformers), ct2 (faster-whisper), or all (default)",
    )
    parser.add_argument(
        "--download", default=None, metavar="ALIAS",
        help="Download a model by alias (e.g. 'large-v3', 'ct2-large-v3-turbo')",
    )

    args = parser.parse_args()

    # Resolve GPU
    gpu_name, gpu_vram = None, 0
    if args.gpu:
        gpu_name = f"Manual override: {args.gpu}"
        gpu_vram = _vram_for_gpu_flag(args.gpu)
        if gpu_vram == 0:
            print(f"{_YELLOW}[WARNING] Unrecognised GPU '{args.gpu}'.{_RESET}")
    else:
        gpu_name, gpu_vram = _detect_gpu()

    # Filter models by format
    if args.format == "hf":
        models = _WHISPER_MODELS + _DISTIL_MODELS
        title = "Whisper Models (HuggingFace Transformers)"
    elif args.format == "ct2":
        models = _CT2_MODELS
        title = "Whisper Models (CTranslate2 / faster-whisper)"
    else:
        models = _ALL_MODELS
        title = "Whisper Models — All Formats"

    # Direct download by alias
    if args.download:
        alias = args.download.strip().lower()
        match = None
        for m in _ALL_MODELS:
            if m["alias"].lower() == alias or m["name"].lower() == alias:
                match = m
                break
        if not match:
            print(f"{_RED}[ERROR] Unknown model alias '{args.download}'.{_RESET}")
            print(f"  Available aliases: {', '.join(m['alias'] for m in _ALL_MODELS)}")
            return
        _download_model(match["repo"], match["alias"], match["format"])
        return

    # Print catalog
    catalog = _print_catalog(models, gpu_name, gpu_vram, title=title)

    if args.list:
        return

    _interactive_loop(catalog, gpu_vram=gpu_vram)


if __name__ == "__main__":
    main()
