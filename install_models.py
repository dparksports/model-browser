#!/usr/bin/env python3
"""
Install Models & Downloader
──────────────────────────
Install models from HuggingFace, show Unsloth optimised models first, and keep track of installed models.

No hardcoded model lists — always shows the latest releases.

Usage::

    python install_models.py                     # top 10 Unsloth + 10 community
    python install_models.py --search "deepseek" # free-form search
    python install_models.py --top 20            # show top 20 per section
    python install_models.py --gpu 5090          # detect GPU for context
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
import time

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
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
# GPU detection
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
# Registry Helpers (installed_models.json)
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "installed_models.json")

def _load_registry():
    """Load installed models from JSON registry."""
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

def _update_registry(repo_id, filename, capabilities=""):
    """Register a newly downloaded model."""
    registry = _load_registry()
    
    # Store by a unique key (repo+filename)
    key = f"{repo_id}/{filename}"
    
    # Generate a simple alias for CLI usage (e.g. "phi-3-mini")
    # We strip "-GGUF" and "Instruct" to keep it short if possible
    alias_candidate = repo_id.split("/")[-1].replace("-GGUF", "").replace("-Instruct", "").replace("-it", "").lower()
    if "unsloth" in repo_id.lower() and not alias_candidate.startswith("unsloth"):
        alias_candidate = f"unsloth-{alias_candidate}"
    
    registry[key] = {
        "repo_id": repo_id,
        "filename": filename,
        "downloaded_at": time.time(),
        "capabilities": capabilities,
        "alias": alias_candidate
    }
    _save_registry(registry)
    print(f"{_GREEN}📝 Registered model in installed_models.json (alias: {alias_candidate}){_RESET}")

def _is_installed(repo_id):
    """Check if any file from this repo is in our registry."""
    registry = _load_registry()
    for k, v in registry.items():
        if v.get("repo_id") == repo_id:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace API helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_api():
    """Return HfApi instance or None."""
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError:
        print(f"{_RED}[ERROR] huggingface-hub not installed. Run: pip install huggingface-hub{_RESET}")
        return None


def _extract_params(repo_id):
    """Extract parameter count (e.g. '14B') from repo name."""
    import re
    # Match patterns like 14b, 7.5b, 70B, etc.
    m = re.search(r"[-_.](\d+(?:\.\d+)?)[bB]\b", repo_id)
    return m.group(1).upper() + "B" if m else "?"


def _infer_capabilities(repo_id, pipeline_tag=None, tags=None):
    """Infer model capabilities from name, pipeline_tag, and tags.
    Returns emoji string like '💬 🧠 💻'.
    """
    name_lower = repo_id.lower()
    caps = []
    tag_set = set(t.lower() for t in (tags or []))

    # Vision
    if any(kw in name_lower for kw in ("vl", "vision", "llava", "image-text")):
        caps.append("👁️")
    elif pipeline_tag in ("image-text-to-text", "image-to-text", "visual-question-answering"):
        caps.append("👁️")

    # Image generation
    if pipeline_tag in ("image-to-image", "text-to-image") or "flux" in name_lower:
        caps.append("🎨")

    # Chat / Conversation
    if any(kw in name_lower for kw in ("instruct", "chat", "it-")) or "conversational" in tag_set:
        caps.append("💬")
    elif pipeline_tag in ("text-generation",):
        caps.append("💬")

    # Code
    if any(kw in name_lower for kw in ("code", "coder", "devstral", "starcoder")):
        caps.append("💻")

    # Reasoning (larger models or known reasoning families)
    if any(kw in name_lower for kw in ("deepseek-r1", "qwen", "phi", "gemma", "llama", "gpt-oss")):
        # Check param size - bigger models get reasoning badge
        param_str = _extract_params(repo_id)
        try:
            param_val = float(param_str.rstrip("B"))
            if param_val >= 3:
                caps.append("🧠")
        except ValueError:
            caps.append("🧠")

    # Summarization (most instruction-tuned models)
    if any(kw in name_lower for kw in ("instruct", "it-", "chat")):
        caps.append("📋")

    # Multilingual
    if any(kw in name_lower for kw in ("qwen", "multilingual")) or "multilingual" in tag_set:
        caps.append("🌍")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return " ".join(unique[:5]) if unique else "💬"


def _fetch_top_models(author=None, limit=10):
    """Fetch top GGUF repos from HuggingFace sorted by downloads.

    Args:
        author: filter to a specific author (e.g. "unsloth"), or None for all.
        limit:  max results to return.

    Returns list of dicts: {repo_id, downloads, updated, params, type}
    """
    api = _get_api()
    if not api:
        return []

    search = f"{author + ' ' if author else ''}GGUF"
    try:
        models = list(api.list_models(
            search=search,
            sort="downloads",
            limit=limit * 3,
            cardData=True,  # fetch metadata for tags
        ))
    except Exception as e:
        print(f"{_RED}[ERROR] HuggingFace query failed: {e}{_RESET}")
        return []

    results = []
    for m in models:
        mid = m.id or ""
        if "gguf" not in mid.lower() and "GGUF" not in mid:
            continue
        if author and not mid.lower().startswith(author.lower() + "/"):
            continue
        
        dl = m.downloads if hasattr(m, "downloads") and m.downloads else 0
        updated = str(getattr(m, 'lastModified', None) or getattr(m, 'last_modified', None) or getattr(m, 'created_at', ''))[:10] or "?"
        
        # Extract params
        params = _extract_params(mid)
        
        # Extract type from pipeline_tag
        task = getattr(m, 'pipeline_tag', '?') or '?'
        if task == "text-generation": task = "text-gen"
        elif task == "image-text-to-text": task = "vision"
        
        # Infer capabilities
        model_tags = getattr(m, 'tags', []) or []
        caps = _infer_capabilities(mid, getattr(m, 'pipeline_tag', None), model_tags)
        
        results.append({
            "repo_id": mid, 
            "downloads": dl, 
            "updated": updated,
            "params": params,
            "type": task,
            "caps": caps,
        })
        if len(results) >= limit:
            break
    return results


def _fetch_model_details(repo_id):
    """Return tuple (files, metadata) where:
       files: list of dicts {name, size}
       metadata: dict with languages, license, tags, etc.
    """
    api = _get_api()
    if not api:
        return [], {}
    try:
        # Use model_info to get file sizes (siblings) + metadata
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        files = []
        for s in info.siblings:
            if s.rfilename.endswith(".gguf"):
                files.append({
                    "name": s.rfilename,
                    "size": s.size or 0
                })
        
        # Extract metadata
        card = info.cardData or {}
        meta = {
            "id": info.id,
            "downloads": info.downloads,
            "pipeline_tag": info.pipeline_tag,
            "languages": card.get("language"),
            "license": card.get("license"),
            "tags": info.tags,
            "updated": str(getattr(info, 'lastModified', ''))[:10]
        }
        return files, meta

    except Exception as e:
        # print(f"{_RED}  [ERROR] Could not fetch details: {e}{_RESET}")
        return [], {}


def _download_file(repo_id, filename, caps=""):
    """Download a single file from HuggingFace. Returns local path or None."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"{_RED}[ERROR] huggingface-hub not installed.{_RESET}")
        return None

    print(f"\n{_CYAN}⬇  Downloading {filename}...{_RESET}")
    print(f"   Repo: {repo_id}")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"{_GREEN}✅ Downloaded to: {path}{_RESET}")
        
        # Update registry!
        _update_registry(repo_id, filename, caps)
        
        return path
    except Exception as e:
        print(f"{_RED}[ERROR] Download failed: {e}{_RESET}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Table renderer
# ──────────────────────────────────────────────────────────────────────────────

def _print_table(unsloth_models, community_models, gpu_name, gpu_vram):
    """Render compact tables for live-fetched model repos."""
    w = 145

    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'🤖  Install Models (live from HuggingFace)':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")

    if gpu_name:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_name}{_RESET} ({gpu_vram} GB)")
    elif gpu_vram > 0:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_vram} GB VRAM{_RESET}")
    else:
        print(f"  {_YELLOW}⚠  No GPU detected.{_RESET}")
    print()

    def _param_sort_key(m):
        """Sort key: parse param string to float, unknowns go last."""
        p = m.get('params', '?')
        try:
            return -float(p.rstrip('B'))  # negative for descending
        except (ValueError, AttributeError):
            return 0  # unknowns at bottom

    def _estimate_q4_size(params_str):
        """Estimate Q4_K_M file size in GB from param count string."""
        try:
            p = float(params_str.rstrip('B'))
            return round(p * 0.6, 1)  # ~0.6 GB per billion params at Q4
        except (ValueError, AttributeError):
            return 0

    def _section(title, models, offset=0):
        if not models:
            return
        models = sorted(models, key=_param_sort_key)

        # Find the largest model that comfortably fits (for recommended tag)
        recommended_idx = -1
        if gpu_vram > 0:
            for j, m in enumerate(models):
                est = _estimate_q4_size(m['params'])
                if est > 0 and est + 2 <= gpu_vram:  # 2GB buffer for KV cache
                    if recommended_idx == -1:
                        recommended_idx = j  # first (largest) that fits

        print(f"{_BOLD}{_MAGENTA}  ── {title} {'─' * max(1, w - len(title) - 7)}{_RESET}")
        print(f"  {_BOLD}{_WHITE}{'#':>3}   {'Repository':<40}  {'Params':<6}  {'~Q4 Size':>8}  {'Capabilities':<14}  {'Status':<14} {'Downloads':>10}  {'Updated':<10}  {'Fit':<18}{_RESET}")
        print(f"  {'─' * (w - 4)}")
        for i, m in enumerate(models, offset + 1):
            est = _estimate_q4_size(m['params'])
            size_str = f"{est:.1f} GB" if est > 0 else "?"

            # GPU fit indicator
            if gpu_vram > 0 and est > 0:
                if est + 2 <= gpu_vram:
                    fit = f"{_GREEN}✅ fits{_RESET}"
                elif est <= gpu_vram:
                    fit = f"{_YELLOW}⚠️ tight{_RESET}"
                else:
                    fit = f"{_RED}❌ too big{_RESET}"
            else:
                fit = ""

            # Check installed status
            is_inst = _is_installed(m["repo_id"])
            status = f"{_GREEN}💾 Installed{_RESET}" if is_inst else ""

            # Recommended tag
            rec = f" {_GREEN}← recommended{_RESET}" if (i - offset - 1) == recommended_idx and not is_inst else ""

            dl = f"{m['downloads']:,}" if m['downloads'] else "?"
            print(
                f"  {_BOLD}{i:>3}{_RESET}   "
                f"{_CYAN}{m['repo_id']:<40}{_RESET}  "
                f"{m['params']:<6}  "
                f"{size_str:>8}  "
                f"{m.get('caps', ''):<14}  "
                f"{status:<14} "
                f"{dl:>10}  {m['updated']:<10}  "
                f"{fit}{rec}"
            )
        print()

    _section("⚡ UNSLOTH — top by downloads", unsloth_models, offset=0)
    _section("COMMUNITY — top by downloads", community_models, offset=len(unsloth_models))

    print(f"  {_DIM}Select a # to browse GGUF files, 's' search, 'c' capabilities, 'b' benchmarks, 'q' quit.{_RESET}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Interactive selection
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Curated Model Capabilities Catalog
# ──────────────────────────────────────────────────────────────────────────────

# Capability legend:  💬 Chat  🧠 Reasoning  💻 Code  📋 Summarize  🌍 Multilingual  👁️ Vision  🎨 Image-Gen
# Benchmark scores: MMLU (knowledge), HumanEval (code), MT-Bench (chat quality) — approximate, from public leaderboards
_MODEL_CATALOG = [
    # ── TINY (≤ 3B) ──
    {"name": "Qwen2.5-0.5B",    "repo": "unsloth/Qwen2.5-0.5B-Instruct-GGUF",    "params": "0.5B", "size_gb": 0.4, "vram_gb": 2,
     "maker": "Alibaba",  "caps": "💬 📋",         "mmlu": 45,  "humaneval": 30, "mtbench": 5.2,
     "desc": "Ultra-lightweight; great for quick tests and edge deployment"},
    {"name": "Qwen2.5-1.5B",    "repo": "unsloth/Qwen2.5-1.5B-Instruct-GGUF",    "params": "1.5B", "size_gb": 1.1, "vram_gb": 3,
     "maker": "Alibaba",  "caps": "💬 📋 🌍",       "mmlu": 58,  "humaneval": 37, "mtbench": 6.0,
     "desc": "Compact with strong multilingual ability; 29 languages"},
    {"name": "Gemma-2-2B",      "repo": "unsloth/gemma-2-2b-it-GGUF",             "params": "2.6B", "size_gb": 1.8, "vram_gb": 4,
     "maker": "Google",   "caps": "💬 🧠 📋",       "mmlu": 52,  "humaneval": 28, "mtbench": 6.4,
     "desc": "Google's efficient small model with solid reasoning"},
    {"name": "Phi-3-Mini",      "repo": "unsloth/Phi-3-mini-4k-instruct-GGUF",    "params": "3.8B", "size_gb": 2.4, "vram_gb": 5,
     "maker": "Microsoft","caps": "💬 🧠 💻 📋",    "mmlu": 69,  "humaneval": 59, "mtbench": 7.6,
     "desc": "Punches way above its weight; strong reasoning + code"},
    {"name": "Gemma-3-4B",      "repo": "unsloth/gemma-3-4b-it-GGUF",             "params": "4B",   "size_gb": 2.3, "vram_gb": 5,
     "maker": "Google",   "caps": "💬 🧠 👁️ 📋",   "mmlu": 60,  "humaneval": 42, "mtbench": 7.2,
     "desc": "Latest small vision+chat model with image understanding"},

    # ── SMALL (4–7B) ──
    {"name": "Qwen2.5-7B",      "repo": "unsloth/Qwen2.5-7B-Instruct-GGUF",       "params": "7B",  "size_gb": 4.7, "vram_gb": 8,
     "maker": "Alibaba",  "caps": "💬 🧠 💻 📋 🌍", "mmlu": 74,  "humaneval": 75, "mtbench": 7.8,
     "desc": "Excellent all-rounder; strong code + 29-language support"},
    {"name": "Llama-3.2-3B",    "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF",     "params": "3.2B", "size_gb": 2.0, "vram_gb": 4,
     "maker": "Meta",     "caps": "💬 🧠 📋",       "mmlu": 63,  "humaneval": 48, "mtbench": 7.0,
     "desc": "Compact Llama; fast inference and strong ecosystem"},
    {"name": "Mistral-7B",      "repo": "unsloth/mistral-7b-instruct-v0.3-GGUF",  "params": "7B",   "size_gb": 4.4, "vram_gb": 8,
     "maker": "Mistral",  "caps": "💬 🧠 💻 📋",    "mmlu": 63,  "humaneval": 55, "mtbench": 7.5,
     "desc": "Pioneered efficient sliding-window attention architecture"},
    {"name": "DeepSeek-R1-8B",  "repo": "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", "params": "8B",   "size_gb": 4.9, "vram_gb": 8,
     "maker": "DeepSeek", "caps": "💬 🧠 💻 📋",    "mmlu": 72,  "humaneval": 67, "mtbench": 7.9,
     "desc": "Reasoning specialist; excellent chain-of-thought ability"},
    {"name": "GLM-4.7-Flash",   "repo": "unsloth/GLM-4.7-Flash-GGUF",             "params": "4.7B", "size_gb": 3.0, "vram_gb": 5,
     "maker": "Zhipu AI", "caps": "💬 🧠 📋",       "mmlu": 62,  "humaneval": 45, "mtbench": 7.1,
     "desc": "Extremely fast inference; optimized for general chat"},

    # ── MEDIUM (8–14B) ──
    {"name": "Gemma-3-12B",     "repo": "unsloth/gemma-3-12b-it-GGUF",            "params": "12B",  "size_gb": 7.6, "vram_gb": 12,
     "maker": "Google",   "caps": "💬 🧠 👁️ 📋 🌍","mmlu": 74,  "humaneval": 56, "mtbench": 8.0,
     "desc": "Mid-range vision model; excellent image + text quality"},
    {"name": "Qwen2.5-14B",    "repo": "unsloth/Qwen2.5-14B-Instruct-GGUF",      "params": "14B",  "size_gb": 9.0, "vram_gb": 14,
     "maker": "Alibaba",  "caps": "💬 🧠 💻 📋 🌍", "mmlu": 80,  "humaneval": 80, "mtbench": 8.2,
     "desc": "Near GPT-4 on many benchmarks; strong code + reasoning"},
    {"name": "Phi-4-14B",      "repo": "unsloth/phi-4-GGUF",                     "params": "14B",  "size_gb": 8.7, "vram_gb": 14,
     "maker": "Microsoft","caps": "💬 🧠 💻 📋",    "mmlu": 81,  "humaneval": 82, "mtbench": 8.3,
     "desc": "Exceptional reasoning + code; top of its class"},
    {"name": "Llama-3.1-8B",   "repo": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF","params": "8B",   "size_gb": 4.9, "vram_gb": 8,
     "maker": "Meta",     "caps": "💬 🧠 💻 📋",    "mmlu": 69,  "humaneval": 62, "mtbench": 7.7,
     "desc": "Industry workhorse; massive fine-tune ecosystem"},

    # ── LARGE (15–32B) ──
    {"name": "Gemma-3-27B",    "repo": "unsloth/gemma-3-27b-it-GGUF",            "params": "27B",  "size_gb": 16.7,"vram_gb": 20,
     "maker": "Google",   "caps": "💬 🧠 👁️ 📋 🌍","mmlu": 82,  "humaneval": 68, "mtbench": 8.5,
     "desc": "Flagship open model; near-frontier quality with vision"},
    {"name": "Qwen3-30B-A3B",  "repo": "unsloth/Qwen3-30B-A3B-GGUF",             "params": "30B",  "size_gb": 2.4, "vram_gb": 5,
     "maker": "Alibaba",  "caps": "💬 🧠 💻 📋 🌍", "mmlu": 79,  "humaneval": 78, "mtbench": 8.1,
     "desc": "MoE: 30B total but only 3B active; blazing fast"},
    {"name": "Devstral-24B",   "repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF","params":"24B","size_gb":15.0,"vram_gb":18,
     "maker": "Mistral",  "caps": "💬 💻 🧠",       "mmlu": 72,  "humaneval": 84, "mtbench": 7.8,
     "desc": "Code specialist built for agentic coding workflows"},
    {"name": "Qwen3-Coder-30B","repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF","params":"30B", "size_gb": 2.4, "vram_gb": 5,
     "maker": "Alibaba",  "caps": "💻 🧠 💬",       "mmlu": 75,  "humaneval": 86, "mtbench": 7.9,
     "desc": "MoE code specialist; 30B sparse, optimized for dev tasks"},
    {"name": "GPT-OSS-20B",    "repo": "unsloth/gpt-oss-20b-GGUF",               "params": "20B",  "size_gb": 12.5,"vram_gb": 16,
     "maker": "OpenAI",   "caps": "💬 🧠 💻 📋",    "mmlu": 78,  "humaneval": 76, "mtbench": 8.4,
     "desc": "OpenAI's first open-weight model; broad capabilities"},

    # ── XL (>32B) ──
    {"name": "Qwen2.5-72B",    "repo": "unsloth/Qwen2.5-72B-Instruct-GGUF",      "params": "72B",  "size_gb": 44.0,"vram_gb": 48,
     "maker": "Alibaba",  "caps": "💬 🧠 💻 📋 🌍", "mmlu": 86,  "humaneval": 86, "mtbench": 8.8,
     "desc": "Frontier-class; rivals GPT-4 across most benchmarks"},
    {"name": "Llama-3.1-70B",  "repo": "unsloth/Meta-Llama-3.1-70B-Instruct-GGUF","params":"70B",  "size_gb": 43.0,"vram_gb": 48,
     "maker": "Meta",     "caps": "💬 🧠 💻 📋",    "mmlu": 84,  "humaneval": 81, "mtbench": 8.7,
     "desc": "Meta's large Llama; industry benchmark standard"},
    {"name": "GPT-OSS-120B",   "repo": "ggml-org/gpt-oss-120b-GGUF",              "params": "120B", "size_gb": 73.0,"vram_gb": 80,
     "maker": "OpenAI",   "caps": "💬 🧠 💻 📋 🌍", "mmlu": 88,  "humaneval": 88, "mtbench": 9.0,
     "desc": "OpenAI's largest open model; frontier quality"},
]

# Commercial model baselines for comparison (cannot run locally)
_COMMERCIAL_BASELINES = [
    {"name": "GPT-4o",          "maker": "OpenAI",    "mmlu": 88, "humaneval": 90, "mtbench": 9.2, "note": "cloud only"},
    {"name": "GPT-4o-mini",     "maker": "OpenAI",    "mmlu": 82, "humaneval": 87, "mtbench": 8.6, "note": "cloud only"},
    {"name": "Claude 3.5 Sonnet","maker": "Anthropic", "mmlu": 89, "humaneval": 92, "mtbench": 9.1, "note": "cloud only"},
    {"name": "Claude 3 Haiku",  "maker": "Anthropic",  "mmlu": 75, "humaneval": 75, "mtbench": 8.0, "note": "cloud only"},
    {"name": "Gemini 2.0 Flash","maker": "Google",     "mmlu": 87, "humaneval": 89, "mtbench": 9.0, "note": "cloud only"},
    {"name": "Gemini 1.5 Pro",  "maker": "Google",     "mmlu": 86, "humaneval": 84, "mtbench": 8.8, "note": "cloud only"},
]


def _print_benchmark_comparison(gpu_vram):
    """Print benchmarks comparing local models vs commercial APIs."""
    w = 120
    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'📊  Model Benchmark Comparison — Local vs Commercial':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"  {_DIM}Scores are approximate, from public leaderboards. Higher = better.{_RESET}")
    print(f"  {_DIM}MMLU = knowledge (0-100)  |  HumanEval = code (0-100)  |  MT-Bench = chat quality (1-10){_RESET}")
    print()

    # Header
    hdr = (
        f"  {_BOLD}{_WHITE}"
        f"{'Model':<22} {'Maker':<12} {'Params':<8} {'MMLU':>6} {'HumanEval':>10} {'MT-Bench':>9}  {'Fit':<14} {'Notes':<20}"
        f"{_RESET}"
    )
    sep = f"  {'─' * (w - 4)}"

    # --- Commercial baselines ---
    print(f"{_BOLD}{_MAGENTA}  ── ☁️  COMMERCIAL APIs (for reference) {'─' * (w - 44)}{_RESET}")
    print(hdr)
    print(sep)
    for m in _COMMERCIAL_BASELINES:
        print(
            f"  {_DIM}{m['name']:<22}{_RESET} "
            f"{m['maker']:<12} "
            f"{'N/A':<8} "
            f"{m['mmlu']:>6} "
            f"{m['humaneval']:>10} "
            f"{m['mtbench']:>9.1f}  "
            f"{'☁️ API only':<14} "
            f"{_DIM}{m.get('note','')}{_RESET}"
        )
    print()

    # --- Local models that fit on this GPU, sorted by MT-Bench descending ---
    fitting = [m for m in _MODEL_CATALOG if m["vram_gb"] + 1 <= gpu_vram] if gpu_vram > 0 else _MODEL_CATALOG
    fitting = sorted(fitting, key=lambda m: m.get("mtbench", 0), reverse=True)[:12]

    print(f"{_BOLD}{_MAGENTA}  ── 🖥️  BEST LOCAL MODELS (fits your GPU) {'─' * (w - 46)}{_RESET}")
    print(hdr)
    print(sep)
    for m in fitting:
        # Color the scores relative to GPT-4o (88, 90, 9.2)
        mmlu_c = _GREEN if m["mmlu"] >= 80 else (_YELLOW if m["mmlu"] >= 70 else _DIM)
        he_c   = _GREEN if m["humaneval"] >= 80 else (_YELLOW if m["humaneval"] >= 60 else _DIM)
        mt_c   = _GREEN if m["mtbench"] >= 8.5 else (_YELLOW if m["mtbench"] >= 7.5 else _DIM)

        fit_str = f"{_GREEN}✅ fits{_RESET}"

        print(
            f"  {_CYAN}{m['name']:<22}{_RESET} "
            f"{m['maker']:<12} "
            f"{m['params']:<8} "
            f"{mmlu_c}{m['mmlu']:>6}{_RESET} "
            f"{he_c}{m['humaneval']:>10}{_RESET} "
            f"{mt_c}{m['mtbench']:>9.1f}{_RESET}  "
            f"{fit_str:<23} "
            f"{_DIM}{m['desc'][:35]}{_RESET}"
        )
    print()
    print(f"  {_DIM}💡 Green = near commercial quality | Yellow = competitive | Gray = basic{_RESET}")
    print()

_SIZE_TIERS = [
    ("TINY (≤ 3B params)",   lambda m: float(m["params"].rstrip("B")) <= 3),
    ("SMALL (4-7B params)",  lambda m: 3 < float(m["params"].rstrip("B")) <= 7),
    ("MEDIUM (8-14B params)",lambda m: 7 < float(m["params"].rstrip("B")) <= 14),
    ("LARGE (15-32B params)",lambda m: 14 < float(m["params"].rstrip("B")) <= 32),
    ("XL (>32B params)",     lambda m: float(m["params"].rstrip("B")) > 32),
]


def _check_downloaded_legacy(repo_id):
    """Check if any GGUF files from this repo exist in the HuggingFace cache."""
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.exists():
            # Windows: also check default HF cache location
            cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if not cache_dir.exists():
            return False
        # HF cache uses repo id format: models--author--name
        safe_name = "models--" + repo_id.replace("/", "--")
        repo_cache = cache_dir / safe_name
        if repo_cache.exists():
            # Check for any .gguf files in snapshots
            for p in repo_cache.rglob("*.gguf"):
                return True
    except Exception:
        pass
    return False


def _print_capabilities_catalog(gpu_vram):
    """Print the curated model capabilities catalog."""
    w = 110
    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'🧭  Model Capabilities Catalog':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"  {_DIM}Legend: 💬 Chat  🧠 Reasoning  💻 Code  📋 Summarize  🌍 Multilingual  👁️ Vision  🎨 Image-Gen{_RESET}")
    print()

    # Header
    hdr = (
        f"  {_BOLD}{_WHITE}"
        f"{'#':>3} │ {'Model':<20} │ {'Maker':<10} │ {'Params':<7} │ {'Size':>8} │ {'Min VRAM':>10} │ {'Capabilities':<22} │ {'Status':<13} │ {'Fit':<6}"
        f"{_RESET}"
    )

    catalog_models = []  # for interactive selection
    idx = 0

    for tier_name, tier_fn in _SIZE_TIERS:
        tier_models = [m for m in _MODEL_CATALOG if tier_fn(m)]
        if not tier_models:
            continue

        print(f"{_BOLD}{_MAGENTA}  ── {tier_name} {'─' * max(1, w - len(tier_name) - 7)}{_RESET}")
        print(hdr)
        print(f"  {'─' * (w - 4)}")

        for m in tier_models:
            idx += 1
            catalog_models.append(m)

            # Check download status
            downloaded = _is_installed(m["repo"]) or _check_downloaded_legacy(m["repo"])
            status = f"{_GREEN}Downloaded{_RESET}" if downloaded else f"{_DIM}Not yet{_RESET}"

            # GPU fit
            if gpu_vram > 0:
                if m["vram_gb"] + 1 <= gpu_vram:
                    fit = f"{_GREEN}✅{_RESET}"
                elif m["vram_gb"] <= gpu_vram:
                    fit = f"{_YELLOW}⚠️{_RESET}"
                else:
                    fit = f"{_RED}❌{_RESET}"
            else:
                fit = f"{_DIM}?{_RESET}"

            print(
                f"  {_BOLD}{idx:>3}{_RESET} │ "
                f"{_CYAN}{m['name']:<20}{_RESET} │ "
                f"{m.get('maker', '?'):<10} │ "
                f"{m['params']:<7} │ "
                f"{m['size_gb']:>7.1f} GB │ "
                f"{m['vram_gb']:>7} GB │ "
                f"{m['caps']:<22} │ "
                f"{status:<22} │ "
                f"{fit}"
            )
            print(f"  {'':>3}   {_DIM}{m['desc']}{_RESET}")

        print()

    return catalog_models


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_size(size_bytes):
    """Format bytes to GB."""
    return f"{size_bytes / (1024**3):.1f} GB"

def _print_model_header(meta):
    """Print a nice header with model capabilities."""
    if not meta: return
    
    model_id = meta.get('id', '?')
    print(f"{_BOLD}{_CYAN}{'═' * 110}{_RESET}")
    print(f"  {_BOLD}Model:{_RESET}   {_GREEN}{model_id}{_RESET}")
    
    # Look up in curated catalog for maker + benchmarks
    catalog_entry = None
    for cm in _MODEL_CATALOG:
        if cm["repo"] == model_id or model_id.endswith(cm["repo"].split("/")[-1]):
            catalog_entry = cm
            break
    
    # Line 2: Maker & Stats
    maker = catalog_entry.get("maker", "?") if catalog_entry else "?"
    dl = f"{meta.get('downloads', 0):,}"
    updated = meta.get('updated', '?')
    task = meta.get('pipeline_tag', '?')
    print(f"  {_DIM}Maker:{_RESET}   {_BOLD}{maker}{_RESET} • {dl} downloads • Updated {updated} • Type: {_BOLD}{task}{_RESET}")
    
    # Line 3: Languages & License
    langs = meta.get('languages')
    if isinstance(langs, list): langs = ", ".join(langs[:5])
    elif not langs: langs = "?"
    
    lic = meta.get('license')
    if isinstance(lic, list): lic = ", ".join(lic)
    elif not lic: lic = "?"
    
    print(f"  {_DIM}Specs:{_RESET}   Languages: {_BOLD}{langs}{_RESET} • License: {_BOLD}{lic}{_RESET}")
    
    # Line 4: Benchmarks (if in catalog)
    if catalog_entry:
        mmlu = catalog_entry.get("mmlu", 0)
        he = catalog_entry.get("humaneval", 0)
        mt = catalog_entry.get("mtbench", 0)
        
        # Color relative to GPT-4o baselines (88, 90, 9.2)
        mmlu_c = _GREEN if mmlu >= 80 else (_YELLOW if mmlu >= 70 else _DIM)
        he_c = _GREEN if he >= 80 else (_YELLOW if he >= 60 else _DIM)
        mt_c = _GREEN if mt >= 8.5 else (_YELLOW if mt >= 7.5 else _DIM)
        
        print(f"  {_DIM}Bench:{_RESET}   "
              f"MMLU: {mmlu_c}{mmlu}{_RESET}/100  "
              f"HumanEval: {he_c}{he}{_RESET}/100  "
              f"MT-Bench: {mt_c}{mt:.1f}{_RESET}/10")
        
        # Comparison line
        gpt4o_mt = 9.2
        pct = int(mt / gpt4o_mt * 100)
        if pct >= 90:
            verdict = f"{_GREEN}🏆 {pct}% of GPT-4o quality — near-commercial grade!{_RESET}"
        elif pct >= 80:
            verdict = f"{_YELLOW}⭐ {pct}% of GPT-4o quality — very competitive{_RESET}"
        elif pct >= 70:
            verdict = f"{_YELLOW}👍 {pct}% of GPT-4o quality — solid for local use{_RESET}"
        else:
            verdict = f"{_DIM}📊 {pct}% of GPT-4o quality — lightweight model{_RESET}"
        print(f"  {_DIM}vs AI:{_RESET}   {verdict}")
        print(f"  {_DIM}About:{_RESET}   {catalog_entry['desc']}")
    
    # Line 5: Tags (filtered)
    tags = meta.get('tags', [])
    ignore = {'gguf', 'transformers', 'text-generation', 'text-generation-inference', 
              'license:other', 'license:apache-2.0', 'license:mit', 'region:us', 
              'safetensors', 'pytorch', 'llama', 'llama-2', 'llama-3', 'facebook', 'meta'}
    interesting = [t for t in tags if t not in ignore and not t.startswith(('dataset:', 'arxiv:', 'transformers:'))]
    if interesting:
        print(f"  {_DIM}Tags:{_RESET}    {', '.join(interesting[:8])}")
    
    print(f"{_BOLD}{_CYAN}{'═' * 110}{_RESET}")
    print()


def _show_and_download(repo_id):
    """List GGUF files in a repo and let the user pick one to download."""
    print(f"\n{_DIM}Fetching details for {repo_id}...{_RESET}")

    # Detect VRAM again to be sure (or pass it in? We'll re-detect for simplicity)
    _, vram_gb = _detect_gpu()

    gguf_files, meta = _fetch_model_details(repo_id)
    
    if not gguf_files and not meta:
         # Error already printed in fetch
         return

    _print_model_header(meta)

    print(f"{_BOLD}📦 GGUF files available:{_RESET}")

    if not gguf_files:
        print(f"{_YELLOW}  No .gguf files found in this repo.{_RESET}")
        return

    # Sort by size (ascending)
    gguf_files.sort(key=lambda x: x["size"])

    for i, f in enumerate(gguf_files, 1):
        size_gb = f["size"] / (1024**3)
        size_str = f"{size_gb:.1f} GB"
        
        # simple heuristic: model size + 0.5GB buffer must fit in VRAM
        # (for partial offload, it allows larger, but this is a "safe" indicator)
        if vram_gb > 0:
            if size_gb + 0.5 <= vram_gb:
                fit = f"{_GREEN}✅ fits{_RESET}"
            elif size_gb <= vram_gb:
                fit = f"{_YELLOW}⚠️ tight{_RESET}"
            else:
                fit = f"{_RED}❌ too big{_RESET}"
        else:
            fit = ""

        tag = f" {_GREEN}← recommended{_RESET}" if "Q4_K_M" in f["name"] else ""
        print(f"  {_BOLD}{i:>3}{_RESET}.  {f['name']:<40}  {size_str:>8}  {fit}{tag}")

    print()
    while True:
        try:
            raw = input(f"{_BOLD}Enter # to download, or 'b' to go back: {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw.lower() in ("b", "back", "q", "quit", ""):
            return
        try:
            choice = int(raw)
        except ValueError:
            continue
        if 1 <= choice <= len(gguf_files):
            # Pass capabilities to download for registry
            caps = _infer_capabilities(repo_id, meta.get("pipeline_tag"), meta.get("tags"))
            _download_file(repo_id, gguf_files[choice - 1]["name"], caps=caps)
            return


def _search_interactive(query):
    """Search HuggingFace for GGUF models and offer to download."""
    results = _fetch_top_models(author=None, limit=20)
    # Re-search with query
    api = _get_api()
    if not api:
        return

    print(f"\n{_CYAN}🔍 Searching for '{query}'...{_RESET}\n")
    try:
        models = list(api.list_models(
            search=f"{query} GGUF",
            sort="downloads",
            limit=20,
            cardData=True,
        ))
    except Exception as e:
        print(f"{_RED}[ERROR] Search failed: {e}{_RESET}")
        return

    gguf_models = [m for m in models if "gguf" in (m.id or "").lower()]
    if not gguf_models:
        print(f"{_YELLOW}  No GGUF models found for '{query}'.{_RESET}")
        print(f"  Try: deepseek, qwen, phi, llama, mistral, unsloth, gemma")
        return

    w = 90
    print(f"  {_BOLD}{_WHITE}{'#':>3}   {'Repository':<40}  {'Params':<6}  {'Capabilities':<14}  {'Downloads':>10}  {'Updated':<10}{_RESET}")
    print(f"  {'─' * (w - 4)}")
    for i, m in enumerate(gguf_models, 1):
        dl = f"{m.downloads:,}" if hasattr(m, "downloads") and m.downloads else "?"
        updated = str(getattr(m, 'lastModified', None) or getattr(m, 'last_modified', None) or getattr(m, 'created_at', ''))[:10] or "?"
        
        # Extract metadata
        mid = m.id
        params = _extract_params(mid)
        model_tags = getattr(m, 'tags', []) or []
        caps = _infer_capabilities(mid, getattr(m, 'pipeline_tag', None), model_tags)

        print(
            f"  {_BOLD}{i:>3}{_RESET}   "
            f"{_CYAN}{mid:<40}{_RESET}  "
            f"{params:<6}  "
            f"{caps:<14}  "
            f"{dl:>10}  {updated:<10}"
        )
    print()

    while True:
        try:
            raw = input(f"{_BOLD}Enter # to browse files, or 'b' to go back: {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw.lower() in ("b", "back", "q", "quit", ""):
            return
        try:
            choice = int(raw)
        except ValueError:
            continue
        if 1 <= choice <= len(gguf_models):
            _show_and_download(gguf_models[choice - 1].id)
            return


def _interactive_loop(all_models, gpu_vram=0):
    """Main interactive loop: pick a model, search, or quit."""
    total = len(all_models)
    if total == 0:
        print(f"{_YELLOW}  No models found. Try --search to look for specific models.{_RESET}")
        return

    while True:
        try:
            raw = input(
                f"{_BOLD}Enter # (1-{total}), 's' search, 'c' capabilities, 'b' benchmarks, or 'q' quit: {_RESET}"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if raw.lower() in ("q", "quit", "exit"):
            print("Bye!")
            return
        if raw.lower() in ("s", "search"):
            try:
                query = input(f"{_BOLD}Search terms: {_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if query:
                _search_interactive(query)
            continue
        if raw.lower() in ("b", "bench", "benchmarks"):
            _print_benchmark_comparison(gpu_vram)
            continue
        if raw.lower() in ("c", "caps", "capabilities"):
            catalog_models = _print_capabilities_catalog(gpu_vram)
            if catalog_models:
                # Let user pick from the catalog to browse/download
                while True:
                    try:
                        pick = input(
                            f"{_BOLD}Enter # to browse/download, or 'b' to go back: {_RESET}"
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if pick.lower() in ("b", "back", "q", ""):
                        break
                    try:
                        ci = int(pick)
                    except ValueError:
                        continue
                    if 1 <= ci <= len(catalog_models):
                        _show_and_download(catalog_models[ci - 1]["repo"])
                        break
            continue

        try:
            choice = int(raw)
        except ValueError:
            print(f"{_YELLOW}  Enter a number 1-{total}, 's', 'c', or 'q'.{_RESET}")
            continue
        if choice < 1 or choice > total:
            print(f"{_YELLOW}  Enter a number 1-{total}, 's', 'c', or 'q'.{_RESET}")
            continue

        repo_id = all_models[choice - 1]["repo_id"]
        _show_and_download(repo_id)
        print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    _enable_ansi_windows()

    parser = argparse.ArgumentParser(
        description="Browse and download GGUF models for local GPU inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python install_models.py                     # top 10 Unsloth + 10 community
              python install_models.py --top 20            # top 20 per section
              python install_models.py --search "deepseek" # free-form search
              python install_models.py --gpu 5090          # show GPU info
              python install_models.py --list              # print table and exit

            Interactive commands:
              #  - browse GGUF files and download
              s  - search for specific models
              c  - show capabilities catalog with GPU fit
              q  - quit
        """),
    )
    parser.add_argument(
        "--gpu", default=None,
        help="Override GPU type (e.g. 3090, 4090, 5090) or raw VRAM in GB",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of models to show per section (default: 10)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the table and exit (no interactive download)",
    )
    parser.add_argument(
        "--search", default=None, metavar="QUERY",
        help="Search HuggingFace for GGUF models (e.g. --search 'llama 3.2')",
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

    # Search mode
    if args.search:
        _search_interactive(args.search)
        return

    # Default: fetch live top models
    print(f"\n{_DIM}  Fetching top {args.top} GGUF models from HuggingFace...{_RESET}")
    unsloth = _fetch_top_models(author="unsloth", limit=args.top)
    community = _fetch_top_models(author=None, limit=args.top)

    # De-duplicate: remove Unsloth repos from community
    unsloth_ids = {m["repo_id"] for m in unsloth}
    community = [m for m in community if m["repo_id"] not in unsloth_ids][:args.top]

    _print_table(unsloth, community, gpu_name, gpu_vram)

    if args.list:
        return

    all_models = unsloth + community
    _interactive_loop(all_models, gpu_vram=gpu_vram)


if __name__ == "__main__":
    main()
