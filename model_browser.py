#!/usr/bin/env python3
"""
Model Browser & Downloader
──────────────────────────
Dynamic CLI that fetches the most popular GGUF models from HuggingFace,
shows Unsloth optimised models first, and lets you download directly.

No hardcoded model lists — always shows the latest releases.

Usage::

    python model_browser.py                     # top 10 Unsloth + 10 community
    python model_browser.py --search "deepseek" # free-form search
    python model_browser.py --top 20            # show top 20 per section
    python model_browser.py --gpu 5090          # detect GPU for context
"""

import argparse
import os
import subprocess
import sys
import textwrap

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
        
        results.append({
            "repo_id": mid, 
            "downloads": dl, 
            "updated": updated,
            "params": params,
            "type": task
        })
        if len(results) >= limit:
            break
    return results


def _list_gguf_files(repo_id):
    """Return list of dicts: {name, size, fit} for .gguf files in a repo."""
    api = _get_api()
    if not api:
        return []
    try:
        # Use model_info to get file sizes (siblings)
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        files = []
        for s in info.siblings:
            if s.rfilename.endswith(".gguf"):
                files.append({
                    "name": s.rfilename,
                    "size": s.size or 0
                })
        return files
    except Exception as e:
        print(f"{_RED}  [ERROR] Could not list files: {e}{_RESET}")
        return []


def _download_file(repo_id, filename):
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
        return path
    except Exception as e:
        print(f"{_RED}[ERROR] Download failed: {e}{_RESET}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Table renderer
# ──────────────────────────────────────────────────────────────────────────────

def _print_table(unsloth_models, community_models, gpu_name, gpu_vram):
    """Render compact tables for live-fetched model repos."""
    w = 90

    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'🤖  Top GGUF Models (live from HuggingFace)':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")

    if gpu_name:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_name}{_RESET} ({gpu_vram} GB)")
    elif gpu_vram > 0:
        print(f"  {_BOLD}GPU:{_RESET} {_GREEN}{gpu_vram} GB VRAM{_RESET}")
    else:
        print(f"  {_YELLOW}⚠  No GPU detected.{_RESET}")
    print()

    def _section(title, models, offset=0):
        if not models:
            return
        print(f"{_BOLD}{_MAGENTA}  ── {title} {'─' * max(1, w - len(title) - 7)}{_RESET}")
        print(f"  {_BOLD}{_WHITE}{'#':>3}   {'Repository':<42}  {'Params':<6}  {'Type':<8}  {'Downloads':>10}  {'Updated':<10}{_RESET}")
        print(f"  {'─' * (w - 4)}")
        for i, m in enumerate(models, offset + 1):
            dl = f"{m['downloads']:,}" if m['downloads'] else "?"
            print(
                f"  {_BOLD}{i:>3}{_RESET}   "
                f"{_CYAN}{m['repo_id']:<42}{_RESET}  "
                f"{m['params']:<6}  "
                f"{m['type']:<8}  "
                f"{dl:>10}  {m['updated']:<10}"
            )
        print()

    _section("⚡ UNSLOTH — top by downloads", unsloth_models, offset=0)
    _section("COMMUNITY — top by downloads", community_models, offset=len(unsloth_models))

    print(f"  {_DIM}Select a # to browse GGUF files and download, 's' to search, 'q' to quit.{_RESET}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Interactive selection
# ──────────────────────────────────────────────────────────────────────────────

def _format_size(size_bytes):
    """Format bytes to GB."""
    return f"{size_bytes / (1024**3):.1f} GB"

def _show_and_download(repo_id):
    """List GGUF files in a repo and let the user pick one to download."""
    print(f"\n{_BOLD}📦 GGUF files in {repo_id}:{_RESET}")

    # Detect VRAM again to be sure (or pass it in? We'll re-detect for simplicity)
    _, vram_gb = _detect_gpu()

    gguf_files = _list_gguf_files(repo_id)
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
            _download_file(repo_id, gguf_files[choice - 1]["name"])
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
    print(f"  {_BOLD}{_WHITE}{'#':>3}   {'Repository':<42}  {'Params':<6}  {'Type':<8}  {'Downloads':>10}  {'Updated':<10}{_RESET}")
    print(f"  {'─' * (w - 4)}")
    for i, m in enumerate(gguf_models, 1):
        dl = f"{m.downloads:,}" if hasattr(m, "downloads") and m.downloads else "?"
        updated = str(getattr(m, 'lastModified', None) or getattr(m, 'last_modified', None) or getattr(m, 'created_at', ''))[:10] or "?"
        
        # Extract metadata
        mid = m.id
        params = _extract_params(mid)
        task = getattr(m, 'pipeline_tag', '?') or '?'
        if task == "text-generation": task = "text-gen"
        elif task == "image-text-to-text": task = "vision"

        print(
            f"  {_BOLD}{i:>3}{_RESET}   "
            f"{_CYAN}{mid:<42}{_RESET}  "
            f"{params:<6}  "
            f"{task:<8}  "
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


def _interactive_loop(all_models):
    """Main interactive loop: pick a model, search, or quit."""
    total = len(all_models)
    if total == 0:
        print(f"{_YELLOW}  No models found. Try --search to look for specific models.{_RESET}")
        return

    while True:
        try:
            raw = input(
                f"{_BOLD}Enter # (1-{total}), 's' to search, or 'q' to quit: {_RESET}"
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

        try:
            choice = int(raw)
        except ValueError:
            print(f"{_YELLOW}  Enter a number 1-{total}, 's', or 'q'.{_RESET}")
            continue
        if choice < 1 or choice > total:
            print(f"{_YELLOW}  Enter a number 1-{total}, 's', or 'q'.{_RESET}")
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
              python model_browser.py                     # top 10 Unsloth + 10 community
              python model_browser.py --top 20            # top 20 per section
              python model_browser.py --search "deepseek" # free-form search
              python model_browser.py --gpu 5090          # show GPU info
              python model_browser.py --list              # print table and exit
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
    _interactive_loop(all_models)


if __name__ == "__main__":
    main()
