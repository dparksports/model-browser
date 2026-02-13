#!/usr/bin/env python3
"""
Meeting Detector -- Standalone CLI extracted from TurboScribe
https://github.com/dparksports/turboscribe

Scans transcript files and uses an LLM (local GGUF or cloud API) to determine
whether each transcript contains a real meeting/conversation or is hallucinated
output from a speech-recognition model.

Usage::

    python detect_meetings.py --dir "C:\\Transcripts" --provider local --model phi-3-mini
    python detect_meetings.py --dir "C:\\Transcripts" --provider gemini --api-key KEY
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# WINDOWS CUDA PATH FIX
# ---------------------------------------------------------------------------
# If the user just installed CUDA, the parent shell won't have the new PATH yet.
# We manually add the default CUDA 12.6 bin directory to environment and DLL search path.
try:
    _cuda_dirs = []
    for _ver in ["v13.1", "v12.6"]:
        _base = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{_ver}"
        # CUDA 13.x puts DLLs in bin\x64\; older versions use bin\
        for _sub in [os.path.join(_base, "bin", "x64"), os.path.join(_base, "bin")]:
            if os.path.exists(_sub):
                _cuda_dirs.append(_sub)

    for _cuda_bin in _cuda_dirs:
        # 1. Update PATH for subprocesses / legacy load
        if _cuda_bin.lower() not in os.environ["PATH"].lower():
            os.environ["PATH"] += ";" + _cuda_bin

        # 2. Update DLL search path (Python 3.8+)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_cuda_bin)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Default transcript directory
# ---------------------------------------------------------------------------

_DEFAULT_TRANSCRIPT_DIR = os.path.join(
    os.path.expanduser("~"), "AppData", "Roaming", "LongAudioApp", "Transcripts"
)

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_model")
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "installed_models.json")


def _load_registry_models():
    """Load installed models from JSON registry and return dict suitable for _gguf_models."""
    models = {}
    if not os.path.exists(_REGISTRY_FILE):
        return models
    try:
        with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                alias = v.get("alias") or k.split("/")[-1].replace(".gguf", "")
                repo = v.get("repo_id")
                filename = v.get("filename")
                if alias and repo and filename:
                    models[alias] = (repo, filename)
    except Exception:
        pass
    return models


def _load_last_model():
    """Load the last-used model alias from .last_model, or None."""
    try:
        if os.path.isfile(_CONFIG_FILE):
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            alias = data.get("model")
            # Check if valid (in hardcoded list or registry)
            if alias and (alias in _gguf_models):
                return alias
    except Exception:
        pass
    return None


def _save_last_model(alias):
    """Persist the chosen model alias to .last_model."""
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"model": alias}, f)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# GGUF Model Registry (HuggingFace repos)
# ---------------------------------------------------------------------------

_gguf_models = {
    "llama-3.1-8b": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "mistral-7b": (
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    ),
    "phi-3-mini": (
        "bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
    ),
    "qwen2-7b": (
        "Qwen/Qwen2-7B-Instruct-GGUF",
        "qwen2-7b-instruct-q4_k_m.gguf",
    ),
    "gemma-2-2b": (
        "bartowski/gemma-2-2b-it-GGUF",
        "gemma-2-2b-it-Q4_K_M.gguf",
    ),
    # Extended catalog (matches install_models.py)
    "qwen2.5-0.5b": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ),
    "qwen2.5-1.5b": (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    "qwen2.5-7b": (
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-7b-instruct-q4_k_m.gguf",
    ),
    "gemma-2-9b": (
        "bartowski/gemma-2-9b-it-GGUF",
        "gemma-2-9b-it-Q4_K_M.gguf",
    ),
    "deepseek-r1-8b": (
        "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    ),
    "qwen2.5-14b": (
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "qwen2.5-14b-instruct-q4_k_m.gguf",
    ),
    "phi-4-14b": (
        "bartowski/phi-4-GGUF",
        "phi-4-Q4_K_M.gguf",
    ),
    "gemma-2-27b": (
        "bartowski/gemma-2-27b-it-GGUF",
        "gemma-2-27b-it-Q4_K_M.gguf",
    ),
    "qwen2.5-32b": (
        "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "qwen2.5-32b-instruct-q4_k_m.gguf",
    ),
    "deepseek-r1-32b": (
        "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    ),
    "mixtral-8x7b": (
        "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    "llama-3.1-70b": (
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    ),
    "deepseek-r1-70b": (
        "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
    ),
    # Unsloth Dynamic 2.0 quantisations
    "unsloth-qwen2.5-1.5b": (
        "unsloth/Qwen2.5-1.5B-Instruct-GGUF",
        "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    ),
    "unsloth-phi-3-mini": (
        "unsloth/Phi-3-mini-4k-instruct-GGUF",
        "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    ),
    "unsloth-llama-3.1-8b": (
        "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    "unsloth-qwen2.5-7b": (
        "unsloth/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    "unsloth-gemma-2-9b": (
        "unsloth/gemma-2-9b-it-GGUF",
        "gemma-2-9b-it-Q4_K_M.gguf",
    ),
    "unsloth-deepseek-r1-8b": (
        "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    ),
    "unsloth-phi-4-14b": (
        "unsloth/phi-4-GGUF",
        "phi-4-Q4_K_M.gguf",
    ),
    "unsloth-qwen2.5-14b": (
        "unsloth/Qwen2.5-14B-Instruct-GGUF",
        "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    ),
    "unsloth-qwen2.5-32b": (
        "unsloth/Qwen2.5-32B-Instruct-GGUF",
        "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    ),
    "unsloth-deepseek-r1-70b": (
        "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
    ),
}

# Merge installed models into the hardcoded list
_installed = _load_registry_models()
if _installed:
    print(f"[INIT] Loaded {len(_installed)} models from installed_models.json")
    _gguf_models.update(_installed)


# ---------------------------------------------------------------------------
# Interactive model picker — shows only already-downloaded models
# ---------------------------------------------------------------------------

def _find_downloaded_models():
    """Scan the model registry and return a list of (alias, repo, file, path)
    for every model that is already present in the HuggingFace cache."""
    downloaded = []
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return downloaded

    for alias, (repo, filename) in _gguf_models.items():
        try:
            path = try_to_load_from_cache(repo_id=repo, filename=filename)
            if path and isinstance(path, str) and os.path.isfile(path):
                downloaded.append((alias, repo, filename, path))
        except Exception:
            pass
    return downloaded


def _pick_model_interactive():
    """Display downloaded models and let the user pick one.

    Returns the chosen model alias, or None if cancelled.
    """
    downloaded = _find_downloaded_models()
    if not downloaded:
        print("\n[WARNING] No models are downloaded yet.")
        print("  Run install_models.py to download a model first:")
        print("    python install_models.py")
        return None

    print("\n" + "=" * 60)
    print("  Downloaded Models — pick one to use")
    print("=" * 60)
    for i, (alias, repo, filename, path) in enumerate(downloaded, 1):
        tag = " ⚡" if alias.startswith("unsloth-") else ""
        print(f"  {i:>3}.  {alias}{tag}")
        print(f"        {filename}")
    print("=" * 60)

    while True:
        try:
            raw = input(f"\nSelect model (1-{len(downloaded)}), or 'q' to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        try:
            choice = int(raw)
        except ValueError:
            print(f"  Please enter a number between 1 and {len(downloaded)}.")
            continue
        if 1 <= choice <= len(downloaded):
            alias = downloaded[choice - 1][0]
            print(f"  → Selected: {alias}")
            return alias
        print(f"  Please enter a number between 1 and {len(downloaded)}.")

# ---------------------------------------------------------------------------
# Local LLM loader (llama-cpp-python) with GPU → CPU fallback
# ---------------------------------------------------------------------------

_cached_llm = None
_cached_llm_name = None


def _load_llm(model_name=None):
    """Load (or return cached) a GGUF model for local inference."""
    global _cached_llm, _cached_llm_name

    if model_name == "pick":
        model_name = _pick_model_interactive()
        if not model_name:
            return None

    if not model_name or model_name not in _gguf_models:
        # Try loading last-used model before falling back to phi-3-mini
        saved = _load_last_model()
        if saved:
            print(f"[LOAD] Using saved model: {saved}")
            model_name = saved
        else:
            print(f"[WARNING] Model '{model_name}' not found. Defaulting to phi-3-mini.")
            model_name = "phi-3-mini"

    # Remember this choice for next time
    _save_last_model(model_name)

    # Return cached if same model
    if _cached_llm is not None and _cached_llm_name == model_name:
        return _cached_llm

    try:
        from llama_cpp import Llama
    except ImportError:
        print("[ERROR] llama-cpp-python not installed. Run install_libraries.ps1 first.")
        return None

    repo_id, filename = _gguf_models[model_name]

    try:
        from huggingface_hub import hf_hub_download

        print(f"[LOAD] Downloading/loading {model_name} ({filename})...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except ImportError:
        print("[ERROR] huggingface-hub not installed")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        return None

    print("[LOAD] Loading LLM into memory (GPU)...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Offload all layers to GPU
            verbose=False,
        )
        print("[LOAD] LLM loaded on GPU successfully")
        _cached_llm = llm
        _cached_llm_name = model_name
        return llm
    except Exception as e:
        print(f"[WARNING] GPU loading failed ({e}), falling back to CPU...")
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=0,  # CPU only
                verbose=False,
            )
            print("[LOAD] LLM loaded on CPU (GPU unavailable)")
            _cached_llm = llm
            _cached_llm_name = model_name
            return llm
        except Exception as e2:
            print(f"[ERROR] Failed to load LLM on both GPU and CPU: {e2}")
            return None


# ---------------------------------------------------------------------------
# LLM provider back-ends
# ---------------------------------------------------------------------------


def _analyze_local(prompt, model_name=None, llm_instance=None):
    """Run inference with a local GGUF model via llama-cpp-python."""
    llm = llm_instance or _load_llm(model_name)
    if llm is None:
        return None

    print("[ANALYZE] Running local inference...")
    try:
        output = llm(
            prompt,
            max_tokens=2048,
            temperature=0.3,
            stop=["\n\n\n"],
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[ERROR] Local inference failed: {e}")
        return None


def _analyze_gemini(prompt, api_key, model="gemini-2.0-flash"):
    """Run inference via Google Gemini (OpenAI-compatible endpoint)."""
    if not api_key:
        print("[ERROR] Gemini API key required")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except ImportError:
        print("[ERROR] openai package not installed")
        return None
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        return None


def _analyze_openai(prompt, api_key, model="gpt-4o"):
    """Run inference via OpenAI API."""
    if not api_key:
        print("[ERROR] OpenAI API key required")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except ImportError:
        print("[ERROR] openai package not installed")
        return None
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return None


def _analyze_claude(prompt, api_key, model="claude-sonnet-4-20250514"):
    """Run inference via Anthropic Claude API."""
    if not api_key:
        print("[ERROR] Claude API key required")
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError:
        print("[ERROR] anthropic package not installed")
        return None
    except Exception as e:
        print(f"[ERROR] Claude API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Core: Detect Meetings
# ---------------------------------------------------------------------------

MEETING_DETECTION_PROMPT = """\
Analyze this transcript and determine if it contains a real conversation or \
meeting, or if it is hallucinated/repetitive nonsense from a speech recognition model.

Signs of HALLUCINATION: identical repeated phrases, single-word loops \
(e.g. "I" repeated many times), no conversational flow, no topic progression, \
very short repeated segments.
Signs of REAL MEETING: varied sentences, questions and answers, topic changes, \
multiple speakers, natural conversation flow, specific details like names/places/plans.

Respond with ONLY this JSON (no other text):
{{
    "has_meeting": true, 
    "confidence": 85, 
    "reason": "one sentence explanation"
}}

The confidence field is an integer from 0 to 100 where 100 means absolute certainty.

Transcript:
{transcript}

JSON:"""


def detect_meetings(
    directory,
    provider="local",
    model_name=None,
    api_key=None,
    cloud_model=None,
    transcript_dir=None,
    skip_checked=False,
    output_dir=".",
):
    """
    Scan all *_transcript*.txt files and classify each as real meeting vs.
    hallucinated via an LLM.  Saves detection_report.json for resumable scans.
    """
    # -- Collect transcript files ------------------------------------------------
    transcript_files = []
    search_dirs = []
    if directory and os.path.isdir(directory):
        search_dirs.append(directory)
    if transcript_dir and os.path.isdir(transcript_dir):
        search_dirs.append(transcript_dir)

    seen = set()
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if "_transcript" in f and f.endswith(".txt"):
                    fpath = os.path.abspath(os.path.join(root, f))
                    if fpath not in seen:
                        seen.add(fpath)
                        transcript_files.append(fpath)

    total = len(transcript_files)
    print(f"[DETECT] Found {total} transcript files to analyze")
    if total == 0:
        print(json.dumps({"status": "complete", "action": "detect_meetings", "results": []}))
        return

    results = []
    meetings_found = 0

    # -- Skip previously-checked files if requested ------------------------------
    previously_checked = set()
    if skip_checked:
        report_path = os.path.join(output_dir, "detection_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as rf:
                    prev_results = json.load(rf)
                    for r in prev_results:
                        previously_checked.add(os.path.abspath(r.get("file", "")))
                    results.extend(prev_results)
                    meetings_found += sum(1 for r in prev_results if r.get("has_meeting"))
                    print(f"[DETECT] Loaded {len(prev_results)} previously checked files, skipping them")
            except Exception as ex:
                print(f"[WARNING] Could not load previous report: {ex}")

        transcript_files = [f for f in transcript_files if os.path.abspath(f) not in previously_checked]
        total = len(transcript_files)
        print(f"[DETECT] {total} remaining files to analyze (after skipping)")
        if total == 0:
            print("[DETECT] All files already checked!")
            print(json.dumps({"status": "complete", "action": "detect_meetings", "results": results}))
            # Still output summary if meaningful results exist
            _write_summary(output_dir, results)
            _write_csv(output_dir, results)
            return

    # -- Pre-load local LLM once for the entire batch ----------------------------
    llm_instance = None
    if provider == "local":
        llm_instance = _load_llm(model_name)
        if llm_instance is None:
            print("[ERROR] Failed to load local LLM model. Aborting.")
            return

    # -- Iterate over transcripts ------------------------------------------------
    for i, fpath in enumerate(transcript_files, 1):
        fname = os.path.basename(fpath)
        print(f"\n[{i}/{total}] Analyzing: {fname}")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            # Strip transcript headers
            lines = [
                l
                for l in content.split("\n")
                if l.strip() and not l.startswith("---") and not l.startswith("Source:")
            ]
            transcript_text = "\n".join(lines)

            if not transcript_text.strip():
                print("  [SKIP] Empty transcript")
                results.append(
                    {"file": fpath, "has_meeting": False, "confidence": 100, "reason": "Empty transcript"}
                )
                continue

            # --- Quick heuristic pre-filter ---
            line_list = [l.strip() for l in lines if l.strip()]
            if len(line_list) > 5:
                texts = []
                for l in line_list:
                    bracket_end = l.rfind("]")
                    if bracket_end >= 0:
                        texts.append(l[bracket_end + 1 :].strip())
                    else:
                        texts.append(l)
                unique_ratio = len(set(texts)) / len(texts) if texts else 0
                if unique_ratio < 0.15:
                    print(f"  [NO_MEETING] Repetition ratio {unique_ratio:.2f} — clearly hallucinated")
                    results.append(
                        {
                            "file": fpath,
                            "has_meeting": False,
                            "confidence": 99,
                            "reason": f"Extreme repetition (unique ratio: {unique_ratio:.2f})",
                        }
                    )
                    continue

            # Truncate for LLM context window
            if len(transcript_text) > 4000:
                transcript_text = transcript_text[:4000] + "\n... (truncated)"

            prompt = MEETING_DETECTION_PROMPT.format(transcript=transcript_text)

            # --- Call LLM -------------------------------------------------------
            if provider == "local":
                result_text = _analyze_local(prompt, model_name, llm_instance=llm_instance)
            elif provider == "gemini":
                result_text = _analyze_gemini(prompt, api_key, cloud_model or "gemini-2.0-flash")
            elif provider == "openai":
                result_text = _analyze_openai(prompt, api_key, cloud_model or "gpt-4o")
            elif provider == "claude":
                result_text = _analyze_claude(prompt, api_key, cloud_model or "claude-sonnet-4-20250514")
            else:
                print(f"  [ERROR] Unknown provider: {provider}")
                continue

            if not result_text:
                print("  [ERROR] LLM returned no result")
                results.append(
                    {"file": fpath, "has_meeting": False, "confidence": 0, "reason": "LLM returned no result"}
                )
                continue

            # --- Parse JSON from LLM response -----------------------------------
            try:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    parsed = json.loads(result_text[json_start:json_end])
                else:
                    parsed = json.loads(result_text)

                has_meeting = parsed.get("has_meeting", False)
                confidence = int(parsed.get("confidence", 50))
                # Normalise 0.0-1.0 → 0-100
                if isinstance(parsed.get("confidence"), float) and parsed["confidence"] <= 1.0:
                    confidence = int(parsed["confidence"] * 100)
                reason = parsed.get("reason", "")
            except json.JSONDecodeError:
                has_meeting = "true" in result_text.lower() and "has_meeting" in result_text.lower()
                confidence = 50
                reason = f"Could not parse JSON: {result_text[:100]}"

            tag = "MEETING_DETECTED" if has_meeting else "NO_MEETING"
            print(f"  [{tag}] confidence={confidence} — {reason}")

            if has_meeting:
                meetings_found += 1

            results.append(
                {"file": fpath, "has_meeting": has_meeting, "confidence": confidence, "reason": reason}
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"file": fpath, "has_meeting": False, "confidence": 0, "reason": str(e)})

    # -- Summary Output ----------------------------------------------------------
    _write_summary(output_dir, results)
    _write_csv(output_dir, results)

    # -- Save report -------------------------------------------------------------
    report_path = os.path.join(output_dir, "detection_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(results, rf, indent=2)
        print(f"\n[DETECT] Detailed JSON report saved to {report_path}")
    except Exception as ex:
        print(f"[WARNING] Could not save report: {ex}")


def _write_csv(directory, results):
    """Write detection results to a CSV file (Only Confirmed Meetings)."""
    import csv
    csv_path = os.path.join(directory, "found_meetings.csv")

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["File", "Has Meeting", "Confidence", "Reason"])

            # Rows - FILTER: Only write if has_meeting is True
            count = 0
            for r in results:
                if r.get("has_meeting"):
                    writer.writerow([
                        os.path.basename(r["file"]),
                        "Yes",
                        f"{r['confidence']}%",
                        r["reason"]
                    ])
                    count += 1

        print(f"[DETECT] CSV export saved to {csv_path} ({count} meetings)")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV file: {e}")


def _write_summary(directory, results):
    """Write a human-readable summary of the detection results."""
    summary_path = os.path.join(directory, "meetings_summary.txt")
    meetings_found = sum(1 for r in results if r.get("has_meeting"))

    print(f"\n{'=' * 60}")
    print("MEETING DETECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total transcripts: {len(results)}")
    print(f"Meetings found:    {meetings_found}")
    print(f"Hallucinated:      {len(results) - meetings_found}")

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("MEETING DETECTION SUMMARY\n")
            f.write("========================\n\n")
            
            if meetings_found > 0:
                f.write("✅ CONFIRMED MEETINGS:\n")
                print("\n--- Files with Real Meetings ---")
                for r in results:
                    if r.get("has_meeting"):
                        line = f"  [{r['confidence']}%] {os.path.basename(r['file'])} — {r['reason']}"
                        print(line)
                        f.write(line + "\n")
            else:
                f.write("No meetings detected.\n")

            f.write("\n\n❌ HALLUCINATIONS / NO MEETINGS:\n")
            for r in results:
                if not r.get("has_meeting"):
                   f.write(f"  [{r['confidence']}%] {os.path.basename(r['file'])} — {r['reason']}\n")

        print(f"\n[DETECT] Summary file created: {summary_path}")
        print(f"         (Open this file to see which transcripts have meetings)")
    except Exception as e:
        print(f"[ERROR] Failed to write summary file: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect real meetings vs. hallucinated transcripts using LLM analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Use default transcript directory (~\\AppData\\Roaming\\LongAudioApp\\Transcripts\\)
  python detect_meetings.py --provider local --model pick

  # Specify a custom directory
  python detect_meetings.py --dir "C:\\Transcripts" --provider local --model phi-3-mini

  # Google Gemini cloud API
  python detect_meetings.py --dir "C:\\Transcripts" --provider gemini --api-key YOUR_KEY

  # Resume a previous scan (skip already-checked files)
  python detect_meetings.py --provider local --skip-checked

  # Install more models:
  python install_models.py
"""
    )
    parser.add_argument(
        "--dir",
        default=_DEFAULT_TRANSCRIPT_DIR,
        help=f"Directory containing transcript files (default: {_DEFAULT_TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "gemini", "openai", "claude"],
        default="local",
        help="LLM provider (default: local)",
    )
    parser.add_argument(
        "--model",
        default="phi-3-mini",
        help="Local GGUF model preset (default: phi-3-mini). "
        "Use 'pick' to interactively choose from downloaded models.",
    )
    parser.add_argument("--api-key", help="API key for cloud provider (gemini/openai/claude)")
    parser.add_argument("--cloud-model", help="Override cloud model name (e.g. gpt-4o, gemini-2.0-flash)")
    parser.add_argument("--transcript-dir", help="Additional directory to search for transcript files")
    parser.add_argument(
        "--skip-checked",
        action="store_true",
        help="Skip files already analysed in a previous detection_report.json",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Directory to save results (default: current directory)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if args.output and not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
        except Exception as e:
            print(f"[ERROR] Could not create output directory: {e}")
            return

    detect_meetings(
        directory=args.dir,
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        cloud_model=args.cloud_model,
        transcript_dir=args.transcript_dir,
        skip_checked=args.skip_checked,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
