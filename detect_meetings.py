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
}

# ---------------------------------------------------------------------------
# Local LLM loader (llama-cpp-python) with GPU → CPU fallback
# ---------------------------------------------------------------------------

_cached_llm = None
_cached_llm_name = None


def _load_llm(model_name=None):
    """Load (or return cached) a GGUF model for local inference."""
    global _cached_llm, _cached_llm_name

    if not model_name or model_name not in _gguf_models:
        model_name = "phi-3-mini"

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
{{"has_meeting": true, "confidence": 85, "reason": "one sentence explanation"}}

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
        report_path = os.path.join(directory, "detection_report.json")
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

    # -- Summary -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MEETING DETECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total transcripts: {len(results)}")
    print(f"Meetings found:    {meetings_found}")
    print(f"Hallucinated:      {len(results) - meetings_found}")

    if meetings_found > 0:
        print("\n--- Files with Real Meetings ---")
        for r in results:
            if r["has_meeting"]:
                print(f"  ✅ {os.path.basename(r['file'])} ({r['confidence']}%) — {r['reason']}")

    # -- Save report -------------------------------------------------------------
    report_path = os.path.join(directory, "detection_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(results, rf, indent=2)
        print(f"\n[DETECT] Report saved to {report_path}")
    except Exception as ex:
        print(f"[WARNING] Could not save report: {ex}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect real meetings vs. hallucinated transcripts using LLM analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Local GGUF model (auto-downloads from HuggingFace)
  python detect_meetings.py --dir "C:\\Transcripts" --provider local --model phi-3-mini

  # Google Gemini cloud API
  python detect_meetings.py --dir "C:\\Transcripts" --provider gemini --api-key YOUR_KEY

  # Resume a previous scan (skip already-checked files)
  python detect_meetings.py --dir "C:\\Transcripts" --provider local --skip-checked

Available local models: """ + ", ".join(sorted(_gguf_models.keys())),
    )
    parser.add_argument("--dir", required=True, help="Directory containing transcript files")
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
        f"Choices: {', '.join(sorted(_gguf_models.keys()))}",
    )
    parser.add_argument("--api-key", help="API key for cloud provider (gemini/openai/claude)")
    parser.add_argument("--cloud-model", help="Override cloud model name (e.g. gpt-4o, gemini-2.0-flash)")
    parser.add_argument("--transcript-dir", help="Additional directory to search for transcript files")
    parser.add_argument(
        "--skip-checked",
        action="store_true",
        help="Skip files already analysed in a previous detection_report.json",
    )

    args = parser.parse_args()

    detect_meetings(
        directory=args.dir,
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        cloud_model=args.cloud_model,
        transcript_dir=args.transcript_dir,
        skip_checked=args.skip_checked,
    )


if __name__ == "__main__":
    main()
