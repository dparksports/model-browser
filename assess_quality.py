#!/usr/bin/env python3
"""
Transcript Quality Assessor -- Uses local LLM to grade transcript quality.

This script loads "Confirmed Meetings" (from detection_report.csv),
groups the transcripts by audio file, and then asks the local LLM to 
rate each transcript version on a scale of 0-10.
"""

import argparse
import json
import os
import sys

# Reuse logic from find_meetings for model loading
# We need to add the current directory to sys.path to import if needed, 
# but find_meetings is a script, not a module. 
# We will replicate the necessary parts or import if possible.
# For simplicity and robustness, I will inline the minimal LLM loading logic here
# to avoid dependency issues with the other script's main execution block.

# ---------------------------------------------------------------------------
# WINDOWS CUDA PATH FIX (Replicated from find_meetings.py)
# ---------------------------------------------------------------------------
try:
    _cuda_dirs = []
    for _ver in ["v13.1", "v12.6"]:
        _base = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{_ver}"
        for _sub in [os.path.join(_base, "bin", "x64"), os.path.join(_base, "bin")]:
            if os.path.exists(_sub):
                _cuda_dirs.append(_sub)
    for _cuda_bin in _cuda_dirs:
        if _cuda_bin.lower() not in os.environ["PATH"].lower():
            os.environ["PATH"] += ";" + _cuda_bin
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_cuda_bin)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------
_GGUF_MODELS = {
    "phi-3-mini": ("bartowski/Phi-3.1-mini-4k-instruct-GGUF", "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf"),
    "llama-3.1-8b": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
}

# Merge installed models from registry
_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "installed_models.json")
if os.path.exists(_REGISTRY_FILE):
    try:
        with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                alias = v.get("alias") or k.split("/")[-1].replace(".gguf", "")
                repo = v.get("repo_id")
                filename = v.get("filename")
                if alias and repo and filename:
                    _GGUF_MODELS[alias] = (repo, filename)
        print(f"[INIT] Loaded extra models from registry: {list(data.keys())}")
    except Exception as e:
        print(f"[WARN] Failed to load local registry: {e}")

DEFAULT_JUDGE_MODEL = "phi-3-mini"

QUALITY_PROMPT = """\
You are an expert transcript evaluator. Grade the following transcript snippet on a scale of 0 to 10.
The audio source is a noisy outdoor environment (wind, distance, etc).

Criteria for High Score (8-10):
- Coherent sentences and natural conversation flow.
- "Recovered" words: recognizable speech even if fragmented.
- Minimal hallucination (no repetitive loops like "I I I" or "You You").

Criteria for Low Score (0-3):
- Complete hallucination (repetitive phrases, single words repeated).
- Gibberish or random symbols.
- Empty or near-empty content.

Transcript:
{transcript}

Return ONLY the score as a single integer (0-10). Do not output any explanation or JSON.
Example:
8
"""

def _load_llm(model_name):
    """Load local GGUF model."""
    try:
        # Import inside function to allow CUDA path fixes to apply first
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("Please run install_libraries.ps1 to install required packages.")
        sys.exit(1)

    if model_name not in _GGUF_MODELS:
        model_name = DEFAULT_JUDGE_MODEL

    
    repo_id, filename = _GGUF_MODELS[model_name]
    print(f"[LOAD] Loading Judge Model: {model_name}...")
    
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1, # GPU
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

def _assess_transcript(llm, text):
    """Run the assessment prompt."""
    
    # Clean input: remove heavy headers that might confuse the model
    # Remove lines starting with "Source:", "---"
    # AND remove timestamps like [00:00] or [00:00.00 - 00:05.00]
    import re
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if line.startswith("Source:"): continue
        if line.startswith("---"): continue
        
        # Remove timestamps
        # pattern: [00:00.00 - 00:05.00] or [00:00] or [0.00 - 5.00]
        # Regex: \[ followed by digits/dots/spaces/hyphens followed by \]
        line = re.sub(r'\[[\d\.\s:-]+\]', '', line).strip()
        
        if line:
            clean_lines.append(line)
        
    clean_text = "\n".join(clean_lines)
    
    # Truncate slightly more to ensure we don't hit context limits with the prompt
    snippet = clean_text[:1200] 
    
    # Use Chat Completion (better for instruct models)
    messages = [
        {"role": "system", "content": "You are a transcript evaluator. Respond with ONLY an integer score 0-10."},
        {"role": "user", "content": QUALITY_PROMPT.format(transcript=snippet)}
    ]
    
    try:
        # 1. Try Chat Completion
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=10, 
            temperature=0.0,
            stop=["\n"] # Less aggressive than "." or space
        )
        response_text = output["choices"][0]["message"]["content"].strip()
        
        # 2. Retry with simple completion if empty (common with some GGUF chat templates)
        if not response_text:
             completion = llm(
                 f"Rate this transcript 0-10:\n{snippet}\nScore:",
                 max_tokens=5,
                 temperature=0.0
             )
             response_text = completion["choices"][0]["text"].strip()

        # Parse Integer
        import re
        match = re.search(r'\d+', response_text)
        if match:
             val = int(match.group(0))
             if 0 <= val <= 10:
                 return val, "LLM Score"
        
        return 0, f"Failed to parse score: '{response_text}'"
            
    except Exception as e:
        return 0, f"Error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Assess transcript quality using local LLM.")
    parser.add_argument("--report", default="detection_report.csv", help="Path to detection report")
    parser.add_argument("--limit", type=int, default=0, help="Number of audio segments to assess (0 = all, default: 0)")
    parser.add_argument("--model", help="Override judge model name")
    args = parser.parse_args()

    # 1. Load Detection Report (CSV)
    if not os.path.exists(args.report):
        print(f"Report not found: {args.report}")
        return

    import csv as _csv
    data = []
    with open(args.report, "r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            has_meeting = row.get("Has Meeting", "").strip().lower() == "yes"
            conf_str = row.get("Confidence", "0").replace("%", "").strip()
            try:
                confidence = int(conf_str)
            except ValueError:
                confidence = 0
            data.append({
                "file": row.get("Full Path", ""),
                "has_meeting": has_meeting,
                "confidence": confidence,
                "reason": row.get("Reason", ""),
            })

    # 2. Group by Audio ID
    from compare_models import parse_filename # Reuse this utility
    audio_groups = {}
    
    for entry in data:
        audio_id, model = parse_filename(entry["file"])
        if audio_id:
            if audio_id not in audio_groups:
                audio_groups[audio_id] = {}
            audio_groups[audio_id][model] = entry

    # 3. Filter for "Confirmed Meetings" (Optimistic Truth)
    # We only want to grade files where we think there IS a meeting.
    # Grading pure silence/noise files is less useful for "recovery" metrics.
    confirmed_ids = []
    for aid, models in audio_groups.items():
        # If ANY model found a meeting with >80 conf
        if any(m.get("has_meeting") and m.get("confidence",0) >= 80 for m in models.values()):
            confirmed_ids.append(aid)

    print(f"[INFO] Found {len(confirmed_ids)} confirmed meeting segments.")
    print(f"[INFO] Assessing a sample of {args.limit} segments...")

    # 4. Load Judge Model
    # 4. Load Judge Model
    # Check .last_model file created by find_meetings.py
    last_model_file = ".last_model"
    model_to_use = DEFAULT_JUDGE_MODEL
    
    if os.path.exists(last_model_file):
        try:
            with open(last_model_file, "r") as f:
                saved_model = f.read().strip()
                if saved_model in _GGUF_MODELS:
                    model_to_use = saved_model
                    print(f"[INFO] Using last selected model: {model_to_use}")
        except Exception:
            pass
            
    if args.model: # Allow override
        model_to_use = args.model

    judge_llm = _load_llm(model_to_use)

    # 5. Run Assessment
    import random
    # Select sample
    keys = sorted(confirmed_ids)
    limit = len(keys)
    
    if args.limit > 0 and args.limit < limit:
        limit = args.limit
        random.shuffle(keys)
        sample_ids = keys[:limit]
    else:
        sample_ids = keys
        
    print(f"[INFO] Assessing {limit} segments...")
    
    model_scores = {} # model -> [scores]

    print(f"\n{'='*60}")
    print(f"QUALITY ASSESSMENT LOG")
    print(f"{'='*60}")

    for i, aid in enumerate(sample_ids, 1):
        print(f"\n[{i}/{len(sample_ids)}] Audio ID: ...{aid[-15:]}")
        
        # Compare all available models for this audio
        models_data = audio_groups[aid]
        
        for model_name, entry in models_data.items():
            fpath = entry["file"]
            if not os.path.exists(fpath):
                continue
                
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Skip if file is trivially empty or just header
            if len(content) < 50:
                score, reason = 0, "Empty file"
            else:
                score, reason = _assess_transcript(judge_llm, content)
            
            print(f"  {model_name:<15} | Score: {score}/10 | {reason}")
            
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(score)

    # 6. Final Leaderboard
    print(f"\n{'='*60}")
    print(f"FINAL QUALITY LEADERBOARD")
    print(f"{'='*60}")
    print(f"{'Model':<15} | {'Avg Score':<10} | {'Samples':<8}")
    print("-" * 40)
    
    avgs = []
    for m, scores in model_scores.items():
        avg = sum(scores) / len(scores)
        avgs.append((m, avg, len(scores)))
    
    avgs.sort(key=lambda x: x[1], reverse=True)
    
    for m, avg, count in avgs:
        print(f"{m:<15} | {avg:<10.1f} | {count:<8}")

if __name__ == "__main__":
    main()
