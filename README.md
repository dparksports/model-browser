# Meeting Detector

Standalone CLI tool that scans transcript files and uses an LLM to classify each
as a **real meeting/conversation** or **hallucinated** output from a speech-recognition
model (e.g. Whisper).

Extracted from [TurboScribe](https://github.com/dparksports/turboscribe).

## Setup

### Prerequisites
- Python 3.10+
- NVIDIA CUDA Toolkit 12.8 (for RTX 5090 GPU acceleration)

### Install
```powershell
.\install_libraries.ps1
```

This creates a `meeting` virtual environment and installs:
| Package | Purpose |
|---|---|
| `llama-cpp-python` (CUDA 12.8) | Local GGUF model inference on GPU |
| `huggingface-hub` | Auto-download GGUF models |
| `openai` | Gemini & OpenAI cloud APIs |
| `anthropic` | Claude cloud API |

---

## Quick Start

```powershell
# 1. Activate the environment
.\meeting\Scripts\activate

# 2. Browse & download a model (live from HuggingFace)
python model_browser.py

# 3. Run detection with interactive model picker
python detect_meetings.py --model pick
```

On subsequent runs, your model selection is remembered automatically.

---

## Model Browser

Fetches the most popular GGUF models **live from HuggingFace** — always up-to-date, no hardcoded lists.

| Command | What it does |
|---|---|
| `python model_browser.py` | Top 10 ⚡Unsloth + top 10 community models |
| `python model_browser.py --top 20` | Show top 20 per section |
| `python model_browser.py --search "deepseek"` | Free-form search for any model |
| `python model_browser.py --gpu 5090` | Show GPU info alongside results |
| `python model_browser.py --list` | Print table and exit (no download) |

Select a model to see its available GGUF quantisations (Q4_K_M recommended) and download.

---

## Detecting Meetings

### Default Transcript Directory
```
%APPDATA%\LongAudioApp\Transcripts\
```
Override with `--dir "C:\MyFolder"` when needed.

### Local LLM (GPU-accelerated, no API key)
```powershell
python detect_meetings.py --model pick       # interactive picker
python detect_meetings.py --model phi-3-mini  # specific model
python detect_meetings.py                     # uses last picked model
```

### Cloud Providers
```powershell
python detect_meetings.py --provider gemini --api-key YOUR_KEY
python detect_meetings.py --provider openai --api-key YOUR_KEY
python detect_meetings.py --provider claude --api-key YOUR_KEY
```

### Resume a Scan
```powershell
python detect_meetings.py --skip-checked
```

---

## How It Works

1. **Discovery** — recursively finds `*_transcript*.txt` files in `--dir`
2. **Heuristic pre-filter** — if >85% of lines are identical → hallucinated
3. **LLM classification** — sends transcript to LLM → `{has_meeting, confidence, reason}`
4. **Report** — saves `detection_report.json` with all results

```json
[
  {
    "file": "C:\\Audio\\call_transcript_large-v3.txt",
    "has_meeting": true,
    "confidence": 92,
    "reason": "Contains varied conversation with multiple speakers"
  }
]
```

## License

Apache-2.0 (same as TurboScribe)
