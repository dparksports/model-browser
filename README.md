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

## Usage

Activate the environment first:
```powershell
.\meeting\Scripts\activate
```

### Local LLM (GPU-accelerated, no API key needed)
```powershell
python detect_meetings.py --dir "C:\Transcripts" --provider local --model phi-3-mini
```

Available local models: `gemma-2-2b`, `llama-3.1-8b`, `mistral-7b`, `phi-3-mini` (default), `qwen2-7b`

Models are auto-downloaded from HuggingFace on first use.

### Cloud Providers
```powershell
# Google Gemini
python detect_meetings.py --dir "C:\Transcripts" --provider gemini --api-key YOUR_KEY

# OpenAI
python detect_meetings.py --dir "C:\Transcripts" --provider openai --api-key YOUR_KEY

# Anthropic Claude
python detect_meetings.py --dir "C:\Transcripts" --provider claude --api-key YOUR_KEY
```

### Resume a Scan
```powershell
python detect_meetings.py --dir "C:\Transcripts" --provider local --skip-checked
```

This skips files already present in the `detection_report.json` from a prior run.

## How It Works

1. **Discovery** — recursively finds all `*_transcript*.txt` files in `--dir`  
2. **Heuristic pre-filter** — if >85% of lines are identical, it's classified as
   hallucinated without calling the LLM  
3. **LLM classification** — sends the transcript to the chosen LLM with a prompt
   asking it to return `{has_meeting, confidence, reason}`  
4. **Report** — saves `detection_report.json` in `--dir` with all results

## Output

Results are printed to the console and saved as `detection_report.json`:

```json
[
  {
    "file": "C:\\Audio\\call_transcript_large-v3.txt",
    "has_meeting": true,
    "confidence": 92,
    "reason": "Contains varied conversation with multiple speakers discussing project plans"
  }
]
```

## License

Apache-2.0 (same as TurboScribe)
