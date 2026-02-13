# Evaluate Whisper Models

**Locally benchmarking Whisper models on challenging noisy outdoor meeting recordings.**

A powerful CLI tool to browse, download, and **evaluate the accuracy** of local Whisper models (GGUF format). It is specifically designed to stress-test models on difficult, real-world audio ("field recordings") to determine which versions can successfully detect speech amidst noise.

## Key Features

### 1. ⚡ Live Model Listing

Fetches the top trending models dynamically from HuggingFace.

- **Smart Sorting:** Sorts by parameter size vs popularity.
- **Heuristic Capabilities:** Auto-detects features like Vision (👁️), Coding (💻), Reasoning (🧠), and Multilingual (🌍) support from model names and metadata.
- **Real-time Stats:** Shows download counts and last update dates.

### 2. 🖥️ Smart GPU Fit

Automatically detects your NVIDIA GPU (e.g., RTX 3090, 4090, 5090) and calculates VRAM usage for every model.

- **Fit Indicators:**
  - ✅ **fits**: Runs comfortably.
  - ⚠️ **tight**: Might fit with reduced context.
  - ❌ **too big**: Exceeds VRAM.
- **Recommendation:** Flags the largest model that fits your hardware (`← recommended`).

### 3. 🧭 Curated Catalog (`c` command)

Built-in offline database of 22+ top-tier models (Qwen, Llama, Gemma, Phi, Mistral, DeepSeek) organized by size:

- **Tiny (≤3B)** to **XL (>32B)** tiers.
- **Maker Info:** See who built it (Google, Meta, Alibaba, Mistral, etc.).
- **Download Status:** Instantly see which models you already have cached.

### 4. 📊 Benchmark Comparison (`b` command)

Compare local models against commercial giants **without leaving your terminal**.

- **Baselines:** Shows GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro scores for reference.
- **Local Rankings:** Lists the best models that fit _your_ specific GPU, ranked by quality.
- **Metrics:**
  - **MMLU** (Knowledge)
  - **HumanEval** (Coding)
  - **MT-Bench** (Chat Quality)
- **Visual Verdicts:** "72% of GPT-4o quality", "Near-commercial grade", etc.

### 5. 📦 Intelligent Downloader

- Browses individual GGUF quantizations (Q4_K_M, Q8_0, IQ2_XS, etc.).
- Shows exact file sizes and fit status for _each quantization_.
- Filters out non-GGUF files automatically.

### 6. 🚀 Unsloth Optimization

The browser prioritizes models from **Unsloth** because they are consistently the best starting point for local AI:

- **Faster Inference:** Often converted with optimal settings for speed.
- **High Integrity:** Verified GGUF conversions that minimize perplexity loss.
- **Broad Compatibility:** Wide range of quantizations (Q4, Q5, Q8, IQ-series) covering most GPU sizes.
- **Latest Architectures:** Unsloth is frequently first to support new architectures (DeepSeek-R1, Llama-3, etc.).

## Usage

```bash
# Browse top models (interactive mode)
python install_models.py

# Search for specific models
python install_models.py --search "deepseek"

# Show more results
python install_models.py --top 20
```

### Interactive Commands

Inside the browser:

- `number`: Select a model to see details & download files.
- `c`: Open the **Capabilities Catalog** (curated list).
- `b`: View **Benchmark Comparisons** (local vs commercial).
- `s`: Search for models.
- `q`: Quit.

## Meeting Detection

To detect meetings in your transcripts, run:

```bash
python detect_meetings.py --provider local --model pick
```

This will generate two summary files in the transcript directory:

1. `meetings_summary.txt`: A quick text overview of files with confirmed meetings.
2. `found_meetings.csv`: A structured CSV file with columns for File, Has Meeting, Confidence, and Reason.

## Model Accuracy Comparison

To find out which model works best for your specific audio data (e.g., finding "hidden" meetings), use the comparison tool:

```bash
python compare_models.py --report detection_report.json
```

**What it does:**

- Analyzes your transcripts to find "Hidden Meetings" (where high-confidence models detected a meeting but others missed it).
- Calculates a **Quality Score** (Recall Rate) for every model.
- explicitly recommends the best model for your data.
- Exports a detailed `model_comparison.csv` for further analysis.

## Transcript Quality Assessment

To grade the _quality_ of the transcripts (coherence, grammar, recoverability), use:

```bash
python assess_quality.py --limit 0
```

**Key Findings: The Discovery vs. Quality Trade-off**

### Benchmark Results (Noisy Outdoor Audio)

| Model       | Discovery Rate (Finding Meetings) | Quality Score (0-10) | Verdict                                |
| :---------- | :-------------------------------- | :------------------- | :------------------------------------- |
| **base.en** | **58%**                           | 4.2                  | 🏆 **Best Overall**                    |
| large-v3    | 31%                               | **5.1** (Best Text)  | High fidelity, but misses 2/3 meetings |
| large-v1    | 44%                               | 4.4                  | Reliable backup choice                 |
| large-v2    | 3%                                | 4.6                  | ❌ Misses almost everything            |
| turbo       | 8%                                | 1.8                  | ❌ Poor sensitivity & quality          |

New users should understand why different models excel at different tasks:

1.  **Discovery (Finding the Meeting)**:
    - **Champion**: **`base.en`**
    - **Why**: It is highly sensitive and "optimistic." It rarely filters out quiet or distant audio, meaning it finds 58% of meetings that larger models miss.
    - **Use Case**: Always start here to ensure you don't miss any conversations.

2.  **Quality (Transcription Fidelity)**:
    - **Champion**: **`large-v3`**
    - **Why**: _If_ it detects speech, it produces the most coherent and grammatically correct text (Score: 5.1/10). However, it is very aggressive at filtering "noise," meaning it completely ignores 70% of the meetings in this dataset (Discovery Rate: 31%).
    - **Use Case**: Use this _only_ on files that `base.en` has already flagged as containing speech, if you need slightly better punctuation/grammar.

### Detailed Model Breakdown (For Power Users)

Here is how the top tiers compare for **noisy outdoor audio**:

- **`large-v3` (Score: 5.1/10)**: The "High Fidelity" expert. It produces the best text when it works, but it is extremely sensitive to noise and will aggressively filter out faint speech. **Use only for high-quality recordings.**
- **`large-v2` (Score: 4.6/10)**: Similar to v3 but significantly prone to hallucination on silence (Discovery Rate: 2.6%). It frequently outputs repetitive garbage instead of silence. **Avoid for this dataset.**
- **`large-v1` (Score: 4.4/10)**: The "Old Reliable." It has a much better Discovery Rate (44%) than its successors, making it the best "Large" model if you absolutely need the large parameter count.
- **`turbo` (Score: 1.8/10)**: Optimized for speed, not sensitivity. It performed poorly on this specific challenging dataset (7.7% Discovery).

**Recommendation:**

For most users, **`base.en`** is the only model you need. It offers the **best balance** (High Discovery + Good Quality). Only use `large-v3` as a second pass on confirmed files.

## Installation

Requires Python 3.8+ and `huggingface_hub`.

```bash
pip install huggingface_hub
```

## Requirements

- Windows, Linux, or macOS.
- NVIDIA GPU recommended (for VRAM detection features), but works on CPU.
- Internet connection (for live listings/downloads). Benchmarks and catalog are offline-available.

## License

[Apache 2.0](LICENSE)
