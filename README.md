# 🤖 Model Browser & Downloader

**Discovery, benchmarks, and smart GPU fit — 100% offline.**

A powerful CLI tool to browse, analyze, and download the best local LLMs (GGUF format) from HuggingFace. It bridges the gap between raw file listings and informed decision-making by providing capability insights, hardware compatibility checks, and benchmark comparisons directly in your terminal.

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
- **Local Rankings:** Lists the best models that fit *your* specific GPU, ranked by quality.
- **Metrics:**
  - **MMLU** (Knowledge)
  - **HumanEval** (Coding)
  - **MT-Bench** (Chat Quality)
- **Visual Verdicts:** "72% of GPT-4o quality", "Near-commercial grade", etc.

### 5. 📦 Intelligent Downloader
- Browses individual GGUF quantizations (Q4_K_M, Q8_0, IQ2_XS, etc.).
- Shows exact file sizes and fit status for *each quantization*.
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
python model_browser.py

# Search for specific models
python model_browser.py --search "deepseek"

# Show more results
python model_browser.py --top 20
```

### Interactive Commands
Inside the browser:
- `number`: Select a model to see details & download files.
- `c`: Open the **Capabilities Catalog** (curated list).
- `b`: View **Benchmark Comparisons** (local vs commercial).
- `s`: Search for models.
- `q`: Quit.

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
