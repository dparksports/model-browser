#!/usr/bin/env python3
"""
verify_gpu.py — Full GPU / CUDA / Model verification for the meetings environment.

Usage:
    python verify_gpu.py
"""

import sys
import os
import time

# ANSI colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _enable_ansi_windows():
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


def main():
    _enable_ansi_windows()

    W = 60
    print(f"\n{BOLD}{CYAN}{'═' * W}{RESET}")
    print(f"{BOLD}{CYAN}{'  🔍  GPU / CUDA Verification':^{W}}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}\n")

    all_ok = True

    # ── 1. Python ──────────────────────────────────────────────
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  {BOLD}Python{RESET}")
    print(f"    Version:       {py_ver}")
    print(f"    Executable:    {sys.executable}")
    print(f"    Platform:      {sys.platform}")
    print()

    # ── 2. NVIDIA Driver ──────────────────────────────────────
    print(f"  {BOLD}NVIDIA Driver{RESET}")
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total,memory.free,temperature.gpu,power.draw,pstate",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 7:
            print(f"    Driver:        {parts[0]}")
            print(f"    GPU:           {parts[1]}")
            print(f"    VRAM total:    {float(parts[2])/1024:.1f} GB ({parts[2]} MB)")
            print(f"    VRAM free:     {float(parts[3])/1024:.1f} GB ({parts[3]} MB)")
            print(f"    Temperature:   {parts[4]}°C")
            print(f"    Power draw:    {parts[5]} W")
            print(f"    P-state:       {parts[6]}")
        else:
            print(f"    Info:          {out}")
    except FileNotFoundError:
        print(f"    {RED}nvidia-smi not found — no NVIDIA driver?{RESET}")
        all_ok = False
    except Exception as e:
        print(f"    {RED}Error: {e}{RESET}")
    print()

    # ── 3. PyTorch ─────────────────────────────────────────────
    print(f"  {BOLD}PyTorch{RESET}")
    try:
        import torch
        print(f"    Version:       {torch.__version__}")
    except ImportError:
        print(f"    {RED}❌ Not installed{RESET}")
        print(f"    {DIM}Run: pip install torch --index-url https://download.pytorch.org/whl/cu128{RESET}")
        return

    # CUDA build
    cuda_build = torch.version.cuda
    if cuda_build:
        print(f"    CUDA (build):  {cuda_build}")
    else:
        print(f"    CUDA (build):  {RED}None — CPU-only build!{RESET}")
        all_ok = False

    # cuDNN
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    if cudnn:
        major, minor, patch = cudnn // 1000, (cudnn % 1000) // 100, cudnn % 100
        print(f"    cuDNN:         {major}.{minor}.{patch} (enabled={torch.backends.cudnn.enabled})")
    else:
        print(f"    cuDNN:         {YELLOW}Not available{RESET}")

    # bf16 support
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16 = torch.cuda.is_bf16_supported()
            print(f"    bfloat16:      {'Supported' if bf16 else 'Not supported'}")
        except Exception:
            pass
    print()

    # ── 4. CUDA Runtime ────────────────────────────────────────
    print(f"  {BOLD}CUDA Runtime{RESET}")
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print(f"    Available:     {GREEN}Yes{RESET}")
        gpu_count = torch.cuda.device_count()
        print(f"    Device count:  {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            cc = f"{props.major}.{props.minor}"
            print(f"    GPU {i}:         {GREEN}{props.name}{RESET}")
            print(f"      VRAM:        {vram_gb:.1f} GB")
            print(f"      Compute:     sm_{props.major}{props.minor}0 (capability {cc})")
            print(f"      Multi-proc:  {props.multi_processor_count} SMs")
    else:
        print(f"    Available:     {RED}No{RESET}")
        if cuda_build:
            print(f"    {DIM}PyTorch has CUDA {cuda_build} but runtime failed.{RESET}")
        else:
            print(f"    {DIM}Install CUDA-enabled PyTorch for GPU support.{RESET}")
        all_ok = False
    print()

    # ── 5. GPU Compute Test ────────────────────────────────────
    print(f"  {BOLD}GPU Compute Test{RESET}")
    if cuda_avail:
        try:
            t0 = time.perf_counter()
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
            b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            result = c.sum().item()
            del a, b, c
            print(f"    FP32 matmul:   {GREEN}PASSED{RESET}  {DIM}(2048×2048 in {elapsed:.1f}ms){RESET}")
        except Exception as e:
            print(f"    FP32 matmul:   {RED}FAILED — {e}{RESET}")
            all_ok = False

        try:
            t0 = time.perf_counter()
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            del a, b, c
            print(f"    BF16 matmul:   {GREEN}PASSED{RESET}  {DIM}(2048×2048 in {elapsed:.1f}ms){RESET}")
        except Exception as e:
            print(f"    BF16 matmul:   {RED}FAILED — {e}{RESET}")

        # Memory after compute test
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"    GPU mem used:  {alloc:.0f} MB allocated, {reserved:.0f} MB reserved")
    else:
        print(f"    {YELLOW}⚠  Skipped (no CUDA){RESET}")
    print()

    # ── 6. Key Libraries ───────────────────────────────────────
    print(f"  {BOLD}Key Libraries{RESET}")
    libs = [
        ("transformers",    "transformers"),
        ("accelerate",      "accelerate"),
        ("PIL",             "Pillow"),
        ("qwen_vl_utils",   "qwen-vl-utils"),
        ("faster_whisper",  "faster-whisper"),
        ("huggingface_hub", "huggingface-hub"),
        ("torchvision",     "torchvision"),
        ("torchaudio",      "torchaudio"),
    ]
    for import_name, display_name in libs:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    {display_name:<20s} {GREEN}{ver}{RESET}")
        except ImportError:
            print(f"    {display_name:<20s} {RED}Not installed{RESET}")
    print()

    # ── 7. Load Qwen VLM Model ─────────────────────────────────
    print(f"  {BOLD}Qwen2.5-VL-7B Model{RESET}")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_loaded = False
    if cuda_avail:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            print(f"    {DIM}Loading {model_name}...{RESET}")
            t0 = time.perf_counter()
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_name)
            load_time = time.perf_counter() - t0
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"    Status:        {GREEN}Loaded{RESET}")
            print(f"    Device:        {device}")
            print(f"    Dtype:         {dtype}")
            print(f"    Load time:     {load_time:.1f}s")
            print(f"    GPU mem used:  {vram_used:.1f} GB")
            model_loaded = True
        except Exception as e:
            print(f"    {RED}❌ Load failed: {e}{RESET}")
            all_ok = False
    else:
        print(f"    {YELLOW}⚠  Skipped (no CUDA){RESET}")
    print()

    # ── 8. Model Inference Test ────────────────────────────────
    print(f"  {BOLD}Model Inference Test{RESET}")
    if model_loaded:
        try:
            from PIL import Image
            from qwen_vl_utils import process_vision_info

            test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": test_img},
                    {"type": "text", "text": "What color is this image? Reply in one word."},
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(model.device)

            t0 = time.perf_counter()
            output_ids = model.generate(**inputs, max_new_tokens=20)
            torch.cuda.synchronize()
            infer_time = time.perf_counter() - t0

            output_text = processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            print(f"    Status:        {GREEN}PASSED{RESET}")
            print(f"    Response:      \"{output_text}\"")
            print(f"    Infer time:    {infer_time:.2f}s")

            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"    Peak GPU mem:  {peak_mem:.1f} GB")

            del model, processor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {RED}❌ Inference failed: {e}{RESET}")
            all_ok = False
    else:
        print(f"    {YELLOW}⚠  Skipped (model not loaded){RESET}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═' * W}{RESET}")
    if all_ok and cuda_avail and model_loaded:
        print(f"  {GREEN}{BOLD}✅ All checks passed — GPU + Model ready!{RESET}")
    elif cuda_avail:
        print(f"  {YELLOW}{BOLD}⚠  CUDA works but some checks failed.{RESET}")
    else:
        print(f"  {RED}{BOLD}❌ GPU not available — running on CPU only.{RESET}")
        print(f"  {DIM}   Run install_whisper_models.py to fix PyTorch CUDA.{RESET}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}\n")


if __name__ == "__main__":
    main()
