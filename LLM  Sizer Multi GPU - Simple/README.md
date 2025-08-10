# LLM Hardware Compatibility Checker — Web Search + Multi‑GPU

This repository contains a **single‑file** script that:
- auto‑detects your **CPU, RAM, storage, and all GPUs**,
- searches the web (Hugging Face API) for **model size & tags**, or falls back to **estimates**,
- checks if a given model/quantization can run on **single GPU**, **multi‑GPU (tensor parallel)**, or **CPU**,
- prints a clear **compatibility report** and **setup instructions** (Ollama / vLLM / LM Studio / llama.cpp).

> Script name used below: `llm_checker_web_multi.py` (you can rename it).

---

## ✨ Features

- **Multi‑GPU aware**: if a model doesn’t fit on one GPU, the tool checks whether it can be **sharded** across multiple GPUs.
- **Quantization catalog**: FP16, Q8_0, Q6_K, Q5_K_M/S, Q4_K_M/S, Q4_0, Q3_K_M/S, Q2_K, IQ3_XXS, IQ2_XXS.
- **Friendly, step‑by‑step UX**: Hardware → Model search → Compatibility → Recommendations → Setup.
- **Cross‑platform**: Windows, Linux, macOS (Apple Silicon heuristic for unified memory).
- **No extra infra**: just Python + small dependencies.

---

## 🧩 Requirements

- Python **3.9+** (tested up to 3.13)
- Install dependencies:
  ```bash
  pip install psutil requests gputil py-cpuinfo
  ```
- NVIDIA users: having `nvidia-smi` on PATH helps as a fallback GPU detector.

---

## 🚀 Quick Start

1) Save the script as `llm_checker_web_multi.py`.
2) Install deps:
   ```bash
   python -m pip install --upgrade pip
   pip install psutil requests gputil py-cpuinfo
   ```
3) Run:
   ```bash
   python llm_checker_web_multi.py
   ```
4) Follow the prompts:
   - STEP 1: Hardware detection (CPU, RAM, storage, **all GPUs**)
   - STEP 2: Model selection → type **only** a model name (e.g. `llama 3 8b`, `mistral 7b`, `mixtral 8x7b`, `phi-3`)
   - STEP 3: Web search (Hugging Face); falls back to estimates if not found
   - STEP 4: Compatibility analysis + **recommendations**
   - Optional: **Setup instructions** tailored to your machine

> Tip: If model search fails (e.g. niche models), try a simpler name (`llama 3 8b` rather than the full repo id).

---

## 🧠 How it Works (high‑level)

1. **Hardware detection**
   - CPU model/cores/threads via `psutil` / `py-cpuinfo`
   - GPUs via `gputil` (preferred) or `nvidia-smi` (fallback)
   - RAM + storage via `psutil`
2. **Model info**
   - Hugging Face API search for `modelId`, tags (parameter count)
   - If missing, estimate FP16 size from params (≈ **2 GB per 1B** parameters)
3. **Quantization sizing**
   - Applies standard ratios (Q4/Q5/Q6/Q8/…)
4. **Device selection**
   - Choose **Single GPU** if the largest GPU can host the quant (with a small overhead)
   - Else consider **Multi‑GPU** sharding (checks total VRAM and per‑GPU allocation)
   - Else **CPU fallback** (RAM check + “slow” warning)
5. **Output**
   - Clear report grouped by Single‑GPU / Multi‑GPU / CPU options
   - Warnings (e.g., HDD will load slowly) and **actionable setup instructions**

---

## 🧪 Simulate Multi‑GPU (optional)

Don’t have multiple GPUs? You can simulate a rig and run the compatibility analysis programmatically:

```python
# Save this in a Python shell near the script, or add the folder to PYTHONPATH
import llm_checker_web_multi as mod

# Build a fake 2x24GB machine
hw = mod.HardwareSpecs(
    cpu_model="Sim CPU", cpu_cores=16, cpu_threads=32,
    gpus=[mod.GPUInfo(0,"NVIDIA RTX 3090",24.0,0,24.0),
          mod.GPUInfo(1,"NVIDIA RTX 3090",24.0,0,24.0)],
    total_gpu_vram_gb=48.0, system_ram_gb=128.0,
    storage_type="SSD", available_storage_gb=2000.0, os_type="Windows"
)

# Create the driver and set the hardware (so the report can print GPU summary)
checker = mod.LLMHardwareChecker()
checker.hardware_specs = hw

# Avoid network: estimate "llama 3 70B"
mi = mod.WebSearcher()._estimate_model_info("llama 3 70B")

compat = checker.check_compatibility(hw, mi)
checker.display_compatibility_report(mi, compat)
```

Change the `gpus=[...]` list to try `2×16GB`, `4×12GB`, uneven VRAM, etc.

---

## 🛠️ Troubleshooting

- **No GPUs detected (Windows/NVIDIA)**  
  - Ensure `pip install gputil` succeeded
  - Update NVIDIA drivers
  - Confirm `nvidia-smi` works in the terminal

- **It only offers CPU mode**  
  - Check VRAM free vs required (other apps may be using VRAM)
  - Try a smaller quant (e.g., `Q4_K_M` instead of `Q6_K`)

- **Search is slow or fails**  
  - Check your internet connection
  - Try a broader model name

- **Apple Silicon**  
  - GPU memory is unified; the script uses conservative estimates based on chip family (M1/M2/M3)

---

## 🔐 Privacy

The script makes **read‑only** calls to Hugging Face’s public API to search models. No personal data is collected or stored.

---

## 📄 License

Add a `LICENSE` file (MIT or Apache‑2.0 recommended).

---

## 🤝 Contributing

PRs welcome! See **[CONTRIBUTE.md](CONTRIBUTE.md)** for guidelines and calibration notes.
