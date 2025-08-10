# LLM Hardware Compatibility Checker — Multi-GPU (single-file)

**`llm_checker_v3_2_multi.py`** is a single-file, interactive tool that:

- auto-detects your **CPU, RAM, storage, and all GPUs**,
- looks up (or estimates) **model sizes** and common **quantizations**,
- decides whether a model will run on **single GPU**, **multi-GPU (tensor parallel)**, or **CPU**,
- estimates **tokens/sec** and shows **trade-offs & recommendations**,
- prints **setup instructions** (Ollama / LM Studio / llama.cpp).

> This is a *standalone* script designed for easy sharing and “download-and-run”.  
> If you prefer a modular library + CLI, see **Contribute** below for how we suggest organizing that.

---

## ✨ Features

- **Multi-GPU aware**: checks shard feasibility with **10% per-shard headroom**.
- **Friendly UX**: step-by-step sections (hardware → model search → analysis → recs → setup).
- **Cross-platform**: Windows, Linux, and Apple Silicon (basic heuristics for M-series).
- **No setup beyond Python**: just install a few pip packages (below).

---

## 🧩 Requirements

- Python **3.9+** (tested up to 3.13)
- Packages:
  ```bash
  pip install psutil requests gputil py-cpuinfo
  ```
- **NVIDIA** users (Windows/Linux): having `nvidia-smi` on PATH helps as a fallback detector.

---

## 🚀 Quick Start

### Windows (PowerShell or CMD)
```bat
python -m pip install --upgrade pip
pip install psutil requests gputil py-cpuinfo

python C:\path\to\llm_checker_v3_2_multi.py
```

### macOS / Linux
```bash
python3 -m pip install --upgrade pip
pip install psutil requests gputil py-cpuinfo

python3 /path/to/llm_checker_v3_2_multi.py
```

You’ll see:

1. **STEP 1: HARDWARE DETECTION** – full report (OS, CPU cores/threads, each GPU with VRAM, RAM, storage).
2. Prompt: `Proceed with this hardware configuration? (y/n)` → type `y`.
3. **STEP 2: MODEL SELECTION** – examples are shown. Type a **model name** (e.g. `llama 3 8b`, `mistral 7b`, `mixtral 8x7b`, `phi-3`) and press Enter.
4. **STEP 3/4** – compatibility report: device choice (single/multi/CPU), speed label, `~ tok/s`, and **Recommendations**.
5. Optional: “**Setup Instructions**” tailored to your machine.

> Tip (Windows): If you accidentally paste a full command like  
> `& C:\...\python.exe "…\llm_checker_v3_2_multi.py" llama 3 8b` into the model prompt, the script sanitizes it and keeps only the model name.

---

## 🧠 How it Works (high-level)

1. **Detect hardware**
   - CPU model + cores/threads via `psutil` / `py-cpuinfo`
   - GPUs via `gputil` (preferred) or `nvidia-smi` (fallback)
   - RAM and storage free space via `psutil`

2. **Model info**
   - Tries Hugging Face search for an ID and tags.
   - If not found, *estimates* base FP16 size from parameter count (e.g., 7B ≈ 14 GB).

3. **Quant sizing**
   - Applies standard size multipliers for popular quants (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, etc.).

4. **Device selection**
   - **Single GPU** if the largest GPU VRAM ≥ model quant size + 0.5 GB headroom.
   - Else **Multi-GPU** if the *sum* of VRAM suffices *and* each chosen GPU can hold its shard with **+10% headroom**.
   - Else **CPU fallback** (with RAM sanity checks).

5. **Performance**
   - Baseline single-GPU TPS table (by common GPU families) + heuristics.
   - Multi-GPU scaling: `single_proxy * num_gpus * efficiency`  
     (efficiency defaults: ~0.78 for 2×, ~0.72 for 3×, ~0.68 for 4×).

6. **Output**
   - Sorted options, warnings/errors, and a **Recommendations** section (Best quality / Best performance / Popular choice).

---

## 🧪 Don’t have multiple GPUs? Simulate them

You can simulate rigs and validate the logic without real hardware:

```python
import sys
sys.path.append(r"C:\path\to\folder")  # where the file lives

import llm_checker_v3_2_multi as chk

# Simulate a 2×24GB NVIDIA box
hw = chk.HardwareSpecs(
    cpu_model="Sim CPU", cpu_cores=16, cpu_threads=32,
    gpus=[chk.GPUInfo(0,"NVIDIA RTX 3090",24.0),
          chk.GPUInfo(1,"NVIDIA RTX 3090",24.0)],
    total_gpu_vram_gb=48.0, system_ram_gb=128.0,
    storage_type="SSD", available_storage_gb=2000.0, os_type="Windows"
)

# Estimate model info offline (no network), or use WebSearcher().search_model_info("llama 3 70b")
mi = chk.WebSearcher()._estimate("llama 3 70B")
res = chk.analyze_compatibility(hw, mi)
chk.display_model_report(mi, res)
```

Try different combos (e.g., `4×12GB`, `2×16GB`, uneven VRAM) to check shard feasibility and device selection.

---

## 🛠️ Troubleshooting

- **No GPUs detected on Windows**
  - Ensure `pip install gputil` succeeded.
  - Update GPU drivers.
  - Check `nvidia-smi` works in the terminal.

- **It says CPU mode even though I have a big GPU**
  - Check VRAM free vs. required (other apps may be using VRAM).
  - Some quants need more headroom; try a smaller quant (e.g., `Q4_K_M` instead of `Q6_K`).

- **I want closer performance numbers**
  - Real TPS depends on framework (Ollama, vLLM, TGI, llama.cpp), context length, batch size, and interconnect.  
  - Open a PR with your measured TPS and setup (see **Contribute**).

---

## 🧭 Roadmap

- Context-window & batch size inputs to refine VRAM math (KV cache).
- Efficiency toggles (PCIe vs NVLink).
- Optional Rich-based tables (pretty TUI).
- Modular library + CLI packaging (keep this single file as a release artifact).

---

## 🤝 Contribute

We welcome PRs! See **contribute.md** for guidelines, dev setup, and how to calibrate performance.

---

## 📄 License

Choose a license you’re comfortable with (e.g., MIT or Apache-2.0) and add a `LICENSE` file.
