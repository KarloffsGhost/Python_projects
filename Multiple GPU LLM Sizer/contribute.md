# Contributing Guide

Thanks for your interest in improving the LLM Hardware Compatibility Checker!  
This project currently ships as a **single-file runner** for easy sharing. We welcome:
- fixes and improvements to detection, device selection, and perf estimates,
- better guidance/wording in the UX,
- calibration data (measured TPS on real hardware).

If you’d like to help modularize it (recommended), see **“Future modular layout”** below.

---

## Code of Conduct

Be kind, constructive, and respectful. Assume good intent.

---

## Dev environment

1. **Clone** your fork and create a venv:
   ```bash
   python -m venv .venv
   . .venv/Scripts/activate   # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Install deps**:
   ```bash
   pip install -U pip
   pip install psutil requests gputil py-cpuinfo
   # optional dev tools (if you plan to refactor / add tests):
   pip install pytest black ruff mypy
   ```

3. **Run locally**:
   ```bash
   python llm_checker_v3_2_multi.py
   ```

---

## Making changes

- Keep the **interactive UX** friendly and consistent (section banners, prompts, clear messages).
- Avoid hard-coding a model; the script must wait for user input.
- Windows users often paste entire command lines into prompts—keep the **input sanitizer** intact (or improve it).
- Be conservative with **performance estimates** and call out assumptions.

### Style

- Use `black` for formatting and `ruff` for lint hints:
  ```bash
  black .
  ruff check .
  ```
- Keep functions short and focused; add docstrings for non-obvious logic.

---

## Testing

We don’t require a GPU to test logic. Please exercise:

### 1) Unit-style tests (optional)

If you add a `tests/` folder, examples:

- **Device selection / headroom**
  - Verify that:
    - single-GPU chosen when largest VRAM ≥ `size + 0.5`
    - multi-GPU chosen only when each shard ≥ `(size / N) * 1.10`
    - CPU fallback otherwise

- **Quant sizes**
  - Q4/Q5/Q6 multipliers are applied correctly from base FP16 size.

- **Perf labels**
  - Thresholds map to labels (`🐌 Slow` … `🚀 Extremely Fast`) correctly.

### 2) Simulation (manual)

Use the built-in dataclasses to simulate GPU rigs:

```python
import llm_checker_v3_2_multi as chk

def fake():
    gpus = [chk.GPUInfo(0, "NVIDIA RTX 3090", 24.0),
            chk.GPUInfo(1, "NVIDIA RTX 3090", 24.0)]
    return chk.HardwareSpecs(
        cpu_model="Sim CPU", cpu_cores=16, cpu_threads=32,
        gpus=gpus, total_gpu_vram_gb=sum(g.vram_gb for g in gpus),
        system_ram_gb=128.0, storage_type="SSD",
        available_storage_gb=2000.0, os_type="Windows"
    )

hw = fake()
mi = chk.WebSearcher()._estimate("llama 3 70B")  # no network required
res = chk.analyze_compatibility(hw, mi)
chk.display_model_report(mi, res)
```

Try various VRAM mixes (2×16, 3×16, 4×12, etc.) and confirm the shard rule holds.

---

## Submitting a PR

- **Scope:** Small, focused changes merge faster.
- **Explain:** In your PR description, describe the problem, your approach, and trade-offs.
- **Test evidence:** Paste a short simulation transcript or your real hardware measurements, if relevant.
- **Checklist:**
  - [ ] No syntax errors (`python -m py_compile llm_checker_v3_2_multi.py`)
  - [ ] Script still runs end-to-end
  - [ ] Banners/prompts readable on Windows and Linux terminals
  - [ ] (If touched) headroom/sharding logic remains correct

---

## Calibrating performance (very welcome!)

If you can test on real hardware, please share:

- GPU(s), VRAM per GPU, interconnect (PCIe / NVLink)
- Framework (Ollama / LM Studio / vLLM / TGI / llama.cpp)
- Context window / batch size
- Measured steady-state **tokens/sec**
- Model + quant (e.g., Llama-3-8B `q4_k_m`)

We’ll use your data to refine the baseline table and multi-GPU efficiency factors.

---

## Future modular layout (optional path)

If you want to help split this into modules while keeping the single-file release:

```
/src
  hardware_detector.py
  model_search.py
  performance.py
  compatibility.py
  display.py
  cli.py
/tools
  build_single_file.py   # bundles modules into one script for releases
```

- Keep the modules as the **source of truth**.
- Use the builder to publish a single-file artifact in GitHub Releases.

---

## License

Please include or reference the repo’s chosen license (MIT/Apache-2.0 recommended).
