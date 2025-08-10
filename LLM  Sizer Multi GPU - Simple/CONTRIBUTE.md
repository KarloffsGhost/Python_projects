# CONTRIBUTE.md — Contributing Guide

Thanks for your interest in improving the LLM Hardware Compatibility Checker!

This project is currently a **single‑file tool** for easy sharing. We gladly accept:
- fixes to detection and device selection,
- improved quantization sizing and warnings,
- UX refinements (clearer prompts/messages),
- **calibration data** (measured tokens/sec for specific models/quantizations on real hardware).

---

## Code of Conduct

Be kind, constructive, and respectful. Assume good intent.

---

## Dev Setup

```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install psutil requests gputil py-cpuinfo

# optional dev tools
pip install pytest black ruff mypy
```

Run locally:
```bash
python llm_checker_web_multi.py
```

---

## Guidelines

- Keep the UX friendly and consistent (section banners, prompts, readable output).
- **Do not hard‑code a model**; always wait for user input.
- Be conservative with performance assumptions; call out limitations where relevant.
- Keep functions short and well‑named; add docstrings for non‑trivial logic.
- Prefer small, focused PRs.

### Style & Quality

- Format with `black` and lint with `ruff`:
  ```bash
  black .
  ruff check .
  ```
- Optional typing with `mypy` if you touch signatures or add new modules.
- Validate basic syntax before opening a PR:
  ```bash
  python -m py_compile llm_checker_web_multi.py
  ```

---

## Testing without Multi‑GPU

You can simulate GPU rigs and exercise the core logic programmatically (no drivers required):

```python
import llm_checker_web_multi as mod

# Fake 3x16GB
hw = mod.HardwareSpecs(
    cpu_model="Sim CPU", cpu_cores=16, cpu_threads=32,
    gpus=[mod.GPUInfo(0,"NVIDIA 4080",16.0,0,16.0),
          mod.GPUInfo(1,"NVIDIA 4080",16.0,0,16.0),
          mod.GPUInfo(2,"NVIDIA 4080",16.0,0,16.0)],
    total_gpu_vram_gb=48.0, system_ram_gb=128.0,
    storage_type="SSD", available_storage_gb=2000.0, os_type="Windows"
)
checker = mod.LLMHardwareChecker()
checker.hardware_specs = hw
mi = mod.WebSearcher()._estimate_model_info("llama 3 70B")
compat = checker.check_compatibility(hw, mi)
checker.display_compatibility_report(mi, compat)
```

Please include a short transcript of your test in the PR description.

---

## Submitting a PR

- Explain the problem, the approach you took, and trade‑offs.
- Include test evidence (simulation or real hardware numbers).
- Checklist:
  - [ ] Syntax OK (`python -m py_compile llm_checker_web_multi.py`)
  - [ ] End‑to‑end run still works
  - [ ] Output is readable on Windows and Linux terminals
  - [ ] Multi‑GPU sharding logic still behaves sensibly

---

## Performance Calibration (Very Welcome)

If you have multi‑GPU hardware, please share:
- GPU models, VRAM per GPU, interconnect (PCIe / NVLink)
- Framework (Ollama / LM Studio / vLLM / TGI / llama.cpp)
- Context window / batch size
- Model + quantization
- Measured steady‑state tokens/sec

We’ll use your data to refine sizing and the multi‑GPU guidance.

---

## License

Match the repository’s `LICENSE` (MIT / Apache‑2.0 recommended).
