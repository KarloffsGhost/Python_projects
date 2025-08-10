#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Hardware Compatibility Checker v3.2 — Multi‑GPU Edition (Fixed)
- Mirrors the single‑GPU UX (sections, prompts, setup)
- Multi‑GPU detection and compatibility
- Windows paste sanitization
"""

import re
import platform
import psutil
import subprocess
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

# Optional deps
try:
    import GPUtil
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except Exception:
    CPUINFO_AVAILABLE = False


@dataclass
class GPUInfo:
    id: int
    name: str
    vram_gb: float
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0


@dataclass
class HardwareSpecs:
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    gpus: List[GPUInfo] = field(default_factory=list)
    total_gpu_vram_gb: float = 0.0
    system_ram_gb: float = 0.0
    storage_type: str = "Unknown"
    available_storage_gb: float = 0.0
    os_type: str = "Unknown"

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        return max(self.gpus, key=lambda g: g.vram_gb) if self.gpus else None


@dataclass
class ModelInfo:
    name: str
    parameter_count: str
    base_size_gb: float
    quantization_info: Dict[str, Dict]
    source_urls: List[str]
    huggingface_id: Optional[str] = None
    description: Optional[str] = None


# ---------------------------
# Banner / UX helpers
# ---------------------------

def print_banner() -> None:
    print("""╔══════════════════════════════════════════════════════════╗
║        🤖 LLM Hardware Compatibility Checker v3.2 🤖      ║
║                   Multi‑GPU Edition                       ║
╚══════════════════════════════════════════════════════════╝
    
This tool will:
  ✓ Automatically detect your hardware (multi‑GPU aware)
  ✓ Search the web for model requirements
  ✓ Check compatibility (single GPU / multi‑GPU / CPU)
  ✓ Provide setup instructions

Note: Internet connection required for model information.
""")


# ---------------------------
# Windows paste sanitization
# ---------------------------

def sanitize_user_input(s: str) -> str:
    """
    Handle accidental pastes like:
      & C:/.../python.exe "c:/.../llm_checker.py" llama 3 8b
    We strip the command/exe/script path and keep only the model name.
    """
    s = s.strip().strip('"').strip("'")
    if not s:
        return s
    # Remove leading PowerShell call operator
    if s.startswith("& "):
        s = s[2:].lstrip()
    # Remove sequences like: python.exe <path>script.py
    s = re.sub(r'(?i)^(python(?:\.exe)?\s+)?"?[a-z]:[^\s"]+\.py"?\s*', '', s)
    # If still contains any .py paths, drop them
    s = re.sub(r'(?i)"?[a-z]:[^\s"]+\.py"?', '', s)
    return s.strip()


# ---------------------------
# Web searcher (HF)
# ---------------------------

class WebSearcher:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def search_model_info(self, model_name: str) -> Optional[ModelInfo]:
        print(f"\n🔍 Searching for information about {model_name}...")
        info = self._search_hf(model_name)
        if not info:
            print("  ⚠️  Using estimated values based on model name...")
            info = self._estimate(model_name)
        return info

    def _search_hf(self, name: str) -> Optional[ModelInfo]:
        patterns = [
            name,
            name.replace(" ", "-"),
            name.lower().replace(" ", "-"),
            f"meta-llama/{name.lower().replace(' ', '-')}",
            f"mistralai/{name.lower().replace(' ', '-')}",
            f"microsoft/{name.lower().replace(' ', '-')}",
            f"google/{name.lower().replace(' ', '-')}",
            f"deepseek-ai/{name.lower().replace(' ', '-')}",
        ]
        for p in patterns:
            try:
                r = self.session.get(
                    f"https://huggingface.co/api/models?search={quote(p)}&limit=3",
                    timeout=5
                )
                if r.status_code == 200:
                    js = r.json()
                    if js:
                        return self._parse_hf(js[0], name)
            except Exception:
                continue
        return None

    def _parse_hf(self, hf: Dict, original: str) -> ModelInfo:
        mid = hf.get('modelId', '')
        param = self._param_from_id(mid, hf.get('tags', []))
        base = self._fp16_size_from_param(param)
        return ModelInfo(
            name=original,
            parameter_count=param,
            base_size_gb=base,
            quantization_info=self._standard_quants(base),
            source_urls=[f"https://huggingface.co/{mid}"],
            huggingface_id=mid,
            description=hf.get('description', '')
        )

    def _estimate(self, name: str) -> ModelInfo:
        m = re.search(r'(\d+\.?\d*)\s*[bB]', name)
        b = float(m.group(1)) if m else 7.0
        param = f"{b:g}B"
        base = b * 2.0
        return ModelInfo(
            name=name,
            parameter_count=param,
            base_size_gb=base,
            quantization_info=self._standard_quants(base),
            source_urls=["Estimated"],
            description="Estimated from name"
        )

    def _param_from_id(self, mid: str, tags: List[str]) -> str:
        m = re.search(r'(\d+\.?\d*)[bB]', mid or '')
        if m:
            return f"{m.group(1)}B"
        for t in (tags or []):
            m2 = re.search(r'(\d+\.?\d*)[bB]', t or '')
            if m2:
                return f"{m2.group(1)}B"
        return "7B"

    def _fp16_size_from_param(self, p: str) -> float:
        m = re.search(r'(\d+\.?\d*)', p or '')
        b = float(m.group(1)) if m else 7.0
        return b * 2.0

    def _standard_quants(self, base: float) -> Dict[str, Dict]:
        return {
            "FP16":    {"size_gb": base,        "quality": "★★★★★", "description": "Full precision (best)"},
            "Q8_0":    {"size_gb": base * 0.53, "quality": "★★★★★", "description": "8‑bit, minimal loss"},
            "Q6_K":    {"size_gb": base * 0.41, "quality": "★★★★☆", "description": "6‑bit, great balance"},
            "Q5_K_M":  {"size_gb": base * 0.36, "quality": "★★★★☆", "description": "5‑bit medium"},
            "Q5_K_S":  {"size_gb": base * 0.34, "quality": "★★★☆☆", "description": "5‑bit small"},
            "Q4_K_M":  {"size_gb": base * 0.30, "quality": "★★★☆☆", "description": "4‑bit sweet spot"},
            "Q4_K_S":  {"size_gb": base * 0.28, "quality": "★★★☆☆", "description": "4‑bit small"},
            "Q4_0":    {"size_gb": base * 0.27, "quality": "★★★☆☆", "description": "Legacy 4‑bit"},
            "Q3_K_M":  {"size_gb": base * 0.24, "quality": "★★☆☆☆", "description": "3‑bit medium"},
            "Q3_K_S":  {"size_gb": base * 0.22, "quality": "★★☆☆☆", "description": "3‑bit small"},
            "Q2_K":    {"size_gb": base * 0.19, "quality": "★☆☆☆☆",  "description": "2‑bit max compression"},
            "IQ3_XXS": {"size_gb": base * 0.21, "quality": "★★☆☆☆", "description": "Improved 3‑bit"},
            "IQ2_XXS": {"size_gb": base * 0.16, "quality": "★☆☆☆☆",  "description": "Improved 2‑bit"},
        }


# ---------------------------
# Hardware detection
# ---------------------------

class HardwareDetector:
    def detect(self) -> HardwareSpecs:
        print("\n============================================================")
        print("STEP 1: HARDWARE DETECTION")
        print("============================================================")
        print("\n🔍 Detecting hardware specifications...")
        cpu_model = self._cpu()
        cpu_cores = psutil.cpu_count(logical=False) or 4
        cpu_threads = psutil.cpu_count(logical=True) or max((cpu_cores or 4) * 2, 8)
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        gpus = self._gpus()
        storage_type, avail_gb = self._storage()
        specs = HardwareSpecs(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            gpus=gpus,
            total_gpu_vram_gb=sum(g.vram_gb for g in gpus),
            system_ram_gb=ram_gb,
            storage_type=storage_type,
            available_storage_gb=avail_gb,
            os_type=platform.system(),
        )
        self.print(specs)
        return specs

    def _cpu(self) -> str:
        try:
            if CPUINFO_AVAILABLE:
                info = cpuinfo.get_cpu_info()
                return info.get('brand_raw', platform.processor() or 'Unknown CPU')
            if platform.system() == "Windows":
                return platform.processor() or "Unknown CPU"
            if platform.system() == "Darwin":
                return subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if platform.system() == "Linux":
                out = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True).decode()
                return out.split(':', 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"

    def _gpus(self) -> List[GPUInfo]:
        gpus: List[GPUInfo] = []
        try:
            if GPU_AVAILABLE:
                for i, g in enumerate(GPUtil.getGPUs()):
                    gpus.append(GPUInfo(
                        id=i,
                        name=g.name,
                        vram_gb=round(g.memoryTotal / 1024, 1),
                        vram_used_gb=round(g.memoryUsed / 1024, 1),
                        vram_free_gb=round(g.memoryFree / 1024, 1),
                    ))
                if gpus:
                    return gpus
        except Exception:
            pass
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3.0
            )
            if r.returncode == 0:
                for line in (l.strip() for l in r.stdout.splitlines() if l.strip()):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append(GPUInfo(
                            id=int(parts[0]),
                            name=parts[1],
                            vram_gb=round(float(parts[2]) / 1024, 1),
                            vram_used_gb=round(float(parts[3]) / 1024, 1),
                            vram_free_gb=round(float(parts[4]) / 1024, 1) if len(parts) > 4 else 0.0,
                        ))
        except Exception:
            pass
        return gpus

    def _storage(self) -> Tuple[str, float]:
        try:
            avail = round(psutil.disk_usage('/').free / (1024**3), 1)
            stype = "SSD"
            if platform.system() == "Windows":
                try:
                    r = subprocess.run(['wmic', 'diskdrive', 'get', 'MediaType'], capture_output=True, text=True, timeout=2.0)
                    stype = "HDD" if "Fixed hard disk" in r.stdout else "SSD"
                except Exception:
                    pass
            elif platform.system() == "Linux":
                try:
                    import glob
                    rot = False
                    for path in glob.glob('/sys/block/*/queue/rotational'):
                        with open(path, 'r') as f:
                            if f.read().strip() == '1':
                                rot = True
                                break
                    stype = "HDD" if rot else "SSD"
                except Exception:
                    pass
            return stype, avail
        except Exception:
            return "Unknown", 100.0

    def print(self, specs: HardwareSpecs) -> None:
        print("\n============================================================")
        print("DETECTED HARDWARE SPECIFICATIONS")
        print("============================================================")
        print(f"\n💻 System: {specs.os_type}")
        print(f"🧠 CPU: {specs.cpu_model}")
        print(f"   - Cores: {specs.cpu_cores} physical, {specs.cpu_threads} logical threads")
        if specs.has_gpu:
            if len(specs.gpus) == 1:
                g = specs.gpus[0]
                print(f"🎮 GPU: {g.name}")
                print(f"   - VRAM: {g.vram_gb:.1f} GB")
            else:
                print(f"🎮 GPUs: {len(specs.gpus)} total (VRAM sum: {specs.total_gpu_vram_gb:.1f} GB)")
                for g in specs.gpus:
                    print(f"   - [{g.id}] {g.name}: {g.vram_gb:.1f} GB (used {g.vram_used_gb:.1f}/free {g.vram_free_gb:.1f})")
        else:
            print("🎮 GPU: Not detected (CPU‑only mode)")
        print(f"🧩 RAM: {specs.system_ram_gb:.1f} GB")
        print(f"💾 Storage: {specs.storage_type} with {specs.available_storage_gb:.1f} GB available")


# ---------------------------
# Compatibility / performance
# ---------------------------

class Perf:
    QUANT_SPEED = {
        "FP16": 1.0, "BF16": 1.0, "FP32": 0.5,
        "Q8_0": 1.2, "Q6_K": 1.35, "Q5_K_M": 1.5, "Q5_K_S": 1.5,
        "Q4_K_M": 1.8, "Q4_K_S": 1.7, "Q4_0": 1.6,
        "Q3_K_M": 1.85, "Q3_K_S": 1.9, "Q2_K": 2.0,
        "IQ3_XXS": 1.7, "IQ2_XXS": 1.9
    }
    GPU_7B_FP16_TPS = {
        "4090": 200, "4080": 160, "4070": 100, "3090": 120, "3080": 90, "3070": 70,
        "3060": 50, "3050": 25, "A100": 220, "H100": 380, "L40S": 200,
        "7900 XTX": 140, "7900 XT": 125,
        "M3 Max": 90, "M3 Pro": 70, "M3": 55, "M2 Max": 70, "M2 Pro": 55, "M2": 40,
        "5070 Ti": 105
    }

    @classmethod
    def _gpu_base(cls, name: str, vram_gb: float) -> float:
        for token, tps in cls.GPU_7B_FP16_TPS.items():
            if token.lower() in (name or "").lower():
                return tps
        return max(20.0, vram_gb / 16.0 * 70.0)

    @classmethod
    def tps_single(cls, gpu: GPUInfo, param_b: float, quant: str) -> float:
        base = cls._gpu_base(gpu.name, gpu.vram_gb)
        size_factor = 7.0 / max(1.0, param_b)
        q = cls.QUANT_SPEED.get(quant, 1.4)
        return max(0.5, base * size_factor * q)

    @classmethod
    def tps_multi(cls, gpus: List[GPUInfo], param_b: float, quant: str) -> float:
        if not gpus:
            return 1.0
        per_vram = sorted([g.vram_gb for g in gpus])[len(gpus)//2]
        single_proxy = cls._gpu_base(gpus[0].name, per_vram) * (per_vram / 16.0)
        size_factor = 7.0 / max(1.0, param_b)
        q = cls.QUANT_SPEED.get(quant, 1.4)
        n = len(gpus)
        eff = 1.0 if n == 1 else 0.78 if n == 2 else 0.72 if n == 3 else 0.68
        return max(0.5, single_proxy * n * eff * size_factor * q)


def pick_device_for_quant(hw: HardwareSpecs, vram_req: float) -> Tuple[str, List[int]]:
    if not hw.has_gpu:
        return ("cpu", [])
    best = max(range(len(hw.gpus)), key=lambda i: hw.gpus[i].vram_gb)
    if hw.gpus[best].vram_gb >= vram_req:
        return ("single", [best])
    if len(hw.gpus) > 1 and hw.total_gpu_vram_gb >= vram_req:
        idxs = sorted(range(len(hw.gpus)), key=lambda i: hw.gpus[i].vram_gb, reverse=True)
        picked: List[int] = []
        for i in idxs:
            picked.append(i)
            n = len(picked)
            need = (vram_req / n) * 1.10  # 10% headroom per shard
            if all(hw.gpus[j].vram_gb >= need for j in picked):
                return ("multi", picked)
    return ("cpu", [])


def analyze_compatibility(hw: HardwareSpecs, mi: ModelInfo) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    m = re.search(r'(\d+\.?\d*)', mi.parameter_count or '7')
    param_b = float(m.group(1)) if m else 7.0
    for quant, meta in mi.quantization_info.items():
        size_gb = float(meta['size_gb'])
        vram_req = size_gb + 0.5
        ram_req = size_gb + 2.0
        mode, idxs = pick_device_for_quant(hw, vram_req)
        entry = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "performance": "",
            "size_gb": size_gb,
            "quality": meta['quality'],
            "description": meta['description'],
            "device": "",
            "tps": 0.0,
        }
        if size_gb > hw.available_storage_gb:
            entry["compatible"] = False
            entry["errors"].append(f"Requires {size_gb:.1f}GB storage, only {hw.available_storage_gb:.1f}GB available")
        if mode == "single":
            g = hw.gpus[idxs[0]]
            tps = Perf.tps_single(g, param_b, quant)
            entry["performance"] = perf_label(tps)
            entry["device"] = f"Single GPU: {g.name}"
            entry["tps"] = round(tps, 1)
        elif mode == "multi":
            sel = [hw.gpus[i] for i in idxs]
            tps = Perf.tps_multi(sel, param_b, quant)
            names = ", ".join(g.name for g in sel)
            entry["performance"] = perf_label(tps)
            entry["device"] = f"Multi‑GPU ({len(sel)}): {names}"
            entry["tps"] = round(tps, 1)
            entry["warnings"].append(f"Requires tensor parallel across {len(sel)} GPU(s)")
        else:
            cpu_need = ram_req * 1.5
            if cpu_need > hw.system_ram_gb:
                entry["compatible"] = False
                entry["errors"].append(f"CPU‑only path needs ~{cpu_need:.1f}GB RAM (have {hw.system_ram_gb:.1f}GB)")
            entry["performance"] = "🐌 Slow"
            entry["device"] = "CPU mode (insufficient GPU VRAM)"
            entry["tps"] = round(max(0.2, (hw.cpu_threads / 8.0) * (7.0 / max(1.0, param_b)) * 1.2), 1)
        out[quant] = entry
    return out


def perf_label(tps: float) -> str:
    if tps >= 100:
        return "🚀 Extremely Fast"
    if tps >= 50:
        return "⚡ Very Fast"
    if tps >= 20:
        return "✅ Fast"
    if tps >= 10:
        return "👍 Good"
    if tps >= 5:
        return "🔄 Moderate"
    return "🐌 Slow"


# ---------------------------
# Display
# ---------------------------

def display_model_report(mi: ModelInfo, results: Dict[str, Dict]) -> None:
    print("\n============================================================")
    print(f"COMPATIBILITY REPORT FOR {mi.name.upper()}")
    print("============================================================")
    print(f"\n📊 Model Size: {mi.parameter_count} parameters")
    print(f"💾 Base Size: {mi.base_size_gb:.1f}GB (FP16)")
    if mi.huggingface_id:
        print(f"🤗 HuggingFace: {mi.huggingface_id}")

    compat = {k: v for k, v in results.items() if v["compatible"]}
    if not compat:
        print("\n❌ This model CANNOT run on your hardware in any configuration.")
        for q, v in results.items():
            if v["errors"]:
                print(f"\n{q}: {v['size_gb']:.1f}GB")
                for e in v["errors"]:
                    print(f"  - {e}")
        return

    print("\n✅ This model CAN run on your hardware!")
    print(f"\nCompatible quantizations: {len(compat)} out of {len(results)}")

    def show(title: str, predicate) -> None:
        group = {k: v for k, v in compat.items() if predicate(v)}
        if not group:
            return
        print("\n----------------------------------------")
        print(title)
        print("----------------------------------------")
        for q, v in sorted(group.items(), key=lambda x: x[1]["size_gb"], reverse=True):
            print(f"\n{q}:")
            print(f"  Size: {v['size_gb']:.1f}GB")
            print(f"  Quality: {v['quality']}")
            print(f"  {v['description']}")
            print(f"  Device: {v['device']}  |  Perf: {v['performance']}  (~{v['tps']:.1f} tok/s)")
            for w in v["warnings"]:
                print(f"  ⚠️  {w}")

    show("⚡ FAST OPTIONS (Recommended for speed)", lambda v: v["performance"] in {"🚀 Extremely Fast", "⚡ Very Fast", "✅ Fast"})
    show("👍 GOOD OPTIONS (Balanced)", lambda v: v["performance"] == "👍 Good")

    slow = {k: v for k, v in compat.items() if v["performance"] in {"🔄 Moderate", "🐌 Slow"}}
    if slow:
        print("\n----------------------------------------")
        print("🐌 SLOW OPTIONS (CPU mode):")
        print("----------------------------------------")
        for q, v in slow.items():
            print(f"\n{q}:")
            print(f"  Size: {v['size_gb']:.1f}GB")
            print(f"  Quality: {v['quality']}")
            print(f"  {v['description']}")
            print(f"  Device: {v['device']}  |  Perf: {v['performance']}  (~{v['tps']:.1f} tok/s)")

    bad = {k: v for k, v in results.items() if not v["compatible"]}
    if bad:
        print("\n----------------------------------------")
        print("❌ INCOMPATIBLE OPTIONS:")
        print("----------------------------------------")
        for q, v in bad.items():
            print(f"\n{q}: {v['size_gb']:.1f}GB")
            for e in v["errors"]:
                print(f"  - {e}")

    print("\n============================================================")
    print("RECOMMENDATIONS")
    print("============================================================")
    best_quality = max(compat.items(), key=lambda x: x[1]["size_gb"])
    best_perf = max(compat.items(), key=lambda x: x[1]["tps"])
    print(f"\n🏆 Best Quality: {best_quality[0]}")
    print(f"   - {best_quality[1]['quality']}  |  Size: {best_quality[1]['size_gb']:.1f}GB")
    print(f"\n⚡ Best Performance: {best_perf[0]}")
    print(f"   - {best_perf[1]['quality']}  |  Size: {best_perf[1]['size_gb']:.1f}GB")
    if "Q4_K_M" in compat:
        v = compat["Q4_K_M"]
        print(f"\n🌟 Popular Choice: Q4_K_M")
        print("   - Good balance; widely supported")
        print(f"   - Perf: {v['performance']} (~{v['tps']:.1f} tok/s)")


def setup_instructions(hw: HardwareSpecs, mi: ModelInfo, compat: Dict[str, Dict]) -> None:
    print("\n============================================================")
    print("SETUP INSTRUCTIONS")
    print("============================================================")
    if not compat:
        print("\n❌ No compatible configurations available.")
        return
    rec = "Q4_K_M" if "Q4_K_M" in compat else max(compat, key=lambda k: compat[k]["tps"])
    tag = mi.name.lower().replace(" ", "-")
    if hw.has_gpu and any("NVIDIA" in g.name.upper() for g in hw.gpus):
        print(f"\n📦 To run {mi.name} on GPU:")
        print("\n🚀 Option 1 — Ollama (easy)")
        print("1. Install Ollama: https://ollama.ai/download/windows")
        print(f"2. ollama search {tag}")
        print(f"3. ollama pull {tag}:{rec.lower()}")
        print(f"4. ollama run {tag}:{rec.lower()}")
        print("\n💡 For multi‑GPU, consider vLLM or TGI with tensor parallel.")
    else:
        print(f"\n📦 To run {mi.name} on CPU:")
        print("1. Install llama.cpp: https://github.com/ggerganov/llama.cpp")
        print("2. Download a GGUF quant (Q3/Q2 for limited RAM)")
        print("3. ./main -m path/to/model.gguf -n 256 -t <threads>")
    print("\n------------------------------------------------------------")
    print("Alternative tools: LM Studio (GUI), Text Generation WebUI.")
    print("Hugging Face search: https://huggingface.co/models?search=" + quote(mi.name + " GGUF"))


# ---------------------------
# Main flow
# ---------------------------

def main() -> None:
    print_banner()
    det = HardwareDetector()
    hw = det.detect()
    resp = input("\nProceed with this hardware configuration? (y/n): ").strip().lower()
    if resp != 'y':
        print("\n💡 Tips: pip install gputil py-cpuinfo  • Ensure 'nvidia-smi' works  • Run in a terminal, not Python REPL")
        return

    print("\n============================================================")
    print("STEP 2: MODEL SELECTION")
    print("============================================================\n")
    print("Enter the name of the model you want to run. Examples:")
    print("  - Llama 3 8B")
    print("  - Mistral 7B")
    print("  - Mixtral 8x7B")
    print("  - Phi-3")
    print("  - Qwen 7B")
    print("  - DeepSeek Coder 6.7B")
    print("\nType 'exit' to quit.\n")

    search = WebSearcher()
    while True:
        raw = input("Model name (or 'exit' to quit): ")
        if not raw:
            continue
        user = sanitize_user_input(raw)
        if not user:
            continue
        if user.lower() in {'exit', 'quit', 'q'}:
            print("\n✨ Thank you for using the LLM Hardware Compatibility Checker!")
            break

        print("\n============================================================")
        print("STEP 3: SEARCHING MODEL INFORMATION")
        print("============================================================")
        mi = search.search_model_info(user)
        if not mi:
            print(f"\n❌ Could not find information for '{user}'. Try a simpler name.")
            continue

        print("\n============================================================")
        print("STEP 4: COMPATIBILITY ANALYSIS")
        print("============================================================")
        res = analyze_compatibility(hw, mi)
        display_model_report(mi, res)

        compat = {k: v for k, v in res.items() if v["compatible"]}
        if compat:
            rsp = input("\n\nWould you like setup instructions? (y/n): ").strip().lower()
            if rsp == 'y':
                setup_instructions(hw, mi, compat)

        again = input("\n\nCheck another model? (y/n): ").strip().lower()
        if again != 'y':
            print("\n✨ Thank you for using the LLM Hardware Compatibility Checker!")
            break


if __name__ == "__main__":
    main()
