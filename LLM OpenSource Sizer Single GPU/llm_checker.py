#!/usr/bin/env python3
"""
LLM Hardware Compatibility Checker - Web Search Version
This script automatically detects hardware and searches the web for model requirements.
"""

import json
import re
import platform
import psutil
import subprocess
import sys
import requests
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote

# Check for required packages
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Note: GPUtil not installed. GPU detection will be limited.")
    print("Install with: pip install gputil")

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False
    print("Note: py-cpuinfo not installed. CPU detection will be limited.")
    print("Install with: pip install py-cpuinfo")

class QuantizationType(Enum):
    """Supported quantization types for LLMs"""
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"
    Q8_0 = "q8_0"
    Q6_K = "q6_k"
    Q5_K_M = "q5_k_m"
    Q5_K_S = "q5_k_s"
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q3_K_M = "q3_k_m"
    Q3_K_S = "q3_k_s"
    Q3_K_L = "q3_k_l"
    Q2_K = "q2_k"
    IQ3_XXS = "iq3_xxs"
    IQ2_XXS = "iq2_xxs"
    IQ2_XS = "iq2_xs"
    IQ2_S = "iq2_s"
    IQ1_S = "iq1_s"

@dataclass
class HardwareSpecs:
    """Hardware specifications"""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    gpu_model: str
    gpu_vram_gb: float
    system_ram_gb: float
    storage_type: str  # SSD or HDD
    available_storage_gb: float
    os_type: str

@dataclass
class ModelInfo:
    """Model information gathered from web search"""
    name: str
    parameter_count: str  # e.g., "7B", "13B", "70B"
    base_size_gb: float
    quantization_info: Dict[str, Dict]  # quant_name -> {size_gb, quality, description}
    source_urls: List[str]
    huggingface_id: Optional[str] = None
    description: Optional[str] = None

class WebSearcher:
    """Handles web searches for model information"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Search for model information using multiple strategies"""
        print(f"\n🔍 Searching for information about {model_name}...")
        
        # Try different search strategies
        model_info = None
        
        # Strategy 1: Search HuggingFace
        model_info = self._search_huggingface(model_name)
        
        # Strategy 2: Search for GGUF quantization info
        if model_info:
            quant_info = self._search_quantization_info(model_name)
            if quant_info:
                model_info.quantization_info.update(quant_info)
        
        # Strategy 3: If no info found, create estimates
        if not model_info:
            model_info = self._estimate_model_info(model_name)
        
        return model_info
    
    def _search_huggingface(self, model_name: str) -> Optional[ModelInfo]:
        """Search HuggingFace for model information"""
        try:
            # Common model name patterns on HuggingFace
            search_patterns = [
                model_name,
                model_name.replace(" ", "-"),
                model_name.lower().replace(" ", "-"),
                f"meta-llama/{model_name.lower().replace(' ', '-')}",
                f"mistralai/{model_name.lower().replace(' ', '-')}",
                f"microsoft/{model_name.lower().replace(' ', '-')}",
                f"google/{model_name.lower().replace(' ', '-')}"
            ]
            
            for pattern in search_patterns:
                url = f"https://huggingface.co/api/models?search={quote(pattern)}"
                
                try:
                    response = self.session.get(url, timeout=5)
                    if response.status_code == 200:
                        models = response.json()
                        if models:
                            # Extract info from the first matching model
                            model = models[0]
                            return self._parse_huggingface_model(model, model_name)
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error searching HuggingFace: {e}")
            return None
    
    def _parse_huggingface_model(self, hf_model: Dict, original_name: str) -> ModelInfo:
        """Parse HuggingFace model data"""
        model_id = hf_model.get('modelId', '')
        
        # Extract parameter count from various sources
        param_count = self._extract_param_count(model_id, hf_model.get('tags', []))
        
        # Estimate base size
        base_size_gb = self._estimate_size_from_params(param_count)
        
        return ModelInfo(
            name=original_name,
            parameter_count=param_count,
            base_size_gb=base_size_gb,
            quantization_info=self._get_standard_quantizations(base_size_gb),
            source_urls=[f"https://huggingface.co/{model_id}"],
            huggingface_id=model_id,
            description=hf_model.get('description', '')
        )
    
    def _search_quantization_info(self, model_name: str) -> Dict[str, Dict]:
        """Search for specific GGUF quantization information"""
        print("  📊 Searching for quantization details...")
        
        # Search for GGUF quantized versions
        search_terms = [
            f"{model_name} GGUF quantization sizes",
            f"{model_name} Q4_K_M Q5_K_M size",
            f"TheBloke {model_name} GGUF"
        ]
        
        quant_info = {}
        
        # This is where you would implement actual web scraping
        # For now, returning standard quantization ratios
        return quant_info
    
    def _extract_param_count(self, model_id: str, tags: List[str]) -> str:
        """Extract parameter count from model ID or tags"""
        # Check model ID for parameter count
        param_patterns = [
            r'(\d+\.?\d*)[bB]',  # 7B, 13B, 1.5B, etc.
            r'(\d+\.?\d*)b-',    # 7b-, 13b-
            r'-(\d+\.?\d*)[bB]',  # -7B, -13B
        ]
        
        for pattern in param_patterns:
            match = re.search(pattern, model_id)
            if match:
                return f"{match.group(1)}B"
        
        # Check tags
        for tag in tags:
            match = re.search(r'(\d+\.?\d*)[bB]', tag)
            if match:
                return f"{match.group(1)}B"
        
        # Default to 7B if not found
        return "7B"
    
    def _estimate_size_from_params(self, param_count: str) -> float:
        """Estimate model size from parameter count"""
        # Extract numeric value
        match = re.search(r'(\d+\.?\d*)', param_count)
        if match:
            params_b = float(match.group(1))
            # Rough estimate: 2GB per billion parameters for fp16
            return params_b * 2
        return 14.0  # Default to 7B model size
    
    def _get_standard_quantizations(self, base_size_gb: float) -> Dict[str, Dict]:
        """Get standard quantization ratios based on research"""
        # Based on typical GGUF quantization ratios
        return {
            "FP16": {
                "size_gb": base_size_gb,
                "quality": "★★★★★",
                "description": "Full 16-bit precision, best quality"
            },
            "Q8_0": {
                "size_gb": base_size_gb * 0.53,
                "quality": "★★★★★",
                "description": "8-bit quantization, minimal quality loss"
            },
            "Q6_K": {
                "size_gb": base_size_gb * 0.41,
                "quality": "★★★★☆",
                "description": "6-bit quantization, excellent quality/size ratio"
            },
            "Q5_K_M": {
                "size_gb": base_size_gb * 0.36,
                "quality": "★★★★☆",
                "description": "5-bit medium quantization, very good quality"
            },
            "Q5_K_S": {
                "size_gb": base_size_gb * 0.34,
                "quality": "★★★☆☆",
                "description": "5-bit small quantization, good quality"
            },
            "Q4_K_M": {
                "size_gb": base_size_gb * 0.30,
                "quality": "★★★☆☆",
                "description": "4-bit medium quantization, good balance"
            },
            "Q4_K_S": {
                "size_gb": base_size_gb * 0.28,
                "quality": "★★★☆☆",
                "description": "4-bit small quantization, good for most uses"
            },
            "Q4_0": {
                "size_gb": base_size_gb * 0.27,
                "quality": "★★★☆☆",
                "description": "Legacy 4-bit quantization"
            },
            "Q3_K_M": {
                "size_gb": base_size_gb * 0.24,
                "quality": "★★☆☆☆",
                "description": "3-bit medium quantization, acceptable quality"
            },
            "Q3_K_S": {
                "size_gb": base_size_gb * 0.22,
                "quality": "★★☆☆☆",
                "description": "3-bit small quantization, lower quality"
            },
            "Q2_K": {
                "size_gb": base_size_gb * 0.19,
                "quality": "★☆☆☆☆",
                "description": "2-bit quantization, significant quality loss"
            },
            "IQ3_XXS": {
                "size_gb": base_size_gb * 0.21,
                "quality": "★★☆☆☆",
                "description": "Improved 3-bit quantization"
            },
            "IQ2_XXS": {
                "size_gb": base_size_gb * 0.16,
                "quality": "★☆☆☆☆",
                "description": "Improved 2-bit quantization, very small"
            }
        }
    
    def _estimate_model_info(self, model_name: str) -> ModelInfo:
        """Create estimated model info when search fails"""
        print("  ⚠️  Using estimated values based on model name...")
        
        # Try to extract parameter count from name
        param_match = re.search(r'(\d+\.?\d*)\s*[bB]', model_name)
        if param_match:
            param_count = f"{param_match.group(1)}B"
            params_b = float(param_match.group(1))
        else:
            param_count = "7B"
            params_b = 7.0
        
        base_size_gb = params_b * 2
        
        return ModelInfo(
            name=model_name,
            parameter_count=param_count,
            base_size_gb=base_size_gb,
            quantization_info=self._get_standard_quantizations(base_size_gb),
            source_urls=["Estimated based on parameter count"],
            description="Model information estimated from name"
        )

class LLMHardwareChecker:
    def __init__(self):
        self.hardware_specs = None
        self.web_searcher = WebSearcher()
        
    def detect_hardware(self) -> HardwareSpecs:
        """Automatically detect hardware specifications"""
        print("\n🔍 Detecting hardware specifications...")
        
        # CPU Detection
        cpu_model = self._detect_cpu()
        cpu_cores = psutil.cpu_count(logical=False) or 4
        cpu_threads = psutil.cpu_count(logical=True) or 8
        
        # Memory Detection
        mem = psutil.virtual_memory()
        system_ram_gb = round(mem.total / (1024**3), 1)
        
        # GPU Detection
        gpu_model, gpu_vram_gb = self._detect_gpu()
        
        # Storage Detection
        storage_type, available_storage_gb = self._detect_storage()
        
        # OS Detection
        os_type = platform.system()
        
        specs = HardwareSpecs(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            gpu_model=gpu_model,
            gpu_vram_gb=gpu_vram_gb,
            system_ram_gb=system_ram_gb,
            storage_type=storage_type,
            available_storage_gb=available_storage_gb,
            os_type=os_type
        )
        
        return specs
    
    def _detect_cpu(self) -> str:
        """Detect CPU model"""
        try:
            if CPUINFO_AVAILABLE:
                info = cpuinfo.get_cpu_info()
                return info.get('brand_raw', 'Unknown CPU')
            elif platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Darwin":  # macOS
                cmd = "sysctl -n machdep.cpu.brand_string"
                return subprocess.check_output(cmd, shell=True).decode().strip()
            elif platform.system() == "Linux":
                cmd = "cat /proc/cpuinfo | grep 'model name' | head -1"
                output = subprocess.check_output(cmd, shell=True).decode()
                return output.split(':')[1].strip()
            else:
                return platform.processor() or "Unknown CPU"
        except:
            return platform.processor() or "Unknown CPU"
    
    def _detect_gpu(self) -> Tuple[str, float]:
        """Detect GPU model and VRAM"""
        try:
            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    return gpu.name, round(gpu.memoryTotal / 1024, 1)
            
            # Try nvidia-smi for NVIDIA GPUs
            if platform.system() in ["Linux", "Windows"]:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                           '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        if output:
                            parts = output.split(', ')
                            if len(parts) >= 2:
                                name = parts[0]
                                vram_mb = float(parts[1])
                                return name, round(vram_mb / 1024, 1)
                except:
                    pass
            
            # Check for Apple Silicon
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    # Detect Apple Silicon model
                    cmd = "sysctl -n machdep.cpu.brand_string"
                    cpu_info = subprocess.check_output(cmd, shell=True).decode().strip()
                    if "M1" in cpu_info:
                        return "Apple M1 GPU", 8.0  # Base M1 has 8GB unified
                    elif "M2" in cpu_info:
                        return "Apple M2 GPU", 8.0  # Base M2 has 8GB unified
                    elif "M3" in cpu_info:
                        return "Apple M3 GPU", 8.0  # Base M3 has 8GB unified
                    # Note: This is simplified - actual unified memory varies by model
                except:
                    pass
            
            return "None", 0.0
            
        except Exception as e:
            return "None", 0.0
    
    def _detect_storage(self) -> Tuple[str, float]:
        """Detect storage type and available space"""
        try:
            # Get disk usage for the current directory
            usage = psutil.disk_usage('/')
            available_gb = round(usage.free / (1024**3), 1)
            
            # Try to detect if SSD or HDD
            storage_type = "SSD"  # Default assumption
            
            if platform.system() == "Windows":
                # On Windows, check if system drive is SSD
                try:
                    result = subprocess.run(['wmic', 'diskdrive', 'get', 'MediaType'], 
                                          capture_output=True, text=True)
                    if "Fixed hard disk" in result.stdout:
                        storage_type = "HDD"
                except:
                    pass
            elif platform.system() == "Linux":
                # Check if root device is rotational
                try:
                    with open('/sys/block/sda/queue/rotational', 'r') as f:
                        if f.read().strip() == '1':
                            storage_type = "HDD"
                except:
                    pass
            elif platform.system() == "Darwin":
                # macOS - most modern Macs have SSDs
                storage_type = "SSD"
            
            return storage_type, available_gb
            
        except:
            return "Unknown", 100.0  # Default fallback
    
    def display_detected_hardware(self, specs: HardwareSpecs):
        """Display the detected hardware specifications"""
        print("\n" + "="*60)
        print("DETECTED HARDWARE SPECIFICATIONS")
        print("="*60)
        
        print(f"\n💻 System: {specs.os_type}")
        print(f"🧠 CPU: {specs.cpu_model}")
        print(f"   - Cores: {specs.cpu_cores} physical, {specs.cpu_threads} logical threads")
        
        if specs.gpu_vram_gb > 0:
            print(f"🎮 GPU: {specs.gpu_model}")
            print(f"   - VRAM: {specs.gpu_vram_gb} GB")
        else:
            print("🎮 GPU: Not detected (CPU-only mode)")
        
        print(f"🧩 RAM: {specs.system_ram_gb} GB")
        print(f"💾 Storage: {specs.storage_type} with {specs.available_storage_gb} GB available")
    
    def get_model_choice(self) -> Optional[str]:
        """Let user enter the model they want to check"""
        print("\n" + "="*60)
        print("MODEL SELECTION")
        print("="*60)
        
        print("\nEnter the name of the model you want to run.")
        print("Examples:")
        print("  - Llama 2 7B")
        print("  - Mistral 7B")
        print("  - Mixtral 8x7B")
        print("  - Phi-2")
        print("  - Qwen 7B")
        print("  - DeepSeek Coder 6.7B")
        print("  - Or any other model name...")
        
        model_name = input("\nModel name (or 'exit' to quit): ").strip()
        
        if model_name.lower() == 'exit':
            return None
        
        return model_name
    
    def check_compatibility(self, hardware: HardwareSpecs, model_info: ModelInfo) -> Dict[str, Dict]:
        """Check which quantizations are compatible"""
        compatibility = {}
        
        for quant_name, quant_data in model_info.quantization_info.items():
            size_gb = quant_data['size_gb']
            
            result = {
                'compatible': True,
                'warnings': [],
                'errors': [],
                'performance': 'Good',
                'size_gb': size_gb,
                'quality': quant_data['quality'],
                'description': quant_data['description']
            }
            
            # Check storage
            if size_gb > hardware.available_storage_gb:
                result['compatible'] = False
                result['errors'].append(f"Requires {size_gb:.1f}GB storage, only {hardware.available_storage_gb}GB available")
            
            # Estimate RAM requirements
            # Rule of thumb: need RAM = model size + overhead
            ram_required = size_gb + 2  # 2GB overhead for system
            vram_required = size_gb + 0.5  # 0.5GB overhead for GPU
            
            # Check GPU mode
            if hardware.gpu_vram_gb > 0:
                if vram_required > hardware.gpu_vram_gb:
                    # Can't fit in VRAM, check CPU fallback
                    cpu_ram_required = ram_required * 1.5
                    if cpu_ram_required > hardware.system_ram_gb:
                        result['compatible'] = False
                        result['errors'].append(f"Requires {vram_required:.1f}GB VRAM (have {hardware.gpu_vram_gb}GB) or {cpu_ram_required:.1f}GB RAM for CPU mode")
                    else:
                        result['warnings'].append(f"Will run in CPU mode (insufficient VRAM)")
                        result['performance'] = 'Slow'
                else:
                    # Determine performance based on quantization
                    if 'Q4' in quant_name or 'Q3' in quant_name:
                        result['performance'] = 'Fast'
                    elif 'Q2' in quant_name or 'IQ' in quant_name:
                        result['performance'] = 'Very Fast'
            else:
                # CPU-only mode
                cpu_ram_required = ram_required * 1.5
                if cpu_ram_required > hardware.system_ram_gb:
                    result['compatible'] = False
                    result['errors'].append(f"Requires {cpu_ram_required:.1f}GB RAM for CPU mode, only {hardware.system_ram_gb}GB available")
                else:
                    result['warnings'].append("Running in CPU-only mode (no GPU detected)")
                    result['performance'] = 'Slow'
            
            # Storage type warning
            if hardware.storage_type == 'HDD' and result['compatible']:
                result['warnings'].append("HDD detected - model loading will be slow")
            
            compatibility[quant_name] = result
        
        return compatibility
    
    def display_compatibility_report(self, model_info: ModelInfo, compatibility: Dict[str, Dict]):
        """Display detailed compatibility report"""
        print("\n" + "="*60)
        print(f"COMPATIBILITY REPORT FOR {model_info.name.upper()}")
        print("="*60)
        
        if model_info.parameter_count:
            print(f"\n📊 Model Size: {model_info.parameter_count} parameters")
        print(f"💾 Base Size: {model_info.base_size_gb:.1f}GB (FP16)")
        
        if model_info.huggingface_id:
            print(f"🤗 HuggingFace: {model_info.huggingface_id}")
        
        compatible_quants = {k: v for k, v in compatibility.items() if v['compatible']}
        
        if not compatible_quants:
            print("\n❌ This model CANNOT run on your hardware in any configuration.")
            print("\nReasons:")
            for quant, result in compatibility.items():
                if result['errors']:
                    print(f"\n{quant}:")
                    for error in result['errors']:
                        print(f"  • {error}")
            return
        
        print(f"\n✅ This model CAN run on your hardware!")
        print(f"\nCompatible quantizations: {len(compatible_quants)} out of {len(compatibility)}")
        
        # Group by performance
        fast_options = {k: v for k, v in compatible_quants.items() if v['performance'] in ['Fast', 'Very Fast']}
        good_options = {k: v for k, v in compatible_quants.items() if v['performance'] == 'Good'}
        slow_options = {k: v for k, v in compatible_quants.items() if v['performance'] == 'Slow'}
        
        # Display options by performance category
        if fast_options:
            print("\n" + "-"*40)
            print("⚡ FAST OPTIONS (Recommended for speed):")
            print("-"*40)
            self._display_options(fast_options)
        
        if good_options:
            print("\n" + "-"*40)
            print("✅ GOOD OPTIONS (Balanced quality/speed):")
            print("-"*40)
            self._display_options(good_options)
        
        if slow_options:
            print("\n" + "-"*40)
            print("🐌 SLOW OPTIONS (CPU mode):")
            print("-"*40)
            self._display_options(slow_options)
        
        # Show incompatible options
        incompatible_quants = {k: v for k, v in compatibility.items() if not v['compatible']}
        
        if incompatible_quants:
            print("\n" + "-"*40)
            print("❌ INCOMPATIBLE OPTIONS:")
            print("-"*40)
            
            for quant, result in incompatible_quants.items():
                print(f"\n{quant}: {result['size_gb']:.1f}GB")
                for error in result['errors']:
                    print(f"  - {error}")
        
        # Recommendations
        self._display_recommendations(compatible_quants, self.hardware_specs)
    
    def _display_options(self, options: Dict[str, Dict]):
        """Display quantization options"""
        for quant, result in sorted(options.items(), key=lambda x: x[1]['size_gb'], reverse=True):
            print(f"\n{quant}:")
            print(f"  Size: {result['size_gb']:.1f}GB")
            print(f"  Quality: {result['quality']}")
            print(f"  {result['description']}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"  ⚠️  {warning}")
    
    def _display_recommendations(self, compatible_quants: Dict[str, Dict], hardware: HardwareSpecs):
        """Display recommendations based on hardware and options"""
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Find best options
        if compatible_quants:
            # Best quality (largest compatible)
            best_quality = max(compatible_quants.items(), key=lambda x: x[1]['size_gb'])
            
            # Best performance (fastest with good quality)
            fast_good_quality = [k for k, v in compatible_quants.items() 
                               if v['performance'] in ['Fast', 'Very Fast'] and '★★★' in v['quality']]
            
            if fast_good_quality:
                best_performance = fast_good_quality[0]
            else:
                # Fallback to smallest size
                best_performance = min(compatible_quants.items(), key=lambda x: x[1]['size_gb'])[0]
            
            print(f"\n🏆 Best Quality: {best_quality[0]}")
            print(f"   - {best_quality[1]['quality']}")
            print(f"   - Size: {best_quality[1]['size_gb']:.1f}GB")
            
            print(f"\n⚡ Best Performance: {best_performance}")
            print(f"   - {compatible_quants[best_performance]['quality']}")
            print(f"   - Size: {compatible_quants[best_performance]['size_gb']:.1f}GB")
            
            # Popular choice
            if 'Q4_K_M' in compatible_quants:
                print(f"\n🌟 Popular Choice: Q4_K_M")
                print(f"   - Good balance of quality and performance")
                print(f"   - Widely supported and tested")
            
            if hardware.gpu_vram_gb == 0:
                print("\n💡 To improve performance:")
                print("   - Add a GPU with at least 8GB VRAM")
                print("   - Use quantized models (Q4_K_M or lower)")
                print("   - Ensure adequate cooling for sustained CPU usage")
                print("   - Close other applications to free RAM")
    
    def provide_setup_instructions(self, hardware: HardwareSpecs, model_info: ModelInfo, compatible_quants: Dict[str, Dict]):
        """Provide setup instructions for running the model"""
        print("\n" + "="*60)
        print("SETUP INSTRUCTIONS")
        print("="*60)
        
        if not compatible_quants:
            print("\n❌ No compatible configurations available.")
            return
        
        print(f"\n📦 To run {model_info.name}:\n")
        
        # Determine best quantization for instructions
        recommended_quant = None
        if 'Q4_K_M' in compatible_quants:
            recommended_quant = 'Q4_K_M'
        elif 'Q5_K_M' in compatible_quants:
            recommended_quant = 'Q5_K_M'
        else:
            # Get the one with best quality/performance balance
            recommended_quant = list(compatible_quants.keys())[0]
        
        # Determine recommended runtime based on hardware
        if hardware.gpu_vram_gb > 0 and 'NVIDIA' in hardware.gpu_model:
            print("🚀 Recommended: Ollama or LM Studio (GPU acceleration supported)\n")
            print("Option 1 - Ollama (Recommended):")
            print("1. Install Ollama:")
            if hardware.os_type == "Windows":
                print("   Download from: https://ollama.ai/download/windows")
            elif hardware.os_type == "Darwin":
                print("   brew install ollama")
            else:
                print("   curl -fsSL https://ollama.ai/install.sh | sh")
            
            model_tag = model_info.name.lower().replace(' ', '-')
            print(f"\n2. Search for the model on Ollama:")
            print(f"   ollama search {model_tag}")
            print(f"\n3. Pull the model (with quantization):")
            print(f"   ollama pull {model_tag}:{recommended_quant.lower()}")
            print(f"\n4. Run the model:")
            print(f"   ollama run {model_tag}:{recommended_quant.lower()}")
            
        elif hardware.os_type == "Darwin" and "Apple" in hardware.gpu_model:
            print("🚀 Recommended: Ollama or LM Studio (Apple Silicon optimized)\n")
            print("Option 1 - Ollama (Recommended for Apple Silicon):")
            print("1. Install Ollama:")
            print("   brew install ollama")
            
            model_tag = model_info.name.lower().replace(' ', '-')
            print(f"\n2. Pull the model:")
            print(f"   ollama pull {model_tag}")
            print(f"\n3. Run the model:")
            print(f"   ollama run {model_tag}")
            
        else:
            print("🖥️  Recommended: llama.cpp (CPU optimized)\n")
            print("1. Install llama.cpp:")
            print("   git clone https://github.com/ggerganov/llama.cpp")
            print("   cd llama.cpp")
            if hardware.os_type == "Windows":
                print("   cmake . && cmake --build . --config Release")
            else:
                print("   make -j")
            
            print(f"\n2. Download GGUF model:")
            print(f"   Visit https://huggingface.co/models?search={quote(model_info.name)}+GGUF")
            print(f"   Look for 'TheBloke' or other GGUF conversions")
            print(f"   Download the {recommended_quant} version")
            
            print(f"\n3. Run the model:")
            print(f"   ./main -m path/to/model.gguf -n 512 -p \"Your prompt here\"")
            print(f"   -t {hardware.cpu_threads}  # Use all CPU threads")
        
        print("\n" + "-"*40)
        print("Alternative Options:")
        print("-"*40)
        
        print("\n📱 LM Studio (User-Friendly GUI):")
        print("   - Download: https://lmstudio.ai")
        print("   - Supports all quantization formats")
        print("   - Built-in model browser")
        print("   - Easy model switching")
        
        print("\n🌐 Text Generation WebUI:")
        print("   - GitHub: oobabooga/text-generation-webui")
        print("   - Web interface for various models")
        print("   - Supports GGUF, GPTQ, AWQ formats")
        
        print("\n🐍 Python with llama-cpp-python:")
        print("   - pip install llama-cpp-python")
        print("   - Python bindings for llama.cpp")
        print("   - Easy integration into Python projects")
        
        # Model download sources
        print("\n" + "="*60)
        print("MODEL DOWNLOAD SOURCES")
        print("="*60)
        
        print(f"\n🤗 HuggingFace (Primary source):")
        if model_info.huggingface_id:
            print(f"   Original: https://huggingface.co/{model_info.huggingface_id}")
        print(f"   GGUF Search: https://huggingface.co/models?search={quote(model_info.name)}+GGUF")
        print(f"   TheBloke: https://huggingface.co/TheBloke")
        
        print("\n📊 Recommended GGUF Quantizations:")
        for quant in ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0']:
            if quant in compatible_quants:
                info = compatible_quants[quant]
                print(f"   • {quant}: {info['size_gb']:.1f}GB - {info['description']}")
        
        # Performance optimization tips
        print("\n" + "="*60)
        print("OPTIMIZATION TIPS")
        print("="*60)
        
        print("\n⚡ For better performance:")
        
        if hardware.storage_type == "HDD":
            print("• ⚠️  Move model files to an SSD if possible (10x faster loading)")
        
        if hardware.gpu_vram_gb > 0:
            print("• Ensure GPU drivers are up to date")
            print("• Use GPU-optimized quantization formats (Q4_K_M, Q5_K_M)")
            print("• Enable GPU offloading in llama.cpp with -ngl flag")
        else:
            print("• Use CPU-optimized builds with AVX2/AVX512 support")
            print("• Close other applications to free up RAM")
            print("• Consider using smaller context sizes (2048 tokens)")
            print("• Use mmap for faster model loading")
        
        print("• Start with smaller context window (2048-4096 tokens)")
        print("• Use lower temperature (0.7) for more consistent output")
        print("• Enable mlock to prevent model swapping to disk")
        
        if hardware.system_ram_gb < 16:
            print("\n💡 RAM Optimization:")
            print("• Use lower quantization (Q3_K_M or Q2_K if needed)")
            print("• Reduce context size to 1024 tokens")
            print("• Close all unnecessary applications")
            print("• Consider upgrading to 16GB+ RAM")

def print_welcome_banner():
    """Print welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        🤖 LLM Hardware Compatibility Checker v3.0 🤖      ║
║                   with Web Search Support                 ║
╚══════════════════════════════════════════════════════════╝
    """)
    print("This tool will:")
    print("  ✓ Automatically detect your hardware")
    print("  ✓ Search the web for model requirements")
    print("  ✓ Check compatibility with various quantizations")
    print("  ✓ Provide setup instructions")
    print("\nNote: Internet connection required for model information.")

def main():
    """Main function"""
    print_welcome_banner()
    
    checker = LLMHardwareChecker()
    
    try:
        # Step 1: Detect hardware automatically
        print("\n" + "="*60)
        print("STEP 1: HARDWARE DETECTION")
        print("="*60)
        
        hardware = checker.detect_hardware()
        checker.hardware_specs = hardware
        
        # Step 2: Display detected hardware
        checker.display_detected_hardware(hardware)
        
        # Ask if user wants to continue
        response = input("\nProceed with this hardware configuration? (y/n): ").strip().lower()
        if response != 'y':
            print("\n💡 Tips for better hardware detection:")
            print("  • Install: pip install gputil py-cpuinfo")
            print("  • Run as administrator/root for full access")
            print("  • Check your system information manually")
            return
        
        while True:
            # Step 3: Let user enter a model name
            print("\n" + "="*60)
            print("STEP 2: MODEL SELECTION")
            print("="*60)
            
            model_name = checker.get_model_choice()
            
            if not model_name:
                print("\nThank you for using the LLM Hardware Compatibility Checker!")
                break
            
            # Step 4: Search for model information
            print("\n" + "="*60)
            print("STEP 3: SEARCHING MODEL INFORMATION")
            print("="*60)
            
            model_info = checker.web_searcher.search_model_info(model_name)
            
            if not model_info:
                print(f"\n❌ Could not find information for '{model_name}'")
                print("Please check the model name and try again.")
                continue
            
            # Step 5: Check compatibility
            print("\n" + "="*60)
            print("STEP 4: COMPATIBILITY ANALYSIS")
            print("="*60)
            
            compatibility = checker.check_compatibility(hardware, model_info)
            
            # Step 6: Display compatibility report
            checker.display_compatibility_report(model_info, compatibility)
            
            # Step 7: Provide setup instructions if compatible
            compatible_quants = {k: v for k, v in compatibility.items() if v['compatible']}
            if compatible_quants:
                response = input("\n\nWould you like setup instructions? (y/n): ").strip().lower()
                if response == 'y':
                    checker.provide_setup_instructions(hardware, model_info, compatible_quants)
            
            # Ask if user wants to check another model
            response = input("\n\nCheck another model? (y/n): ").strip().lower()
            if response != 'y':
                print("\n✨ Thank you for using the LLM Hardware Compatibility Checker!")
                break
        
    except KeyboardInterrupt:
        print("\n\nExiting... Thank you for using the LLM Hardware Compatibility Checker!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("  • Check your internet connection")
        print("  • Ensure required packages are installed:")
        print("    pip install psutil requests")
        print("    pip install gputil py-cpuinfo  # Optional")

if __name__ == "__main__":
    main()