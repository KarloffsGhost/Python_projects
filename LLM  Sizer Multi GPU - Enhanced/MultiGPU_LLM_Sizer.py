#!/usr/bin/env python3
"""
LLM Hardware Compatibility Checker - Web Search Version with Multi-GPU Support
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
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote
import traceback

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
class GPUInfo:
    """Individual GPU information"""
    id: int
    name: str
    vram_gb: float
    vram_used_gb: float
    vram_free_gb: float
    temperature: Optional[float] = None
    utilization: Optional[float] = None

@dataclass
class HardwareSpecs:
    """Hardware specifications with multi-GPU support"""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    gpus: List[GPUInfo] = field(default_factory=list)  # List of GPUs
    total_gpu_vram_gb: float = 0.0  # Combined VRAM of all GPUs
    system_ram_gb: float = 0.0
    storage_type: str = "Unknown"  # SSD or HDD
    available_storage_gb: float = 0.0
    os_type: str = "Unknown"
    
    @property
    def gpu_count(self) -> int:
        """Number of GPUs detected"""
        return len(self.gpus)
    
    @property
    def has_gpu(self) -> bool:
        """Whether any GPU is available"""
        return self.gpu_count > 0
    
    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        """Get the primary (first) GPU if available"""
        return self.gpus[0] if self.gpus else None
    
    @property
    def gpu_model(self) -> str:
        """Legacy property for backward compatibility"""
        if self.primary_gpu:
            if self.gpu_count > 1:
                return f"{self.primary_gpu.name} (+{self.gpu_count-1} more)"
            return self.primary_gpu.name
        return "None"
    
    @property
    def gpu_vram_gb(self) -> float:
        """Legacy property for backward compatibility - returns primary GPU VRAM"""
        return self.primary_gpu.vram_gb if self.primary_gpu else 0.0

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
        """Automatically detect hardware specifications with multi-GPU support"""
        print("\n🔍 Detecting hardware specifications...")
        
        # CPU Detection
        cpu_model = self._detect_cpu()
        cpu_cores = psutil.cpu_count(logical=False) or 4
        cpu_threads = psutil.cpu_count(logical=True) or 8
        
        # Memory Detection
        mem = psutil.virtual_memory()
        system_ram_gb = round(mem.total / (1024**3), 1)
        
        # GPU Detection (now supports multiple GPUs)
        gpus = self._detect_all_gpus()
        total_vram = sum(gpu.vram_gb for gpu in gpus)
        
        # Storage Detection
        storage_type, available_storage_gb = self._detect_storage()
        
        # OS Detection
        os_type = platform.system()
        
        specs = HardwareSpecs(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            gpus=gpus,
            total_gpu_vram_gb=total_vram,
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
    
    def _detect_all_gpus(self) -> List[GPUInfo]:
        """Detect all GPUs in the system"""
        gpus = []
        
        try:
            # Try GPUtil first
            if GPU_AVAILABLE:
                gpu_list = GPUtil.getGPUs()
                for i, gpu in enumerate(gpu_list):
                    gpu_info = GPUInfo(
                        id=i,
                        name=gpu.name,
                        vram_gb=round(gpu.memoryTotal / 1024, 1),
                        vram_used_gb=round(gpu.memoryUsed / 1024, 1),
                        vram_free_gb=round(gpu.memoryFree / 1024, 1),
                        temperature=gpu.temperature,
                        utilization=gpu.load * 100 if gpu.load else None
                    )
                    gpus.append(gpu_info)
                
                if gpus:
                    return gpus
            
            # Try nvidia-smi for NVIDIA GPUs
            if platform.system() in ["Linux", "Windows"]:
                try:
                    # Get detailed GPU info
                    result = subprocess.run([
                        'nvidia-smi', 
                        '--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if line:
                                parts = [p.strip() for p in line.split(',')]
                                if len(parts) >= 5:
                                    gpu_info = GPUInfo(
                                        id=int(parts[0]),
                                        name=parts[1],
                                        vram_gb=round(float(parts[2]) / 1024, 1),
                                        vram_used_gb=round(float(parts[3]) / 1024, 1),
                                        vram_free_gb=round(float(parts[4]) / 1024, 1),
                                        temperature=float(parts[5]) if len(parts) > 5 and parts[5] else None,
                                        utilization=float(parts[6]) if len(parts) > 6 and parts[6] else None
                                    )
                                    gpus.append(gpu_info)
                        
                        if gpus:
                            return gpus
                except Exception as e:
                    print(f"Debug: nvidia-smi error: {e}")
                    pass
            
            # Check for Apple Silicon
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    # Detect Apple Silicon model
                    cmd = "sysctl -n machdep.cpu.brand_string"
                    cpu_info = subprocess.check_output(cmd, shell=True).decode().strip()
                    
                    # Get total memory (unified memory on Apple Silicon)
                    cmd_mem = "sysctl -n hw.memsize"
                    total_mem_bytes = int(subprocess.check_output(cmd_mem, shell=True).decode().strip())
                    total_mem_gb = round(total_mem_bytes / (1024**3), 1)
                    
                    # Estimate GPU memory allocation (Apple Silicon uses unified memory)
                    # Typically, up to 75% can be used for GPU on higher-end models
                    if "M1" in cpu_info:
                        if "Max" in cpu_info:
                            gpu_mem = min(32.0, total_mem_gb * 0.75)  # M1 Max
                        elif "Pro" in cpu_info:
                            gpu_mem = min(16.0, total_mem_gb * 0.66)  # M1 Pro
                        else:
                            gpu_mem = min(8.0, total_mem_gb * 0.5)   # M1 base
                        gpu_name = f"Apple {cpu_info.split()[1]} GPU"
                    elif "M2" in cpu_info:
                        if "Max" in cpu_info:
                            gpu_mem = min(48.0, total_mem_gb * 0.75)  # M2 Max
                        elif "Pro" in cpu_info:
                            gpu_mem = min(19.0, total_mem_gb * 0.66)  # M2 Pro
                        else:
                            gpu_mem = min(10.0, total_mem_gb * 0.5)   # M2 base
                        gpu_name = f"Apple {cpu_info.split()[1]} GPU"
                    elif "M3" in cpu_info:
                        if "Max" in cpu_info:
                            gpu_mem = min(64.0, total_mem_gb * 0.75)  # M3 Max
                        elif "Pro" in cpu_info:
                            gpu_mem = min(18.0, total_mem_gb * 0.66)  # M3 Pro
                        else:
                            gpu_mem = min(10.0, total_mem_gb * 0.5)   # M3 base
                        gpu_name = f"Apple {cpu_info.split()[1]} GPU"
                    else:
                        gpu_mem = min(8.0, total_mem_gb * 0.5)
                        gpu_name = "Apple Silicon GPU"
                    
                    gpu_info = GPUInfo(
                        id=0,
                        name=gpu_name,
                        vram_gb=gpu_mem,
                        vram_used_gb=0.0,  # Can't easily determine on macOS
                        vram_free_gb=gpu_mem,
                        temperature=None,
                        utilization=None
                    )
                    gpus.append(gpu_info)
                except:
                    pass
            
            # Try AMD GPUs on Linux
            if platform.system() == "Linux" and not gpus:
                try:
                    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse AMD GPU info (this is a simplified version)
                        # Real implementation would need proper parsing
                        lines = result.stdout.strip().split('\n')
                        gpu_id = 0
                        for line in lines:
                            if 'GPU' in line and 'MB' in line:
                                # Extract memory info
                                match = re.search(r'(\d+)\s*MB', line)
                                if match:
                                    vram_mb = float(match.group(1))
                                    gpu_info = GPUInfo(
                                        id=gpu_id,
                                        name=f"AMD GPU {gpu_id}",
                                        vram_gb=round(vram_mb / 1024, 1),
                                        vram_used_gb=0.0,
                                        vram_free_gb=round(vram_mb / 1024, 1),
                                        temperature=None,
                                        utilization=None
                                    )
                                    gpus.append(gpu_info)
                                    gpu_id += 1
                except:
                    pass
            
            return gpus
            
        except Exception as e:
            print(f"Debug: GPU detection error: {e}")
            return []
    
    def _detect_gpu(self) -> Tuple[str, float]:
        """Legacy method for backward compatibility"""
        gpus = self._detect_all_gpus()
        if gpus:
            primary = gpus[0]
            name = primary.name
            if len(gpus) > 1:
                name += f" (+{len(gpus)-1} more)"
            return name, primary.vram_gb
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
        """Display the detected hardware specifications with multi-GPU support"""
        print("\n" + "="*60)
        print("DETECTED HARDWARE SPECIFICATIONS")
        print("="*60)
        
        print(f"\n💻 System: {specs.os_type}")
        print(f"🧠 CPU: {specs.cpu_model}")
        print(f"   - Cores: {specs.cpu_cores} physical, {specs.cpu_threads} logical threads")
        
        if specs.has_gpu:
            print(f"\n🎮 GPU Configuration:")
            print(f"   - Total GPUs: {specs.gpu_count}")
            print(f"   - Total VRAM: {specs.total_gpu_vram_gb} GB")
            
            for i, gpu in enumerate(specs.gpus):
                print(f"\n   GPU {i}: {gpu.name}")
                print(f"   - VRAM: {gpu.vram_gb} GB total")
                print(f"   - Available: {gpu.vram_free_gb} GB free")
                if gpu.temperature:
                    print(f"   - Temperature: {gpu.temperature}°C")
                if gpu.utilization is not None:
                    print(f"   - Utilization: {gpu.utilization:.1f}%")
        else:
            print("🎮 GPU: Not detected (CPU-only mode)")
        
        print(f"\n🧩 RAM: {specs.system_ram_gb} GB")
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
        """Check which quantizations are compatible with multi-GPU support"""
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
                'description': quant_data['description'],
                'gpu_config': None  # Will store multi-GPU configuration if applicable
            }
            
            # Check storage
            if size_gb > hardware.available_storage_gb:
                result['compatible'] = False
                result['errors'].append(f"Requires {size_gb:.1f}GB storage, only {hardware.available_storage_gb}GB available")
            
            # Estimate RAM requirements
            # Rule of thumb: need RAM = model size + overhead
            ram_required = size_gb + 2  # 2GB overhead for system
            vram_required = size_gb + 0.5  # 0.5GB overhead for GPU
            
            # Check GPU mode with multi-GPU support
            if hardware.has_gpu:
                # Check single GPU first
                largest_gpu = max(hardware.gpus, key=lambda g: g.vram_gb)
                
                if vram_required <= largest_gpu.vram_gb:
                    # Fits on single GPU
                    result['gpu_config'] = f"Single GPU: {largest_gpu.name}"
                    # Determine performance based on quantization
                    if 'Q4' in quant_name or 'Q3' in quant_name:
                        result['performance'] = 'Fast'
                    elif 'Q2' in quant_name or 'IQ' in quant_name:
                        result['performance'] = 'Very Fast'
                elif hardware.total_gpu_vram_gb >= vram_required and hardware.gpu_count > 1:
                    # Can use multi-GPU
                    result['gpu_config'] = f"Multi-GPU: {hardware.gpu_count} GPUs with {hardware.total_gpu_vram_gb}GB total VRAM"
                    result['warnings'].append(f"Requires multi-GPU setup (model splitting across {hardware.gpu_count} GPUs)")
                    result['performance'] = 'Good (Multi-GPU)'
                    
                    # Calculate how the model would be split
                    gpus_needed = []
                    remaining_size = vram_required
                    for gpu in sorted(hardware.gpus, key=lambda g: g.vram_gb, reverse=True):
                        if remaining_size > 0:
                            allocation = min(gpu.vram_gb - 0.5, remaining_size)  # Leave 0.5GB headroom
                            if allocation > 0:
                                gpus_needed.append((gpu, allocation))
                                remaining_size -= allocation
                    
                    if gpus_needed:
                        split_info = ", ".join([f"{gpu.name}: {alloc:.1f}GB" for gpu, alloc in gpus_needed])
                        result['warnings'].append(f"Model split: {split_info}")
                else:
                    # Can't fit even with multi-GPU, check CPU fallback
                    cpu_ram_required = ram_required * 1.5
                    if cpu_ram_required > hardware.system_ram_gb:
                        result['compatible'] = False
                        result['errors'].append(f"Requires {vram_required:.1f}GB VRAM (have {hardware.total_gpu_vram_gb}GB across {hardware.gpu_count} GPUs) or {cpu_ram_required:.1f}GB RAM for CPU mode")
                    else:
                        result['warnings'].append(f"Will run in CPU mode (insufficient total VRAM across all GPUs)")
                        result['performance'] = 'Slow'
                        result['gpu_config'] = "CPU mode (insufficient GPU VRAM)"
            else:
                # CPU-only mode
                cpu_ram_required = ram_required * 1.5
                if cpu_ram_required > hardware.system_ram_gb:
                    result['compatible'] = False
                    result['errors'].append(f"Requires {cpu_ram_required:.1f}GB RAM for CPU mode, only {hardware.system_ram_gb}GB available")
                else:
                    result['warnings'].append("Running in CPU-only mode (no GPU detected)")
                    result['performance'] = 'Slow'
                    result['gpu_config'] = "CPU only"
            
            # Storage type warning
            if hardware.storage_type == 'HDD' and result['compatible']:
                result['warnings'].append("HDD detected - model loading will be slow")
            
            compatibility[quant_name] = result
        
        return compatibility
    
    def display_compatibility_report(self, model_info: ModelInfo, compatibility: Dict[str, Dict]):
        """Display detailed compatibility report with multi-GPU information"""
        print("\n" + "="*60)
        print(f"COMPATIBILITY REPORT FOR {model_info.name.upper()}")
        print("="*60)
        
        if model_info.parameter_count:
            print(f"\n📊 Model Size: {model_info.parameter_count} parameters")
        print(f"💾 Base Size: {model_info.base_size_gb:.1f}GB (FP16)")
        
        if model_info.huggingface_id:
            print(f"🤗 HuggingFace: {model_info.huggingface_id}")
        
        # Display GPU configuration summary
        if self.hardware_specs.has_gpu:
            print(f"\n🎮 GPU Configuration: {self.hardware_specs.gpu_count} GPU(s) with {self.hardware_specs.total_gpu_vram_gb}GB total VRAM")
            for i, gpu in enumerate(self.hardware_specs.gpus):
                print(f"   - GPU {i}: {gpu.name} ({gpu.vram_gb}GB)")
        
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
        
        # Group by GPU configuration
        single_gpu_options = {k: v for k, v in compatible_quants.items() 
                             if v['gpu_config'] and 'Single GPU' in v['gpu_config']}
        multi_gpu_options = {k: v for k, v in compatible_quants.items() 
                            if v['gpu_config'] and 'Multi-GPU' in v['gpu_config']}
        cpu_options = {k: v for k, v in compatible_quants.items() 
                      if v['gpu_config'] and 'CPU' in v['gpu_config']}
        
        # Display options by configuration type
        if single_gpu_options:
            print("\n" + "-"*40)
            print("🎮 SINGLE GPU OPTIONS (Best performance):")
            print("-"*40)
            self._display_options(single_gpu_options)
        
        if multi_gpu_options:
            print("\n" + "-"*40)
            print("🎮🎮 MULTI-GPU OPTIONS (Requires setup):")
            print("-"*40)
            self._display_options(multi_gpu_options)
        
        if cpu_options:
            print("\n" + "-"*40)
            print("🖥️  CPU OPTIONS (Slower but works):")
            print("-"*40)
            self._display_options(cpu_options)
        
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
        """Display quantization options with GPU configuration info"""
        for quant, result in sorted(options.items(), key=lambda x: x[1]['size_gb'], reverse=True):
            print(f"\n{quant}:")
            print(f"  Size: {result['size_gb']:.1f}GB")
            print(f"  Quality: {result['quality']}")
            print(f"  {result['description']}")
            if result['gpu_config']:
                print(f"  GPU Config: {result['gpu_config']}")
            
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
            
            # Best performance (single GPU options first)
            single_gpu_fast = [k for k, v in compatible_quants.items() 
                              if v['gpu_config'] and 'Single GPU' in v['gpu_config'] 
                              and v['performance'] in ['Fast', 'Very Fast']]
            
            if single_gpu_fast:
                best_performance = single_gpu_fast[0]
            else:
                # Fallback to smallest size
                best_performance = min(compatible_quants.items(), key=lambda x: x[1]['size_gb'])[0]
            
            print(f"\n🏆 Best Quality: {best_quality[0]}")
            print(f"   - {best_quality[1]['quality']}")
            print(f"   - Size: {best_quality[1]['size_gb']:.1f}GB")
            print(f"   - Config: {best_quality[1]['gpu_config']}")
            
            print(f"\n⚡ Best Performance: {best_performance}")
            print(f"   - {compatible_quants[best_performance]['quality']}")
            print(f"   - Size: {compatible_quants[best_performance]['size_gb']:.1f}GB")
            print(f"   - Config: {compatible_quants[best_performance]['gpu_config']}")
            
            # Popular choice
            if 'Q4_K_M' in compatible_quants:
                print(f"\n🌟 Popular Choice: Q4_K_M")
                print(f"   - Good balance of quality and performance")
                print(f"   - Widely supported and tested")
                print(f"   - Config: {compatible_quants['Q4_K_M']['gpu_config']}")
            
            # Multi-GPU specific recommendations
            if hardware.gpu_count > 1:
                print("\n🎮🎮 Multi-GPU Tips:")
                print("   - Use frameworks that support model parallelism (vLLM, DeepSpeed)")
                print("   - Consider using PCIe 4.0 or better for GPU interconnect")
                print("   - NVLink/SLI provides better multi-GPU performance if available")
                print("   - Some frameworks work better with identical GPUs")
            
            if hardware.gpu_count == 0:
                print("\n💡 To improve performance:")
                print("   - Add a GPU with at least 8GB VRAM")
                print("   - Use quantized models (Q4_K_M or lower)")
                print("   - Ensure adequate cooling for sustained CPU usage")
                print("   - Close other applications to free RAM")
    
    def provide_setup_instructions(self, hardware: HardwareSpecs, model_info: ModelInfo, compatible_quants: Dict[str, Dict]):
        """Provide setup instructions for running the model with multi-GPU support"""
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
        
        # Check if multi-GPU setup is needed
        needs_multi_gpu = any('Multi-GPU' in v.get('gpu_config', '') for v in compatible_quants.values())
        
        # Determine recommended runtime based on hardware
        if hardware.has_gpu and hardware.gpu_count > 1 and needs_multi_gpu:
            print("🎮🎮 Multi-GPU Setup Required\n")
            print("For multi-GPU inference, you'll need specialized frameworks:\n")
            
            print("Option 1 - vLLM (Recommended for multi-GPU):")
            print("1. Install vLLM:")
            print("   pip install vllm")
            print("\n2. Run with tensor parallelism:")
            print(f"   python -m vllm.entrypoints.openai.api_server \\")
            print(f"     --model {model_info.huggingface_id or model_info.name} \\")
            print(f"     --tensor-parallel-size {hardware.gpu_count}")
            
            print("\n\nOption 2 - DeepSpeed:")
            print("1. Install DeepSpeed:")
            print("   pip install deepspeed")
            print("\n2. Configure for model parallelism")
            print("   (Requires custom implementation)")
            
            print("\n\nOption 3 - Accelerate (Hugging Face):")
            print("1. Install accelerate:")
            print("   pip install accelerate")
            print("\n2. Run with device_map='auto' for automatic splitting")
            
        elif hardware.has_gpu and 'NVIDIA' in hardware.gpus[0].name:
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
            
            if hardware.gpu_count > 1:
                print(f"\n   Note: Ollama will use GPU 0 by default. To use a different GPU:")
                print(f"   CUDA_VISIBLE_DEVICES=1 ollama run {model_tag}")
            
        elif hardware.os_type == "Darwin" and hardware.has_gpu and "Apple" in hardware.gpus[0].name:
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
        if hardware.gpu_count > 1:
            print("   - Note: LM Studio currently uses single GPU only")
        
        print("\n🌐 Text Generation WebUI:")
        print("   - GitHub: oobabooga/text-generation-webui")
        print("   - Web interface for various models")
        print("   - Supports GGUF, GPTQ, AWQ formats")
        if hardware.gpu_count > 1:
            print("   - Supports multi-GPU with --gpu-memory flag")
        
        print("\n🐍 Python with llama-cpp-python:")
        print("   - pip install llama-cpp-python")
        print("   - Python bindings for llama.cpp")
        print("   - Easy integration into Python projects")
        
        # Multi-GPU specific tools
        if hardware.gpu_count > 1:
            print("\n🎮🎮 Multi-GPU Specific Tools:")
            print("\n🚀 vLLM (High-performance multi-GPU):")
            print("   - pip install vllm")
            print("   - Optimized for throughput")
            print("   - Automatic tensor parallelism")
            
            print("\n⚡ TensorRT-LLM (NVIDIA only):")
            print("   - Optimized for NVIDIA multi-GPU")
            print("   - Requires model conversion")
            print("   - Best performance for production")
        
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
        
        if hardware.has_gpu:
            print("• Ensure GPU drivers are up to date")
            if hardware.gpu_count > 1:
                print("• For multi-GPU setups:")
                print("  - Use NVLink if available for better GPU-to-GPU bandwidth")
                print("  - Match GPU models for best performance")
                print("  - Monitor GPU memory usage with nvidia-smi or gpu-z")
                print("  - Consider pipeline parallelism for very large models")
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
║      🤖 LLM Hardware Compatibility Checker v3.1 🤖        ║
║         with Web Search & Multi-GPU Support              ║
╚══════════════════════════════════════════════════════════╝
    """)
    print("This tool will:")
    print("  ✓ Automatically detect your hardware (including multiple GPUs)")
    print("  ✓ Search the web for model requirements")
    print("  ✓ Check compatibility with various quantizations")
    print("  ✓ Provide setup instructions for single and multi-GPU configs")
    print("\nNote: Internet connection required for model information.")

def main():
    """Main function"""
    try:
        print_welcome_banner()
        
        checker = LLMHardwareChecker()
        
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
        print("\n💡 Troubleshooting:")
        print("  • Check your internet connection")
        print("  • Ensure required packages are installed:")
        print("    pip install psutil requests")
        print("    pip install gputil py-cpuinfo  # Optional")
        print("\nDetailed error information:")
        traceback.print_exc()

if __name__ == "__main__":
    main()