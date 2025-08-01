# 🤖 LLM Hardware Compatibility Checker

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Cross Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/yourusername/llm-hardware-checker)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python tool that automatically detects your hardware specifications and determines which Large Language Models (LLMs) you can run locally. The tool searches the web for model requirements, analyzes various quantization options, and provides detailed setup instructions.

## ✨ Features

- **🔍 Automatic Hardware Detection**: Detects CPU, GPU, RAM, storage, and OS specifications
- **🌐 Web-Based Model Search**: Searches HuggingFace and other sources for model information
- **📊 Quantization Analysis**: Supports 20+ quantization formats (GGUF, INT4, INT8, etc.)
- **⚡ Performance Predictions**: Estimates inference speed and quality for each configuration
- **🎯 Smart Recommendations**: Suggests optimal quantization levels for your hardware
- **📋 Setup Instructions**: Provides step-by-step installation guides for multiple runtimes
- **🔄 Cross-Platform Support**: Works on Windows, macOS, and Linux

## 📋 Requirements

### Required Dependencies
```bash
pip install psutil requests
```

### Optional Dependencies (for enhanced detection)
```bash
pip install gputil py-cpuinfo
```

**Note**: The script will work without optional dependencies but with limited GPU and CPU detection capabilities.

## 🚀 Quick Start

1. **Clone or download the script**:
   ```bash
   git clone https://github.com/yourusername/llm-hardware-checker.git
   cd llm-hardware-checker
   ```

2. **Install dependencies**:
   ```bash
   pip install psutil requests
   pip install gputil py-cpuinfo  # Optional but recommended
   ```

3. **Run the checker**:
   ```bash
   python llm_hardware_checker.py
   ```

4. **Follow the interactive prompts**:
   - Review your detected hardware
   - Enter the name of the model you want to run
   - Get compatibility analysis and setup instructions

## 💻 Supported Hardware

### CPUs
- ✅ Intel (all generations)
- ✅ AMD (all generations) 
- ✅ Apple Silicon (M1, M2, M3)
- ✅ ARM processors

### GPUs
- ✅ NVIDIA (with CUDA support)
- ✅ Apple Silicon integrated graphics
- ✅ AMD GPUs (limited support)
- ✅ CPU-only mode (no GPU required)

### Operating Systems
- ✅ Windows 10/11
- ✅ macOS (Intel and Apple Silicon)
- ✅ Linux (Ubuntu, CentOS, Arch, etc.)

## 🎯 Supported Models

The tool can analyze compatibility for popular LLM families including:

- **Meta Llama**: Llama 2, Code Llama, Llama-2-Chat
- **Mistral AI**: Mistral 7B, Mixtral 8x7B, Mistral-Instruct
- **Microsoft**: Phi-2, DialoGPT, CodeBERT
- **Google**: T5, FLAN-T5, Gemma
- **Alibaba**: Qwen, Qwen-Chat
- **DeepSeek**: DeepSeek Coder, DeepSeek Chat
- **Anthropic**: Claude (when available locally)
- **OpenAI**: GPT variants (when available locally)
- **Custom models**: Any model with parameter count information

## 📊 Quantization Support

The tool analyzes **20+ quantization formats**:

| Format | Size Reduction | Quality | Best For |
|--------|---------------|---------|----------|
| FP16 | 0% | ★★★★★ | Maximum quality |
| Q8_0 | ~47% | ★★★★★ | Minimal quality loss |
| Q6_K | ~59% | ★★★★☆ | Excellent balance |
| Q5_K_M | ~64% | ★★★★☆ | Popular choice |
| Q4_K_M | ~70% | ★★★☆☆ | Good balance |
| Q4_0 | ~73% | ★★★☆☆ | Legacy standard |
| Q3_K_M | ~76% | ★★☆☆☆ | Aggressive compression |
| Q2_K | ~81% | ★☆☆☆☆ | Maximum compression |

## 🎮 Example Usage

```bash
$ python llm_hardware_checker.py

╔══════════════════════════════════════════════════════════╗
║        🤖 LLM Hardware Compatibility Checker v3.0 🤖      ║
║                   with Web Search Support                 ║
╚══════════════════════════════════════════════════════════╝

🔍 Detecting hardware specifications...

============================================================
DETECTED HARDWARE SPECIFICATIONS
============================================================

💻 System: Linux
🧠 CPU: AMD Ryzen 9 5900X 12-Core Processor
   - Cores: 12 physical, 24 logical threads
🎮 GPU: NVIDIA GeForce RTX 4090
   - VRAM: 24.0 GB
🧩 RAM: 32.0 GB
💾 Storage: SSD with 2048.0 GB available

Model name (or 'exit' to quit): Llama 2 7B

🔍 Searching for information about Llama 2 7B...

============================================================
COMPATIBILITY REPORT FOR LLAMA 2 7B
============================================================

📊 Model Size: 7B parameters
💾 Base Size: 14.0GB (FP16)
🤗 HuggingFace: meta-llama/Llama-2-7b-hf

✅ This model CAN run on your hardware!

Compatible quantizations: 13 out of 15

⚡ FAST OPTIONS (Recommended for speed):
----------------------------------------

Q2_K: 2.7GB
  Quality: ★☆☆☆☆
  2-bit quantization, significant quality loss

Q3_K_M: 3.4GB
  Quality: ★★☆☆☆
  3-bit medium quantization, acceptable quality

...
```

## 🛠️ Supported Runtimes

The tool provides setup instructions for multiple LLM runtimes:

### 🥇 Recommended
- **[Ollama](https://ollama.ai)**: Simple, fast, and user-friendly
- **[LM Studio](https://lmstudio.ai)**: GUI-based with model browser

### 🔧 Advanced
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: High-performance C++ implementation
- **[Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)**: Feature-rich web interface
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)**: Python bindings

## 🎯 Hardware Recommendations

### Minimum Requirements (7B models)
- **CPU**: 4+ cores, 2.0+ GHz
- **RAM**: 8GB (for Q3_K quantizations)
- **Storage**: 5GB available space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended (13B models)
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 16GB
- **GPU**: 8GB+ VRAM (RTX 3070, RTX 4060 Ti, etc.)
- **Storage**: 20GB+ SSD space

### Optimal (70B models)
- **CPU**: 16+ cores, 3.5+ GHz
- **RAM**: 32GB+
- **GPU**: 24GB+ VRAM (RTX 4090, A6000, etc.)
- **Storage**: 100GB+ NVMe SSD

## 🔧 Advanced Usage

### Command Line Options
The script currently runs interactively, but you can modify it for programmatic use:

```python
from llm_hardware_checker import LLMHardwareChecker

checker = LLMHardwareChecker()
hardware = checker.detect_hardware()
model_info = checker.web_searcher.search_model_info("Llama 2 7B")
compatibility = checker.check_compatibility(hardware, model_info)
```

### Custom Model URLs
You can extend the web searcher to include custom model repositories by modifying the `WebSearcher` class.

## 🐛 Troubleshooting

### Common Issues

**GPU not detected**:
```bash
pip install gputil
# For NVIDIA GPUs, ensure nvidia-smi is working:
nvidia-smi
```

**CPU detection limited**:
```bash
pip install py-cpuinfo
```

**Web search failing**:
- Check your internet connection
- Verify the model name spelling
- Try simpler model names (e.g., "Llama 7B" instead of "Llama-2-7b-chat-hf")

**Permission errors on Linux/macOS**:
```bash
sudo python llm_hardware_checker.py
```

### Error Messages

| Error | Solution |
|-------|----------|
| `GPUtil not installed` | `pip install gputil` |
| `py-cpuinfo not installed` | `pip install py-cpuinfo` |
| `Could not find information for model` | Check model name spelling |
| `This model CANNOT run` | Upgrade hardware or use smaller model |

## 📈 Performance Tips

### For GPU Users
- Update to latest GPU drivers
- Use Q4_K_M or Q5_K_M quantizations for best balance
- Enable GPU offloading with `-ngl` flag in llama.cpp
- Ensure adequate cooling for sustained workloads

### For CPU-Only Users
- Use models with aggressive quantization (Q3_K_M, Q2_K)
- Close unnecessary applications before running
- Use CPU builds with AVX2/AVX512 support
- Consider smaller context windows (1024-2048 tokens)

### For Apple Silicon Users
- Ollama provides excellent Apple Silicon optimization
- Unified memory allows larger models than traditional setups
- Metal Performance Shaders acceleration available

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTE.md](https://github.com/KarloffsGhost/Python_projects/blob/main/contribute.md)** for guidelines on:
- 🐛 Bug reports
- ✨ Feature requests  
- 🔧 Code contributions
- 📚 Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - High-performance LLM inference
- **[Ollama](https://ollama.ai)** - Simplified local LLM deployment
- **[HuggingFace](https://huggingface.co)** - Model repository and API
- **[TheBloke](https://huggingface.co/TheBloke)** - GGUF quantized model conversions

## 🔗 Related Projects

- [Ollama](https://ollama.ai) - Run LLMs locally
- [LM Studio](https://lmstudio.ai) - GUI for local LLMs
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) - Web interface
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ LLM inference

## 📊 Changelog

### v3.0 (Current)
- ✅ Added web search for model information
- ✅ Enhanced quantization support (20+ formats)
- ✅ Improved hardware detection
- ✅ Cross-platform compatibility
- ✅ Setup instruction generation

### v2.0
- ✅ Added GPU detection
- ✅ Quantization analysis
- ✅ Performance predictions

### v1.0
- ✅ Basic hardware detection
- ✅ Simple compatibility checking

---

**Made with ❤️ for the open source AI community**
