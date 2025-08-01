# 🤝 Contributing to LLM Hardware Compatibility Checker

Thank you for your interest in contributing to the LLM Hardware Compatibility Checker! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Areas for Contribution](#areas-for-contribution)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- Git
- Basic knowledge of Python and hardware concepts
- Understanding of LLM deployment (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/llm-hardware-checker.git
   cd llm-hardware-checker
   ```

3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/originaluser/llm-hardware-checker.git
   ```

## 🛠️ Development Setup

### Install Dependencies

```bash
# Required dependencies
pip install psutil requests

# Development dependencies
pip install pytest black flake8 mypy

# Optional dependencies (for full functionality)
pip install gputil py-cpuinfo
```

### Environment Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=llm_hardware_checker

# Run specific test file
python -m pytest tests/test_hardware_detection.py
```

## 📝 Contributing Guidelines

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Commit Message Format

Use clear, descriptive commit messages:

```
type(scope): brief description

Longer description if needed

- List specific changes
- Reference issues: Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat(gpu): add AMD GPU detection support

- Implement ROCm detection for AMD GPUs
- Add VRAM detection for Radeon cards
- Update compatibility matrix for AMD hardware
- Fixes #45

fix(search): handle network timeouts gracefully

- Add timeout handling for HuggingFace API calls
- Fallback to estimated values on network failure
- Improve error messages for connection issues
```

## 🎨 Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specific requirements:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Type hints**: Required for all new functions
- **Docstrings**: Google style for all public methods

### Formatting Tools

Before submitting, run:

```bash
# Format code
black llm_hardware_checker.py

# Check style
flake8 llm_hardware_checker.py

# Type checking
mypy llm_hardware_checker.py
```

### Code Organization

```python
# File structure example
"""
Module docstring explaining purpose
"""

# Standard library imports
import json
import platform
from typing import Dict, List

# Third-party imports
import psutil
import requests

# Local imports
from .utils import helper_function

# Constants
DEFAULT_TIMEOUT = 5

# Classes and functions
class ExampleClass:
    """Class docstring with purpose and usage examples."""
    
    def __init__(self, parameter: str) -> None:
        """Initialize with clear parameter descriptions."""
        self.parameter = parameter
    
    def public_method(self, input_data: Dict) -> List[str]:
        """Public method with type hints and docstring."""
        return self._private_method(input_data)
    
    def _private_method(self, data: Dict) -> List[str]:
        """Private method for internal use."""
        # Implementation here
        pass
```

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── test_hardware_detection.py
├── test_web_search.py
├── test_compatibility.py
├── test_integration.py
└── fixtures/
    ├── mock_responses.json
    └── sample_hardware.json
```

### Writing Tests

```python
import pytest
from unittest.mock import patch, MagicMock
from llm_hardware_checker import LLMHardwareChecker

class TestHardwareDetection:
    def test_cpu_detection(self):
        """Test CPU detection accuracy."""
        checker = LLMHardwareChecker()
        cpu_model = checker._detect_cpu()
        assert isinstance(cpu_model, str)
        assert len(cpu_model) > 0

    @patch('psutil.virtual_memory')
    def test_memory_detection(self, mock_memory):
        """Test memory detection with mocked data."""
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        checker = LLMHardwareChecker()
        hardware = checker.detect_hardware()
        assert hardware.system_ram_gb == 16.0

    def test_compatibility_analysis(self):
        """Test compatibility analysis logic."""
        # Test implementation here
        pass
```

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Hardware Tests**: Test on different hardware configurations
- **Network Tests**: Test web search functionality (with mocking)

## 📋 Areas for Contribution

### 🔥 High Priority

1. **GPU Support Expansion**
   - AMD GPU detection and VRAM measurement
   - Intel Arc GPU support
   - Multi-GPU configurations
   - GPU memory fragmentation analysis

2. **Model Database Enhancement**
   - Expanded model coverage beyond HuggingFace
   - Local model database for offline use
   - Model performance benchmarks
   - Memory usage patterns for different architectures

3. **Advanced Compatibility Logic**
   - Context length impact on memory usage
   - Batch size recommendations
   - Multi-modal model support (vision, audio)
   - Fine-tuning memory requirements

### 🌟 Medium Priority

4. **User Experience Improvements**
   - Configuration file support
   - Command-line interface options
   - Progress bars for long operations
   - Better error handling and recovery

5. **Platform-Specific Features**
   - Windows GPU detection improvements
   - Apple Silicon optimization detection
   - Linux package manager integration
   - Docker container support

6. **Performance Analysis**
   - Inference speed predictions
   - Token/second estimations
   - Power consumption estimates
   - Thermal throttling considerations

### 💡 Nice to Have

7. **Web Interface**
   - Flask/FastAPI web frontend
   - Real-time hardware monitoring
   - Model comparison tools
   - Community hardware database

8. **Integration Features**
   - Jupyter notebook integration
   - VS Code extension
   - GitHub Actions for CI testing
   - Cloud instance recommendations

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Update to latest version** to see if bug is fixed
3. **Test on clean environment** to rule out local issues

### Bug Report Template

```markdown
**Bug Description**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Hardware Information**
- OS: [e.g. Windows 11, macOS 13.0, Ubuntu 22.04]
- CPU: [e.g. Intel i7-12700K, AMD Ryzen 5 5600X]
- GPU: [e.g. NVIDIA RTX 4070, Apple M2, None]
- RAM: [e.g. 16GB DDR4]
- Python version: [e.g. 3.9.7]

**Error Logs**
```
Paste any error messages or logs here
```

**Additional Context**
Any other context about the problem.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How would you like this feature to work?

**Alternatives Considered**
Other solutions you've considered.

**Use Cases**
Specific scenarios where this would be helpful.

**Priority**
How important is this feature to you? (Low/Medium/High)
```

## 📤 Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

2. **Make your changes**:
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**:
   ```bash
   python -m pytest
   black --check .
   flake8 .
   mypy .
   ```

4. **Commit with clear messages**:
   ```bash
   git add .
   git commit -m "feat(gpu): add AMD GPU detection support"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-new-feature
   ```

6. **Create Pull Request**:
   - Use the PR template
   - Reference related issues
   - Provide detailed description
   - Include screenshots if UI changes

### Pull Request Template

```markdown
## 📋 Description
Brief description of changes and motivation.

## 🔗 Related Issues
Fixes #123
Relates to #456

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Hardware tested: [list configurations]

## 📸 Screenshots (if applicable)
Before/after screenshots for UI changes.

## ✅ Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

## 🔍 Code Review Process

### For Reviewers

- **Functionality**: Does the code work as intended?
- **Performance**: Are there any performance implications?
- **Security**: Any security concerns with web requests?
- **Compatibility**: Does it work across platforms?
- **Tests**: Are tests comprehensive and meaningful?
- **Documentation**: Is the code well-documented?

### For Contributors

- Be responsive to feedback
- Make requested changes promptly
- Ask questions if feedback is unclear
- Test thoroughly before requesting review

## 🌐 Development Areas

### Core Components

```python
# Key classes to understand:
- HardwareSpecs: Hardware specification data structure
- ModelInfo: Model information from web search
- QuantizationType: Supported quantization formats
- WebSearcher: Handles model information retrieval
- LLMHardwareChecker: Main compatibility analysis engine
```

### Extension Points

1. **Hardware Detection** (`_detect_gpu`, `_detect_cpu`):
   - Add support for new GPU vendors
   - Improve detection accuracy
   - Add hardware-specific optimizations

2. **Web Search** (`WebSearcher` class):
   - Add new model repositories
   - Improve search algorithms
   - Cache search results

3. **Compatibility Analysis** (`check_compatibility`):
   - Refine memory usage calculations
   - Add performance prediction models
   - Support new quantization formats

4. **Output Generation**:
   - Add export formats (JSON, CSV, HTML)
   - Improve terminal output formatting
   - Add logging capabilities

## 📚 Resources for Contributors

### Learning Resources
- [LLM Inference Optimization](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Quantization Techniques](https://huggingface.co/docs/transformers/main/en/quantization)

### Tools and Libraries
- [psutil documentation](https://psutil.readthedocs.io/)
- [GPUtil documentation](https://github.com/anderskm/gputil)
- [HuggingFace API](https://huggingface.co/docs/api-inference/index)

## 💬 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community chat
- **Discord**: [Add Discord link if available]

### Communication Guidelines

- Be respectful and constructive
- Search before asking questions
- Provide context and details
- Help others when possible
- Follow the code of conduct

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributor graphs
- Special thanks for major features

## 📄 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

### Quick Summary

- **Be respectful** of differing viewpoints and experiences
- **Accept constructive criticism** gracefully
- **Focus on community benefit** over personal gain
- **Show empathy** towards other community members

---

**Happy contributing! 🎉**

*If you have questions about contributing, feel free to open an issue or reach out to the maintainers.*
