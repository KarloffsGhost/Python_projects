#!/usr/bin/env python3
"""
Build script to combine modular LLM Hardware Checker into a single file
Creates a standalone script that can be distributed easily
"""

import os
import re
from pathlib import Path
from datetime import datetime


class SingleFileBuilder:
    """Builds a single-file version of the LLM Hardware Checker"""
    
    def __init__(self, source_dir="llm_hardware_checker", output_file="llm_checker_standalone.py"):
        self.source_dir = Path(source_dir)
        self.output_file = Path(output_file)
        self.module_order = [
            "hardware_detector.py",
            "model_database.py", 
            "performance.py",
            "compatibility.py",
            "display.py",
            "main.py"
        ]
        self.combined_imports = set()
        self.combined_code = []
    
    def build(self):
        """Build the single file"""
        print(f"🔨 Building single file from {self.source_dir}/...")
        
        # Add header
        self._add_header()
        
        # Process each module
        for module_file in self.module_order:
            self._process_module(module_file)
        
        # Add main execution block
        self._add_main_block()
        
        # Write output
        self._write_output()
        
        print(f"✅ Successfully built {self.output_file}")
        print(f"📦 File size: {self.output_file.stat().st_size / 1024:.1f} KB")
    
    def _add_header(self):
        """Add file header with metadata"""
        header = f'''#!/usr/bin/env python3
"""
LLM Hardware Compatibility Checker v4.0 - Standalone Version
Built on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is an auto-generated single-file version for easy distribution.
Source: https://github.com/yourusername/llm-hardware-checker

Features:
- Hardware detection (multi-GPU support)
- Model search with HuggingFace integration
- Performance estimation
- Compatibility checking
- Setup instructions
"""

'''
        self.combined_code.append(header)
    
    def _process_module(self, module_file):
        """Process a single module file"""
        module_path = self.source_dir / module_file
        
        if not module_path.exists():
            print(f"⚠️  Warning: {module_path} not found, skipping...")
            return
        
        print(f"  📄 Processing {module_file}...")
        
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract imports and code
        imports, code = self._split_imports_and_code(content)
        
        # Process imports
        for imp in imports:
            self._process_import(imp)
        
        # Add module comment
        self.combined_code.append(f"\n# {'='*60}")
        self.combined_code.append(f"# Module: {module_file}")
        self.combined_code.append(f"# {'='*60}\n")
        
        # Add processed code
        self.combined_code.append(code)
    
    def _split_imports_and_code(self, content):
        """Split module content into imports and code"""
        lines = content.split('\n')
        imports = []
        code_lines = []
        in_imports = True
        skip_next = False
        
        for i, line in enumerate(lines):
            # Skip module docstring
            if i < 10 and (line.startswith('"""') or line.startswith("'''")):
                skip_next = not skip_next
                continue
            if skip_next:
                continue
            
            # Process line
            if in_imports:
                if line.strip() == '' and imports:
                    # Empty line after imports
                    continue
                elif line.startswith(('import ', 'from ')) and not line.startswith('from .'):
                    imports.append(line)
                elif line.startswith('from .'):
                    # Internal import - skip as we're combining
                    continue
                else:
                    # End of imports section
                    in_imports = False
                    code_lines.append(line)
            else:
                code_lines.append(line)
        
        return imports, '\n'.join(code_lines)
    
    def _process_import(self, import_line):
        """Process an import statement"""
        # Normalize import
        import_line = import_line.strip()
        
        # Skip internal imports
        if 'from .' in import_line:
            return
        
        # Skip package-level imports from our own package
        if 'from llm_hardware_checker' in import_line:
            return
        
        # Add to combined imports if not already present
        self.combined_imports.add(import_line)
    
    def _add_main_block(self):
        """Add the main execution block"""
        main_block = '''

# ============================================================
# Main Execution
# ============================================================

def main():
    """Entry point for the standalone application"""
    # Create and run the checker
    checker = LLMHardwareChecker()
    checker.run()


if __name__ == "__main__":
    # Check for optional packages and provide warnings
    try:
        import GPUtil
    except ImportError:
        print("Note: GPUtil not installed. GPU detection will be limited.")
        print("Install with: pip install gputil")
    
    try:
        import cpuinfo
    except ImportError:
        print("Note: py-cpuinfo not installed. CPU detection will be limited.")
        print("Install with: pip install py-cpuinfo")
    
    # Run main
    main()
'''
        self.combined_code.append(main_block)
    
    def _write_output(self):
        """Write the combined output file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            # Write all imports first
            f.write("# Standard library imports\n")
            stdlib_imports = sorted([imp for imp in self.combined_imports 
                                   if not imp.startswith('from ') or 
                                   imp.split()[1] in ['typing', 'dataclasses', 'enum', 'pathlib']])
            for imp in stdlib_imports:
                f.write(imp + '\n')
            
            f.write("\n# Third-party imports\n")
            third_party = sorted([imp for imp in self.combined_imports 
                                if imp not in stdlib_imports])
            for imp in third_party:
                f.write(imp + '\n')
            
            # Write combined code
            f.write('\n'.join(self.combined_code))
        
        # Make executable on Unix-like systems
        try:
            os.chmod(self.output_file, 0o755)
        except:
            pass


class MinimalBuilder:
    """Creates a minimal version with embedded model database"""
    
    def __init__(self, source_dir="llm_hardware_checker", output_file="llm_checker_minimal.py"):
        self.source_dir = Path(source_dir)
        self.output_file = Path(output_file)
        self.builder = SingleFileBuilder(source_dir, output_file)
    
    def build(self):
        """Build minimal version with embedded database"""
        print(f"🔨 Building minimal version...")
        
        # First build the full version
        self.builder.build()
        
        # Then optimize it
        self._optimize_output()
        
        print(f"✅ Minimal version created: {self.output_file}")
    
    def _optimize_output(self):
        """Optimize the output file"""
        with open(self.output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove unnecessary comments
        lines = content.split('\n')
        optimized = []
        
        for line in lines:
            # Skip module separator comments
            if line.startswith('# ====') and 'Module:' in lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else False:
                continue
            # Skip other unnecessary comments
            if line.strip().startswith('#') and not line.startswith('#!'):
                # Keep important comments
                if any(keep in line for keep in ['TODO', 'FIXME', 'NOTE', 'WARNING']):
                    optimized.append(line)
            else:
                optimized.append(line)
        
        # Write back
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(optimized))


def create_setup_script():
    """Create a setup.py for the modular version"""
    setup_content = '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-hardware-checker",
    version="4.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Check LLM compatibility with your hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-hardware-checker",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "psutil",
        "requests",
    ],
    extras_require={
        "full": ["GPUtil", "py-cpuinfo"],
    },
    entry_points={
        "console_scripts": [
            "llm-checker=llm_hardware_checker.main:main",
        ],
    },
)
'''
    
    with open("setup.py", "w") as f:
        f.write(setup_content)
    
    print("✅ Created setup.py for package installation")


def create_requirements_file():
    """Create requirements.txt files"""
    # Basic requirements
    with open("requirements.txt", "w") as f:
        f.write("""# Core requirements
psutil>=5.8.0
requests>=2.25.0

# Optional but recommended
GPUtil>=1.4.0
py-cpuinfo>=8.0.0
""")
    
    # Development requirements
    with open("requirements-dev.txt", "w") as f:
        f.write("""# Development requirements
-r requirements.txt

# Testing
pytest>=6.0.0
pytest-cov>=2.10.0

# Linting
flake8>=3.8.0
black>=20.8b1

# Type checking
mypy>=0.800
""")
    
    print("✅ Created requirements.txt files")


def create_readme():
    """Create a README for the project structure"""
    readme_content = '''# LLM Hardware Checker - Modular Version

This is the modular version of the LLM Hardware Checker, organized for better maintainability and development.

## Project Structure

```
llm_hardware_checker/
├── __init__.py          # Package initialization
├── hardware_detector.py # Hardware detection logic
├── model_database.py    # Model information and search
├── performance.py       # Performance estimation
├── compatibility.py     # Compatibility checking
├── display.py          # Output formatting
└── main.py             # Entry point
```

## Installation

### For Development

```bash
pip install -r requirements-dev.txt
pip install -e .
```

### For Users

```bash
pip install -r requirements.txt
python -m llm_hardware_checker
```

## Building Single File Version

To create a standalone single-file version for distribution:

```bash
python build_single_file.py
```

This creates `llm_checker_standalone.py` which can be distributed as a single file.

## Usage

### Modular Version
```python
from llm_hardware_checker import LLMHardwareChecker

checker = LLMHardwareChecker()
checker.run()
```

### Command Line
```bash
# After installation
llm-checker

# Or directly
python -m llm_hardware_checker
```

## Adding New Models

Edit `model_database.py` or create a `models.json` file:

```json
{
  "new_model": {
    "patterns": ["new-model", "newmodel"],
    "capabilities": {
      "primary_strength": "Your Model Strength",
      "strengths": ["Strength 1", "Strength 2"],
      "weaknesses": ["Weakness 1"],
      "use_cases": ["Use case 1", "Use case 2"]
    },
    "hf_models": {
      "7B": ["org/model-7b", "org/model-7b-instruct"]
    }
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License
'''
    
    with open("README_MODULAR.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README_MODULAR.md")


def main():
    """Main build function"""
    print("🚀 LLM Hardware Checker Build Tool\n")
    
    # Check if source directory exists
    if not Path("llm_hardware_checker").exists():
        print("❌ Error: llm_hardware_checker/ directory not found!")
        print("Please ensure you have the modular source code in place.")
        return
    
    print("Select build option:")
    print("1. Build standalone file (full version)")
    print("2. Build minimal standalone file")
    print("3. Create setup files for modular version")
    print("4. Build everything")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        builder = SingleFileBuilder()
        builder.build()
    
    elif choice == "2":
        builder = MinimalBuilder()
        builder.build()
    
    elif choice == "3":
        create_setup_script()
        create_requirements_file()
        create_readme()
    
    elif choice == "4":
        # Build everything
        print("\n📦 Building all versions...\n")
        
        # Standalone
        builder = SingleFileBuilder()
        builder.build()
        
        # Minimal
        minimal = MinimalBuilder()
        minimal.build()
        
        # Setup files
        create_setup_script()
        create_requirements_file()
        create_readme()
        
        print("\n✅ All builds completed!")
        print("\nGenerated files:")
        print("  - llm_checker_standalone.py (full single file)")
        print("  - llm_checker_minimal.py (optimized single file)")
        print("  - setup.py (for pip install)")
        print("  - requirements.txt")
        print("  - requirements-dev.txt")
        print("  - README_MODULAR.md")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()