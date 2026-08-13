@echo off
REM AI/ML Toolchain Cleaner - Clean Ollama, Docker, pip, npm, NVIDIA caches
REM Run without arguments for interactive mode

echo.
echo ========================================
echo     AI/ML TOOLCHAIN CLEANER
echo ========================================
echo.
echo Starting AI toolchain scan...
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0AIToolchainCleaner.ps1" %*

echo.
