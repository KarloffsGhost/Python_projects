@echo off
REM System Cleaner - Scan and clean temp files, caches, and junk
REM Run without arguments for interactive mode

echo.
echo ========================================
echo     SYSTEM CLEANER
echo ========================================
echo.
echo Starting system scan...
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0SystemCleaner.ps1" %*

echo.
