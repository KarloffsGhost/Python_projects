@echo off
REM Startup Manager - View and manage startup items
REM Run without arguments for interactive mode
REM Run as Administrator to manage all startup items

echo.
echo ========================================
echo     STARTUP MANAGER
echo ========================================
echo.
echo Starting startup scan...
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0StartupManager.ps1" %*

echo.
