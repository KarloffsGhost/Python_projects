@echo off
REM System Maintenance Toolkit - Main Launcher
REM Your one-stop shop for system updates, cleaning, and maintenance

echo.
echo ========================================
echo     SYSTEM MAINTENANCE TOOLKIT
echo ========================================
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0SystemMaintenance.ps1" %*
