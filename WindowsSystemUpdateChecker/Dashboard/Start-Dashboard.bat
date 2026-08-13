@echo off
REM System Maintenance Dashboard - Web Interface
REM Opens dashboard in your default browser

echo.
echo ========================================
echo     SYSTEM MAINTENANCE DASHBOARD
echo ========================================
echo.
echo Starting web dashboard...
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%~dp0Start-Dashboard.ps1"
