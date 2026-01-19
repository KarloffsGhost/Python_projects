@echo off
REM Install Windows and Driver updates - Requires Administrator

echo.
echo ========================================
echo   WINDOWS UPDATE INSTALLER
echo ========================================
echo.
echo This requires Administrator privileges...
echo.

PowerShell -Command "Start-Process PowerShell -ArgumentList '-ExecutionPolicy Bypass -File ""%~dp0InstallUpdates-Windows.ps1""' -Verb RunAs"
