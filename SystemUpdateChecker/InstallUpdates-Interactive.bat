@echo off
REM Interactive Application Updater - Choose what to update

echo.
echo ========================================
echo   INTERACTIVE APPLICATION UPDATER
echo ========================================
echo.
echo This lets you choose which apps to update...
echo.

PowerShell -ExecutionPolicy Bypass -File "%~dp0InstallUpdates-Interactive.ps1"
