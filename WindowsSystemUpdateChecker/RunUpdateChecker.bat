@echo off
REM Daily Update Checker - Quick Launcher
REM Double-click this file to run the update checker

echo.
echo ========================================
echo     DAILY SYSTEM UPDATE CHECKER
echo ========================================
echo.
echo Starting update check...
echo.

PowerShell.exe -ExecutionPolicy Bypass -File "%USERPROFILE%\SystemUpdateChecker\DailyUpdateChecker.ps1"

echo.
echo Check complete! Press any key to close...
pause >nul
