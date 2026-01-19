@echo off
REM Auto-install application updates via winget

echo.
echo ========================================
echo   INSTALLING APPLICATION UPDATES
echo ========================================
echo.
echo This will update all applications via winget...
echo.
pause

PowerShell -Command "winget upgrade --all --accept-source-agreements --accept-package-agreements"

echo.
echo ========================================
echo Application updates complete!
echo ========================================
echo.
pause
