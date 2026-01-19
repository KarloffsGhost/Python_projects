@echo off
REM Run UpdateTaskLocation with Administrator privileges

echo Requesting Administrator privileges...
echo.

PowerShell -Command "Start-Process PowerShell -ArgumentList '-ExecutionPolicy Bypass -File ""%~dp0UpdateTaskLocation.ps1""' -Verb RunAs"
