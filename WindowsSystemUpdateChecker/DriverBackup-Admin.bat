@echo off
REM Driver Backup and Restore - Requires Administrator for backup/restore operations
REM Run without arguments for interactive mode

echo.
echo ========================================
echo     DRIVER BACKUP ^& RESTORE
echo ========================================
echo.
echo Requesting Administrator privileges...
echo.

PowerShell -Command "Start-Process PowerShell -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0DriverBackup.ps1\" %*' -Verb RunAs"
