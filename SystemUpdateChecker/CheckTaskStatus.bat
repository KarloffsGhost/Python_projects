@echo off
echo.
echo ========================================
echo    CHECKING UPDATE TASK STATUS
echo ========================================
echo.

PowerShell -Command "Get-ScheduledTask -TaskName 'DailySystemUpdateChecker' | Format-Table TaskName, State -AutoSize"
echo.

PowerShell -Command "$info = Get-ScheduledTask -TaskName 'DailySystemUpdateChecker' | Get-ScheduledTaskInfo; Write-Host 'Last Run Time: ' $info.LastRunTime; Write-Host 'Next Run Time: ' $info.NextRunTime; Write-Host 'Last Result:  ' $info.LastTaskResult"
echo.

PowerShell -Command "$task = Get-ScheduledTask -TaskName 'DailySystemUpdateChecker'; $action = $task.Actions[0]; Write-Host 'Script Location: ' $action.Arguments"
echo.
echo.
echo Task Status:
echo   State = Ready means it will run automatically
echo   State = Disabled means it's turned off
echo.
echo Last Result = 0 means successful
echo.
echo ========================================
pause
