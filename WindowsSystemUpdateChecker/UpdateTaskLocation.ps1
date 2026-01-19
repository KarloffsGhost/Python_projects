# Update the scheduled task to use the correct location
$ScriptVersion = "1.0.0"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    exit
}

$TaskName = "DailySystemUpdateChecker"
$NewScriptPath = "$env:USERPROFILE\SystemUpdateChecker\DailyUpdateChecker.ps1"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "UPDATE TASK LOCATION v$ScriptVersion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Updating scheduled task location..." -ForegroundColor Cyan
Write-Host "New path: $NewScriptPath" -ForegroundColor Yellow
Write-Host ""

# Remove old task
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Old task removed." -ForegroundColor Green
}

# Get time preference
do {
    $hourInput = Read-Host "Enter hour (0-23), default is 9 for 9 AM"
    if ([string]::IsNullOrWhiteSpace($hourInput)) {
        $hour = 9
        break
    }
    if ($hourInput -match '^\d+$' -and [int]$hourInput -ge 0 -and [int]$hourInput -le 23) {
        $hour = [int]$hourInput
        break
    }
    Write-Host "Invalid hour. Please enter a number between 0 and 23." -ForegroundColor Red
} while ($true)

do {
    $minuteInput = Read-Host "Enter minute (0-59), default is 0"
    if ([string]::IsNullOrWhiteSpace($minuteInput)) {
        $minute = 0
        break
    }
    if ($minuteInput -match '^\d+$' -and [int]$minuteInput -ge 0 -and [int]$minuteInput -le 59) {
        $minute = [int]$minuteInput
        break
    }
    Write-Host "Invalid minute. Please enter a number between 0 and 59." -ForegroundColor Red
} while ($true)

$timeString = "{0:D2}:{1:D2}" -f $hour, $minute

# Create new task with correct location
$actionArgs = "-ExecutionPolicy Bypass -WindowStyle Normal -File `"$NewScriptPath`""
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument $actionArgs
$trigger = New-ScheduledTaskTrigger -Daily -At $timeString
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Daily system update checker" | Out-Null

Write-Host ""
Write-Host "Task updated successfully!" -ForegroundColor Green
Write-Host "Location: $NewScriptPath" -ForegroundColor Cyan
Write-Host "Schedule: Daily at $timeString" -ForegroundColor Cyan
Write-Host ""
Write-Host "Logs will appear on your Desktop: UpdateCheck_yyyy-MM-dd.log" -ForegroundColor Yellow
Write-Host ""
