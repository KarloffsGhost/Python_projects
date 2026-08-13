$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\SystemMaintenance_$(Get-Date -Format 'yyyy-MM-dd').log"

# Import shared library
Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force

# ============================================
# QUICK SCAN FUNCTIONS
# ============================================

function Get-QuickSystemStatus {
    $status = @{
        WindowsUpdates = 0
        DriverUpdates = 0
        AppUpdates = 0
        CriticalUpdates = 0
        SystemCleanable = 0
        AICleanable = 0
        StartupItems = 0
        DriverIssues = 0
    }

    # Windows Updates (quick check)
    Write-Host "`r  Checking Windows updates...     " -NoNewline -ForegroundColor Gray
    try {
        $updateSession = New-Object -ComObject Microsoft.Update.Session
        $updateSearcher = $updateSession.CreateUpdateSearcher()
        $searchResult = $updateSearcher.Search("IsInstalled=0")
        $status.WindowsUpdates = $searchResult.Updates.Count

        foreach ($update in $searchResult.Updates) {
            if ($update.MsrcSeverity -eq "Critical") {
                $status.CriticalUpdates++
            }
        }
    }
    catch {}

    # App Updates (quick check via winget)
    Write-Host "`r  Checking app updates...         " -NoNewline -ForegroundColor Gray
    try {
        $wingetPath = Get-Command winget -ErrorAction SilentlyContinue
        if ($wingetPath) {
            $output = winget upgrade 2>&1 | Out-String
            $lines = $output -split "`n" | Where-Object { $_ -match "^\S+\s+\S+\s+[\d\.]+\s+[\d\.]+\s+" }
            $status.AppUpdates = $lines.Count
        }
    }
    catch {}

    # System cleanable (estimate)
    Write-Host "`r  Scanning cleanable space...     " -NoNewline -ForegroundColor Gray
    try {
        if (Test-Path $env:TEMP) { $status.SystemCleanable += Get-FolderSize -Path $env:TEMP }
        if (Test-Path "$env:WINDIR\Temp") { $status.SystemCleanable += Get-FolderSize -Path "$env:WINDIR\Temp" }
        if (Test-Path "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache") {
            $status.SystemCleanable += Get-FolderSize -Path "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache"
        }
    }
    catch {}

    # AI/ML cleanable (estimate)
    Write-Host "`r  Scanning AI toolchain...        " -NoNewline -ForegroundColor Gray
    try {
        $aiPaths = @(
            "$env:LOCALAPPDATA\pip\cache",
            "$env:APPDATA\npm-cache",
            "$env:LOCALAPPDATA\NVIDIA\DXCache"
        )
        foreach ($path in $aiPaths) {
            if (Test-Path $path) { $status.AICleanable += Get-FolderSize -Path $path }
        }
    }
    catch {}

    # Startup items
    Write-Host "`r  Checking startup items...       " -NoNewline -ForegroundColor Gray
    try {
        $runPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
        if (Test-Path $runPath) {
            $key = Get-Item $runPath -ErrorAction SilentlyContinue
            if ($key) { $status.StartupItems = $key.GetValueNames().Count }
        }
    }
    catch {}

    Write-Host "`r                                  `r" -NoNewline

    return $status
}

# ============================================
# MENU DISPLAY
# ============================================

function Show-MainMenu {
    param([hashtable]$Status)

    Clear-Host

    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor Cyan
    Write-Host "    SYSTEM MAINTENANCE TOOLKIT v$ScriptVersion" -ForegroundColor Cyan
    Write-Host "  ========================================" -ForegroundColor Cyan
    Write-Host ""

    # Quick status line
    $statusParts = @()
    if ($Status.CriticalUpdates -gt 0) {
        $statusParts += "$($Status.CriticalUpdates) CRITICAL"
    }
    $totalUpdates = $Status.WindowsUpdates + $Status.AppUpdates
    if ($totalUpdates -gt 0) {
        $statusParts += "$totalUpdates updates"
    }
    $totalCleanable = $Status.SystemCleanable + $Status.AICleanable
    if ($totalCleanable -gt 100MB) {
        $statusParts += "$(Format-FileSize $totalCleanable) cleanable"
    }

    if ($Status.CriticalUpdates -gt 0) {
        Write-Host "  Status: " -NoNewline -ForegroundColor White
        Write-Host ($statusParts -join " | ") -ForegroundColor Red
    }
    elseif ($statusParts.Count -gt 0) {
        Write-Host "  Status: " -NoNewline -ForegroundColor White
        Write-Host ($statusParts -join " | ") -ForegroundColor Yellow
    }
    else {
        Write-Host "  Status: " -NoNewline -ForegroundColor White
        Write-Host "System looks good!" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "  SCAN / CHECK (read-only)" -ForegroundColor Gray
    Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [1] Scan for Updates" -NoNewline -ForegroundColor White
    Write-Host " (Windows + Drivers + Apps)" -ForegroundColor DarkGray
    Write-Host "  [2] Scan for Cleanable Items" -NoNewline -ForegroundColor White
    Write-Host " (System + AI/ML)" -ForegroundColor DarkGray
    Write-Host "  [3] Scan Everything" -NoNewline -ForegroundColor White
    Write-Host " (Full System Report)" -ForegroundColor DarkGray
    Write-Host ""

    Write-Host "  UPDATE (install updates)" -ForegroundColor Gray
    Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [4] Update Windows & Drivers" -ForegroundColor White
    Write-Host "  [5] Update Applications" -NoNewline -ForegroundColor White
    Write-Host " (Interactive)" -ForegroundColor DarkGray
    Write-Host "  [6] Update ALL" -NoNewline -ForegroundColor White
    Write-Host " (Windows + Drivers + Apps)" -ForegroundColor DarkGray
    Write-Host ""

    Write-Host "  CLEAN (remove junk)" -ForegroundColor Gray
    Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [7] Clean System" -NoNewline -ForegroundColor White
    Write-Host " (Temp, Cache, Browser)" -ForegroundColor DarkGray
    Write-Host "  [8] Clean AI/ML Toolchain" -NoNewline -ForegroundColor White
    Write-Host " (Ollama, Docker, pip, npm)" -ForegroundColor DarkGray
    Write-Host "  [9] Clean ALL" -NoNewline -ForegroundColor White
    Write-Host " (System + AI/ML)" -ForegroundColor DarkGray
    Write-Host ""

    Write-Host "  FULL MAINTENANCE" -ForegroundColor Gray
    Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [F] FULL MAINTENANCE" -NoNewline -ForegroundColor Yellow
    Write-Host " (Update ALL + Clean ALL)" -ForegroundColor DarkGray
    Write-Host ""

    Write-Host "  TOOLS" -ForegroundColor Gray
    Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [D] Driver Backup/Restore" -ForegroundColor White
    Write-Host "  [S] Startup Manager" -ForegroundColor White
    Write-Host "  [R] Generate HTML Report" -ForegroundColor White
    Write-Host "  [W] Open Web Dashboard" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "  [Q] Quit" -ForegroundColor DarkGray
    Write-Host ""

    return (Read-Host "  Enter choice").ToUpper()
}

function Show-SectionHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "  ========================================" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================
# ACTION FUNCTIONS
# ============================================

function Invoke-UpdateScan {
    Show-SectionHeader "SCANNING FOR UPDATES"

    # Run the existing DailyUpdateChecker
    $checkerPath = Join-Path $ScriptRoot "DailyUpdateChecker.ps1"
    if (Test-Path $checkerPath) {
        & $checkerPath
    }
    else {
        Write-Host "  DailyUpdateChecker.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-CleanerScan {
    Show-SectionHeader "SCANNING FOR CLEANABLE ITEMS"

    # Run SystemCleaner in scan mode
    $cleanerPath = Join-Path $ScriptRoot "SystemCleaner.ps1"
    if (Test-Path $cleanerPath) {
        & $cleanerPath -scan
    }

    Write-Host ""

    # Run AIToolchainCleaner in scan mode
    $aiCleanerPath = Join-Path $ScriptRoot "AIToolchainCleaner.ps1"
    if (Test-Path $aiCleanerPath) {
        & $aiCleanerPath -scan
    }
}

function Invoke-FullScan {
    Show-SectionHeader "GENERATING FULL SYSTEM REPORT"

    $reportPath = Join-Path $ScriptRoot "SystemReport.ps1"
    if (Test-Path $reportPath) {
        & $reportPath -generate
    }
    else {
        Write-Host "  SystemReport.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-WindowsUpdate {
    Show-SectionHeader "UPDATING WINDOWS & DRIVERS"

    $updatePath = Join-Path $ScriptRoot "InstallUpdates-Windows.ps1"
    if (Test-Path $updatePath) {
        # This requires admin, so we'll launch elevated
        Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -File `"$updatePath`"" -Verb RunAs -Wait
    }
    else {
        Write-Host "  InstallUpdates-Windows.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-AppUpdate {
    Show-SectionHeader "UPDATING APPLICATIONS"

    $appUpdatePath = Join-Path $ScriptRoot "InstallUpdates-Interactive.ps1"
    if (Test-Path $appUpdatePath) {
        & $appUpdatePath
    }
    else {
        Write-Host "  InstallUpdates-Interactive.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-AllUpdates {
    Show-SectionHeader "UPDATING ALL (WINDOWS + DRIVERS + APPS)"

    Write-Host "  Step 1: Windows & Drivers" -ForegroundColor Yellow
    Invoke-WindowsUpdate

    Write-Host ""
    Write-Host "  Step 2: Applications" -ForegroundColor Yellow
    Invoke-AppUpdate
}

function Invoke-SystemClean {
    Show-SectionHeader "CLEANING SYSTEM"

    $cleanerPath = Join-Path $ScriptRoot "SystemCleaner.ps1"
    if (Test-Path $cleanerPath) {
        & $cleanerPath
    }
    else {
        Write-Host "  SystemCleaner.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-AIClean {
    Show-SectionHeader "CLEANING AI/ML TOOLCHAIN"

    $aiCleanerPath = Join-Path $ScriptRoot "AIToolchainCleaner.ps1"
    if (Test-Path $aiCleanerPath) {
        & $aiCleanerPath
    }
    else {
        Write-Host "  AIToolchainCleaner.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-AllClean {
    Show-SectionHeader "CLEANING ALL (SYSTEM + AI/ML)"

    Write-Host "  Step 1: System Cleanup" -ForegroundColor Yellow
    $cleanerPath = Join-Path $ScriptRoot "SystemCleaner.ps1"
    if (Test-Path $cleanerPath) {
        & $cleanerPath -safe
    }

    Write-Host ""
    Write-Host "  Step 2: AI/ML Cleanup" -ForegroundColor Yellow
    $aiCleanerPath = Join-Path $ScriptRoot "AIToolchainCleaner.ps1"
    if (Test-Path $aiCleanerPath) {
        & $aiCleanerPath -safe
    }
}

function Invoke-FullMaintenance {
    Show-SectionHeader "FULL MAINTENANCE"

    Write-Host "  This will:" -ForegroundColor Yellow
    Write-Host "    1. Install all Windows & Driver updates" -ForegroundColor White
    Write-Host "    2. Install all application updates" -ForegroundColor White
    Write-Host "    3. Clean system temp files and caches" -ForegroundColor White
    Write-Host "    4. Clean AI/ML toolchain caches" -ForegroundColor White
    Write-Host ""

    $confirm = Read-Host "  Proceed with full maintenance? (Y/N)"
    if ($confirm.ToUpper() -ne "Y") {
        Write-Host "  Cancelled." -ForegroundColor Gray
        return
    }

    Write-Host ""
    Write-Host "  [1/4] Updating Windows & Drivers..." -ForegroundColor Yellow
    Invoke-WindowsUpdate

    Write-Host ""
    Write-Host "  [2/4] Updating Applications..." -ForegroundColor Yellow
    Invoke-AppUpdate

    Write-Host ""
    Write-Host "  [3/4] Cleaning System..." -ForegroundColor Yellow
    $cleanerPath = Join-Path $ScriptRoot "SystemCleaner.ps1"
    if (Test-Path $cleanerPath) {
        & $cleanerPath -safe
    }

    Write-Host ""
    Write-Host "  [4/4] Cleaning AI/ML Toolchain..." -ForegroundColor Yellow
    $aiCleanerPath = Join-Path $ScriptRoot "AIToolchainCleaner.ps1"
    if (Test-Path $aiCleanerPath) {
        & $aiCleanerPath -safe
    }

    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor Green
    Write-Host "  FULL MAINTENANCE COMPLETE" -ForegroundColor Green
    Write-Host "  ========================================" -ForegroundColor Green

    # Send notification
    Show-ToastNotification -Title "Maintenance Complete" -Message "Full system maintenance finished successfully" -Type "Success" | Out-Null
}

function Invoke-DriverBackup {
    Show-SectionHeader "DRIVER BACKUP/RESTORE"

    $driverPath = Join-Path $ScriptRoot "DriverBackup.ps1"
    if (Test-Path $driverPath) {
        # Launch elevated
        Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -File `"$driverPath`"" -Verb RunAs -Wait
    }
    else {
        Write-Host "  DriverBackup.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-StartupManager {
    Show-SectionHeader "STARTUP MANAGER"

    $startupPath = Join-Path $ScriptRoot "StartupManager.ps1"
    if (Test-Path $startupPath) {
        & $startupPath
    }
    else {
        Write-Host "  StartupManager.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-GenerateReport {
    Show-SectionHeader "GENERATING HTML REPORT"

    $reportPath = Join-Path $ScriptRoot "SystemReport.ps1"
    if (Test-Path $reportPath) {
        & $reportPath -generate
    }
    else {
        Write-Host "  SystemReport.ps1 not found" -ForegroundColor Red
    }
}

function Invoke-WebDashboard {
    $dashboardScript = Join-Path $ScriptRoot "Dashboard\Start-Dashboard.ps1"
    if (Test-Path $dashboardScript) {
        & $dashboardScript
    }
    else {
        Write-Host ""
        Write-Host "  Web Dashboard not found at: $dashboardScript" -ForegroundColor Red
        Write-Host "  Make sure Dashboard\Start-Dashboard.ps1 exists." -ForegroundColor Yellow
    }
}

# ============================================
# MAIN EXECUTION
# ============================================

# Check for command line arguments
$QuickMode = $null
foreach ($arg in $args) {
    switch ($arg.ToLower()) {
        "-scan" { $QuickMode = "Scan" }
        "-update" { $QuickMode = "Update" }
        "-clean" { $QuickMode = "Clean" }
        "-full" { $QuickMode = "Full" }
        "-report" { $QuickMode = "Report" }
        "-dashboard" { $QuickMode = "Dashboard" }
    }
}

# Handle quick mode
if ($QuickMode) {
    switch ($QuickMode) {
        "Scan" { Invoke-UpdateScan }
        "Update" { Invoke-AllUpdates }
        "Clean" { Invoke-AllClean }
        "Full" { Invoke-FullMaintenance }
        "Report" { Invoke-GenerateReport }
        "Dashboard" { Invoke-WebDashboard }
    }
    exit
}

# Interactive mode
$continue = $true

while ($continue) {
    # Quick status check
    Write-Host ""
    Write-Host "  Loading system status..." -ForegroundColor Gray
    $status = Get-QuickSystemStatus

    $choice = Show-MainMenu -Status $status

    switch ($choice) {
        "1" { Invoke-UpdateScan; pause }
        "2" { Invoke-CleanerScan; pause }
        "3" { Invoke-FullScan; pause }
        "4" { Invoke-WindowsUpdate; pause }
        "5" { Invoke-AppUpdate; pause }
        "6" { Invoke-AllUpdates; pause }
        "7" { Invoke-SystemClean; pause }
        "8" { Invoke-AIClean; pause }
        "9" { Invoke-AllClean; pause }
        "F" { Invoke-FullMaintenance; pause }
        "D" { Invoke-DriverBackup }
        "S" { Invoke-StartupManager; pause }
        "R" { Invoke-GenerateReport; pause }
        "W" { Invoke-WebDashboard }
        "Q" { $continue = $false }
        default {
            Write-Host ""
            Write-Host "  Invalid choice. Press any key to continue..." -ForegroundColor Red
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
    }
}

Write-Host ""
Write-Host "  Goodbye!" -ForegroundColor Cyan
Write-Host ""
