$ErrorActionPreference = "Stop"
$ScriptVersion = "1.0.0"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click and select 'Run with PowerShell as Administrator'" -ForegroundColor Yellow
    pause
    exit
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  WINDOWS & DRIVER UPDATE INSTALLER v$ScriptVersion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Searching for updates..." -ForegroundColor Yellow

$updateSession = New-Object -ComObject Microsoft.Update.Session
$updateSearcher = $updateSession.CreateUpdateSearcher()
$searchResult = $updateSearcher.Search("IsInstalled=0")

if ($searchResult.Updates.Count -eq 0) {
    Write-Host "No updates found!" -ForegroundColor Green
    pause
    exit
}

Write-Host "Found $($searchResult.Updates.Count) update(s):" -ForegroundColor Yellow
Write-Host ""

$updatesToInstall = New-Object -ComObject Microsoft.Update.UpdateColl

foreach ($update in $searchResult.Updates) {
    $install = "Yes"

    # Skip drivers where user already has a newer version installed
    # You can customize this section for your specific hardware
    # Example: if ($update.Title -match "Specific.Driver.Name") { ... }
    
    Write-Host "  [INSTALL] $($update.Title)" -ForegroundColor Green
    $updatesToInstall.Add($update) | Out-Null
}

if ($updatesToInstall.Count -eq 0) {
    Write-Host ""
    Write-Host "No updates need to be installed." -ForegroundColor Yellow
    pause
    exit
}

Write-Host ""
Write-Host "Installing $($updatesToInstall.Count) update(s)..." -ForegroundColor Cyan

# Download
Write-Host "Downloading..." -ForegroundColor Yellow
$downloader = $updateSession.CreateUpdateDownloader()
$downloader.Updates = $updatesToInstall
$downloadResult = $downloader.Download()

if ($downloadResult.ResultCode -eq 2) {
    Write-Host "Download complete!" -ForegroundColor Green

    # Install
    Write-Host "Installing..." -ForegroundColor Yellow
    $installer = $updateSession.CreateUpdateInstaller()
    $installer.Updates = $updatesToInstall
    $installResult = $installer.Install()

    if ($installResult.ResultCode -eq 2) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  INSTALLATION SUCCESSFUL!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green

        if ($installResult.RebootRequired) {
            Write-Host ""
            Write-Host "REBOOT REQUIRED to complete installation." -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "Installation failed with code: $($installResult.ResultCode)" -ForegroundColor Red
    }
} else {
    Write-Host "Download failed with code: $($downloadResult.ResultCode)" -ForegroundColor Red
}

Write-Host ""
pause
