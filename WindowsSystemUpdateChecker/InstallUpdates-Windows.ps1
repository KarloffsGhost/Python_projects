$ErrorActionPreference = "Stop"
$ScriptVersion = "1.1.0"

function Get-InstalledDriverForUpdate {
    <#
    .SYNOPSIS
        Finds the installed driver that an offered driver update would replace.
    .DESCRIPTION
        Matched on hardware ID. The update reports a base ID such as
        "pci\ven_8086&dev_7d67&subsys_88ef1043" while the installed driver
        reports the same ID with a revision suffix, so this matches on prefix.
        Returns the newest match, or $null when the device is not present.
    #>
    param(
        [string]$HardwareId,
        [array]$InstalledDrivers
    )

    if (-not $HardwareId) { return $null }

    # Not named $matches - that is a PowerShell automatic variable set by -match.
    $matchingDrivers = $InstalledDrivers | Where-Object {
        $_.HardWareID -and $_.HardWareID.StartsWith($HardwareId, [StringComparison]::OrdinalIgnoreCase)
    }

    return $matchingDrivers | Sort-Object DriverDate -Descending | Select-Object -First 1
}

function Test-DriverUpdateSuperseded {
    <#
    .SYNOPSIS
        Decides whether an offered driver update is older than what is installed.
    .DESCRIPTION
        Windows Update routinely offers drivers that are older than the vendor
        driver already on the machine. This replaces a hardcoded title match on
        one specific Intel version, which only stayed correct by coincidence and
        would have silently skipped a genuinely newer driver of the same name.

        Version is compared first where both sides parse, since a vendor can
        ship a higher version with an older signing date. Otherwise the driver
        date decides.
    .OUTPUTS
        Hashtable: Superseded (bool), Reason (string)
    #>
    param(
        $Update,
        [array]$InstalledDrivers
    )

    $result = @{ Superseded = $false; Reason = "" }

    $hardwareId = $null
    $offeredDate = $null
    try { $hardwareId = $Update.DriverHardwareID } catch { }
    try { $offeredDate = $Update.DriverVerDate } catch { }

    if (-not $hardwareId) { return $result }

    $installed = Get-InstalledDriverForUpdate -HardwareId $hardwareId -InstalledDrivers $InstalledDrivers
    if (-not $installed) { return $result }

    # The offered version is the last dotted-numeric token of the update title,
    # e.g. "Vendor Name - Display - 20.0.100.4000".
    $offeredVersion = $null
    if ($Update.Title -match '(\d+(?:\.\d+){2,})\s*$') {
        [version]::TryParse($Matches[1], [ref]$offeredVersion) | Out-Null
    }

    $installedVersion = $null
    if ($installed.DriverVersion) {
        [version]::TryParse($installed.DriverVersion, [ref]$installedVersion) | Out-Null
    }

    if ($offeredVersion -and $installedVersion) {
        if ($installedVersion -ge $offeredVersion) {
            $result.Superseded = $true
            $result.Reason = "installed v$installedVersion is newer than offered v$offeredVersion"
        }
        return $result
    }

    if ($offeredDate -and $installed.DriverDate -and $installed.DriverDate -gt $offeredDate) {
        $result.Superseded = $true
        $result.Reason = "installed driver dated $($installed.DriverDate.ToString('yyyy-MM-dd')) is newer than offered $($offeredDate.ToString('yyyy-MM-dd'))"
    }

    return $result
}

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
$supersededUpdates = @()

$installedDrivers = @(Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
                      Select-Object DeviceName, DriverVersion, DriverDate, HardWareID)

foreach ($update in $searchResult.Updates) {
    $supersedeCheck = Test-DriverUpdateSuperseded -Update $update -InstalledDrivers $installedDrivers

    if ($supersedeCheck.Superseded) {
        Write-Host "  [SKIP] $($update.Title)" -ForegroundColor Gray
        Write-Host "         $($supersedeCheck.Reason)" -ForegroundColor Gray
        $supersededUpdates += $update
        continue
    }

    Write-Host "  [INSTALL] $($update.Title)" -ForegroundColor Green
    $updatesToInstall.Add($update) | Out-Null
}

# Superseded drivers are re-offered on every scan and keep the daily report
# showing a pending update forever. Hiding them stops that, and is reversible
# from Windows Update's "Restore hidden updates".
if ($supersededUpdates.Count -gt 0) {
    Write-Host ""
    Write-Host "$($supersededUpdates.Count) update(s) were skipped as older than what you have installed." -ForegroundColor Yellow
    Write-Host "Windows will keep offering them on every scan unless they are hidden." -ForegroundColor Gray
    $hideChoice = Read-Host "Hide these updates? (Y/N)"

    if ($hideChoice.ToUpper() -eq "Y") {
        foreach ($update in $supersededUpdates) {
            try {
                $update.IsHidden = $true
                Write-Host "  [HIDDEN] $($update.Title)" -ForegroundColor Gray
            }
            catch {
                Write-Host "  [FAILED] Could not hide $($update.Title): $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
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
