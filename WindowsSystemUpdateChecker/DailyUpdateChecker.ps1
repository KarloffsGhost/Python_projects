$ErrorActionPreference = "Continue"
$ScriptVersion = "1.1.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\UpdateCheck_$(Get-Date -Format 'yyyy-MM-dd').log"

Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force -ErrorAction Stop

# Remove previous days' log files. Today's is excluded so that a second run on
# the same day appends to the existing log instead of destroying the first run's.
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "UpdateCheck_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne (Split-Path $LogFile -Leaf) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Deliberately overrides the module's Write-Log with a two-argument form bound to
# this script's log file. Defined after the import so precedence is unambiguous.
function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $logMessage -Encoding UTF8
}

Write-Log "========================================" "Cyan"
Write-Log "DAILY SYSTEM UPDATE CHECKER v$ScriptVersion" "Cyan"
Write-Log "$(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm')" "Cyan"
Write-Log "========================================" "Cyan"

$summary = @{
    WindowsUpdates = 0
    DriverUpdates = 0
    AppUpdates = 0
    CriticalUpdates = 0
}

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Log "WARNING: Not running as Administrator. Some checks may be limited." "Yellow"
}

# CHECK WINDOWS & DRIVER UPDATES
#
# A single search. "IsInstalled=0" already returns driver updates alongside
# everything else, so searching again for Type='Driver' and adding the result to
# the total counted every driver update twice - one pending Intel driver was
# being reported as "2 updates available".
Write-Log "" "White"
Write-Log "[1] CHECKING WINDOWS & DRIVER UPDATES..." "Cyan"
Write-Log "----------------------------------------" "Gray"

$windowsUpdateTitles = @()
$driverUpdateTitles = @()

try {
    $updateSession = New-Object -ComObject Microsoft.Update.Session
    $updateSearcher = $updateSession.CreateUpdateSearcher()

    Write-Log "Searching for Windows updates..." "Yellow"
    $searchResult = $updateSearcher.Search("IsInstalled=0")

    foreach ($update in $searchResult.Updates) {
        $isDriver = $false
        foreach ($category in $update.Categories) {
            if ($category.Name -match "Driver") {
                $isDriver = $true
                break
            }
        }

        $importance = ""
        if ($update.MsrcSeverity -eq "Critical") {
            $importance = "[CRITICAL] "
            $summary.CriticalUpdates++
        }

        if ($isDriver) {
            $driverUpdateTitles += "$importance$($update.Title)"
            $summary.DriverUpdates++
        }
        else {
            $windowsUpdateTitles += "$importance$($update.Title)"
            $summary.WindowsUpdates++
        }
    }
}
catch {
    Write-Log "ERROR: Windows Update search failed: $($_.Exception.Message)" "Red"
    Write-Log "  (Windows Update service may be stopped, or a scan is already running)" "Gray"
}

if ($windowsUpdateTitles.Count -eq 0) {
    Write-Log "Windows is up to date!" "Green"
} else {
    Write-Log "Found $($windowsUpdateTitles.Count) Windows update(s):" "Yellow"
    foreach ($title in $windowsUpdateTitles) {
        Write-Log "  $title" "White"
    }
}

Write-Log "" "White"
if ($driverUpdateTitles.Count -eq 0) {
    Write-Log "No driver updates offered by Windows Update" "Green"
} else {
    Write-Log "Found $($driverUpdateTitles.Count) driver update(s):" "Yellow"
    foreach ($title in $driverUpdateTitles) {
        Write-Log "  $title" "White"
    }
}

# CHECK DRIVER AGES
Write-Log "" "White"
Write-Log "[2] CHECKING DRIVER AGES..." "Cyan"
Write-Log "----------------------------------------" "Gray"
$oldDriverCount = 0
$cutoffDate = (Get-Date).AddMonths(-6)

# Microsoft's inbox drivers are excluded. They carry a placeholder date of
# 2006-06-21 that never changes, so they always look "old" - this listed 40+
# USB/Bluetooth entries the report itself then told you to ignore.
#
# Get-CimInstance returns DriverDate already typed as [DateTime], and unlike
# Get-WmiObject it exists in PowerShell 7 without the WinPS compatibility shim.
$drivers = Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue | Where-Object {
    $_.DriverDate -and
    $_.DeviceName -match "Graphics|Network|Audio|Bluetooth|USB|Chipset" -and
    $_.DriverProviderName -notmatch "^Microsoft"
} | Select-Object DeviceName, DriverDate, DriverProviderName, DriverVersion

foreach ($driver in $drivers) {
    if ($driver.DriverDate -lt $cutoffDate) {
        if ($oldDriverCount -eq 0) {
            Write-Log "Drivers older than 6 months:" "Yellow"
        }
        Write-Log ("  {0} - {1} (v{2}, {3})" -f $driver.DeviceName,
                                                $driver.DriverDate.ToString('yyyy-MM-dd'),
                                                $driver.DriverVersion,
                                                $driver.DriverProviderName) "Yellow"
        $oldDriverCount++
    }
}

if ($oldDriverCount -eq 0) {
    Write-Log "All third-party drivers are less than 6 months old" "Green"
}

# CHECK APPLICATION UPDATES
Write-Log "" "White"
Write-Log "[3] CHECKING APPLICATION UPDATES (via winget)..." "Cyan"
Write-Log "----------------------------------------" "Gray"

Write-Log "Checking for application updates..." "Yellow"
$wingetResult = Get-WingetUpgrades -IncludeUnknown

if ($wingetResult.Error) {
    Write-Log $wingetResult.Error "Yellow"
}
elseif ($wingetResult.Count -eq 0) {
    Write-Log "All applications are up to date!" "Green"
}
else {
    # Count every upgrade, but only print the first 10. The counter used to live
    # inside the display loop, which capped the reported total at 10 forever.
    $summary.AppUpdates = $wingetResult.Count

    Write-Log "Found $($wingetResult.Count) application update(s):" "Yellow"

    $displayLimit = 10
    foreach ($package in $wingetResult.Packages | Select-Object -First $displayLimit) {
        Write-Log ("  {0}  ({1} -> {2})" -f $package.Name, $package.CurrentVersion, $package.NewVersion) "White"
    }

    if ($wingetResult.Packages.Count -gt $displayLimit) {
        Write-Log "  ... and $($wingetResult.Packages.Count - $displayLimit) more" "Gray"
    }
}

# SYSTEM HEALTH CHECK
Write-Log "" "White"
Write-Log "[4] SYSTEM HEALTH CHECK..." "Cyan"
Write-Log "----------------------------------------" "Gray"

$problemDevices = @(Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object {$_.ConfigManagerErrorCode -ne 0})

if ($problemDevices) {
    Write-Log "Found $($problemDevices.Count) device(s) with errors:" "Yellow"
    foreach ($device in $problemDevices | Select-Object -First 5) {
        Write-Log "  $($device.Name) - Error Code: $($device.ConfigManagerErrorCode)" "Yellow"
    }
} else {
    Write-Log "No device errors detected" "Green"
}

# Antivirus status.
#
# This used to read Get-MpComputerStatus().AntivirusSignatureLastUpdated and
# subtract it from the current date. When a third-party AV is active, Defender
# leaves that property null, the subtraction threw, and $null.TotalDays -lt 2
# evaluates to $true - so the report printed "signatures up to date" every single
# day regardless of the actual state. Security Center is asked instead, because
# it knows which product is actually protecting the machine.
Write-Log "" "White"
$avProducts = @(Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct -ErrorAction SilentlyContinue)

if ($avProducts.Count -eq 0) {
    Write-Log "WARNING: No antivirus product is registered with Windows Security" "Red"
}
else {
    foreach ($av in $avProducts) {
        # productState packs two flags: byte 1 is the scanner state (0x10/0x11
        # mean enabled) and byte 0 is the signature state (0x00 means current).
        $scannerEnabled = ((($av.productState -shr 8) -band 0xFF) -in @(0x10, 0x11))
        $signaturesCurrent = (($av.productState -band 0xFF) -eq 0x00)

        if ($scannerEnabled -and $signaturesCurrent) {
            Write-Log "$($av.displayName): active, signatures up to date" "Green"
        }
        elseif ($scannerEnabled) {
            Write-Log "$($av.displayName): active, SIGNATURES OUT OF DATE" "Yellow"
        }
        else {
            Write-Log "$($av.displayName): not active (disabled or superseded)" "Gray"
        }
    }

    if (-not ($avProducts | Where-Object { ((($_.productState -shr 8) -band 0xFF) -in @(0x10, 0x11)) })) {
        Write-Log "WARNING: No antivirus product is currently active on this machine" "Red"
    }
}

# SUMMARY
Write-Log "" "White"
Write-Log "========================================" "Cyan"
Write-Log "UPDATE SUMMARY" "Cyan"
Write-Log "========================================" "Cyan"
$totalUpdates = $summary.WindowsUpdates + $summary.DriverUpdates + $summary.AppUpdates
Write-Log "Windows Updates:     $($summary.WindowsUpdates)" "White"
Write-Log "Driver Updates:      $($summary.DriverUpdates)" "White"
Write-Log "Application Updates: $($summary.AppUpdates)" "White"
Write-Log "----------------------------------------" "Gray"
Write-Log "TOTAL:               $totalUpdates" "White"
Write-Log "  of which critical: $($summary.CriticalUpdates)" "Gray"
Write-Log "========================================" "Cyan"

Write-Log "" "White"
Write-Log "========================================" "Cyan"
Write-Log "WHAT YOU SHOULD ACTUALLY DO" "Cyan"
Write-Log "========================================" "Cyan"

# Analyze Windows/Driver updates
if ($summary.WindowsUpdates -gt 0 -or $summary.DriverUpdates -gt 0) {
    Write-Log "" "White"
    Write-Log "WINDOWS/DRIVER UPDATES:" "Yellow"
    if ($summary.CriticalUpdates -gt 0) {
        Write-Log "  You have $($summary.CriticalUpdates) CRITICAL update(s)" "Red"
        Write-Log "  -> INSTALL THESE - Security important" "Yellow"
    } else {
        Write-Log "  You have $($summary.WindowsUpdates + $summary.DriverUpdates) update(s) available" "Cyan"
        Write-Log "  -> Review the list above for specific recommendations" "White"
    }
    Write-Log "" "White"
    Write-Log "  Note: Windows Update sometimes offers a driver older than the one" "Gray"
    Write-Log "  you already have. InstallUpdates-Windows.ps1 checks versions and" "Gray"
    Write-Log "  skips those automatically." "Gray"
}

# Applications
if ($summary.AppUpdates -gt 0) {
    Write-Log "" "White"
    Write-Log "APPLICATIONS TO UPDATE:" "Yellow"
    Write-Log "  Found $($summary.AppUpdates) application update(s)" "Cyan"
    Write-Log "" "White"
    Write-Log "  Priority recommendations:" "White"
    Write-Log "    - Security/Browsers (Chrome, Firefox, Edge): RECOMMENDED" "Yellow"
    Write-Log "    - Development tools: Update if you use them" "Cyan"
    Write-Log "    - Other apps: Update when convenient" "Gray"
}

# Device errors
if ($problemDevices -and $problemDevices.Count -gt 0) {
    Write-Log "" "White"
    Write-Log "DEVICE ERRORS:" "Yellow"
    Write-Log "  Found $($problemDevices.Count) device(s) with errors" "Yellow"
    Write-Log "" "White"
    Write-Log "  Common error codes:" "Gray"
    Write-Log "    - Error 22: Device is disabled (not an actual error)" "Gray"
    Write-Log "    - Error 52: Unsigned driver (usually antivirus drivers)" "Gray"
    Write-Log "  -> Review the errors above. If devices work fine, ignore them" "Green"
}

Write-Log "" "White"
Write-Log "========================================" "Cyan"
Write-Log "BOTTOM LINE" "Cyan"
Write-Log "========================================" "Cyan"

if ($summary.AppUpdates -gt 0) {
    Write-Log "" "White"
    Write-Log "WHAT TO DO NOW:" "Yellow"
    Write-Log "  1. Run: InstallUpdates-Interactive.bat" "Cyan"
    Write-Log "  2. Choose [S] for Security apps (Chrome, VPN)" "White"
    Write-Log "  3. Or choose [C] to pick specific apps" "White"
    Write-Log "" "White"
    Write-Log "Everything else can be ignored!" "Green"
} else {
    Write-Log "" "White"
    Write-Log "YOUR SYSTEM IS GOOD!" "Green"
    Write-Log "  No important updates needed" "White"
}

Write-Log "" "White"
Write-Log "========================================" "Cyan"
Write-Log "Log saved to: $LogFile" "Gray"
Write-Log "Scan completed at $(Get-Date -Format 'HH:mm:ss')" "Gray"
Write-Log "========================================" "Cyan"
