$ErrorActionPreference = "Continue"
$ScriptVersion = "1.0.0"
$LogFile = "$env:USERPROFILE\Desktop\UpdateCheck_$(Get-Date -Format 'yyyy-MM-dd').log"

# Remove previous day's log files (keep only today's)
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "UpdateCheck_*.log" -ErrorAction SilentlyContinue | Remove-Item -Force

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $logMessage
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

# CHECK WINDOWS UPDATES
Write-Log "" "White"
Write-Log "[1] CHECKING WINDOWS UPDATES..." "Cyan"
Write-Log "----------------------------------------" "Gray"

$updateSession = New-Object -ComObject Microsoft.Update.Session
$updateSearcher = $updateSession.CreateUpdateSearcher()

Write-Log "Searching for Windows updates..." "Yellow"
$searchResult = $updateSearcher.Search("IsInstalled=0")

if ($searchResult.Updates.Count -eq 0) {
    Write-Log "Windows is up to date!" "Green"
} else {
    Write-Log "Found $($searchResult.Updates.Count) Windows update(s):" "Yellow"
    foreach ($update in $searchResult.Updates) {
        $importance = ""
        if ($update.MsrcSeverity -eq "Critical") {
            $importance = "[CRITICAL] "
            $summary.CriticalUpdates++
        }
        Write-Log "  $importance$($update.Title)" "White"
        $summary.WindowsUpdates++
    }
}

# CHECK DRIVER UPDATES
Write-Log "" "White"
Write-Log "[2] CHECKING DRIVER UPDATES..." "Cyan"
Write-Log "----------------------------------------" "Gray"

$driverSearchResult = $updateSearcher.Search("IsInstalled=0 and Type='Driver'")

if ($driverSearchResult.Updates.Count -eq 0) {
    Write-Log "All drivers are up to date!" "Green"
} else {
    Write-Log "Found $($driverSearchResult.Updates.Count) driver update(s):" "Yellow"
    foreach ($update in $driverSearchResult.Updates) {
        Write-Log "  $($update.Title)" "White"
        $summary.DriverUpdates++
    }
}

# Check for old drivers
Write-Log "" "White"
Write-Log "Checking driver ages..." "Yellow"
$oldDriverCount = 0
$cutoffDate = (Get-Date).AddMonths(-6)

$drivers = Get-WmiObject Win32_PnPSignedDriver | Where-Object {
    $_.DriverDate -and $_.DeviceName -match "Graphics|Network|Audio|Bluetooth|USB|Chipset"
} | Select-Object DeviceName, @{Name="DriverDate";Expression={[Management.ManagementDateTimeConverter]::ToDateTime($_.DriverDate)}}

foreach ($driver in $drivers) {
    if ($driver.DriverDate -lt $cutoffDate) {
        if ($oldDriverCount -eq 0) {
            Write-Log "Drivers older than 6 months:" "Yellow"
        }
        Write-Log "  $($driver.DeviceName) - $($driver.DriverDate.ToString('yyyy-MM-dd'))" "Yellow"
        $oldDriverCount++
    }
}

if ($oldDriverCount -eq 0) {
    Write-Log "All critical drivers are less than 6 months old" "Green"
}

# CHECK APPLICATION UPDATES
Write-Log "" "White"
Write-Log "[3] CHECKING APPLICATION UPDATES (via winget)..." "Cyan"
Write-Log "----------------------------------------" "Gray"

$wingetPath = Get-Command winget -ErrorAction SilentlyContinue

if ($wingetPath) {
    Write-Log "Checking for application updates..." "Yellow"
    $wingetOutput = winget upgrade 2>&1 | Out-String
    $upgradeLines = $wingetOutput -split "`n" | Where-Object { $_ -match "^\S+\s+\S+" } | Select-Object -Skip 2

    if ($upgradeLines.Count -gt 0) {
        Write-Log "Found application updates available:" "Yellow"
        foreach ($line in $upgradeLines | Select-Object -First 10) {
            if ($line.Trim()) {
                Write-Log "  $line" "White"
                $summary.AppUpdates++
            }
        }
        if ($upgradeLines.Count -gt 10) {
            Write-Log "  ... and $($upgradeLines.Count - 10) more" "Gray"
        }
    } else {
        Write-Log "All applications are up to date!" "Green"
    }
} else {
    Write-Log "winget not found. Install Windows Package Manager for app update checks." "Yellow"
}

# SYSTEM HEALTH CHECK
Write-Log "" "White"
Write-Log "[4] SYSTEM HEALTH CHECK..." "Cyan"
Write-Log "----------------------------------------" "Gray"

$problemDevices = Get-WmiObject Win32_PnPEntity | Where-Object {$_.ConfigManagerErrorCode -ne 0}

if ($problemDevices) {
    Write-Log "Found $($problemDevices.Count) device(s) with errors:" "Yellow"
    foreach ($device in $problemDevices | Select-Object -First 5) {
        Write-Log "  $($device.Name) - Error Code: $($device.ConfigManagerErrorCode)" "Yellow"
    }
} else {
    Write-Log "No device errors detected" "Green"
}

$defenderStatus = Get-MpComputerStatus -ErrorAction SilentlyContinue
if ($defenderStatus) {
    $defenderAge = (Get-Date) - $defenderStatus.AntivirusSignatureLastUpdated
    if ($defenderAge.TotalDays -lt 2) {
        Write-Log "Windows Defender signatures up to date" "Green"
    } else {
        Write-Log "Windows Defender signatures outdated" "Yellow"
    }
}

# SUMMARY
Write-Log "" "White"
Write-Log "========================================" "Cyan"
Write-Log "UPDATE SUMMARY" "Cyan"
Write-Log "========================================" "Cyan"
Write-Log "Windows Updates:     $($summary.WindowsUpdates)" "White"
Write-Log "Driver Updates:      $($summary.DriverUpdates)" "White"
Write-Log "Application Updates: $($summary.AppUpdates)" "White"
Write-Log "Critical Updates:    $($summary.CriticalUpdates)" "White"
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
    Write-Log "  Note: Old drivers (2006 dates) for USB/Bluetooth are usually" "Gray"
    Write-Log "  Windows built-in drivers and can be ignored unless you have issues" "Gray"
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
