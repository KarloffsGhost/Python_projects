$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\SystemReport_$(Get-Date -Format 'yyyy-MM-dd').log"
$ReportsPath = Join-Path $ScriptRoot "Reports"

# Import shared library
Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force

# ============================================
# SYSTEM INFO FUNCTIONS
# ============================================

function Get-SystemOverview {
    $info = @{
        ComputerName = $env:COMPUTERNAME
        UserName = $env:USERNAME
        OSVersion = ""
        OSBuild = ""
        Processor = ""
        RAM = ""
        SystemDrive = ""
        SystemDriveFree = ""
        SystemDriveUsed = ""
        Uptime = ""
    }

    try {
        # OS Info
        $os = Get-CimInstance Win32_OperatingSystem
        $info.OSVersion = $os.Caption
        $info.OSBuild = $os.BuildNumber

        # CPU
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        $info.Processor = $cpu.Name

        # RAM
        $totalRAM = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        $freeRAM = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
        $info.RAM = "$freeRAM GB free of $totalRAM GB"

        # System Drive
        $sysDrive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$env:SystemDrive'"
        $info.SystemDrive = $env:SystemDrive
        $info.SystemDriveFree = Format-FileSize $sysDrive.FreeSpace
        $info.SystemDriveUsed = Format-FileSize ($sysDrive.Size - $sysDrive.FreeSpace)
        $info.SystemDriveTotal = Format-FileSize $sysDrive.Size
        $info.SystemDrivePercentFree = [math]::Round(($sysDrive.FreeSpace / $sysDrive.Size) * 100, 1)

        # Uptime
        $uptime = (Get-Date) - $os.LastBootUpTime
        $info.Uptime = "$($uptime.Days)d $($uptime.Hours)h $($uptime.Minutes)m"
    }
    catch {
        # Silent fail for individual items
    }

    return $info
}

function Get-UpdateSummary {
    $summary = @{
        WindowsUpdates = 0
        DriverUpdates = 0
        AppUpdates = 0
        CriticalUpdates = 0
        LastChecked = Get-Date -Format "o"
    }

    try {
        # Windows Updates
        $updateSession = New-Object -ComObject Microsoft.Update.Session
        $updateSearcher = $updateSession.CreateUpdateSearcher()
        $searchResult = $updateSearcher.Search("IsInstalled=0")

        $summary.WindowsUpdates = $searchResult.Updates.Count

        foreach ($update in $searchResult.Updates) {
            if ($update.MsrcSeverity -eq "Critical") {
                $summary.CriticalUpdates++
            }
        }

        # Driver updates
        $driverResult = $updateSearcher.Search("IsInstalled=0 and Type='Driver'")
        $summary.DriverUpdates = $driverResult.Updates.Count
    }
    catch {}

    try {
        # App updates via winget
        $wingetPath = Get-Command winget -ErrorAction SilentlyContinue
        if ($wingetPath) {
            $wingetOutput = winget upgrade 2>&1 | Out-String
            $lines = $wingetOutput -split "`n" | Where-Object { $_ -match "^\S+\s+\S+\s+[\d\.]+\s+[\d\.]+\s+" }
            $summary.AppUpdates = $lines.Count
        }
    }
    catch {}

    return $summary
}

function Get-CleanerSummary {
    $summary = @{
        TempFiles = 0
        BrowserCache = 0
        RecycleBin = 0
        TotalReclaimable = 0
    }

    # User temp
    if (Test-Path $env:TEMP) {
        $summary.TempFiles += Get-FolderSize -Path $env:TEMP
    }

    # Windows temp
    if (Test-Path "$env:WINDIR\Temp") {
        $summary.TempFiles += Get-FolderSize -Path "$env:WINDIR\Temp"
    }

    # Browser caches
    $browserPaths = @(
        "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache",
        "$env:LOCALAPPDATA\Mozilla\Firefox\Profiles",
        "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"
    )

    foreach ($path in $browserPaths) {
        if (Test-Path $path) {
            $summary.BrowserCache += Get-FolderSize -Path $path
        }
    }

    # Recycle Bin (approximate)
    try {
        $rbPath = "$env:SystemDrive\`$Recycle.Bin"
        if (Test-Path $rbPath) {
            $summary.RecycleBin = Get-FolderSize -Path $rbPath
        }
    }
    catch {}

    $summary.TotalReclaimable = $summary.TempFiles + $summary.BrowserCache + $summary.RecycleBin

    return $summary
}

function Get-StartupSummary {
    $summary = @{
        TotalItems = 0
        HighImpact = 0
        MediumImpact = 0
        LowImpact = 0
    }

    try {
        # Count registry Run keys
        $runPaths = @(
            "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run",
            "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run"
        )

        foreach ($path in $runPaths) {
            if (Test-Path $path) {
                $key = Get-Item $path -ErrorAction SilentlyContinue
                if ($key) {
                    $summary.TotalItems += $key.GetValueNames().Count
                }
            }
        }

        # Startup folders
        $startupPaths = @(
            "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup",
            "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\Startup"
        )

        foreach ($path in $startupPaths) {
            if (Test-Path $path) {
                $summary.TotalItems += (Get-ChildItem $path -ErrorAction SilentlyContinue).Count
            }
        }
    }
    catch {}

    return $summary
}

function Get-DriverStatus {
    $status = @{
        ThirdPartyCount = 0
        ProblemsCount = 0
        Status = "OK"
    }

    try {
        # Third-party drivers
        $drivers = Get-WmiObject Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
            Where-Object { $_.DriverProviderName -and $_.DriverProviderName -notmatch "Microsoft" }
        $status.ThirdPartyCount = ($drivers | Measure-Object).Count

        # Problem devices
        $problems = Get-WmiObject Win32_PnPEntity -ErrorAction SilentlyContinue |
            Where-Object { $_.ConfigManagerErrorCode -ne 0 }
        $status.ProblemsCount = ($problems | Measure-Object).Count

        if ($status.ProblemsCount -gt 0) {
            $status.Status = "Issues"
        }
    }
    catch {}

    return $status
}

# ============================================
# REPORT GENERATION
# ============================================

function New-FullSystemReport {
    param(
        [string]$OutputPath = $null
    )

    if (-not $OutputPath) {
        if (-not (Test-Path $ReportsPath)) {
            New-Item -Path $ReportsPath -ItemType Directory -Force | Out-Null
        }
        $OutputPath = Join-Path $ReportsPath "SystemReport_$(Get-Date -Format 'yyyy-MM-dd_HHmmss').html"
    }

    Write-Log "  Gathering system information..." "Yellow" $LogFile
    $sysInfo = Get-SystemOverview

    Write-Log "  Checking for updates..." "Yellow" $LogFile
    $updates = Get-UpdateSummary

    Write-Log "  Scanning cleanable items..." "Yellow" $LogFile
    $cleaner = Get-CleanerSummary

    Write-Log "  Analyzing startup items..." "Yellow" $LogFile
    $startup = Get-StartupSummary

    Write-Log "  Checking driver status..." "Yellow" $LogFile
    $drivers = Get-DriverStatus

    # Determine overall status
    $overallStatus = "Good"
    $statusColor = "#27ae60"
    $statusEmoji = "&#9989;"

    if ($updates.CriticalUpdates -gt 0) {
        $overallStatus = "Critical Updates Needed"
        $statusColor = "#e74c3c"
        $statusEmoji = "&#9888;"
    }
    elseif ($updates.WindowsUpdates -gt 5 -or $cleaner.TotalReclaimable -gt 5GB) {
        $overallStatus = "Maintenance Recommended"
        $statusColor = "#f39c12"
        $statusEmoji = "&#9888;"
    }

    # Generate HTML
    $html = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Report - $($sysInfo.ComputerName)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 30px;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .subtitle { opacity: 0.9; font-size: 1.1em; }
        .header .timestamp { opacity: 0.7; margin-top: 15px; }
        .status-banner {
            background: $statusColor;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.3em;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
            font-size: 1.3em;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stat:last-child { border-bottom: none; }
        .stat-label { opacity: 0.7; }
        .stat-value { font-weight: 600; color: #667eea; }
        .stat-value.warning { color: #f39c12; }
        .stat-value.danger { color: #e74c3c; }
        .stat-value.success { color: #27ae60; }
        .big-number {
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            text-align: center;
            margin: 20px 0;
        }
        .big-number.warning { color: #f39c12; }
        .big-number.danger { color: #e74c3c; }
        .big-number.success { color: #27ae60; }
        .big-label {
            text-align: center;
            opacity: 0.7;
            margin-bottom: 10px;
        }
        .progress-bar {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 15px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.5s;
        }
        .progress-fill.warning { background: linear-gradient(90deg, #f39c12, #e67e22); }
        .progress-fill.danger { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .action-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }
        .action-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .footer {
            text-align: center;
            padding: 30px;
            opacity: 0.5;
            font-size: 0.9em;
        }
        .recommendations {
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }
        .recommendations h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .recommendations ul {
            list-style: none;
            padding: 0;
        }
        .recommendations li {
            padding: 10px 0;
            padding-left: 30px;
            position: relative;
        }
        .recommendations li:before {
            content: '>';
            position: absolute;
            left: 10px;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>System Maintenance Report</h1>
            <div class="subtitle">$($sysInfo.ComputerName) - $($sysInfo.OSVersion)</div>
            <div class="timestamp">Generated: $(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm:ss')</div>
        </div>

        <div class="status-banner">
            $statusEmoji Overall Status: $overallStatus
        </div>

        <div class="grid">
            <div class="card">
                <h2>System Overview</h2>
                <div class="stat">
                    <span class="stat-label">Computer</span>
                    <span class="stat-value">$($sysInfo.ComputerName)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">User</span>
                    <span class="stat-value">$($sysInfo.UserName)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Processor</span>
                    <span class="stat-value">$($sysInfo.Processor)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Memory</span>
                    <span class="stat-value">$($sysInfo.RAM)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Uptime</span>
                    <span class="stat-value">$($sysInfo.Uptime)</span>
                </div>
            </div>

            <div class="card">
                <h2>Storage ($($sysInfo.SystemDrive))</h2>
                <div class="big-number $(if ($sysInfo.SystemDrivePercentFree -lt 10) { 'danger' } elseif ($sysInfo.SystemDrivePercentFree -lt 20) { 'warning' } else { 'success' })">
                    $($sysInfo.SystemDrivePercentFree)%
                </div>
                <div class="big-label">Free Space</div>
                <div class="progress-bar">
                    <div class="progress-fill $(if ($sysInfo.SystemDrivePercentFree -lt 10) { 'danger' } elseif ($sysInfo.SystemDrivePercentFree -lt 20) { 'warning' })" style="width: $(100 - $sysInfo.SystemDrivePercentFree)%"></div>
                </div>
                <div class="stat">
                    <span class="stat-label">Free</span>
                    <span class="stat-value success">$($sysInfo.SystemDriveFree)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Used</span>
                    <span class="stat-value">$($sysInfo.SystemDriveUsed)</span>
                </div>
            </div>

            <div class="card">
                <h2>Updates Available</h2>
                <div class="big-number $(if ($updates.CriticalUpdates -gt 0) { 'danger' } elseif ($updates.WindowsUpdates -gt 0) { 'warning' } else { 'success' })">
                    $($updates.WindowsUpdates + $updates.AppUpdates)
                </div>
                <div class="big-label">Total Updates</div>
                <div class="stat">
                    <span class="stat-label">Windows Updates</span>
                    <span class="stat-value">$($updates.WindowsUpdates)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Driver Updates</span>
                    <span class="stat-value">$($updates.DriverUpdates)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">App Updates</span>
                    <span class="stat-value">$($updates.AppUpdates)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Critical</span>
                    <span class="stat-value $(if ($updates.CriticalUpdates -gt 0) { 'danger' } else { 'success' })">$($updates.CriticalUpdates)</span>
                </div>
            </div>

            <div class="card">
                <h2>Space Reclaimable</h2>
                <div class="big-number $(if ($cleaner.TotalReclaimable -gt 5GB) { 'warning' } else { 'success' })">
                    $(Format-FileSize $cleaner.TotalReclaimable)
                </div>
                <div class="big-label">Can Be Freed</div>
                <div class="stat">
                    <span class="stat-label">Temp Files</span>
                    <span class="stat-value">$(Format-FileSize $cleaner.TempFiles)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Browser Cache</span>
                    <span class="stat-value">$(Format-FileSize $cleaner.BrowserCache)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Recycle Bin</span>
                    <span class="stat-value">$(Format-FileSize $cleaner.RecycleBin)</span>
                </div>
            </div>

            <div class="card">
                <h2>Startup Items</h2>
                <div class="big-number">$($startup.TotalItems)</div>
                <div class="big-label">Programs at Startup</div>
                <div class="stat">
                    <span class="stat-label">Status</span>
                    <span class="stat-value $(if ($startup.TotalItems -gt 20) { 'warning' } else { 'success' })">$(if ($startup.TotalItems -gt 20) { 'Review Recommended' } else { 'OK' })</span>
                </div>
            </div>

            <div class="card">
                <h2>Drivers</h2>
                <div class="big-number $(if ($drivers.ProblemsCount -gt 0) { 'warning' } else { 'success' })">
                    $(if ($drivers.ProblemsCount -gt 0) { $drivers.ProblemsCount } else { '&#10003;' })
                </div>
                <div class="big-label">$(if ($drivers.ProblemsCount -gt 0) { 'Issues Found' } else { 'All OK' })</div>
                <div class="stat">
                    <span class="stat-label">Third-Party Drivers</span>
                    <span class="stat-value">$($drivers.ThirdPartyCount)</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Problem Devices</span>
                    <span class="stat-value $(if ($drivers.ProblemsCount -gt 0) { 'warning' } else { 'success' })">$($drivers.ProblemsCount)</span>
                </div>
            </div>
        </div>

        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
"@

    # Add recommendations
    if ($updates.CriticalUpdates -gt 0) {
        $html += "                <li><strong style='color:#e74c3c'>CRITICAL:</strong> Install $($updates.CriticalUpdates) critical security update(s) immediately</li>`n"
    }
    if ($updates.WindowsUpdates -gt 0) {
        $html += "                <li>Install $($updates.WindowsUpdates) pending Windows update(s)</li>`n"
    }
    if ($cleaner.TotalReclaimable -gt 1GB) {
        $html += "                <li>Run System Cleaner to free $(Format-FileSize $cleaner.TotalReclaimable) of disk space</li>`n"
    }
    if ($startup.TotalItems -gt 15) {
        $html += "                <li>Review startup items - $($startup.TotalItems) programs may be slowing boot time</li>`n"
    }
    if ($drivers.ProblemsCount -gt 0) {
        $html += "                <li>Check Device Manager for $($drivers.ProblemsCount) device(s) with issues</li>`n"
    }
    if ($sysInfo.SystemDrivePercentFree -lt 20) {
        $html += "                <li style='color:#f39c12'>Low disk space warning - only $($sysInfo.SystemDrivePercentFree)% free on system drive</li>`n"
    }

    if ($updates.CriticalUpdates -eq 0 -and $updates.WindowsUpdates -eq 0 -and $cleaner.TotalReclaimable -lt 1GB -and $startup.TotalItems -le 15 -and $drivers.ProblemsCount -eq 0) {
        $html += "                <li style='color:#27ae60'>Your system is in great shape! No immediate actions needed.</li>`n"
    }

    $html += @"
            </ul>
        </div>

        <div class="footer">
            <p>System Maintenance Toolkit v$ScriptVersion</p>
            <p>Report generated on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')</p>
        </div>
    </div>
</body>
</html>
"@

    try {
        $html | Set-Content $OutputPath -Encoding UTF8 -Force
        return $OutputPath
    }
    catch {
        return $null
    }
}

# ============================================
# NOTIFICATION FUNCTIONS
# ============================================

function Send-MaintenanceNotification {
    param(
        [hashtable]$Summary
    )

    $title = "System Maintenance"
    $message = ""
    $type = "Info"

    if ($Summary.CriticalUpdates -gt 0) {
        $message = "$($Summary.CriticalUpdates) critical update(s) need attention!"
        $type = "Warning"
    }
    elseif ($Summary.TotalUpdates -gt 0 -or $Summary.SpaceReclaimable -gt 1GB) {
        $parts = @()
        if ($Summary.TotalUpdates -gt 0) { $parts += "$($Summary.TotalUpdates) updates" }
        if ($Summary.SpaceReclaimable -gt 1GB) { $parts += "$(Format-FileSize $Summary.SpaceReclaimable) reclaimable" }
        $message = $parts -join ", "
        $type = "Info"
    }
    else {
        $message = "System is up to date and clean"
        $type = "Success"
    }

    Show-ToastNotification -Title $title -Message $message -Type $type | Out-Null
}

# ============================================
# MAIN EXECUTION
# ============================================

$Mode = "Interactive"
$OpenReport = $true

foreach ($arg in $args) {
    switch ($arg.ToLower()) {
        "-generate" { $Mode = "Generate" }
        "-notify" { $Mode = "Notify" }
        "-noopen" { $OpenReport = $false }
        "-quiet" { $Mode = "Quiet" }
    }
}

# Remove previous logs
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "SystemReport_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "SystemReport_$(Get-Date -Format 'yyyy-MM-dd').log" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Display banner
Write-Banner "SYSTEM REPORT GENERATOR" $ScriptVersion -LogFile $LogFile
Write-Log "$(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm')" "Gray" $LogFile

# Ensure reports directory exists
if (-not (Test-Path $ReportsPath)) {
    New-Item -Path $ReportsPath -ItemType Directory -Force | Out-Null
}

switch ($Mode) {
    "Generate" {
        Write-Section "GENERATING REPORT" -Number 1 -LogFile $LogFile

        $reportPath = New-FullSystemReport

        if ($reportPath) {
            Write-Log "" "White" $LogFile
            Write-Log "  Report saved to: $reportPath" "Green" $LogFile

            if ($OpenReport) {
                Start-Process $reportPath
            }
        }
        else {
            Write-Log "  Failed to generate report" "Red" $LogFile
        }
    }

    "Notify" {
        Write-Log "  Gathering status..." "Yellow" $LogFile

        $updates = Get-UpdateSummary
        $cleaner = Get-CleanerSummary

        Send-MaintenanceNotification -Summary @{
            CriticalUpdates = $updates.CriticalUpdates
            TotalUpdates = $updates.WindowsUpdates + $updates.AppUpdates
            SpaceReclaimable = $cleaner.TotalReclaimable
        }

        Write-Log "  Notification sent" "Green" $LogFile
    }

    "Quiet" {
        # Generate report without output
        $reportPath = New-FullSystemReport

        if ($reportPath -and $OpenReport) {
            Start-Process $reportPath
        }
    }

    "Interactive" {
        Write-Log "" "White" $LogFile
        Write-Host "  [G] Generate full HTML report" -ForegroundColor White
        Write-Host "  [N] Send notification with current status" -ForegroundColor Cyan
        Write-Host "  [V] View existing reports" -ForegroundColor White
        Write-Host "  [Q] Quit" -ForegroundColor Gray
        Write-Host ""

        $choice = (Read-Host "Enter your choice (G/N/V/Q)").ToUpper()

        switch ($choice) {
            "G" {
                Write-Log "" "White" $LogFile
                Write-Section "GENERATING REPORT" -Number 1 -LogFile $LogFile

                $reportPath = New-FullSystemReport

                if ($reportPath) {
                    Write-Log "" "White" $LogFile
                    Write-Log "  Report saved to: $reportPath" "Green" $LogFile
                    Start-Process $reportPath
                }
                else {
                    Write-Log "  Failed to generate report" "Red" $LogFile
                }
            }

            "N" {
                Write-Log "" "White" $LogFile
                Write-Log "  Gathering status..." "Yellow" $LogFile

                $updates = Get-UpdateSummary
                $cleaner = Get-CleanerSummary

                Send-MaintenanceNotification -Summary @{
                    CriticalUpdates = $updates.CriticalUpdates
                    TotalUpdates = $updates.WindowsUpdates + $updates.AppUpdates
                    SpaceReclaimable = $cleaner.TotalReclaimable
                }

                Write-Log "  Notification sent!" "Green" $LogFile
            }

            "V" {
                Write-Log "" "White" $LogFile
                Write-Section "EXISTING REPORTS" -Number 1 -LogFile $LogFile

                $reports = Get-ChildItem -Path $ReportsPath -Filter "*.html" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending

                if ($reports.Count -eq 0) {
                    Write-Log "  No reports found" "Gray" $LogFile
                }
                else {
                    $index = 1
                    foreach ($report in $reports | Select-Object -First 10) {
                        Write-Log "  [$index] $($report.Name)" "White" $LogFile
                        Write-Log "      Created: $($report.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))" "Gray" $LogFile
                        $index++
                    }

                    Write-Host ""
                    $selection = Read-Host "Enter number to open (or Enter to skip)"

                    if ($selection -match "^\d+$") {
                        $reportIndex = [int]$selection - 1
                        if ($reportIndex -ge 0 -and $reportIndex -lt $reports.Count) {
                            Start-Process $reports[$reportIndex].FullName
                        }
                    }
                }
            }

            "Q" {
                # Exit
            }
        }
    }
}

Write-Log "" "White" $LogFile
Write-Log "========================================" "Cyan" $LogFile
Write-Log "Log saved to: $LogFile" "Gray" $LogFile
Write-Log "========================================" "Cyan" $LogFile
Write-Log "" "White" $LogFile

if ($Mode -eq "Interactive") {
    pause
}
