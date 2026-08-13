# SystemMaintenanceLib.psm1
# Shared functions for System Maintenance Toolkit
# Version: 2.0.0

$script:LibVersion = "2.0.0"

# ============================================
# LOGGING FUNCTIONS
# ============================================

function Write-Log {
    <#
    .SYNOPSIS
        Writes a message to console with color and optionally to a log file.
    #>
    param(
        [string]$Message,
        [string]$Color = "White",
        [string]$LogFile = $null
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"

    Write-Host $Message -ForegroundColor $Color

    if ($LogFile) {
        Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
    }
}

function Write-Section {
    <#
    .SYNOPSIS
        Writes a section header with consistent formatting.
    #>
    param(
        [string]$Title,
        [int]$Number = 0,
        [string]$LogFile = $null
    )

    Write-Log "" "White" $LogFile
    if ($Number -gt 0) {
        Write-Log "[$Number] $Title" "Cyan" $LogFile
    } else {
        Write-Log $Title "Cyan" $LogFile
    }
    Write-Log "----------------------------------------" "Gray" $LogFile
}

function Write-Banner {
    <#
    .SYNOPSIS
        Writes a banner header with consistent formatting.
    #>
    param(
        [string]$Title,
        [string]$Version = "",
        [string]$LogFile = $null
    )

    Write-Log "" "White" $LogFile
    Write-Log "========================================" "Cyan" $LogFile
    if ($Version) {
        Write-Log "  $Title v$Version" "Cyan" $LogFile
    } else {
        Write-Log "  $Title" "Cyan" $LogFile
    }
    Write-Log "========================================" "Cyan" $LogFile
    Write-Log "" "White" $LogFile
}

function Write-Summary {
    <#
    .SYNOPSIS
        Writes a summary section with consistent formatting.
    #>
    param(
        [string]$Title,
        [string]$LogFile = $null
    )

    Write-Log "" "White" $LogFile
    Write-Log "========================================" "Cyan" $LogFile
    Write-Log $Title "Cyan" $LogFile
    Write-Log "========================================" "Cyan" $LogFile
}

# ============================================
# ADMIN & SYSTEM FUNCTIONS
# ============================================

function Test-IsAdmin {
    <#
    .SYNOPSIS
        Checks if the current session is running as Administrator.
    #>
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Request-AdminElevation {
    <#
    .SYNOPSIS
        Requests admin elevation if not already running as admin.
    #>
    param(
        [string]$ScriptPath,
        [string]$Arguments = ""
    )

    if (-not (Test-IsAdmin)) {
        Write-Log "Requesting Administrator privileges..." "Yellow"
        $argList = "-ExecutionPolicy Bypass -File `"$ScriptPath`""
        if ($Arguments) {
            $argList += " $Arguments"
        }
        Start-Process PowerShell -ArgumentList $argList -Verb RunAs
        return $false
    }
    return $true
}

function New-SystemRestorePoint {
    <#
    .SYNOPSIS
        Creates a system restore point.
    #>
    param(
        [string]$Description = "System Maintenance Toolkit Restore Point"
    )

    if (-not (Test-IsAdmin)) {
        Write-Log "ERROR: Creating restore point requires Administrator privileges" "Red"
        return $false
    }

    try {
        Write-Log "Creating System Restore Point..." "Yellow"

        # Enable System Restore if not enabled
        $drive = $env:SystemDrive
        Enable-ComputerRestore -Drive $drive -ErrorAction SilentlyContinue

        # Create restore point
        Checkpoint-Computer -Description $Description -RestorePointType "MODIFY_SETTINGS" -ErrorAction Stop

        Write-Log "Restore point created successfully" "Green"
        return $true
    }
    catch {
        if ($_.Exception.Message -match "A new system restore point cannot be created") {
            Write-Log "Restore point skipped (one was created recently)" "Yellow"
            return $true
        }
        Write-Log "Failed to create restore point: $($_.Exception.Message)" "Red"
        return $false
    }
}

# ============================================
# SIZE & FORMAT FUNCTIONS
# ============================================

function Format-FileSize {
    <#
    .SYNOPSIS
        Formats a byte count as human-readable size.
    #>
    param(
        [long]$Bytes
    )

    if ($Bytes -ge 1TB) {
        return "{0:N2} TB" -f ($Bytes / 1TB)
    }
    elseif ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    }
    elseif ($Bytes -ge 1MB) {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    }
    elseif ($Bytes -ge 1KB) {
        return "{0:N2} KB" -f ($Bytes / 1KB)
    }
    else {
        return "$Bytes Bytes"
    }
}

function Get-FolderSize {
    <#
    .SYNOPSIS
        Gets the total size of a folder in bytes.
    #>
    param(
        [string]$Path,
        [switch]$Recurse = $true
    )

    if (-not (Test-Path $Path)) {
        return 0
    }

    try {
        $size = (Get-ChildItem -Path $Path -Recurse:$Recurse -File -Force -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        return [long]($size -as [long])
    }
    catch {
        return 0
    }
}

function Get-FileCount {
    <#
    .SYNOPSIS
        Gets the count of files in a folder.
    #>
    param(
        [string]$Path,
        [switch]$Recurse = $true
    )

    if (-not (Test-Path $Path)) {
        return 0
    }

    try {
        return (Get-ChildItem -Path $Path -Recurse:$Recurse -File -Force -ErrorAction SilentlyContinue).Count
    }
    catch {
        return 0
    }
}

# ============================================
# CLEANUP FUNCTIONS
# ============================================

function Remove-FolderContents {
    <#
    .SYNOPSIS
        Removes contents of a folder with error handling.
    #>
    param(
        [string]$Path,
        [int]$OlderThanDays = 0,
        [switch]$WhatIf
    )

    if (-not (Test-Path $Path)) {
        return @{
            Success = $true
            FilesDeleted = 0
            BytesFreed = 0
            Errors = @()
        }
    }

    $result = @{
        Success = $true
        FilesDeleted = 0
        BytesFreed = 0
        Errors = @()
    }

    try {
        $files = Get-ChildItem -Path $Path -Recurse -File -Force -ErrorAction SilentlyContinue

        if ($OlderThanDays -gt 0) {
            $cutoff = (Get-Date).AddDays(-$OlderThanDays)
            $files = $files | Where-Object { $_.LastWriteTime -lt $cutoff }
        }

        foreach ($file in $files) {
            try {
                $size = $file.Length
                if (-not $WhatIf) {
                    Remove-Item -Path $file.FullName -Force -ErrorAction Stop
                }
                $result.FilesDeleted++
                $result.BytesFreed += $size
            }
            catch {
                $result.Errors += $file.FullName
            }
        }

        # Also remove empty directories if not WhatIf
        if (-not $WhatIf) {
            Get-ChildItem -Path $Path -Recurse -Directory -Force -ErrorAction SilentlyContinue |
                Sort-Object { $_.FullName.Length } -Descending |
                ForEach-Object {
                    if ((Get-ChildItem -Path $_.FullName -Force -ErrorAction SilentlyContinue).Count -eq 0) {
                        Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
                    }
                }
        }
    }
    catch {
        $result.Success = $false
        $result.Errors += $_.Exception.Message
    }

    return $result
}

# ============================================
# TOAST NOTIFICATION FUNCTIONS
# ============================================

function Show-ToastNotification {
    <#
    .SYNOPSIS
        Shows a Windows Toast notification.
    #>
    param(
        [string]$Title,
        [string]$Message,
        [ValidateSet("Info", "Warning", "Error", "Success")]
        [string]$Type = "Info"
    )

    try {
        # Try using BurntToast module if available
        if (Get-Module -ListAvailable -Name BurntToast) {
            Import-Module BurntToast -ErrorAction SilentlyContinue
            New-BurntToastNotification -Text $Title, $Message -ErrorAction Stop
            return $true
        }

        # Fallback to native Windows toast
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

        $template = @"
<toast>
    <visual>
        <binding template="ToastText02">
            <text id="1">$Title</text>
            <text id="2">$Message</text>
        </binding>
    </visual>
</toast>
"@

        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($template)

        $toast = New-Object Windows.UI.Notifications.ToastNotification($xml)
        $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("System Maintenance Toolkit")
        $notifier.Show($toast)

        return $true
    }
    catch {
        # Silent fail - notifications are optional
        return $false
    }
}

# ============================================
# WINGET FUNCTIONS
# ============================================

function Get-WingetUpgrades {
    <#
    .SYNOPSIS
        Returns the application upgrades winget has available.
    .DESCRIPTION
        winget has no machine-readable output for `winget upgrade`, so its table
        has to be parsed. Splitting each row on runs of whitespace (the approach
        this toolkit used previously) mis-parses any package whose name contains
        two or more consecutive spaces, and counts the header, separator and
        summary lines as if they were packages. This parses positionally instead,
        using the column offsets taken from winget's own header row.

        Count comes from winget's own "N upgrades available" line when present,
        so a single unparseable row cannot silently shrink the reported total.
    .PARAMETER IncludeUnknown
        Include packages whose installed version winget cannot determine. These
        are genuinely out of date but are excluded from `winget upgrade` by
        default, so omitting this under-reports.
    .OUTPUTS
        Hashtable: Available (bool), Count (int), Packages (array), Error (string)
    #>
    param(
        [switch]$IncludeUnknown
    )

    $result = @{
        Available = $false
        Count     = 0
        Packages  = @()
        Error     = $null
    }

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        $result.Error = "winget not found. Install Windows Package Manager (App Installer) to check application updates."
        return $result
    }

    # winget emits UTF-8. Without this the console decodes it as the OEM code
    # page and names come back mangled (e.g. "HWiNFO-? 64").
    $previousEncoding = [Console]::OutputEncoding
    $output = ""

    try {
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

        $arguments = @('upgrade', '--accept-source-agreements')
        if ($IncludeUnknown) { $arguments += '--include-unknown' }

        $output = & winget @arguments 2>&1 | Out-String
    }
    catch {
        $result.Error = "winget upgrade failed: $($_.Exception.Message)"
        return $result
    }
    finally {
        [Console]::OutputEncoding = $previousEncoding
    }

    $lines = $output -split "`r?`n"

    # The table is introduced by a run of dashes; the header sits directly above
    # it. Locating the header this way avoids depending on English column names.
    $separatorIndex = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^-{5,}\s*$') {
            $separatorIndex = $i
            break
        }
    }

    if ($separatorIndex -lt 1) {
        # No table at all - either nothing to upgrade, or output we don't understand.
        if ($output -match '(?m)^\s*(\d+)\s+upgrade') { $result.Count = [int]$Matches[1] }
        $result.Available = $result.Count -gt 0
        return $result
    }

    $header = $lines[$separatorIndex - 1]

    # A column begins at each non-space character that follows a space.
    $columnStarts = @()
    for ($c = 0; $c -lt $header.Length; $c++) {
        if ($header[$c] -ne ' ' -and ($c -eq 0 -or $header[$c - 1] -eq ' ')) {
            $columnStarts += $c
        }
    }

    if ($columnStarts.Count -lt 4) {
        $result.Error = "Could not parse the winget upgrade table header."
        return $result
    }

    for ($i = $separatorIndex + 1; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]

        # Blank line, the trailing summary, or the "requires explicit targeting"
        # notice all mark the end of the table.
        if ([string]::IsNullOrWhiteSpace($line)) { break }
        if ($line -match '^\s*\d+\s+upgrade') { break }
        if ($line.Length -le $columnStarts[1]) { break }

        $fields = @()
        for ($col = 0; $col -lt $columnStarts.Count; $col++) {
            $start = $columnStarts[$col]
            if ($start -ge $line.Length) {
                $fields += ""
                continue
            }

            if ($col -eq $columnStarts.Count - 1) {
                $fields += $line.Substring($start).Trim()
            }
            else {
                $end = [Math]::Min($columnStarts[$col + 1], $line.Length)
                $fields += $line.Substring($start, $end - $start).Trim()
            }
        }

        if (-not $fields[0] -or -not $fields[1]) { continue }

        $result.Packages += [PSCustomObject]@{
            Name           = $fields[0]
            Id             = $fields[1]
            CurrentVersion = $fields[2]
            NewVersion     = $fields[3]
            Source         = if ($fields.Count -ge 5) { $fields[4] } else { "winget" }
        }
    }

    if ($output -match '(?m)^\s*(\d+)\s+upgrade') {
        $result.Count = [int]$Matches[1]
    }
    else {
        $result.Count = $result.Packages.Count
    }

    $result.Available = $result.Count -gt 0
    return $result
}

function Get-WingetCategory {
    <#
    .SYNOPSIS
        Classifies a package name so browsers and security tools can be
        prioritised over everything else.
    .OUTPUTS
        Hashtable: Category (string), Priority (high|medium|low)
    #>
    param(
        [string]$Name
    )

    $securityApps = @('Chrome', 'Firefox', 'Edge', 'Brave', 'VPN', 'Proton', 'Security',
                      'Malwarebytes', 'Bitwarden', '1Password', 'KeePass')
    $devTools = @('Git', 'Node', 'Python', 'Visual Studio', 'VS Code', 'Docker', 'Go',
                  'Rust', 'Java', 'dotnet', 'PowerShell', 'Cursor', 'WSL', 'Windows Subsystem')

    foreach ($app in $securityApps) {
        if ($Name -match [regex]::Escape($app)) {
            return @{ Category = "Security/Browser"; Priority = "high" }
        }
    }

    foreach ($app in $devTools) {
        if ($Name -match [regex]::Escape($app)) {
            return @{ Category = "Development"; Priority = "medium" }
        }
    }

    return @{ Category = "Other"; Priority = "low" }
}

# ============================================
# CONFIG FUNCTIONS
# ============================================

function Get-ToolkitConfig {
    <#
    .SYNOPSIS
        Loads a configuration file.
    #>
    param(
        [string]$ConfigName
    )

    $configPath = Join-Path $PSScriptRoot "..\Config\$ConfigName"

    if (Test-Path $configPath) {
        try {
            return Get-Content $configPath -Raw | ConvertFrom-Json
        }
        catch {
            return $null
        }
    }

    return $null
}

function Save-ToolkitConfig {
    <#
    .SYNOPSIS
        Saves a configuration file.
    #>
    param(
        [string]$ConfigName,
        [object]$Config
    )

    $configPath = Join-Path $PSScriptRoot "..\Config\$ConfigName"

    try {
        $Config | ConvertTo-Json -Depth 10 | Set-Content $configPath -Force
        return $true
    }
    catch {
        return $false
    }
}

function Save-ScanData {
    <#
    .SYNOPSIS
        Saves scan results to the Data folder for dashboard use.
    #>
    param(
        [object]$ScanData,
        [string]$FileName = "last-scan.json"
    )

    $dataDir = Join-Path $PSScriptRoot "..\Data"
    $dataPath = Join-Path $dataDir $FileName

    try {
        # Data/ is gitignored, so it will not exist in a fresh clone.
        if (-not (Test-Path $dataDir)) {
            New-Item -Path $dataDir -ItemType Directory -Force | Out-Null
        }

        $ScanData | Add-Member -NotePropertyName "Timestamp" -NotePropertyValue (Get-Date -Format "o") -Force
        $ScanData | ConvertTo-Json -Depth 10 | Set-Content $dataPath -Force
        return $true
    }
    catch {
        Write-Warning "Save-ScanData failed for '$FileName': $($_.Exception.Message)"
        return $false
    }
}

function Get-ScanData {
    <#
    .SYNOPSIS
        Loads scan results from the Data folder.
    #>
    param(
        [string]$FileName = "last-scan.json"
    )

    $dataPath = Join-Path $PSScriptRoot "..\Data\$FileName"

    if (Test-Path $dataPath) {
        try {
            return Get-Content $dataPath -Raw | ConvertFrom-Json
        }
        catch {
            return $null
        }
    }

    return $null
}

function Add-ScanHistory {
    <#
    .SYNOPSIS
        Appends scan results to history file for charts.
    #>
    param(
        [object]$ScanSummary
    )

    $dataDir = Join-Path $PSScriptRoot "..\Data"
    $historyPath = Join-Path $dataDir "scan-history.json"

    try {
        if (-not (Test-Path $dataDir)) {
            New-Item -Path $dataDir -ItemType Directory -Force | Out-Null
        }

        $history = @()
        if (Test-Path $historyPath) {
            $existing = Get-Content $historyPath -Raw | ConvertFrom-Json
            if ($existing) {
                $history = @($existing)
            }
        }

        $entry = @{
            Timestamp = Get-Date -Format "o"
            SpaceReclaimable = $ScanSummary.TotalReclaimable
            UpdatesAvailable = $ScanSummary.TotalUpdates
            ItemsCleaned = $ScanSummary.ItemsCleaned
            SpaceFreed = $ScanSummary.SpaceFreed
        }

        $history += $entry

        # Keep only last 30 days
        $cutoff = (Get-Date).AddDays(-30)
        $history = $history | Where-Object {
            [DateTime]::Parse($_.Timestamp) -gt $cutoff
        }

        $history | ConvertTo-Json -Depth 10 | Set-Content $historyPath -Force
        return $true
    }
    catch {
        Write-Warning "Add-ScanHistory failed: $($_.Exception.Message)"
        return $false
    }
}

# ============================================
# HTML REPORT FUNCTIONS
# ============================================

function New-HtmlReport {
    <#
    .SYNOPSIS
        Generates an HTML report from scan data.
    #>
    param(
        [object]$ReportData,
        [string]$Title = "System Maintenance Report",
        [string]$OutputPath = $null
    )

    if (-not $OutputPath) {
        $reportsDir = Join-Path $PSScriptRoot "..\Reports"
        $OutputPath = Join-Path $reportsDir "Report_$(Get-Date -Format 'yyyy-MM-dd_HHmmss').html"
    }

    $html = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$Title</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header .timestamp { opacity: 0.8; }
        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-box {
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box .value { font-size: 2em; color: #667eea; font-weight: bold; }
        .stat-box .label { opacity: 0.7; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #0f3460; }
        th { background: #0f3460; color: #667eea; }
        tr:hover { background: #0f3460; }
        .tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
        }
        .tag-safe { background: #27ae60; }
        .tag-warning { background: #f39c12; color: #000; }
        .tag-critical { background: #e74c3c; }
        .footer {
            text-align: center;
            padding: 20px;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>$Title</h1>
            <div class="timestamp">Generated: $(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm:ss')</div>
        </div>

        <div class="card">
            <h2>Summary</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="value">$($ReportData.Summary.UpdatesAvailable)</div>
                    <div class="label">Updates Available</div>
                </div>
                <div class="stat-box">
                    <div class="value">$(Format-FileSize $ReportData.Summary.SpaceReclaimable)</div>
                    <div class="label">Space Reclaimable</div>
                </div>
                <div class="stat-box">
                    <div class="value">$($ReportData.Summary.StartupItems)</div>
                    <div class="label">Startup Items</div>
                </div>
                <div class="stat-box">
                    <div class="value">$($ReportData.Summary.DriverStatus)</div>
                    <div class="label">Driver Status</div>
                </div>
            </div>
        </div>
"@

    # Add sections based on available data
    if ($ReportData.Updates) {
        $html += @"

        <div class="card">
            <h2>Available Updates</h2>
            <table>
                <tr><th>Name</th><th>Type</th><th>Current</th><th>Available</th></tr>
"@
        foreach ($update in $ReportData.Updates) {
            $html += "                <tr><td>$($update.Name)</td><td>$($update.Type)</td><td>$($update.Current)</td><td>$($update.Available)</td></tr>`n"
        }
        $html += "            </table>`n        </div>`n"
    }

    if ($ReportData.Cleanable) {
        $html += @"

        <div class="card">
            <h2>Cleanable Items</h2>
            <table>
                <tr><th>Category</th><th>Size</th><th>Files</th><th>Status</th></tr>
"@
        foreach ($item in $ReportData.Cleanable) {
            $tagClass = switch ($item.Risk) {
                "Safe" { "tag-safe" }
                "Review" { "tag-warning" }
                "Critical" { "tag-critical" }
                default { "tag-safe" }
            }
            $html += "                <tr><td>$($item.Category)</td><td>$(Format-FileSize $item.Size)</td><td>$($item.Files)</td><td><span class='tag $tagClass'>$($item.Risk)</span></td></tr>`n"
        }
        $html += "            </table>`n        </div>`n"
    }

    $html += @"

        <div class="footer">
            System Maintenance Toolkit v$script:LibVersion
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
# EXPORT MODULE MEMBERS
# ============================================

Export-ModuleMember -Function @(
    'Write-Log',
    'Write-Section',
    'Write-Banner',
    'Write-Summary',
    'Test-IsAdmin',
    'Request-AdminElevation',
    'New-SystemRestorePoint',
    'Format-FileSize',
    'Get-FolderSize',
    'Get-FileCount',
    'Remove-FolderContents',
    'Get-WingetUpgrades',
    'Get-WingetCategory',
    'Show-ToastNotification',
    'Get-ToolkitConfig',
    'Save-ToolkitConfig',
    'Save-ScanData',
    'Get-ScanData',
    'Add-ScanHistory',
    'New-HtmlReport'
)
