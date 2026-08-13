$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\StartupManager_$(Get-Date -Format 'yyyy-MM-dd').log"
$BackupPath = Join-Path $ScriptRoot "Config\startup-backup"

# Import shared library
Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force

# ============================================
# CONFIGURATION
# ============================================

$StartupLocations = @{
    HKCU_Run = @{
        Path = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
        Type = "Registry (User)"
        RequiresAdmin = $false
    }
    HKCU_RunOnce = @{
        Path = "HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce"
        Type = "Registry (User)"
        RequiresAdmin = $false
    }
    HKLM_Run = @{
        Path = "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run"
        Type = "Registry (System)"
        RequiresAdmin = $true
    }
    HKLM_RunOnce = @{
        Path = "HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce"
        Type = "Registry (System)"
        RequiresAdmin = $true
    }
    UserStartup = @{
        Path = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
        Type = "Startup Folder (User)"
        RequiresAdmin = $false
    }
    AllUsersStartup = @{
        Path = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs\Startup"
        Type = "Startup Folder (All Users)"
        RequiresAdmin = $true
    }
}

# Known application categories for impact estimation
$KnownApps = @{
    "High" = @{
        Apps = @("OneDrive", "Dropbox", "Google Drive", "iCloud", "Teams", "Slack", "Discord", "Steam", "Epic Games", "Origin", "Ubisoft")
        Description = "Cloud sync, gaming platforms, communication apps"
    }
    "Medium" = @{
        Apps = @("Spotify", "iTunes", "Adobe", "Java", "Updater", "Update", "Helper", "Agent")
        Description = "Media players, updaters, background agents"
    }
    "Low" = @{
        Apps = @("Clipboard", "Notification", "Tray", "Monitor")
        Description = "Utility apps, system tray items"
    }
    "Security" = @{
        Apps = @("Defender", "Antivirus", "Norton", "McAfee", "Kaspersky", "Avast", "AVG", "Bitdefender", "Malwarebytes", "Windows Security")
        Description = "Security software - DO NOT DISABLE"
    }
    "Essential" = @{
        Apps = @("SecurityHealth", "ctfmon", "NVIDIA", "AMD", "Realtek", "Intel")
        Description = "System components and drivers"
    }
}

# ============================================
# STARTUP ITEM FUNCTIONS
# ============================================

function Get-StartupItems {
    <#
    .SYNOPSIS
        Gets all startup items from registry, startup folders, and scheduled tasks.
    #>

    $items = @()
    $index = 1
    $isAdmin = Test-IsAdmin

    # Registry locations
    foreach ($locationKey in $StartupLocations.Keys) {
        $location = $StartupLocations[$locationKey]

        # Skip admin-required locations if not admin
        if ($location.RequiresAdmin -and -not $isAdmin) {
            continue
        }

        if ($location.Path -match "^HK") {
            # Registry key
            if (Test-Path $location.Path) {
                $regKey = Get-Item $location.Path -ErrorAction SilentlyContinue
                if ($regKey) {
                    foreach ($valueName in $regKey.GetValueNames()) {
                        if (-not $valueName) { continue }

                        $value = $regKey.GetValue($valueName)
                        $impact = Get-StartupImpact -Name $valueName -Command $value
                        $category = Get-StartupCategory -Name $valueName -Command $value

                        $items += [PSCustomObject]@{
                            Index = $index
                            Name = $valueName
                            Command = $value
                            Location = $locationKey
                            LocationPath = $location.Path
                            Type = $location.Type
                            Impact = $impact
                            Category = $category
                            Enabled = $true
                            RequiresAdmin = $location.RequiresAdmin
                        }
                        $index++
                    }
                }
            }
        }
        else {
            # Startup folder
            if (Test-Path $location.Path) {
                Get-ChildItem -Path $location.Path -ErrorAction SilentlyContinue | ForEach-Object {
                    $file = $_
                    $impact = Get-StartupImpact -Name $file.Name -Command $file.FullName
                    $category = Get-StartupCategory -Name $file.Name -Command $file.FullName

                    $items += [PSCustomObject]@{
                        Index = $index
                        Name = $file.BaseName
                        Command = $file.FullName
                        Location = $locationKey
                        LocationPath = $location.Path
                        Type = $location.Type
                        Impact = $impact
                        Category = $category
                        Enabled = $true
                        RequiresAdmin = $location.RequiresAdmin
                        IsFile = $true
                        FileName = $file.Name
                    }
                    $index++
                }
            }
        }
    }

    # Scheduled Tasks (user-created, run at logon)
    try {
        $tasks = Get-ScheduledTask -ErrorAction SilentlyContinue |
            Where-Object {
                $_.State -ne "Disabled" -and
                $_.Triggers | Where-Object { $_.CimClass.CimClassName -eq "MSFT_TaskLogonTrigger" }
            }

        foreach ($task in $tasks) {
            # Skip system tasks
            if ($task.TaskPath -match "^\\Microsoft\\") { continue }

            $action = $task.Actions | Select-Object -First 1
            $command = if ($action.Execute) {
                "$($action.Execute) $($action.Arguments)"
            } else { "N/A" }

            $impact = Get-StartupImpact -Name $task.TaskName -Command $command
            $category = Get-StartupCategory -Name $task.TaskName -Command $command

            $items += [PSCustomObject]@{
                Index = $index
                Name = $task.TaskName
                Command = $command
                Location = "ScheduledTask"
                LocationPath = $task.TaskPath
                Type = "Scheduled Task"
                Impact = $impact
                Category = $category
                Enabled = ($task.State -eq "Ready")
                RequiresAdmin = $true
                TaskName = $task.TaskName
                TaskPath = $task.TaskPath
            }
            $index++
        }
    }
    catch {
        # Scheduled task enumeration may fail without admin
    }

    return $items
}

function Get-StartupImpact {
    param(
        [string]$Name,
        [string]$Command
    )

    $combined = "$Name $Command".ToLower()

    # Security apps - always mark as essential
    foreach ($app in $KnownApps["Security"].Apps) {
        if ($combined -match [regex]::Escape($app.ToLower())) {
            return "Essential"
        }
    }

    # Essential system components
    foreach ($app in $KnownApps["Essential"].Apps) {
        if ($combined -match [regex]::Escape($app.ToLower())) {
            return "Essential"
        }
    }

    # High impact
    foreach ($app in $KnownApps["High"].Apps) {
        if ($combined -match [regex]::Escape($app.ToLower())) {
            return "High"
        }
    }

    # Medium impact
    foreach ($app in $KnownApps["Medium"].Apps) {
        if ($combined -match [regex]::Escape($app.ToLower())) {
            return "Medium"
        }
    }

    # Low impact
    foreach ($app in $KnownApps["Low"].Apps) {
        if ($combined -match [regex]::Escape($app.ToLower())) {
            return "Low"
        }
    }

    return "Unknown"
}

function Get-StartupCategory {
    param(
        [string]$Name,
        [string]$Command
    )

    $combined = "$Name $Command".ToLower()

    if ($combined -match "security|defender|antivirus|norton|mcafee|kaspersky|avast|avg|bitdefender|malwarebytes") {
        return "Security"
    }
    if ($combined -match "onedrive|dropbox|google drive|icloud|sync") {
        return "Cloud Sync"
    }
    if ($combined -match "steam|epic|origin|ubisoft|game") {
        return "Gaming"
    }
    if ($combined -match "teams|slack|discord|skype|zoom") {
        return "Communication"
    }
    if ($combined -match "spotify|itunes|music") {
        return "Media"
    }
    if ($combined -match "nvidia|amd|intel|realtek|audio|display") {
        return "Hardware"
    }
    if ($combined -match "update|updater") {
        return "Updater"
    }
    if ($combined -match "adobe|creative|photoshop|acrobat") {
        return "Adobe"
    }
    if ($combined -match "java") {
        return "Java"
    }

    return "Other"
}

function Get-DisabledItems {
    <#
    .SYNOPSIS
        Gets list of items that have been disabled (backed up).
    #>

    $disabled = @()

    if (-not (Test-Path $BackupPath)) {
        return $disabled
    }

    Get-ChildItem -Path $BackupPath -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $item = Get-Content $_.FullName -Raw | ConvertFrom-Json
            $item | Add-Member -NotePropertyName "BackupFile" -NotePropertyValue $_.FullName -Force
            $disabled += $item
        }
        catch {}
    }

    return $disabled
}

# ============================================
# ENABLE/DISABLE FUNCTIONS
# ============================================

function Disable-StartupItem {
    param(
        [PSCustomObject]$Item,
        [switch]$WhatIf
    )

    $result = @{ Success = $true; Error = "" }

    # Check admin requirement
    if ($Item.RequiresAdmin -and -not (Test-IsAdmin)) {
        $result.Success = $false
        $result.Error = "Requires Administrator privileges"
        return $result
    }

    # Warn about security/essential items
    if ($Item.Impact -in @("Essential", "Security")) {
        Write-Log "    WARNING: This is a $($Item.Impact) item!" "Red" $LogFile
    }

    # Create backup
    $backupData = @{
        Name = $Item.Name
        Command = $Item.Command
        Location = $Item.Location
        LocationPath = $Item.LocationPath
        Type = $Item.Type
        DisabledDate = Get-Date -Format "o"
    }

    if ($Item.IsFile) {
        $backupData.IsFile = $true
        $backupData.FileName = $Item.FileName
    }

    if ($Item.TaskName) {
        $backupData.TaskName = $Item.TaskName
        $backupData.TaskPath = $Item.TaskPath
    }

    $backupFile = Join-Path $BackupPath "$($Item.Name -replace '[^\w]', '_')_$(Get-Date -Format 'yyyyMMddHHmmss').json"

    if (-not $WhatIf) {
        # Ensure backup directory exists
        if (-not (Test-Path $BackupPath)) {
            New-Item -Path $BackupPath -ItemType Directory -Force | Out-Null
        }

        # Save backup
        $backupData | ConvertTo-Json | Set-Content $backupFile -Force
    }

    try {
        if ($Item.Location -eq "ScheduledTask") {
            # Disable scheduled task
            if (-not $WhatIf) {
                Disable-ScheduledTask -TaskName $Item.TaskName -ErrorAction Stop | Out-Null
            }
        }
        elseif ($Item.IsFile) {
            # Move file to backup location
            if (-not $WhatIf) {
                $destPath = Join-Path $BackupPath $Item.FileName
                Move-Item -Path $Item.Command -Destination $destPath -Force
                $backupData.BackedUpFile = $destPath
                $backupData | ConvertTo-Json | Set-Content $backupFile -Force
            }
        }
        else {
            # Registry - remove the value
            if (-not $WhatIf) {
                Remove-ItemProperty -Path $Item.LocationPath -Name $Item.Name -ErrorAction Stop
            }
        }
    }
    catch {
        $result.Success = $false
        $result.Error = $_.Exception.Message

        # Remove backup file if operation failed
        if (Test-Path $backupFile) {
            Remove-Item $backupFile -Force -ErrorAction SilentlyContinue
        }
    }

    return $result
}

function Enable-StartupItem {
    param(
        [PSCustomObject]$BackupItem,
        [switch]$WhatIf
    )

    $result = @{ Success = $true; Error = "" }

    try {
        if ($BackupItem.TaskName) {
            # Re-enable scheduled task
            if (-not $WhatIf) {
                Enable-ScheduledTask -TaskName $BackupItem.TaskName -ErrorAction Stop | Out-Null
            }
        }
        elseif ($BackupItem.IsFile) {
            # Move file back from backup
            if (-not $WhatIf) {
                $sourcePath = $BackupItem.BackedUpFile
                if (-not $sourcePath) {
                    $sourcePath = Join-Path $BackupPath $BackupItem.FileName
                }
                $destPath = Join-Path $BackupItem.LocationPath $BackupItem.FileName
                Move-Item -Path $sourcePath -Destination $destPath -Force
            }
        }
        else {
            # Registry - restore the value
            if (-not $WhatIf) {
                Set-ItemProperty -Path $BackupItem.LocationPath -Name $BackupItem.Name -Value $BackupItem.Command -ErrorAction Stop
            }
        }

        # Remove backup file
        if (-not $WhatIf -and $BackupItem.BackupFile) {
            Remove-Item $BackupItem.BackupFile -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        $result.Success = $false
        $result.Error = $_.Exception.Message
    }

    return $result
}

# ============================================
# DISPLAY FUNCTIONS
# ============================================

function Show-StartupItems {
    param(
        [array]$Items,
        [switch]$Detailed
    )

    if ($Items.Count -eq 0) {
        Write-Log "  No startup items found" "Gray" $LogFile
        return
    }

    # Group by category
    $grouped = $Items | Group-Object -Property Category | Sort-Object Name

    foreach ($group in $grouped) {
        Write-Log "" "White" $LogFile
        Write-Log "  $($group.Name) ($($group.Count))" "Yellow" $LogFile

        foreach ($item in $group.Group | Sort-Object Name) {
            $impactColor = switch ($item.Impact) {
                "Essential" { "Red" }
                "High" { "Yellow" }
                "Medium" { "Cyan" }
                "Low" { "Green" }
                default { "Gray" }
            }

            $impactTag = "[$($item.Impact)]"
            $statusTag = if ($item.Enabled) { "" } else { " [DISABLED]" }

            Write-Log "  [$($item.Index)] $($item.Name)$statusTag" "White" $LogFile
            Write-Log "      Impact: $impactTag | Type: $($item.Type)" $impactColor $LogFile

            if ($Detailed) {
                Write-Log "      Command: $($item.Command)" "Gray" $LogFile
            }
        }
    }
}

function Show-ImpactSummary {
    param([array]$Items)

    Write-Log "" "White" $LogFile
    Write-Log "  IMPACT SUMMARY:" "Yellow" $LogFile
    Write-Log "" "White" $LogFile

    $essential = ($Items | Where-Object { $_.Impact -eq "Essential" }).Count
    $high = ($Items | Where-Object { $_.Impact -eq "High" }).Count
    $medium = ($Items | Where-Object { $_.Impact -eq "Medium" }).Count
    $low = ($Items | Where-Object { $_.Impact -eq "Low" }).Count
    $unknown = ($Items | Where-Object { $_.Impact -eq "Unknown" }).Count

    if ($essential -gt 0) { Write-Log "    Essential (DO NOT DISABLE): $essential" "Red" $LogFile }
    if ($high -gt 0) { Write-Log "    High impact: $high" "Yellow" $LogFile }
    if ($medium -gt 0) { Write-Log "    Medium impact: $medium" "Cyan" $LogFile }
    if ($low -gt 0) { Write-Log "    Low impact (safe to disable): $low" "Green" $LogFile }
    if ($unknown -gt 0) { Write-Log "    Unknown: $unknown" "Gray" $LogFile }
}

function Show-DisabledItems {
    param([array]$Items)

    if ($Items.Count -eq 0) {
        Write-Log "  No disabled items found" "Gray" $LogFile
        return
    }

    Write-Log "" "White" $LogFile

    $index = 1
    foreach ($item in $Items) {
        Write-Log "  [$index] $($item.Name)" "White" $LogFile
        Write-Log "      Type: $($item.Type) | Disabled: $($item.DisabledDate)" "Gray" $LogFile
        $index++
    }
}

function Show-MainMenu {
    Write-Log "" "White" $LogFile
    Write-Summary "STARTUP MANAGER OPTIONS" -LogFile $LogFile
    Write-Log "" "White" $LogFile

    Write-Host "  [L] LIST all startup items" -ForegroundColor White
    Write-Host "  [D] DISABLE startup item(s)" -ForegroundColor Yellow
    Write-Host "  [E] ENABLE (restore) disabled item(s)" -ForegroundColor Green
    Write-Host "  [A] ANALYZE startup impact" -ForegroundColor Cyan
    Write-Host "  [Q] Quit" -ForegroundColor Gray
    Write-Host ""

    return (Read-Host "Enter your choice (L/D/E/A/Q)").ToUpper()
}

# ============================================
# MAIN EXECUTION
# ============================================

$Mode = "Interactive"
$DryRun = $false

foreach ($arg in $args) {
    switch ($arg.ToLower()) {
        "-list" { $Mode = "List" }
        "-analyze" { $Mode = "Analyze" }
        "-dryrun" { $DryRun = $true }
        "-whatif" { $DryRun = $true }
    }
}

# Remove previous logs
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "StartupManager_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "StartupManager_$(Get-Date -Format 'yyyy-MM-dd').log" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Display banner
Write-Banner "STARTUP MANAGER" $ScriptVersion -LogFile $LogFile
Write-Log "$(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm')" "Gray" $LogFile

# Check admin
$isAdmin = Test-IsAdmin
if (-not $isAdmin) {
    Write-Log "" "White" $LogFile
    Write-Log "Note: Running without admin - some items may be hidden" "Yellow" $LogFile
    Write-Log "Run as Administrator to see and manage all startup items." "Gray" $LogFile
}

if ($DryRun) {
    Write-Log "" "White" $LogFile
    Write-Log "*** DRY RUN MODE - No changes will be made ***" "Yellow" $LogFile
}

# Ensure backup directory exists
if (-not (Test-Path $BackupPath)) {
    New-Item -Path $BackupPath -ItemType Directory -Force | Out-Null
}

# Handle modes
switch ($Mode) {
    "List" {
        Write-Section "STARTUP ITEMS" -Number 1 -LogFile $LogFile

        Write-Log "  Scanning startup locations..." "Yellow" $LogFile
        $items = Get-StartupItems

        Write-Log "" "White" $LogFile
        Write-Log "  Found $($items.Count) startup item(s)" "Cyan" $LogFile

        Show-StartupItems -Items $items -Detailed
        Show-ImpactSummary -Items $items

        # Save scan data
        Save-ScanData -ScanData @{
            Type = "StartupManager"
            ItemCount = $items.Count
            Items = $items | Select-Object Name, Category, Impact, Type, Enabled
        } -FileName "startup-scan.json" | Out-Null
    }

    "Analyze" {
        Write-Section "STARTUP ANALYSIS" -Number 1 -LogFile $LogFile

        Write-Log "  Scanning startup locations..." "Yellow" $LogFile
        $items = Get-StartupItems

        Show-ImpactSummary -Items $items

        Write-Log "" "White" $LogFile
        Write-Log "  RECOMMENDATIONS:" "Yellow" $LogFile
        Write-Log "" "White" $LogFile

        $safeToDisable = $items | Where-Object { $_.Impact -in @("Low", "Medium") -and $_.Category -notin @("Security", "Hardware") }

        if ($safeToDisable.Count -gt 0) {
            Write-Log "  Items you could safely disable to speed up boot:" "Cyan" $LogFile
            foreach ($item in $safeToDisable | Select-Object -First 10) {
                Write-Log "    - $($item.Name) ($($item.Category))" "Green" $LogFile
            }
        }
        else {
            Write-Log "  Your startup is already optimized!" "Green" $LogFile
        }
    }

    "Interactive" {
        $continue = $true

        while ($continue) {
            $choice = Show-MainMenu

            switch ($choice) {
                "L" {
                    Write-Log "" "White" $LogFile
                    Write-Section "STARTUP ITEMS" -Number 1 -LogFile $LogFile

                    Write-Log "  Scanning..." "Yellow" $LogFile
                    $items = Get-StartupItems

                    Write-Log "" "White" $LogFile
                    Write-Log "  Found $($items.Count) startup item(s)" "Cyan" $LogFile

                    Show-StartupItems -Items $items

                    # Save scan data
                    Save-ScanData -ScanData @{
                        Type = "StartupManager"
                        ItemCount = $items.Count
                        Items = $items | Select-Object Name, Category, Impact, Type, Enabled
                    } -FileName "startup-scan.json" | Out-Null
                }

                "D" {
                    Write-Log "" "White" $LogFile
                    Write-Section "DISABLE STARTUP ITEMS" -Number 1 -LogFile $LogFile

                    $items = Get-StartupItems
                    Show-StartupItems -Items $items

                    Write-Host ""
                    Write-Host "Enter item numbers to disable (e.g., 1 3 5):" -ForegroundColor Yellow
                    Write-Host "Or enter 'low' to disable all Low impact items" -ForegroundColor Gray
                    $selection = Read-Host "Selection"

                    $toDisable = @()

                    if ($selection.ToLower() -eq "low") {
                        $toDisable = $items | Where-Object { $_.Impact -eq "Low" }
                    }
                    else {
                        $numbers = $selection -split '\s+' | ForEach-Object { [int]$_ }
                        foreach ($num in $numbers) {
                            $item = $items | Where-Object { $_.Index -eq $num }
                            if ($item) {
                                $toDisable += $item
                            }
                        }
                    }

                    if ($toDisable.Count -eq 0) {
                        Write-Log "  No items selected" "Gray" $LogFile
                        continue
                    }

                    Write-Log "" "White" $LogFile
                    Write-Log "  Disabling $($toDisable.Count) item(s)..." "Yellow" $LogFile

                    foreach ($item in $toDisable) {
                        Write-Log "    $($item.Name)..." "Gray" $LogFile
                        $result = Disable-StartupItem -Item $item -WhatIf:$DryRun

                        if ($result.Success) {
                            Write-Log "      -> Disabled" "Green" $LogFile
                        }
                        else {
                            Write-Log "      -> Failed: $($result.Error)" "Red" $LogFile
                        }
                    }

                    Write-Log "" "White" $LogFile
                    Write-Log "  Done! Changes take effect on next login/reboot." "Cyan" $LogFile
                }

                "E" {
                    Write-Log "" "White" $LogFile
                    Write-Section "RESTORE DISABLED ITEMS" -Number 1 -LogFile $LogFile

                    $disabled = Get-DisabledItems
                    Show-DisabledItems -Items $disabled

                    if ($disabled.Count -eq 0) {
                        continue
                    }

                    Write-Host ""
                    Write-Host "Enter item numbers to restore (e.g., 1 3 5) or 'all':" -ForegroundColor Yellow
                    $selection = Read-Host "Selection"

                    $toEnable = @()

                    if ($selection.ToLower() -eq "all") {
                        $toEnable = $disabled
                    }
                    else {
                        $numbers = $selection -split '\s+' | ForEach-Object { [int]$_ }
                        $index = 1
                        foreach ($item in $disabled) {
                            if ($index -in $numbers) {
                                $toEnable += $item
                            }
                            $index++
                        }
                    }

                    if ($toEnable.Count -eq 0) {
                        Write-Log "  No items selected" "Gray" $LogFile
                        continue
                    }

                    Write-Log "" "White" $LogFile
                    Write-Log "  Restoring $($toEnable.Count) item(s)..." "Yellow" $LogFile

                    foreach ($item in $toEnable) {
                        Write-Log "    $($item.Name)..." "Gray" $LogFile
                        $result = Enable-StartupItem -BackupItem $item -WhatIf:$DryRun

                        if ($result.Success) {
                            Write-Log "      -> Restored" "Green" $LogFile
                        }
                        else {
                            Write-Log "      -> Failed: $($result.Error)" "Red" $LogFile
                        }
                    }

                    Write-Log "" "White" $LogFile
                    Write-Log "  Done! Changes take effect on next login/reboot." "Cyan" $LogFile
                }

                "A" {
                    Write-Log "" "White" $LogFile
                    Write-Section "STARTUP ANALYSIS" -Number 1 -LogFile $LogFile

                    Write-Log "  Scanning..." "Yellow" $LogFile
                    $items = Get-StartupItems

                    Show-ImpactSummary -Items $items

                    Write-Log "" "White" $LogFile
                    Write-Log "  RECOMMENDATIONS:" "Yellow" $LogFile
                    Write-Log "" "White" $LogFile

                    $safeToDisable = $items | Where-Object {
                        $_.Impact -in @("Low", "Medium") -and
                        $_.Category -notin @("Security", "Hardware")
                    }

                    if ($safeToDisable.Count -gt 0) {
                        Write-Log "  Consider disabling these to speed up boot:" "Cyan" $LogFile
                        foreach ($item in $safeToDisable | Select-Object -First 10) {
                            Write-Log "    [$($item.Index)] $($item.Name) ($($item.Category))" "Green" $LogFile
                        }

                        Write-Log "" "White" $LogFile
                        Write-Log "  Tip: Use [D] to disable, then test. Use [E] to restore if needed." "Gray" $LogFile
                    }
                    else {
                        Write-Log "  Your startup is already well-optimized!" "Green" $LogFile
                    }
                }

                "Q" {
                    $continue = $false
                }

                default {
                    Write-Log "  Invalid choice" "Red" $LogFile
                }
            }
        }
    }
}

Write-Log "" "White" $LogFile
Write-Log "========================================" "Cyan" $LogFile
Write-Log "Log saved to: $LogFile" "Gray" $LogFile
Write-Log "========================================" "Cyan" $LogFile
Write-Log "" "White" $LogFile
