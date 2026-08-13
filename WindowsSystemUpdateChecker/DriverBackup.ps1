$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\DriverBackup_$(Get-Date -Format 'yyyy-MM-dd').log"
$BackupRoot = Join-Path $ScriptRoot "DriverBackups"

# Import shared library
Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force

# ============================================
# DRIVER FUNCTIONS
# ============================================

function Get-ThirdPartyDrivers {
    <#
    .SYNOPSIS
        Gets all third-party (non-Microsoft) drivers installed on the system.
    #>

    $drivers = @()

    try {
        # Get drivers from DISM
        $dismOutput = dism /online /get-drivers /format:table 2>&1

        $inDriverList = $false
        foreach ($line in $dismOutput) {
            if ($line -match "Published Name\s+Original File Name") {
                $inDriverList = $true
                continue
            }

            if ($inDriverList -and $line -match "^(oem\d+\.inf)\s+(.+)$") {
                $publishedName = $Matches[1]
                $originalName = $Matches[2].Trim()

                # Get detailed info for this driver
                $driverInfo = dism /online /get-driverinfo /driver:$publishedName 2>&1 | Out-String

                $provider = ""
                $className = ""
                $version = ""
                $date = ""

                if ($driverInfo -match "Provider Name\s*:\s*(.+)") { $provider = $Matches[1].Trim() }
                if ($driverInfo -match "Class Name\s*:\s*(.+)") { $className = $Matches[1].Trim() }
                if ($driverInfo -match "Driver Version\s*:\s*(.+)") { $version = $Matches[1].Trim() }
                if ($driverInfo -match "Date\s*:\s*(.+)") { $date = $Matches[1].Trim() }

                # Skip Microsoft drivers
                if ($provider -notmatch "Microsoft") {
                    $drivers += [PSCustomObject]@{
                        PublishedName = $publishedName
                        OriginalName = $originalName
                        Provider = $provider
                        ClassName = $className
                        Version = $version
                        Date = $date
                    }
                }
            }
        }
    }
    catch {
        Write-Log "Error getting drivers: $($_.Exception.Message)" "Red" $LogFile
    }

    return $drivers | Sort-Object ClassName, Provider
}

function Get-AllDriversDetailed {
    <#
    .SYNOPSIS
        Gets detailed information about all drivers using WMI.
    #>

    $drivers = @()

    try {
        $wmiDrivers = Get-WmiObject Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
            Where-Object { $_.DriverProviderName -and $_.DriverProviderName -notmatch "Microsoft" } |
            Select-Object DeviceName, DriverProviderName, DriverVersion,
                @{Name="DriverDate";Expression={
                    if ($_.DriverDate) {
                        try { [Management.ManagementDateTimeConverter]::ToDateTime($_.DriverDate) }
                        catch { $null }
                    }
                }},
                DeviceClass, InfName

        foreach ($driver in $wmiDrivers) {
            $drivers += [PSCustomObject]@{
                DeviceName = $driver.DeviceName
                Provider = $driver.DriverProviderName
                Version = $driver.DriverVersion
                Date = if ($driver.DriverDate) { $driver.DriverDate.ToString("yyyy-MM-dd") } else { "Unknown" }
                Class = $driver.DeviceClass
                InfName = $driver.InfName
            }
        }
    }
    catch {
        Write-Log "Error getting detailed drivers: $($_.Exception.Message)" "Red" $LogFile
    }

    return $drivers | Sort-Object Class, Provider
}

function Export-Drivers {
    <#
    .SYNOPSIS
        Exports all third-party drivers to a backup folder.
    #>
    param(
        [string]$BackupPath,
        [switch]$WhatIf
    )

    $result = @{
        Success = $true
        DriversExported = 0
        BackupPath = $BackupPath
        Errors = @()
    }

    if (-not (Test-IsAdmin)) {
        $result.Success = $false
        $result.Errors += "Administrator privileges required for driver export"
        return $result
    }

    # Create backup directory
    if (-not $WhatIf) {
        if (-not (Test-Path $BackupPath)) {
            New-Item -Path $BackupPath -ItemType Directory -Force | Out-Null
        }
    }

    Write-Log "  Exporting drivers to: $BackupPath" "Yellow" $LogFile

    try {
        if (-not $WhatIf) {
            # Use DISM to export all third-party drivers
            $exportResult = dism /online /export-driver /destination:"$BackupPath" 2>&1

            if ($LASTEXITCODE -eq 0) {
                # Count exported drivers
                $exportedInfs = Get-ChildItem -Path $BackupPath -Filter "*.inf" -Recurse -ErrorAction SilentlyContinue
                $result.DriversExported = $exportedInfs.Count
                Write-Log "    -> Exported $($result.DriversExported) driver(s)" "Green" $LogFile
            }
            else {
                $result.Success = $false
                $result.Errors += "DISM export failed: $exportResult"
            }
        }
        else {
            $drivers = Get-ThirdPartyDrivers
            $result.DriversExported = $drivers.Count
            Write-Log "    -> Would export $($result.DriversExported) driver(s)" "Cyan" $LogFile
        }
    }
    catch {
        $result.Success = $false
        $result.Errors += $_.Exception.Message
    }

    return $result
}

function Get-DriverBackups {
    <#
    .SYNOPSIS
        Gets list of available driver backups.
    #>

    $backups = @()

    if (-not (Test-Path $BackupRoot)) {
        return $backups
    }

    Get-ChildItem -Path $BackupRoot -Directory | ForEach-Object {
        $backupDir = $_
        $infCount = (Get-ChildItem -Path $backupDir.FullName -Filter "*.inf" -Recurse -ErrorAction SilentlyContinue).Count

        $backups += [PSCustomObject]@{
            Name = $backupDir.Name
            Path = $backupDir.FullName
            Date = $backupDir.CreationTime
            DriverCount = $infCount
            Size = Get-FolderSize -Path $backupDir.FullName
            SizeFormatted = Format-FileSize (Get-FolderSize -Path $backupDir.FullName)
        }
    }

    return $backups | Sort-Object Date -Descending
}

function Restore-DriversFromBackup {
    <#
    .SYNOPSIS
        Restores drivers from a backup folder.
    #>
    param(
        [string]$BackupPath,
        [array]$SpecificDrivers = @(),
        [switch]$WhatIf
    )

    $result = @{
        Success = $true
        DriversRestored = 0
        Errors = @()
    }

    if (-not (Test-IsAdmin)) {
        $result.Success = $false
        $result.Errors += "Administrator privileges required for driver restore"
        return $result
    }

    if (-not (Test-Path $BackupPath)) {
        $result.Success = $false
        $result.Errors += "Backup path not found: $BackupPath"
        return $result
    }

    # Get all INF files in backup
    $infFiles = Get-ChildItem -Path $BackupPath -Filter "*.inf" -Recurse -ErrorAction SilentlyContinue

    if ($SpecificDrivers.Count -gt 0) {
        $infFiles = $infFiles | Where-Object { $_.BaseName -in $SpecificDrivers }
    }

    foreach ($inf in $infFiles) {
        Write-Log "    Installing: $($inf.Name)" "Gray" $LogFile

        if (-not $WhatIf) {
            try {
                $pnpResult = pnputil /add-driver "$($inf.FullName)" /install 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $result.DriversRestored++
                }
                else {
                    $result.Errors += "Failed to install $($inf.Name): $pnpResult"
                }
            }
            catch {
                $result.Errors += "Error installing $($inf.Name): $($_.Exception.Message)"
            }
        }
        else {
            $result.DriversRestored++
        }
    }

    return $result
}

# ============================================
# DISPLAY FUNCTIONS
# ============================================

function Show-DriverList {
    param([array]$Drivers)

    if ($Drivers.Count -eq 0) {
        Write-Log "  No third-party drivers found" "Gray" $LogFile
        return
    }

    $currentClass = ""
    $index = 1

    foreach ($driver in $Drivers) {
        $class = if ($driver.ClassName) { $driver.ClassName } elseif ($driver.Class) { $driver.Class } else { "Other" }

        if ($class -ne $currentClass) {
            Write-Log "" "White" $LogFile
            Write-Log "  $class" "Yellow" $LogFile
            $currentClass = $class
        }

        $provider = if ($driver.Provider) { $driver.Provider } elseif ($driver.DriverProviderName) { $driver.DriverProviderName } else { "Unknown" }
        $name = if ($driver.DeviceName) { $driver.DeviceName } elseif ($driver.OriginalName) { $driver.OriginalName } else { $driver.PublishedName }
        $version = if ($driver.Version) { $driver.Version } else { "N/A" }
        $date = if ($driver.Date) { $driver.Date } else { "N/A" }

        Write-Log "  [$index] $name" "White" $LogFile
        Write-Log "      Provider: $provider | Version: $version | Date: $date" "Gray" $LogFile
        $index++
    }
}

function Show-BackupList {
    param([array]$Backups)

    if ($Backups.Count -eq 0) {
        Write-Log "  No backups found" "Gray" $LogFile
        return
    }

    $index = 1
    foreach ($backup in $Backups) {
        Write-Log "  [$index] $($backup.Name)" "White" $LogFile
        Write-Log "      Date: $($backup.Date.ToString('yyyy-MM-dd HH:mm')) | Drivers: $($backup.DriverCount) | Size: $($backup.SizeFormatted)" "Gray" $LogFile
        $index++
    }
}

function Show-MainMenu {
    Write-Log "" "White" $LogFile
    Write-Summary "DRIVER BACKUP OPTIONS" -LogFile $LogFile
    Write-Log "" "White" $LogFile

    Write-Host "  [B] BACKUP all third-party drivers" -ForegroundColor Green
    Write-Host "  [R] RESTORE drivers from backup" -ForegroundColor Cyan
    Write-Host "  [L] LIST installed third-party drivers" -ForegroundColor White
    Write-Host "  [V] VIEW available backups" -ForegroundColor White
    Write-Host "  [D] DELETE old backups" -ForegroundColor Yellow
    Write-Host "  [Q] Quit" -ForegroundColor Gray
    Write-Host ""

    return (Read-Host "Enter your choice (B/R/L/V/D/Q)").ToUpper()
}

# ============================================
# MAIN EXECUTION
# ============================================

$Mode = "Interactive"
$DryRun = $false

foreach ($arg in $args) {
    switch ($arg.ToLower()) {
        "-backup" { $Mode = "Backup" }
        "-restore" { $Mode = "Restore" }
        "-list" { $Mode = "List" }
        "-dryrun" { $DryRun = $true }
        "-whatif" { $DryRun = $true }
    }
}

# Remove previous logs
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "DriverBackup_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "DriverBackup_$(Get-Date -Format 'yyyy-MM-dd').log" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Display banner
Write-Banner "DRIVER BACKUP & RESTORE" $ScriptVersion -LogFile $LogFile
Write-Log "$(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm')" "Gray" $LogFile

# Check admin
$isAdmin = Test-IsAdmin
if (-not $isAdmin) {
    Write-Log "" "White" $LogFile
    Write-Log "WARNING: Not running as Administrator" "Yellow" $LogFile
    Write-Log "Driver backup/restore requires admin privileges." "Yellow" $LogFile
    Write-Log "You can still list drivers and view backups." "Gray" $LogFile
}

if ($DryRun) {
    Write-Log "" "White" $LogFile
    Write-Log "*** DRY RUN MODE - No changes will be made ***" "Yellow" $LogFile
}

# Ensure backup directory exists
if (-not (Test-Path $BackupRoot)) {
    New-Item -Path $BackupRoot -ItemType Directory -Force | Out-Null
}

# Handle modes
switch ($Mode) {
    "Backup" {
        if (-not $isAdmin) {
            Write-Log "ERROR: Administrator privileges required for backup" "Red" $LogFile
            exit 1
        }

        Write-Section "CREATING DRIVER BACKUP" -Number 1 -LogFile $LogFile

        # Create restore point first
        Write-Log "  Creating System Restore Point..." "Yellow" $LogFile
        New-SystemRestorePoint -Description "Before Driver Backup - $(Get-Date -Format 'yyyy-MM-dd')" | Out-Null

        $backupName = "Backup_$(Get-Date -Format 'yyyy-MM-dd_HHmmss')"
        $backupPath = Join-Path $BackupRoot $backupName

        $exportResult = Export-Drivers -BackupPath $backupPath -WhatIf:$DryRun

        Write-Log "" "White" $LogFile
        Write-Summary "BACKUP COMPLETE" -LogFile $LogFile
        Write-Log "" "White" $LogFile
        Write-Log "  Drivers backed up: $($exportResult.DriversExported)" "Green" $LogFile
        Write-Log "  Location: $backupPath" "Cyan" $LogFile
    }

    "Restore" {
        if (-not $isAdmin) {
            Write-Log "ERROR: Administrator privileges required for restore" "Red" $LogFile
            exit 1
        }

        Write-Section "AVAILABLE BACKUPS" -Number 1 -LogFile $LogFile
        $backups = Get-DriverBackups
        Show-BackupList -Backups $backups

        if ($backups.Count -eq 0) {
            Write-Log "" "White" $LogFile
            Write-Log "No backups available to restore." "Yellow" $LogFile
            exit 0
        }

        Write-Host ""
        $selection = Read-Host "Enter backup number to restore (or Q to quit)"

        if ($selection.ToUpper() -eq "Q") {
            exit 0
        }

        $backupIndex = [int]$selection - 1
        if ($backupIndex -lt 0 -or $backupIndex -ge $backups.Count) {
            Write-Log "Invalid selection" "Red" $LogFile
            exit 1
        }

        $selectedBackup = $backups[$backupIndex]

        Write-Log "" "White" $LogFile
        Write-Section "RESTORING DRIVERS" -Number 2 -LogFile $LogFile

        # Create restore point first
        Write-Log "  Creating System Restore Point..." "Yellow" $LogFile
        New-SystemRestorePoint -Description "Before Driver Restore - $(Get-Date -Format 'yyyy-MM-dd')" | Out-Null

        Write-Log "  Restoring from: $($selectedBackup.Name)" "Yellow" $LogFile
        $restoreResult = Restore-DriversFromBackup -BackupPath $selectedBackup.Path -WhatIf:$DryRun

        Write-Log "" "White" $LogFile
        Write-Summary "RESTORE COMPLETE" -LogFile $LogFile
        Write-Log "" "White" $LogFile
        Write-Log "  Drivers restored: $($restoreResult.DriversRestored)" "Green" $LogFile

        if ($restoreResult.Errors.Count -gt 0) {
            Write-Log "  Errors: $($restoreResult.Errors.Count)" "Yellow" $LogFile
        }
    }

    "List" {
        Write-Section "INSTALLED THIRD-PARTY DRIVERS" -Number 1 -LogFile $LogFile

        Write-Log "  Scanning drivers..." "Yellow" $LogFile
        $drivers = Get-AllDriversDetailed

        Write-Log "" "White" $LogFile
        Write-Log "  Found $($drivers.Count) third-party driver(s)" "Cyan" $LogFile

        Show-DriverList -Drivers $drivers
    }

    "Interactive" {
        $continue = $true

        while ($continue) {
            $choice = Show-MainMenu

            switch ($choice) {
                "B" {
                    if (-not $isAdmin) {
                        Write-Log "ERROR: Run as Administrator for backup" "Red" $LogFile
                        continue
                    }

                    Write-Log "" "White" $LogFile
                    Write-Section "CREATING DRIVER BACKUP" -Number 1 -LogFile $LogFile

                    # Create restore point
                    Write-Log "  Creating System Restore Point..." "Yellow" $LogFile
                    $rpResult = New-SystemRestorePoint -Description "Before Driver Backup - $(Get-Date -Format 'yyyy-MM-dd')"

                    $backupName = "Backup_$(Get-Date -Format 'yyyy-MM-dd_HHmmss')"
                    $backupPath = Join-Path $BackupRoot $backupName

                    $exportResult = Export-Drivers -BackupPath $backupPath -WhatIf:$DryRun

                    Write-Log "" "White" $LogFile
                    if ($exportResult.Success) {
                        Write-Log "  Backup complete! $($exportResult.DriversExported) driver(s) saved." "Green" $LogFile
                        Write-Log "  Location: $backupPath" "Cyan" $LogFile

                        # Toast notification
                        Show-ToastNotification -Title "Driver Backup" -Message "Backed up $($exportResult.DriversExported) drivers" -Type "Success" | Out-Null
                    }
                    else {
                        Write-Log "  Backup failed: $($exportResult.Errors -join ', ')" "Red" $LogFile
                    }
                }

                "R" {
                    if (-not $isAdmin) {
                        Write-Log "ERROR: Run as Administrator for restore" "Red" $LogFile
                        continue
                    }

                    Write-Log "" "White" $LogFile
                    Write-Section "AVAILABLE BACKUPS" -Number 1 -LogFile $LogFile

                    $backups = Get-DriverBackups
                    Show-BackupList -Backups $backups

                    if ($backups.Count -eq 0) {
                        Write-Log "" "White" $LogFile
                        Write-Log "  No backups available." "Yellow" $LogFile
                        continue
                    }

                    Write-Host ""
                    $selection = Read-Host "Enter backup number to restore (or Enter to cancel)"

                    if (-not $selection) { continue }

                    $backupIndex = [int]$selection - 1
                    if ($backupIndex -lt 0 -or $backupIndex -ge $backups.Count) {
                        Write-Log "  Invalid selection" "Red" $LogFile
                        continue
                    }

                    $selectedBackup = $backups[$backupIndex]

                    Write-Log "" "White" $LogFile
                    Write-Section "RESTORING DRIVERS" -Number 2 -LogFile $LogFile

                    # Create restore point
                    Write-Log "  Creating System Restore Point..." "Yellow" $LogFile
                    New-SystemRestorePoint -Description "Before Driver Restore - $(Get-Date -Format 'yyyy-MM-dd')" | Out-Null

                    Write-Log "  Restoring from: $($selectedBackup.Name)" "Yellow" $LogFile
                    $restoreResult = Restore-DriversFromBackup -BackupPath $selectedBackup.Path -WhatIf:$DryRun

                    Write-Log "" "White" $LogFile
                    Write-Log "  Restored $($restoreResult.DriversRestored) driver(s)" "Green" $LogFile

                    if ($restoreResult.Errors.Count -gt 0) {
                        Write-Log "  Some errors occurred (see log)" "Yellow" $LogFile
                    }

                    Show-ToastNotification -Title "Driver Restore" -Message "Restored $($restoreResult.DriversRestored) drivers" -Type "Success" | Out-Null
                }

                "L" {
                    Write-Log "" "White" $LogFile
                    Write-Section "INSTALLED THIRD-PARTY DRIVERS" -Number 1 -LogFile $LogFile

                    Write-Log "  Scanning..." "Yellow" $LogFile
                    $drivers = Get-AllDriversDetailed

                    Write-Log "" "White" $LogFile
                    Write-Log "  Found $($drivers.Count) third-party driver(s)" "Cyan" $LogFile

                    Show-DriverList -Drivers $drivers
                }

                "V" {
                    Write-Log "" "White" $LogFile
                    Write-Section "AVAILABLE BACKUPS" -Number 1 -LogFile $LogFile

                    $backups = Get-DriverBackups
                    Show-BackupList -Backups $backups

                    if ($backups.Count -eq 0) {
                        Write-Log "" "White" $LogFile
                        Write-Log "  No backups found in: $BackupRoot" "Gray" $LogFile
                    }
                    else {
                        $totalSize = ($backups | Measure-Object -Property Size -Sum).Sum
                        Write-Log "" "White" $LogFile
                        Write-Log "  Total backup size: $(Format-FileSize $totalSize)" "Cyan" $LogFile
                    }
                }

                "D" {
                    Write-Log "" "White" $LogFile
                    Write-Section "DELETE BACKUPS" -Number 1 -LogFile $LogFile

                    $backups = Get-DriverBackups
                    Show-BackupList -Backups $backups

                    if ($backups.Count -eq 0) {
                        Write-Log "  No backups to delete." "Gray" $LogFile
                        continue
                    }

                    Write-Host ""
                    Write-Host "Enter backup numbers to delete (e.g., 1 3 5) or 'old' for backups > 30 days:" -ForegroundColor Yellow
                    $selection = Read-Host "Selection"

                    $toDelete = @()

                    if ($selection.ToLower() -eq "old") {
                        $cutoff = (Get-Date).AddDays(-30)
                        $toDelete = $backups | Where-Object { $_.Date -lt $cutoff }
                    }
                    else {
                        $numbers = $selection -split '\s+' | ForEach-Object { [int]$_ }
                        foreach ($num in $numbers) {
                            if ($num -gt 0 -and $num -le $backups.Count) {
                                $toDelete += $backups[$num - 1]
                            }
                        }
                    }

                    if ($toDelete.Count -eq 0) {
                        Write-Log "  No backups selected for deletion." "Gray" $LogFile
                        continue
                    }

                    Write-Log "" "White" $LogFile
                    foreach ($backup in $toDelete) {
                        Write-Log "  Deleting: $($backup.Name)" "Yellow" $LogFile
                        if (-not $DryRun) {
                            Remove-Item -Path $backup.Path -Recurse -Force -ErrorAction SilentlyContinue
                        }
                    }

                    $freedSpace = ($toDelete | Measure-Object -Property Size -Sum).Sum
                    Write-Log "  Deleted $($toDelete.Count) backup(s), freed $(Format-FileSize $freedSpace)" "Green" $LogFile
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

if ($Mode -eq "Interactive") {
    # Don't pause - user chose to quit
}
elseif ($Mode -ne "List") {
    pause
}
