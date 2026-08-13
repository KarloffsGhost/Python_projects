$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.1"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\SystemCleaner_$(Get-Date -Format 'yyyy-MM-dd').log"

# Import shared library
Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force

# ============================================
# CONFIGURATION
# ============================================

$CleaningTargets = @{
    WindowsTemp = @{
        Name = "Windows Temp Files"
        Path = "$env:WINDIR\Temp"
        RequiresAdmin = $true
        Risk = "Safe"
        Description = "Temporary files created by Windows"
    }
    UserTemp = @{
        Name = "User Temp Files"
        Path = $env:TEMP
        RequiresAdmin = $false
        Risk = "Safe"
        Description = "Temporary files created by user applications"
    }
    Prefetch = @{
        Name = "Windows Prefetch"
        Path = "$env:WINDIR\Prefetch"
        RequiresAdmin = $true
        Risk = "Safe"
        Description = "Application prefetch data (rebuilds automatically)"
    }
    WindowsUpdateCache = @{
        Name = "Windows Update Cache"
        Path = "$env:WINDIR\SoftwareDistribution\Download"
        RequiresAdmin = $true
        Risk = "Safe"
        Description = "Downloaded Windows Update files"
    }
    ThumbnailCache = @{
        Name = "Thumbnail Cache"
        Path = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"
        Pattern = "thumbcache_*.db"
        RequiresAdmin = $false
        Risk = "Safe"
        Description = "Windows Explorer thumbnail database"
    }
    WindowsErrorReports = @{
        Name = "Windows Error Reports"
        Path = "$env:LOCALAPPDATA\Microsoft\Windows\WER"
        RequiresAdmin = $false
        Risk = "Safe"
        Description = "Windows Error Reporting files"
    }
    ChromeCache = @{
        Name = "Chrome Cache"
        Path = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache"
        RequiresAdmin = $false
        Risk = "Safe"
        Description = "Google Chrome browser cache"
        Paths = @(
            "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache",
            "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Code Cache",
            "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\GPUCache"
        )
    }
    FirefoxCache = @{
        Name = "Firefox Cache"
        Path = "$env:LOCALAPPDATA\Mozilla\Firefox\Profiles"
        Pattern = "cache2"
        RequiresAdmin = $false
        Risk = "Safe"
        Description = "Mozilla Firefox browser cache"
    }
    EdgeCache = @{
        Name = "Edge Cache"
        Path = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"
        RequiresAdmin = $false
        Risk = "Safe"
        Description = "Microsoft Edge browser cache"
        Paths = @(
            "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache",
            "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Code Cache",
            "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\GPUCache"
        )
    }
    RecycleBin = @{
        Name = "Recycle Bin"
        Special = "RecycleBin"
        RequiresAdmin = $false
        Risk = "Review"
        Description = "Deleted files in Recycle Bin"
    }
}

# ============================================
# SCANNING FUNCTIONS
# ============================================

function Get-RecycleBinInfo {
    try {
        $shell = New-Object -ComObject Shell.Application
        $recycleBin = $shell.NameSpace(0x0a)
        $items = $recycleBin.Items()

        $totalSize = 0
        $itemCount = $items.Count

        # $item.Size is the size in bytes. GetDetailsOf() was used here before,
        # but it returns the localised display string from the size column
        # ("1.2 MB"), and stripping non-digits from that yields 12.
        foreach ($item in $items) {
            $itemSize = $item.Size -as [long]
            if ($itemSize) { $totalSize += $itemSize }
        }

        # Alternative method using COM
        if ($totalSize -eq 0 -and $itemCount -gt 0) {
            $recycleBinPath = "$env:SystemDrive\`$Recycle.Bin"
            if (Test-Path $recycleBinPath) {
                $totalSize = Get-FolderSize -Path $recycleBinPath
            }
        }

        return @{
            Count = $itemCount
            Size = $totalSize
        }
    }
    catch {
        return @{ Count = 0; Size = 0 }
    }
}

function Get-FirefoxCacheSize {
    $totalSize = 0
    $profilesPath = "$env:LOCALAPPDATA\Mozilla\Firefox\Profiles"

    if (Test-Path $profilesPath) {
        Get-ChildItem -Path $profilesPath -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $cachePath = Join-Path $_.FullName "cache2"
            if (Test-Path $cachePath) {
                $totalSize += Get-FolderSize -Path $cachePath
            }
        }
    }

    return $totalSize
}

function Get-BrowserCacheSize {
    param([hashtable]$Target)

    $totalSize = 0

    if ($Target.Paths) {
        foreach ($path in $Target.Paths) {
            if (Test-Path $path) {
                $totalSize += Get-FolderSize -Path $path
            }
        }
    }
    elseif ($Target.Path -and (Test-Path $Target.Path)) {
        $totalSize = Get-FolderSize -Path $Target.Path
    }

    return $totalSize
}

function Invoke-SystemScan {
    param(
        [switch]$Detailed
    )

    $results = @()
    $totalReclaimable = 0

    Write-Section "SCANNING SYSTEM FOR CLEANABLE ITEMS" -Number 1 -LogFile $LogFile

    $isAdmin = Test-IsAdmin
    if (-not $isAdmin) {
        Write-Log "Note: Running without admin - some locations will be skipped" "Yellow" $LogFile
    }

    $index = 1
    $totalTargets = $CleaningTargets.Count

    foreach ($key in $CleaningTargets.Keys) {
        $target = $CleaningTargets[$key]
        $progress = [math]::Round(($index / $totalTargets) * 100)
        Write-Host "`r  Scanning ($progress%): $($target.Name)...                    " -NoNewline -ForegroundColor Gray

        $size = 0
        $fileCount = 0
        $accessible = $true

        # Check admin requirement
        if ($target.RequiresAdmin -and -not $isAdmin) {
            $accessible = $false
        }

        if ($accessible) {
            switch ($key) {
                "RecycleBin" {
                    $rbInfo = Get-RecycleBinInfo
                    $size = $rbInfo.Size
                    $fileCount = $rbInfo.Count
                }
                "FirefoxCache" {
                    $size = Get-FirefoxCacheSize
                    $fileCount = -1  # Unknown
                }
                { $_ -in @("ChromeCache", "EdgeCache") } {
                    $size = Get-BrowserCacheSize -Target $target
                    $fileCount = -1
                }
                "ThumbnailCache" {
                    $thumbPath = $target.Path
                    if (Test-Path $thumbPath) {
                        $thumbFiles = Get-ChildItem -Path $thumbPath -Filter $target.Pattern -ErrorAction SilentlyContinue
                        $size = ($thumbFiles | Measure-Object -Property Length -Sum).Sum
                        $fileCount = $thumbFiles.Count
                    }
                }
                default {
                    if (Test-Path $target.Path) {
                        $size = Get-FolderSize -Path $target.Path
                        $fileCount = Get-FileCount -Path $target.Path
                    }
                }
            }
        }

        $results += [PSCustomObject]@{
            Key = $key
            Name = $target.Name
            Size = [long]$size
            SizeFormatted = Format-FileSize $size
            FileCount = $fileCount
            Risk = $target.Risk
            RequiresAdmin = $target.RequiresAdmin
            Accessible = $accessible
            Description = $target.Description
        }

        $totalReclaimable += $size
        $index++
    }

    Write-Host "`r                                                              `r" -NoNewline

    return @{
        Items = $results
        TotalReclaimable = $totalReclaimable
        TotalReclaimableFormatted = Format-FileSize $totalReclaimable
        ScanTime = Get-Date -Format "o"
        IsAdmin = $isAdmin
    }
}

# ============================================
# CLEANING FUNCTIONS
# ============================================

function Clear-FirefoxCache {
    param([switch]$WhatIf)

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }
    $profilesPath = "$env:LOCALAPPDATA\Mozilla\Firefox\Profiles"

    if (Test-Path $profilesPath) {
        Get-ChildItem -Path $profilesPath -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $cachePath = Join-Path $_.FullName "cache2"
            if (Test-Path $cachePath) {
                $cleanResult = Remove-FolderContents -Path $cachePath -WhatIf:$WhatIf
                $result.BytesFreed += $cleanResult.BytesFreed
                $result.Errors += $cleanResult.Errors
            }
        }
    }

    return $result
}

function Clear-BrowserCache {
    param(
        [hashtable]$Target,
        [switch]$WhatIf
    )

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }

    if ($Target.Paths) {
        foreach ($path in $Target.Paths) {
            if (Test-Path $path) {
                $cleanResult = Remove-FolderContents -Path $path -WhatIf:$WhatIf
                $result.BytesFreed += $cleanResult.BytesFreed
                $result.Errors += $cleanResult.Errors
            }
        }
    }
    elseif ($Target.Path -and (Test-Path $Target.Path)) {
        $cleanResult = Remove-FolderContents -Path $Target.Path -WhatIf:$WhatIf
        $result.BytesFreed += $cleanResult.BytesFreed
        $result.Errors += $cleanResult.Errors
    }

    return $result
}

function Clear-SystemRecycleBin {
    <#
    .SYNOPSIS
        Empties the Recycle Bin and reports the space actually reclaimed.
    .DESCRIPTION
        This function used to be named Clear-RecycleBin, which shadowed the
        built-in cmdlet of the same name. Its own call to "Clear-RecycleBin
        -Force" therefore resolved back to itself, failed on the unknown -Force
        parameter, and silently fell through to the COM path - so the documented
        primary path never ran at all.

        BytesFreed is now measured from the difference before and after. It was
        previously set to the size of the bin before deletion, so a failed or
        partial empty still reported the full amount as reclaimed.
    #>
    param([switch]$WhatIf)

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }

    $before = Get-RecycleBinInfo

    if ($WhatIf) {
        $result.BytesFreed = $before.Size
        return $result
    }

    try {
        # Module-qualified so it cannot resolve back to this function.
        Microsoft.PowerShell.Management\Clear-RecycleBin -Force -Confirm:$false -ErrorAction Stop
    }
    catch {
        Write-Log "  Clear-RecycleBin failed ($($_.Exception.Message)); trying shell fallback" "Gray" $LogFile

        try {
            $shell = New-Object -ComObject Shell.Application
            $recycleBin = $shell.NameSpace(0x0a)
            foreach ($item in @($recycleBin.Items())) {
                Remove-Item -LiteralPath $item.Path -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            $result.Success = $false
            $result.Errors += "Recycle Bin could not be emptied: $($_.Exception.Message)"
            return $result
        }
    }

    $after = Get-RecycleBinInfo
    $result.BytesFreed = [Math]::Max(0, $before.Size - $after.Size)

    if ($after.Count -gt 0) {
        $result.Errors += "$($after.Count) item(s) remain in the Recycle Bin (in use or access denied)"
    }

    return $result
}

function Clear-ThumbnailCache {
    param([switch]$WhatIf)

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }
    $thumbPath = "$env:LOCALAPPDATA\Microsoft\Windows\Explorer"

    if (Test-Path $thumbPath) {
        $thumbFiles = Get-ChildItem -Path $thumbPath -Filter "thumbcache_*.db" -ErrorAction SilentlyContinue

        foreach ($file in $thumbFiles) {
            try {
                $result.BytesFreed += $file.Length
                if (-not $WhatIf) {
                    Remove-Item $file.FullName -Force -ErrorAction Stop
                }
            }
            catch {
                $result.Errors += $file.FullName
            }
        }
    }

    return $result
}

function Invoke-Cleaning {
    param(
        [array]$SelectedItems,
        [switch]$WhatIf
    )

    $totalFreed = 0
    $results = @()

    $action = if ($WhatIf) { "Would clean" } else { "Cleaning" }

    foreach ($item in $SelectedItems) {
        Write-Log "  $action $($item.Name)..." "Yellow" $LogFile

        $cleanResult = @{ Success = $true; BytesFreed = 0; Errors = @() }
        $target = $CleaningTargets[$item.Key]

        switch ($item.Key) {
            "RecycleBin" {
                $cleanResult = Clear-SystemRecycleBin -WhatIf:$WhatIf
            }
            "FirefoxCache" {
                $cleanResult = Clear-FirefoxCache -WhatIf:$WhatIf
            }
            { $_ -in @("ChromeCache", "EdgeCache") } {
                $cleanResult = Clear-BrowserCache -Target $target -WhatIf:$WhatIf
            }
            "ThumbnailCache" {
                $cleanResult = Clear-ThumbnailCache -WhatIf:$WhatIf
            }
            default {
                if (Test-Path $target.Path) {
                    $cleanResult = Remove-FolderContents -Path $target.Path -WhatIf:$WhatIf
                }
            }
        }

        $totalFreed += $cleanResult.BytesFreed

        $status = if ($cleanResult.Success) { "Green" } else { "Yellow" }
        $statusText = if ($cleanResult.Success) { "Done" } else { "Partial" }
        Write-Log "    -> $statusText ($(Format-FileSize $cleanResult.BytesFreed))" $status $LogFile

        $results += @{
            Name = $item.Name
            BytesFreed = $cleanResult.BytesFreed
            Success = $cleanResult.Success
            Errors = $cleanResult.Errors
        }
    }

    return @{
        Results = $results
        TotalFreed = $totalFreed
        TotalFreedFormatted = Format-FileSize $totalFreed
    }
}

# ============================================
# DISPLAY FUNCTIONS
# ============================================

function Show-ScanResults {
    param([hashtable]$ScanResults)

    Write-Log "" "White" $LogFile
    Write-Summary "SCAN RESULTS" -LogFile $LogFile

    Write-Log "" "White" $LogFile
    Write-Log "  SYSTEM CLEANUP ITEMS:" "Yellow" $LogFile
    Write-Log "" "White" $LogFile

    $index = 1
    foreach ($item in $ScanResults.Items | Where-Object { $_.Key -notmatch "Cache$|RecycleBin" }) {
        $sizeColor = if ($item.Size -gt 100MB) { "Yellow" } elseif ($item.Size -gt 10MB) { "Cyan" } else { "White" }
        $adminNote = if ($item.RequiresAdmin -and -not $ScanResults.IsAdmin) { " [Requires Admin]" } else { "" }
        $accessNote = if (-not $item.Accessible) { " (Skipped)" } else { "" }

        Write-Log "  [$index] $($item.Name)$adminNote$accessNote" "White" $LogFile
        Write-Log "      Size: $($item.SizeFormatted)" $sizeColor $LogFile
        $index++
    }

    Write-Log "" "White" $LogFile
    Write-Log "  BROWSER CACHES:" "Yellow" $LogFile
    Write-Log "" "White" $LogFile

    foreach ($item in $ScanResults.Items | Where-Object { $_.Key -match "Cache$" }) {
        $sizeColor = if ($item.Size -gt 500MB) { "Yellow" } elseif ($item.Size -gt 100MB) { "Cyan" } else { "White" }

        Write-Log "  [$index] $($item.Name)" "White" $LogFile
        Write-Log "      Size: $($item.SizeFormatted)" $sizeColor $LogFile
        $index++
    }

    Write-Log "" "White" $LogFile
    Write-Log "  RECYCLE BIN:" "Yellow" $LogFile
    Write-Log "" "White" $LogFile

    $rbItem = $ScanResults.Items | Where-Object { $_.Key -eq "RecycleBin" }
    if ($rbItem) {
        $sizeColor = if ($rbItem.Size -gt 1GB) { "Yellow" } elseif ($rbItem.Size -gt 100MB) { "Cyan" } else { "White" }
        Write-Log "  [$index] $($rbItem.Name) ($($rbItem.FileCount) items)" "White" $LogFile
        Write-Log "      Size: $($rbItem.SizeFormatted)" $sizeColor $LogFile
    }

    Write-Log "" "White" $LogFile
    Write-Log "----------------------------------------" "Gray" $LogFile
    Write-Log "  TOTAL RECLAIMABLE: $($ScanResults.TotalReclaimableFormatted)" "Green" $LogFile
    Write-Log "----------------------------------------" "Gray" $LogFile
}

function Show-InteractiveMenu {
    param([hashtable]$ScanResults)

    Write-Log "" "White" $LogFile
    Write-Summary "CLEANING OPTIONS" -LogFile $LogFile
    Write-Log "" "White" $LogFile

    Write-Host "  [A] Clean ALL items ($(Format-FileSize $ScanResults.TotalReclaimable))" -ForegroundColor White
    Write-Host "  [S] Clean SAFE items only (System temp, caches)" -ForegroundColor Green
    Write-Host "  [B] Clean BROWSER caches only" -ForegroundColor Cyan
    Write-Host "  [C] Choose specific items" -ForegroundColor White
    Write-Host "  [R] Empty Recycle Bin only" -ForegroundColor Yellow
    Write-Host "  [N] Don't clean anything (scan only)" -ForegroundColor Gray
    Write-Host ""

    $choice = Read-Host "Enter your choice (A/S/B/C/R/N)"

    $selectedItems = @()

    switch ($choice.ToUpper()) {
        "A" {
            $selectedItems = $ScanResults.Items | Where-Object { $_.Accessible -and $_.Size -gt 0 }
        }
        "S" {
            $selectedItems = $ScanResults.Items | Where-Object {
                $_.Accessible -and $_.Size -gt 0 -and $_.Risk -eq "Safe" -and $_.Key -ne "RecycleBin"
            }
        }
        "B" {
            $selectedItems = $ScanResults.Items | Where-Object {
                $_.Accessible -and $_.Size -gt 0 -and $_.Key -match "Cache$"
            }
        }
        "C" {
            Write-Host ""
            Write-Host "Enter item numbers to clean (e.g., 1 3 5 7):" -ForegroundColor White
            $selection = Read-Host "Numbers"
            $numbers = $selection -split '\s+' | ForEach-Object { [int]$_ }

            $allItems = @($ScanResults.Items)
            foreach ($num in $numbers) {
                if ($num -gt 0 -and $num -le $allItems.Count) {
                    $item = $allItems[$num - 1]
                    if ($item.Accessible -and $item.Size -gt 0) {
                        $selectedItems += $item
                    }
                }
            }
        }
        "R" {
            $selectedItems = $ScanResults.Items | Where-Object { $_.Key -eq "RecycleBin" -and $_.Size -gt 0 }
        }
        "N" {
            Write-Log "" "White" $LogFile
            Write-Log "No items will be cleaned." "Gray" $LogFile
            return @()
        }
        default {
            Write-Log "" "White" $LogFile
            Write-Log "Invalid choice. No items will be cleaned." "Red" $LogFile
            return @()
        }
    }

    return $selectedItems
}

# ============================================
# MAIN EXECUTION
# ============================================

# Parse command line arguments
$Mode = "Interactive"  # Default mode
$DryRun = $false

foreach ($arg in $args) {
    switch ($arg.ToLower()) {
        "-scan" { $Mode = "ScanOnly" }
        "-clean" { $Mode = "CleanAll" }
        "-dryrun" { $DryRun = $true }
        "-whatif" { $DryRun = $true }
        "-safe" { $Mode = "CleanSafe" }
        "-browser" { $Mode = "CleanBrowser" }
    }
}

# Remove previous day's log files
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "SystemCleaner_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "SystemCleaner_$(Get-Date -Format 'yyyy-MM-dd').log" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Display banner
Write-Banner "SYSTEM CLEANER" $ScriptVersion -LogFile $LogFile
Write-Log "$(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm')" "Gray" $LogFile

if ($DryRun) {
    Write-Log "" "White" $LogFile
    Write-Log "*** DRY RUN MODE - No files will be deleted ***" "Yellow" $LogFile
}

# Run scan
$scanResults = Invoke-SystemScan

# Display results
Show-ScanResults -ScanResults $scanResults

# Save scan data for dashboard
$scanData = @{
    Type = "SystemCleaner"
    Items = $scanResults.Items
    TotalReclaimable = $scanResults.TotalReclaimable
    IsAdmin = $scanResults.IsAdmin
}
Save-ScanData -ScanData $scanData -FileName "system-cleaner-scan.json" | Out-Null

# Determine what to clean based on mode
$selectedItems = @()

switch ($Mode) {
    "ScanOnly" {
        Write-Log "" "White" $LogFile
        Write-Log "Scan complete. Use -clean, -safe, or -browser to clean items." "Cyan" $LogFile
    }
    "CleanAll" {
        $selectedItems = $scanResults.Items | Where-Object { $_.Accessible -and $_.Size -gt 0 }
    }
    "CleanSafe" {
        $selectedItems = $scanResults.Items | Where-Object {
            $_.Accessible -and $_.Size -gt 0 -and $_.Risk -eq "Safe" -and $_.Key -ne "RecycleBin"
        }
    }
    "CleanBrowser" {
        $selectedItems = $scanResults.Items | Where-Object {
            $_.Accessible -and $_.Size -gt 0 -and $_.Key -match "Cache$"
        }
    }
    "Interactive" {
        $selectedItems = Show-InteractiveMenu -ScanResults $scanResults
    }
}

# Perform cleaning if items selected
if ($selectedItems.Count -gt 0) {
    Write-Log "" "White" $LogFile
    Write-Section "CLEANING" -Number 2 -LogFile $LogFile

    $cleanResults = Invoke-Cleaning -SelectedItems $selectedItems -WhatIf:$DryRun

    # Summary
    Write-Log "" "White" $LogFile
    Write-Summary "CLEANING COMPLETE" -LogFile $LogFile
    Write-Log "" "White" $LogFile

    if ($DryRun) {
        Write-Log "  Would have freed: $($cleanResults.TotalFreedFormatted)" "Cyan" $LogFile
        Write-Log "" "White" $LogFile
        Write-Log "  Run without -dryrun to actually clean these items." "Yellow" $LogFile
    } else {
        Write-Log "  Space freed: $($cleanResults.TotalFreedFormatted)" "Green" $LogFile

        # Update history
        Add-ScanHistory -ScanSummary @{
            TotalReclaimable = $scanResults.TotalReclaimable
            TotalUpdates = 0
            ItemsCleaned = $selectedItems.Count
            SpaceFreed = $cleanResults.TotalFreed
        } | Out-Null

        # Show toast notification
        Show-ToastNotification -Title "System Cleaner" -Message "Freed $($cleanResults.TotalFreedFormatted) of disk space" -Type "Success" | Out-Null
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
