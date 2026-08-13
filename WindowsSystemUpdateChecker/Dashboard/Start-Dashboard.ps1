$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentRoot = Split-Path -Parent $ScriptRoot

# Configuration
$Port = 8080
$MaxPort = 8090

# Import shared library
Import-Module "$ParentRoot\Modules\SystemMaintenanceLib.psm1" -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SYSTEM MAINTENANCE DASHBOARD v$ScriptVersion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================
# DETAILED API FUNCTIONS
# ============================================

function Get-WindowsUpdatesDetailed {
    $updates = @()

    try {
        $updateSession = New-Object -ComObject Microsoft.Update.Session
        $updateSearcher = $updateSession.CreateUpdateSearcher()
        $searchResult = $updateSearcher.Search("IsInstalled=0")

        foreach ($update in $searchResult.Updates) {
            $severity = "Optional"
            $severityClass = "optional"

            if ($update.MsrcSeverity -eq "Critical") {
                $severity = "CRITICAL"
                $severityClass = "critical"
            }
            elseif ($update.MsrcSeverity -eq "Important") {
                $severity = "Important"
                $severityClass = "important"
            }

            $isDriver = $update.Categories | Where-Object { $_.Name -match "Driver" }

            $updates += @{
                id = $update.Identity.UpdateID
                title = $update.Title
                description = $update.Description
                severity = $severity
                severityClass = $severityClass
                type = if ($isDriver) { "Driver" } else { "Windows" }
                size = $update.MaxDownloadSize
                sizeFormatted = Format-FileSize $update.MaxDownloadSize
                kbArticles = @($update.KBArticleIDs)
                isDownloaded = $update.IsDownloaded
            }
        }
    }
    catch {
        # Silent fail
    }

    return $updates
}

function Get-AppUpdatesDetailed {
    $updates = @()

    try {
        $wingetPath = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $wingetPath) { return $updates }

        $wingetOutput = winget upgrade 2>&1 | Out-String
        $lines = $wingetOutput -split "`n"

        $headerFound = $false

        foreach ($line in $lines) {
            if ($line -match "^Name\s+Id\s+Version\s+Available") {
                $headerFound = $true
                continue
            }

            if ($line -match "^-+") { continue }

            if ($headerFound -and $line.Trim() -and $line -match "\S+\s+\S+\s+[\d\.]+\s+[\d\.]+") {
                $parts = $line -split '\s{2,}'
                if ($parts.Count -ge 4) {
                    $name = $parts[0].Trim()
                    $id = $parts[1].Trim()
                    $current = $parts[2].Trim()
                    $available = $parts[3].Trim()

                    # Categorize
                    $category = "Other"
                    $priority = "low"

                    $securityApps = @("Chrome", "Firefox", "Edge", "VPN", "Proton", "Security", "Brave")
                    $devTools = @("Git", "Node", "Python", "Visual Studio", "VS Code", "Docker", "Go", "Rust", "Java", "dotnet")

                    foreach ($app in $securityApps) {
                        if ($name -match $app) {
                            $category = "Security/Browser"
                            $priority = "high"
                            break
                        }
                    }

                    if ($category -eq "Other") {
                        foreach ($app in $devTools) {
                            if ($name -match $app) {
                                $category = "Development"
                                $priority = "medium"
                                break
                            }
                        }
                    }

                    $updates += @{
                        id = $id
                        name = $name
                        currentVersion = $current
                        availableVersion = $available
                        category = $category
                        priority = $priority
                    }
                }
            }
        }
    }
    catch {}

    return $updates
}

function Get-CleanableItemsDetailed {
    $items = @()

    # Windows Temp
    $tempPath = "$env:WINDIR\Temp"
    if (Test-Path $tempPath) {
        $size = Get-FolderSize -Path $tempPath
        $items += @{
            id = "windows-temp"
            name = "Windows Temp Files"
            path = $tempPath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "System"
            risk = "safe"
            description = "Temporary files created by Windows"
        }
    }

    # User Temp
    if (Test-Path $env:TEMP) {
        $size = Get-FolderSize -Path $env:TEMP
        $items += @{
            id = "user-temp"
            name = "User Temp Files"
            path = $env:TEMP
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "System"
            risk = "safe"
            description = "Temporary files from your applications"
        }
    }

    # Prefetch
    $prefetchPath = "$env:WINDIR\Prefetch"
    if (Test-Path $prefetchPath) {
        $size = Get-FolderSize -Path $prefetchPath
        $items += @{
            id = "prefetch"
            name = "Windows Prefetch"
            path = $prefetchPath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "System"
            risk = "safe"
            description = "Application prefetch data (rebuilds automatically)"
        }
    }

    # Windows Update Cache
    $wuPath = "$env:WINDIR\SoftwareDistribution\Download"
    if (Test-Path $wuPath) {
        $size = Get-FolderSize -Path $wuPath
        $items += @{
            id = "windows-update"
            name = "Windows Update Cache"
            path = $wuPath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "System"
            risk = "safe"
            description = "Downloaded Windows Update files"
        }
    }

    # Chrome Cache
    $chromePath = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache"
    if (Test-Path $chromePath) {
        $size = Get-FolderSize -Path $chromePath
        $chromeCodeCache = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Code Cache"
        if (Test-Path $chromeCodeCache) { $size += Get-FolderSize -Path $chromeCodeCache }
        $items += @{
            id = "chrome-cache"
            name = "Chrome Browser Cache"
            path = $chromePath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "Browser"
            risk = "safe"
            description = "Google Chrome cached files"
        }
    }

    # Firefox Cache
    $firefoxPath = "$env:LOCALAPPDATA\Mozilla\Firefox\Profiles"
    if (Test-Path $firefoxPath) {
        $size = 0
        Get-ChildItem -Path $firefoxPath -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $cachePath = Join-Path $_.FullName "cache2"
            if (Test-Path $cachePath) { $size += Get-FolderSize -Path $cachePath }
        }
        if ($size -gt 0) {
            $items += @{
                id = "firefox-cache"
                name = "Firefox Browser Cache"
                path = $firefoxPath
                size = $size
                sizeFormatted = Format-FileSize $size
                category = "Browser"
                risk = "safe"
                description = "Mozilla Firefox cached files"
            }
        }
    }

    # Edge Cache
    $edgePath = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"
    if (Test-Path $edgePath) {
        $size = Get-FolderSize -Path $edgePath
        $items += @{
            id = "edge-cache"
            name = "Edge Browser Cache"
            path = $edgePath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "Browser"
            risk = "safe"
            description = "Microsoft Edge cached files"
        }
    }

    # NVIDIA Shader Cache
    $nvidiaSize = 0
    $nvidiaPaths = @(
        "$env:LOCALAPPDATA\NVIDIA\DXCache",
        "$env:LOCALAPPDATA\NVIDIA\GLCache"
    )
    foreach ($path in $nvidiaPaths) {
        if (Test-Path $path) { $nvidiaSize += Get-FolderSize -Path $path }
    }
    if ($nvidiaSize -gt 0) {
        $items += @{
            id = "nvidia-cache"
            name = "NVIDIA Shader Cache"
            path = $nvidiaPaths[0]
            size = $nvidiaSize
            sizeFormatted = Format-FileSize $nvidiaSize
            category = "AI/ML"
            risk = "safe"
            description = "Compiled shader cache (rebuilds automatically)"
        }
    }

    # pip cache
    $pipPath = "$env:LOCALAPPDATA\pip\cache"
    if (Test-Path $pipPath) {
        $size = Get-FolderSize -Path $pipPath
        $items += @{
            id = "pip-cache"
            name = "Python pip Cache"
            path = $pipPath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "AI/ML"
            risk = "safe"
            description = "Python package cache"
        }
    }

    # npm cache
    $npmPath = "$env:APPDATA\npm-cache"
    if (Test-Path $npmPath) {
        $size = Get-FolderSize -Path $npmPath
        $items += @{
            id = "npm-cache"
            name = "npm Cache"
            path = $npmPath
            size = $size
            sizeFormatted = Format-FileSize $size
            category = "AI/ML"
            risk = "safe"
            description = "Node.js package cache"
        }
    }

    # Recycle Bin (estimate)
    try {
        $rbPath = "$env:SystemDrive\`$Recycle.Bin"
        if (Test-Path $rbPath) {
            $size = Get-FolderSize -Path $rbPath
            $items += @{
                id = "recycle-bin"
                name = "Recycle Bin"
                path = $rbPath
                size = $size
                sizeFormatted = Format-FileSize $size
                category = "System"
                risk = "review"
                description = "Deleted files (review before emptying)"
            }
        }
    }
    catch {}

    return $items | Sort-Object { -$_.size }
}

function Get-SystemStatus {
    $windowsUpdates = Get-WindowsUpdatesDetailed
    $appUpdates = Get-AppUpdatesDetailed
    $cleanable = Get-CleanableItemsDetailed

    $criticalCount = ($windowsUpdates | Where-Object { $_.severityClass -eq "critical" }).Count
    $totalCleanable = ($cleanable | Measure-Object -Property size -Sum).Sum

    # Disk info
    $sysDrive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$env:SystemDrive'" -ErrorAction SilentlyContinue

    $status = @{
        computerName = $env:COMPUTERNAME
        userName = $env:USERNAME
        timestamp = Get-Date -Format "o"
        summary = @{
            windowsUpdates = $windowsUpdates.Count
            appUpdates = $appUpdates.Count
            criticalUpdates = $criticalCount
            totalCleanable = $totalCleanable
            totalCleanableFormatted = Format-FileSize $totalCleanable
        }
        disk = @{
            drive = $env:SystemDrive
            freeBytes = if ($sysDrive) { $sysDrive.FreeSpace } else { 0 }
            totalBytes = if ($sysDrive) { $sysDrive.Size } else { 0 }
            percentFree = if ($sysDrive) { [math]::Round(($sysDrive.FreeSpace / $sysDrive.Size) * 100, 1) } else { 0 }
        }
    }

    return $status
}

function Invoke-UpdateAction {
    param(
        [string]$Type,
        [array]$Ids
    )

    $result = @{ success = $false; message = "" }

    switch ($Type) {
        "windows" {
            # Launch Windows Update installer
            $scriptPath = Join-Path $ParentRoot "InstallUpdates-Windows.ps1"
            if (Test-Path $scriptPath) {
                Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`"" -Verb RunAs
                $result.success = $true
                $result.message = "Windows Update installer launched"
            }
        }
        "apps" {
            if ($Ids -and $Ids.Count -gt 0) {
                # Update specific apps
                $idList = $Ids -join ","
                Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -Command `"foreach (`$id in '$idList' -split ',') { winget upgrade --id `$id --accept-source-agreements --accept-package-agreements }`"; pause"
                $result.success = $true
                $result.message = "Updating $($Ids.Count) application(s)"
            }
            else {
                # Launch interactive updater
                $scriptPath = Join-Path $ParentRoot "InstallUpdates-Interactive.ps1"
                if (Test-Path $scriptPath) {
                    Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`""
                    $result.success = $true
                    $result.message = "Interactive app updater launched"
                }
            }
        }
        "clean" {
            if ($Ids -and $Ids.Count -gt 0) {
                # Clean specific items - launch cleaner with items
                $scriptPath = Join-Path $ParentRoot "SystemCleaner.ps1"
                Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`" -safe"
                $result.success = $true
                $result.message = "System cleaner launched"
            }
            else {
                $scriptPath = Join-Path $ParentRoot "SystemCleaner.ps1"
                Start-Process PowerShell -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`""
                $result.success = $true
                $result.message = "System cleaner launched"
            }
        }
    }

    return $result
}

# ============================================
# HTTP SERVER
# ============================================

function Start-HttpServer {
    param([int]$Port)

    $listener = $null
    $currentPort = $Port
    $started = $false

    while ($currentPort -le $MaxPort -and -not $started) {
        try {
            $listener = New-Object System.Net.HttpListener
            $listener.Prefixes.Add("http://localhost:$currentPort/")
            $listener.Start()
            $started = $true
            Write-Host "  Dashboard running at: " -NoNewline -ForegroundColor White
            Write-Host "http://localhost:$currentPort" -ForegroundColor Green
        }
        catch {
            Write-Host "  Port $currentPort in use, trying next..." -ForegroundColor Yellow
            if ($listener) {
                try { $listener.Close() } catch {}
                $listener = $null
            }
            $currentPort++
        }
    }

    if (-not $started -or $null -eq $listener) {
        Write-Host "  Failed to start server on ports $Port-$MaxPort" -ForegroundColor Red
        Write-Host "  Make sure no other process is using these ports." -ForegroundColor Yellow
        return
    }

    Start-Process "http://localhost:$currentPort"

    Write-Host ""
    Write-Host "  Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
    Write-Host ""

    try {
        while ($listener.IsListening) {
            $context = $listener.GetContext()
            $request = $context.Request
            $response = $context.Response

            $path = $request.Url.LocalPath
            $method = $request.HttpMethod

            # CORS and Cache headers
            $response.Headers.Add("Access-Control-Allow-Origin", "*")
            $response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            $response.Headers.Add("Access-Control-Allow-Headers", "Content-Type")
            $response.Headers.Add("Cache-Control", "no-cache, no-store, must-revalidate")
            $response.Headers.Add("Pragma", "no-cache")
            $response.Headers.Add("Expires", "0")

            if ($method -eq "OPTIONS") {
                $response.StatusCode = 200
                $response.Close()
                continue
            }

            $content = ""
            $contentType = "text/html"

            # Parse POST body if present
            $body = $null
            if ($method -eq "POST" -and $request.HasEntityBody) {
                $reader = New-Object System.IO.StreamReader($request.InputStream)
                $bodyText = $reader.ReadToEnd()
                $reader.Close()
                try {
                    $body = $bodyText | ConvertFrom-Json
                }
                catch {}
            }

            switch -Regex ($path) {
                "^/$" {
                    $indexPath = Join-Path $ScriptRoot "index.html"
                    if (Test-Path $indexPath) {
                        $content = Get-Content $indexPath -Raw -Encoding UTF8
                    }
                }

                "^/css/(.+)$" {
                    $cssPath = Join-Path $ScriptRoot "css\$($Matches[1])"
                    if (Test-Path $cssPath) {
                        $content = Get-Content $cssPath -Raw -Encoding UTF8
                        $contentType = "text/css"
                    }
                }

                "^/js/(.+)$" {
                    $jsPath = Join-Path $ScriptRoot "js\$($Matches[1])"
                    if (Test-Path $jsPath) {
                        $content = Get-Content $jsPath -Raw -Encoding UTF8
                        $contentType = "application/javascript"
                    }
                }

                "^/api/status$" {
                    $content = Get-SystemStatus | ConvertTo-Json -Depth 10
                    $contentType = "application/json"
                }

                "^/api/updates/windows$" {
                    $content = @{ items = @(Get-WindowsUpdatesDetailed) } | ConvertTo-Json -Depth 10
                    $contentType = "application/json"
                }

                "^/api/updates/apps$" {
                    $content = @{ items = @(Get-AppUpdatesDetailed) } | ConvertTo-Json -Depth 10
                    $contentType = "application/json"
                }

                "^/api/cleanable$" {
                    $content = @{ items = @(Get-CleanableItemsDetailed) } | ConvertTo-Json -Depth 10
                    $contentType = "application/json"
                }

                "^/api/action/update$" {
                    $type = if ($body.type) { $body.type } else { "apps" }
                    $ids = if ($body.ids) { $body.ids } else { @() }
                    $result = Invoke-UpdateAction -Type $type -Ids $ids
                    $content = $result | ConvertTo-Json
                    $contentType = "application/json"
                }

                "^/api/action/clean$" {
                    $ids = if ($body.ids) { $body.ids } else { @() }
                    $result = Invoke-UpdateAction -Type "clean" -Ids $ids
                    $content = $result | ConvertTo-Json
                    $contentType = "application/json"
                }

                "^/api/history$" {
                    $historyPath = Join-Path $ParentRoot "Data\scan-history.json"
                    if (Test-Path $historyPath) {
                        $content = Get-Content $historyPath -Raw
                    }
                    else {
                        $content = "[]"
                    }
                    $contentType = "application/json"
                }

                default {
                    $response.StatusCode = 404
                    $content = "Not Found"
                }
            }

            $buffer = [System.Text.Encoding]::UTF8.GetBytes($content)
            $response.ContentType = $contentType
            $response.ContentLength64 = $buffer.Length
            $response.OutputStream.Write($buffer, 0, $buffer.Length)
            $response.Close()
        }
    }
    catch {}
    finally {
        $listener.Stop()
        $listener.Close()
    }
}

Start-HttpServer -Port $Port
