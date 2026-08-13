param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Continue"
$ScriptVersion = "2.1.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentRoot = Split-Path -Parent $ScriptRoot

# Configuration (Config\dashboard-settings.json overrides these)
$Port = 8080
$MaxPort = 8090

$dashboardConfig = $null
$configPath = Join-Path $ParentRoot "Config\dashboard-settings.json"
if (Test-Path $configPath) {
    try {
        $dashboardConfig = Get-Content $configPath -Raw | ConvertFrom-Json
        if ($dashboardConfig.Server.Port) { $Port = [int]$dashboardConfig.Server.Port }
        if ($dashboardConfig.Server.MaxPort) { $MaxPort = [int]$dashboardConfig.Server.MaxPort }
    }
    catch {
        Write-Host "  Warning: Config\dashboard-settings.json could not be read ($($_.Exception.Message)); using defaults" -ForegroundColor Yellow
    }
}

# Per-session token. The dashboard can launch elevated processes, so every /api
# request must present it. It is injected into index.html when the page is
# served, which means only a document actually loaded from this server can
# obtain it - a page on another origin cannot read our HTML, so it cannot forge
# an accepted request even though the browser would happily let it POST here.
$SessionToken = [System.Guid]::NewGuid().ToString('N') + [System.Guid]::NewGuid().ToString('N')

# Import shared library. Stop rather than continue - the dashboard depends on
# Format-FileSize, Get-FolderSize and Get-WingetUpgrades, and silently carrying
# on without them produced a dashboard that showed zeros for everything.
Import-Module "$ParentRoot\Modules\SystemMaintenanceLib.psm1" -Force -ErrorAction Stop

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
        # Surfaced in the server console rather than swallowed - a failed
        # Windows Update search used to render as "0 updates available".
        Write-Host "  Windows Update search failed: $($_.Exception.Message)" -ForegroundColor Red
    }

    return $updates
}

function Get-AppUpdatesDetailed {
    # Parsing lives in SystemMaintenanceLib so the dashboard, the daily checker
    # and the interactive updater all report the same set of packages. The
    # copy that used to live here split rows on whitespace runs and dropped any
    # package whose installed version winget could not determine.
    $updates = @()

    $wingetResult = Get-WingetUpgrades -IncludeUnknown

    if ($wingetResult.Error) {
        Write-Host "  winget: $($wingetResult.Error)" -ForegroundColor Yellow
        return $updates
    }

    foreach ($package in $wingetResult.Packages) {
        $classification = Get-WingetCategory -Name $package.Name

        $updates += @{
            id = $package.Id
            name = $package.Name
            currentVersion = $package.CurrentVersion
            availableVersion = $package.NewVersion
            category = $classification.Category
            priority = $classification.Priority
        }
    }

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
    catch {
        Write-Host "  Could not size the Recycle Bin: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    return $items | Sort-Object { -$_.size }
}

function Get-SystemStatus {
    # @() forces array semantics regardless of item count. Without it, PowerShell
    # unwraps a single-item result: when there is exactly one Windows update,
    # $windowsUpdates becomes the bare hashtable itself rather than a one-element
    # array, and .Count then returns that hashtable's *key* count (10 keys) in
    # place of the update count (1). The same applies to $appUpdates and
    # $cleanable, and to the Where-Object pipe below - piping a bare hashtable
    # enumerates its entries instead of testing it as a single object.
    #
    # This was live and reproducible: with exactly one Windows update present,
    # the dashboard reported "10 Windows updates" on the Overview tab. The list
    # views were unaffected because their API routes already wrap with @(...).
    $windowsUpdates = @(Get-WindowsUpdatesDetailed)
    $appUpdates = @(Get-AppUpdatesDetailed)
    $cleanable = @(Get-CleanableItemsDetailed)

    $criticalCount = @($windowsUpdates | Where-Object { $_.severityClass -eq "critical" }).Count

    # Not "Measure-Object -Property size -Sum": Windows PowerShell 5.1 - what
    # every .bat launcher and the scheduled task actually run under - does not
    # resolve a hashtable key as a "property" for that parameter. It fails with
    # "The property 'size' cannot be found in the input for any objects", which
    # is a non-terminating error under $ErrorActionPreference="Continue", so
    # execution continued past it with a null Sum and the dashboard silently
    # showed "0 Bytes reclaimable" on every real run. PowerShell 7 resolves
    # hashtable keys as properties for this parameter, which is why this only
    # surfaced when tested under actual PowerShell 5.1 rather than pwsh.
    # $_.size (member access, not the cmdlet parameter) works on a hashtable in
    # both versions, so summing that way is used instead.
    $totalCleanable = 0
    foreach ($item in $cleanable) { $totalCleanable += $item.size }

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

function Test-WingetPackageId {
    <#
    .SYNOPSIS
        Checks that a package id is one winget currently offers as an upgrade.
    .DESCRIPTION
        The id arrives from an HTTP request body, so it is untrusted input that
        ends up on a command line. Allowing only ids present in the current
        upgrade list means an attacker cannot introduce an id of their choosing
        even if the character filter were bypassed. The character check is a
        second layer, not the primary defence.
    #>
    param([string]$Id)

    if ([string]::IsNullOrWhiteSpace($Id)) { return $false }
    if ($Id -notmatch '^[A-Za-z0-9][A-Za-z0-9._+\-]{0,127}$') { return $false }

    $offered = Get-AppUpdatesDetailed
    return [bool]($offered | Where-Object { $_.id -eq $Id })
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
                # Ids were previously joined into a string and interpolated into
                # a -Command block. An id containing a single quote closed the
                # string and ran arbitrary commands, so any web page the user had
                # open could POST here and execute code. Ids are now validated
                # against the live upgrade list and passed as discrete arguments,
                # never as text to be parsed.
                $validIds = @()
                $rejectedIds = @()

                foreach ($id in $Ids) {
                    if (Test-WingetPackageId -Id $id) { $validIds += $id }
                    else { $rejectedIds += $id }
                }

                if ($rejectedIds.Count -gt 0) {
                    Write-Host "  Rejected $($rejectedIds.Count) package id(s) not present in the current upgrade list" -ForegroundColor Yellow
                }

                if ($validIds.Count -eq 0) {
                    $result.message = "None of the requested package ids are available to upgrade"
                    return $result
                }

                foreach ($id in $validIds) {
                    Start-Process -FilePath "winget" -ArgumentList @(
                        'upgrade', '--id', $id, '--exact',
                        '--accept-source-agreements', '--accept-package-agreements'
                    )
                }

                $result.success = $true
                $result.message = "Updating $($validIds.Count) application(s)"
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

function Resolve-StaticFilePath {
    <#
    .SYNOPSIS
        Resolves a requested static file, refusing anything outside its folder.
    .DESCRIPTION
        The request path was previously concatenated straight onto the folder
        name, so "/css/../../../../Windows/win.ini" resolved to a file well
        outside the dashboard and was served. The resolved full path is now
        required to sit under the intended directory.
    #>
    param(
        [string]$BaseDirectory,
        [string]$RelativePath
    )

    if ([string]::IsNullOrWhiteSpace($RelativePath)) { return $null }

    try {
        $decoded = [System.Uri]::UnescapeDataString($RelativePath)
    }
    catch {
        return $null
    }

    if ($decoded.Contains([char]0)) { return $null }

    $baseFull = [System.IO.Path]::GetFullPath($BaseDirectory)
    if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFull += [System.IO.Path]::DirectorySeparatorChar
    }

    try {
        $candidate = [System.IO.Path]::GetFullPath((Join-Path $baseFull $decoded))
    }
    catch {
        return $null
    }

    if (-not $candidate.StartsWith($baseFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }

    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $null }

    return $candidate
}

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

    # Config\dashboard-settings.json has always carried an AutoOpenBrowser flag
    # that nothing read. It is honoured now, and -NoBrowser overrides it.
    $autoOpen = $true
    if ($null -ne $dashboardConfig -and $null -ne $dashboardConfig.Server.AutoOpenBrowser) {
        $autoOpen = [bool]$dashboardConfig.Server.AutoOpenBrowser
    }
    if ($NoBrowser) { $autoOpen = $false }

    if ($autoOpen) {
        Start-Process "http://localhost:$currentPort"
    }
    else {
        Write-Host "  Browser not opened automatically." -ForegroundColor Gray
    }

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

            # No Access-Control-Allow-Origin. The page is served from this same
            # origin so it never needed CORS, and the previous wildcard let any
            # site the user was browsing read these responses. Without it the
            # browser blocks cross-origin reads outright.
            $response.Headers.Add("Cache-Control", "no-cache, no-store, must-revalidate")
            $response.Headers.Add("Pragma", "no-cache")
            $response.Headers.Add("Expires", "0")
            $response.Headers.Add("X-Content-Type-Options", "nosniff")
            $response.Headers.Add("Referrer-Policy", "no-referrer")

            # Cross-origin preflights are refused rather than approved.
            if ($method -eq "OPTIONS") {
                $response.StatusCode = 405
                $response.Close()
                continue
            }

            # Every API route requires the token that was injected into the page
            # at load time. A cross-origin page cannot read our HTML, so it
            # cannot obtain the token, which stops a malicious site from POSTing
            # actions into this server on the user's behalf.
            if ($path -like "/api/*") {
                $providedToken = $request.Headers["X-Dashboard-Token"]

                if ($providedToken -ne $SessionToken) {
                    Write-Host "  Rejected unauthenticated $method $path from $($request.RemoteEndPoint)" -ForegroundColor Yellow
                    $response.StatusCode = 403
                    $buffer = [System.Text.Encoding]::UTF8.GetBytes('{"error":"Invalid or missing dashboard token"}')
                    $response.ContentType = "application/json"
                    $response.ContentLength64 = $buffer.Length
                    $response.OutputStream.Write($buffer, 0, $buffer.Length)
                    $response.Close()
                    continue
                }
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
                catch {
                    Write-Host "  Ignoring malformed JSON body on $method $path" -ForegroundColor Yellow
                }
            }

            switch -Regex ($path) {
                "^/$" {
                    $indexPath = Join-Path $ScriptRoot "index.html"
                    if (Test-Path $indexPath) {
                        $content = Get-Content $indexPath -Raw -Encoding UTF8
                        # Hand this session's token to the page. Only a document
                        # served from here receives it.
                        $content = $content.Replace("{{DASHBOARD_TOKEN}}", $SessionToken)
                    }
                }

                "^/css/(.+)$" {
                    $cssPath = Resolve-StaticFilePath -BaseDirectory (Join-Path $ScriptRoot "css") -RelativePath $Matches[1]
                    if ($cssPath) {
                        $content = Get-Content $cssPath -Raw -Encoding UTF8
                        $contentType = "text/css"
                    }
                    else {
                        $response.StatusCode = 404
                        $content = "Not Found"
                    }
                }

                "^/js/(.+)$" {
                    $jsPath = Resolve-StaticFilePath -BaseDirectory (Join-Path $ScriptRoot "js") -RelativePath $Matches[1]
                    if ($jsPath) {
                        $content = Get-Content $jsPath -Raw -Encoding UTF8
                        $contentType = "application/javascript"
                    }
                    else {
                        $response.StatusCode = 404
                        $content = "Not Found"
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
    catch {
        # Ctrl+C closes the listener mid-GetContext, which is expected shutdown.
        if ($_.Exception -isnot [System.Net.HttpListenerException]) {
            Write-Host ""
            Write-Host "  Dashboard stopped after an error: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    finally {
        Write-Host ""
        Write-Host "  Shutting down dashboard..." -ForegroundColor Gray
        $listener.Stop()
        $listener.Close()
    }
}

Start-HttpServer -Port $Port
