$ErrorActionPreference = "Continue"
$ScriptVersion = "2.0.0"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = "$env:USERPROFILE\Desktop\AIToolchainCleaner_$(Get-Date -Format 'yyyy-MM-dd').log"

# Import shared library
Import-Module "$ScriptRoot\Modules\SystemMaintenanceLib.psm1" -Force

# ============================================
# CONFIGURATION
# ============================================

# Try to load custom paths from config, fall back to defaults
$configPaths = Get-ToolkitConfig -ConfigName "ai-tools-paths.json"

$AIToolPaths = @{
    Ollama = @{
        Name = "Ollama Models"
        ModelsPath = if ($configPaths.Ollama.ModelsPath) { $configPaths.Ollama.ModelsPath } else { "$env:USERPROFILE\.ollama\models" }
        BlobsPath = if ($configPaths.Ollama.BlobsPath) { $configPaths.Ollama.BlobsPath } else { "$env:USERPROFILE\.ollama\models\blobs" }
        ManifestsPath = if ($configPaths.Ollama.ManifestsPath) { $configPaths.Ollama.ManifestsPath } else { "$env:USERPROFILE\.ollama\models\manifests" }
        Description = "Ollama LLM models and blobs"
        Risk = "Review"
    }
    ComfyUI = @{
        Name = "ComfyUI Cache"
        TempPath = if ($configPaths.ComfyUI.TempPath) { $configPaths.ComfyUI.TempPath } else { "$env:USERPROFILE\ComfyUI\temp" }
        OutputPath = if ($configPaths.ComfyUI.OutputPath) { $configPaths.ComfyUI.OutputPath } else { "$env:USERPROFILE\ComfyUI\output" }
        PycachePaths = @(
            "$env:USERPROFILE\ComfyUI\__pycache__",
            "$env:USERPROFILE\ComfyUI\custom_nodes\__pycache__"
        )
        Description = "ComfyUI temporary files and Python cache"
        Risk = "Safe"
    }
    LMStudio = @{
        Name = "LM Studio Cache"
        CachePath = if ($configPaths.LMStudio.CachePath) { $configPaths.LMStudio.CachePath } else { "$env:USERPROFILE\.cache\lm-studio" }
        ModelsPath = if ($configPaths.LMStudio.ModelsPath) { $configPaths.LMStudio.ModelsPath } else { "$env:USERPROFILE\.lmstudio\models" }
        Description = "LM Studio model cache and downloads"
        Risk = "Review"
    }
    NVIDIAShaderCache = @{
        Name = "NVIDIA Shader Cache"
        Paths = @(
            "$env:LOCALAPPDATA\NVIDIA\DXCache",
            "$env:LOCALAPPDATA\NVIDIA\GLCache",
            "$env:TEMP\NVIDIA Corporation"
        )
        Description = "NVIDIA compiled shader cache (rebuilds automatically)"
        Risk = "Safe"
    }
    PipCache = @{
        Name = "pip Cache"
        Path = "$env:LOCALAPPDATA\pip\cache"
        Description = "Python pip package cache"
        Risk = "Safe"
    }
    NpmCache = @{
        Name = "npm Cache"
        Path = "$env:APPDATA\npm-cache"
        Description = "Node.js npm package cache"
        Risk = "Safe"
    }
    Conda = @{
        Name = "Conda Package Cache"
        Path = "$env:USERPROFILE\.conda\pkgs"
        Description = "Conda/Anaconda package cache"
        Risk = "Safe"
    }
    HuggingFace = @{
        Name = "HuggingFace Cache"
        Path = "$env:USERPROFILE\.cache\huggingface"
        Description = "HuggingFace model and dataset cache"
        Risk = "Review"
    }
    PyTorch = @{
        Name = "PyTorch Cache"
        Path = "$env:USERPROFILE\.cache\torch"
        Description = "PyTorch model cache"
        Risk = "Review"
    }
}

# ============================================
# OLLAMA FUNCTIONS
# ============================================

function Get-OllamaModels {
    $models = @()
    $manifestsPath = $AIToolPaths.Ollama.ManifestsPath
    $blobsPath = $AIToolPaths.Ollama.BlobsPath

    if (-not (Test-Path $manifestsPath)) {
        return $models
    }

    # Get all manifest directories (registry.ollama.ai/library/modelname)
    Get-ChildItem -Path $manifestsPath -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        $manifestFile = $_
        try {
            $manifest = Get-Content $manifestFile.FullName -Raw | ConvertFrom-Json

            # Calculate model size from layers
            $totalSize = 0
            $blobFiles = @()

            foreach ($layer in $manifest.layers) {
                $digest = $layer.digest -replace "sha256:", ""
                $blobPath = Join-Path $blobsPath "sha256-$digest"
                if (Test-Path $blobPath) {
                    $blobSize = (Get-Item $blobPath -ErrorAction SilentlyContinue).Length
                    $totalSize += $blobSize
                    $blobFiles += $blobPath
                }
            }

            # Also add config blob
            if ($manifest.config.digest) {
                $configDigest = $manifest.config.digest -replace "sha256:", ""
                $configPath = Join-Path $blobsPath "sha256-$configDigest"
                if (Test-Path $configPath) {
                    $totalSize += (Get-Item $configPath -ErrorAction SilentlyContinue).Length
                    $blobFiles += $configPath
                }
            }

            # Parse model name from path
            $relativePath = $manifestFile.FullName.Replace($manifestsPath, "").TrimStart("\", "/")
            $pathParts = $relativePath -split "[/\\]"
            $modelName = if ($pathParts.Count -ge 3) {
                "$($pathParts[-2]):$($pathParts[-1])"
            } else {
                $manifestFile.BaseName
            }

            # Get last access time (approximate usage)
            $lastAccess = $manifestFile.LastAccessTime

            $models += [PSCustomObject]@{
                Name = $modelName
                Size = $totalSize
                SizeFormatted = Format-FileSize $totalSize
                LastAccess = $lastAccess
                DaysSinceAccess = [math]::Round(((Get-Date) - $lastAccess).TotalDays)
                ManifestPath = $manifestFile.FullName
                BlobFiles = $blobFiles
            }
        }
        catch {
            # Skip malformed manifests
        }
    }

    return $models | Sort-Object -Property Size -Descending
}

function Remove-OllamaModel {
    param(
        [PSCustomObject]$Model,
        [switch]$WhatIf
    )

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }

    try {
        # First try using ollama CLI if available
        $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        if ($ollamaCmd -and -not $WhatIf) {
            $modelBaseName = ($Model.Name -split ":")[0]
            & ollama rm $Model.Name 2>&1 | Out-Null
            $result.BytesFreed = $Model.Size
            return $result
        }

        # Manual removal
        foreach ($blobFile in $Model.BlobFiles) {
            if (Test-Path $blobFile) {
                $fileSize = (Get-Item $blobFile).Length
                if (-not $WhatIf) {
                    Remove-Item $blobFile -Force -ErrorAction Stop
                }
                $result.BytesFreed += $fileSize
            }
        }

        if (-not $WhatIf) {
            Remove-Item $Model.ManifestPath -Force -ErrorAction Stop
            # Clean up empty parent directories
            $parentDir = Split-Path $Model.ManifestPath -Parent
            while ($parentDir -and (Test-Path $parentDir)) {
                if ((Get-ChildItem $parentDir -Force -ErrorAction SilentlyContinue).Count -eq 0) {
                    Remove-Item $parentDir -Force -ErrorAction SilentlyContinue
                    $parentDir = Split-Path $parentDir -Parent
                } else {
                    break
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
# DOCKER FUNCTIONS
# ============================================

function Get-DockerCleanableInfo {
    $result = @{
        Available = $false
        DanglingImages = @{ Count = 0; Size = 0 }
        UnusedVolumes = @{ Count = 0; Size = 0 }
        BuildCache = @{ Count = 0; Size = 0 }
        TotalReclaimable = 0
    }

    # Check if Docker is available
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        return $result
    }

    # Check if Docker daemon is running
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $result
        }
        $result.Available = $true
    }
    catch {
        return $result
    }

    # Get dangling images
    try {
        $danglingImages = docker images -f "dangling=true" --format "{{.Size}}" 2>&1
        if ($LASTEXITCODE -eq 0 -and $danglingImages) {
            $result.DanglingImages.Count = ($danglingImages | Measure-Object).Count
            # Parse sizes (simplified - Docker returns human-readable)
            foreach ($size in $danglingImages) {
                $bytes = Convert-DockerSizeToBytes $size
                $result.DanglingImages.Size += $bytes
            }
        }
    }
    catch {}

    # Get dangling volumes
    try {
        $danglingVolumes = docker volume ls -f "dangling=true" --format "{{.Name}}" 2>&1
        if ($LASTEXITCODE -eq 0 -and $danglingVolumes) {
            $result.UnusedVolumes.Count = ($danglingVolumes | Measure-Object).Count
            # Volumes don't report size easily, estimate
            $result.UnusedVolumes.Size = $result.UnusedVolumes.Count * 100MB  # Rough estimate
        }
    }
    catch {}

    # Get build cache
    try {
        $buildCache = docker system df --format "{{.Type}}\t{{.Size}}\t{{.Reclaimable}}" 2>&1
        if ($LASTEXITCODE -eq 0) {
            foreach ($line in $buildCache) {
                if ($line -match "Build Cache") {
                    $parts = $line -split "\t"
                    if ($parts.Count -ge 3) {
                        $result.BuildCache.Size = Convert-DockerSizeToBytes $parts[2]
                    }
                }
            }
        }
    }
    catch {}

    $result.TotalReclaimable = $result.DanglingImages.Size + $result.UnusedVolumes.Size + $result.BuildCache.Size
    return $result
}

function Convert-DockerSizeToBytes {
    param([string]$SizeString)

    if (-not $SizeString) { return 0 }

    $SizeString = $SizeString.Trim()

    if ($SizeString -match "^([\d.]+)\s*(B|KB|MB|GB|TB|kB)") {
        $value = [double]$Matches[1]
        $unit = $Matches[2].ToUpper()

        switch ($unit) {
            "B" { return [long]$value }
            "KB" { return [long]($value * 1KB) }
            "MB" { return [long]($value * 1MB) }
            "GB" { return [long]($value * 1GB) }
            "TB" { return [long]($value * 1TB) }
            default { return [long]$value }
        }
    }

    return 0
}

function Invoke-DockerCleanup {
    param(
        [switch]$Images,
        [switch]$Volumes,
        [switch]$BuildCache,
        [switch]$All,
        [switch]$WhatIf
    )

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }

    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) {
        $result.Success = $false
        $result.Errors += "Docker not found"
        return $result
    }

    if ($WhatIf) {
        Write-Log "    Would run: docker system prune" "Gray" $LogFile
        return $result
    }

    try {
        if ($All) {
            $output = docker system prune -a -f --volumes 2>&1
        }
        else {
            if ($Images) {
                docker image prune -f 2>&1 | Out-Null
            }
            if ($Volumes) {
                docker volume prune -f 2>&1 | Out-Null
            }
            if ($BuildCache) {
                docker builder prune -f 2>&1 | Out-Null
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
# PIP/NPM FUNCTIONS
# ============================================

function Get-PipCacheInfo {
    $result = @{ Available = $false; Size = 0; Path = "" }

    # Check if pip is available
    $pipCmd = Get-Command pip -ErrorAction SilentlyContinue
    if (-not $pipCmd) {
        $pipCmd = Get-Command pip3 -ErrorAction SilentlyContinue
    }

    if ($pipCmd) {
        $result.Available = $true
        try {
            $cacheDir = (pip cache dir 2>&1).Trim()
            if (Test-Path $cacheDir) {
                $result.Path = $cacheDir
                $result.Size = Get-FolderSize -Path $cacheDir
            }
        }
        catch {
            # Fall back to default path
            $defaultPath = $AIToolPaths.PipCache.Path
            if (Test-Path $defaultPath) {
                $result.Path = $defaultPath
                $result.Size = Get-FolderSize -Path $defaultPath
            }
        }
    }
    else {
        # Check default path even without pip command
        $defaultPath = $AIToolPaths.PipCache.Path
        if (Test-Path $defaultPath) {
            $result.Available = $true
            $result.Path = $defaultPath
            $result.Size = Get-FolderSize -Path $defaultPath
        }
    }

    return $result
}

function Clear-PipCache {
    param([switch]$WhatIf)

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }

    $pipCmd = Get-Command pip -ErrorAction SilentlyContinue
    if (-not $pipCmd) {
        $pipCmd = Get-Command pip3 -ErrorAction SilentlyContinue
    }

    if ($pipCmd) {
        if (-not $WhatIf) {
            try {
                pip cache purge 2>&1 | Out-Null
                $result.Success = $true
            }
            catch {
                $result.Errors += $_.Exception.Message
            }
        }
    }
    else {
        # Manual cleanup
        $pipPath = $AIToolPaths.PipCache.Path
        if (Test-Path $pipPath) {
            $cleanResult = Remove-FolderContents -Path $pipPath -WhatIf:$WhatIf
            $result.BytesFreed = $cleanResult.BytesFreed
            $result.Errors = $cleanResult.Errors
        }
    }

    return $result
}

function Get-NpmCacheInfo {
    $result = @{ Available = $false; Size = 0; Path = "" }

    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmCmd) {
        $result.Available = $true
        try {
            $cacheDir = (npm config get cache 2>&1).Trim()
            if (Test-Path $cacheDir) {
                $result.Path = $cacheDir
                $result.Size = Get-FolderSize -Path $cacheDir
            }
        }
        catch {}
    }

    # Fall back to default
    if ($result.Size -eq 0) {
        $defaultPath = $AIToolPaths.NpmCache.Path
        if (Test-Path $defaultPath) {
            $result.Available = $true
            $result.Path = $defaultPath
            $result.Size = Get-FolderSize -Path $defaultPath
        }
    }

    return $result
}

function Clear-NpmCache {
    param([switch]$WhatIf)

    $result = @{ Success = $true; BytesFreed = 0; Errors = @() }

    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmCmd -and -not $WhatIf) {
        try {
            npm cache clean --force 2>&1 | Out-Null
            $result.Success = $true
        }
        catch {
            $result.Errors += $_.Exception.Message
        }
    }

    return $result
}

# ============================================
# SCANNING FUNCTIONS
# ============================================

function Invoke-AIToolchainScan {
    $results = @{
        Ollama = @{ Available = $false; Models = @(); TotalSize = 0 }
        ComfyUI = @{ Available = $false; TempSize = 0; PycacheSize = 0 }
        LMStudio = @{ Available = $false; CacheSize = 0 }
        NVIDIAShader = @{ Available = $false; Size = 0 }
        Docker = @{ Available = $false; Info = $null }
        PipCache = @{ Available = $false; Size = 0 }
        NpmCache = @{ Available = $false; Size = 0 }
        Conda = @{ Available = $false; Size = 0 }
        HuggingFace = @{ Available = $false; Size = 0 }
        PyTorch = @{ Available = $false; Size = 0 }
        TotalReclaimable = 0
    }

    Write-Section "SCANNING AI/ML TOOLCHAIN" -Number 1 -LogFile $LogFile

    # Ollama
    Write-Host "  Scanning Ollama models..." -NoNewline -ForegroundColor Gray
    if (Test-Path $AIToolPaths.Ollama.ModelsPath) {
        $results.Ollama.Available = $true
        $results.Ollama.Models = Get-OllamaModels
        $results.Ollama.TotalSize = ($results.Ollama.Models | Measure-Object -Property Size -Sum).Sum
    }
    Write-Host "`r  Ollama:          $(Format-FileSize $results.Ollama.TotalSize)                    " -ForegroundColor $(if ($results.Ollama.TotalSize -gt 0) { "Cyan" } else { "Gray" })

    # ComfyUI
    Write-Host "  Scanning ComfyUI..." -NoNewline -ForegroundColor Gray
    $comfyTempPath = $AIToolPaths.ComfyUI.TempPath
    if (Test-Path (Split-Path $comfyTempPath -Parent)) {
        $results.ComfyUI.Available = $true
        if (Test-Path $comfyTempPath) {
            $results.ComfyUI.TempSize = Get-FolderSize -Path $comfyTempPath
        }
        foreach ($pycachePath in $AIToolPaths.ComfyUI.PycachePaths) {
            if (Test-Path $pycachePath) {
                $results.ComfyUI.PycacheSize += Get-FolderSize -Path $pycachePath
            }
        }
    }
    $comfyTotal = $results.ComfyUI.TempSize + $results.ComfyUI.PycacheSize
    Write-Host "`r  ComfyUI:         $(Format-FileSize $comfyTotal)                    " -ForegroundColor $(if ($comfyTotal -gt 0) { "Cyan" } else { "Gray" })

    # LM Studio
    Write-Host "  Scanning LM Studio..." -NoNewline -ForegroundColor Gray
    if (Test-Path $AIToolPaths.LMStudio.CachePath) {
        $results.LMStudio.Available = $true
        $results.LMStudio.CacheSize = Get-FolderSize -Path $AIToolPaths.LMStudio.CachePath
    }
    Write-Host "`r  LM Studio:       $(Format-FileSize $results.LMStudio.CacheSize)                    " -ForegroundColor $(if ($results.LMStudio.CacheSize -gt 0) { "Cyan" } else { "Gray" })

    # NVIDIA Shader Cache
    Write-Host "  Scanning NVIDIA cache..." -NoNewline -ForegroundColor Gray
    foreach ($path in $AIToolPaths.NVIDIAShaderCache.Paths) {
        if (Test-Path $path) {
            $results.NVIDIAShader.Available = $true
            $results.NVIDIAShader.Size += Get-FolderSize -Path $path
        }
    }
    Write-Host "`r  NVIDIA Shader:   $(Format-FileSize $results.NVIDIAShader.Size)                    " -ForegroundColor $(if ($results.NVIDIAShader.Size -gt 0) { "Cyan" } else { "Gray" })

    # Docker
    Write-Host "  Scanning Docker..." -NoNewline -ForegroundColor Gray
    $results.Docker.Info = Get-DockerCleanableInfo
    $results.Docker.Available = $results.Docker.Info.Available
    Write-Host "`r  Docker:          $(Format-FileSize $results.Docker.Info.TotalReclaimable)                    " -ForegroundColor $(if ($results.Docker.Info.TotalReclaimable -gt 0) { "Cyan" } else { "Gray" })

    # pip cache
    Write-Host "  Scanning pip cache..." -NoNewline -ForegroundColor Gray
    $pipInfo = Get-PipCacheInfo
    $results.PipCache.Available = $pipInfo.Available
    $results.PipCache.Size = $pipInfo.Size
    Write-Host "`r  pip cache:       $(Format-FileSize $results.PipCache.Size)                    " -ForegroundColor $(if ($results.PipCache.Size -gt 0) { "Cyan" } else { "Gray" })

    # npm cache
    Write-Host "  Scanning npm cache..." -NoNewline -ForegroundColor Gray
    $npmInfo = Get-NpmCacheInfo
    $results.NpmCache.Available = $npmInfo.Available
    $results.NpmCache.Size = $npmInfo.Size
    Write-Host "`r  npm cache:       $(Format-FileSize $results.NpmCache.Size)                    " -ForegroundColor $(if ($results.NpmCache.Size -gt 0) { "Cyan" } else { "Gray" })

    # Conda
    Write-Host "  Scanning Conda..." -NoNewline -ForegroundColor Gray
    if (Test-Path $AIToolPaths.Conda.Path) {
        $results.Conda.Available = $true
        $results.Conda.Size = Get-FolderSize -Path $AIToolPaths.Conda.Path
    }
    Write-Host "`r  Conda:           $(Format-FileSize $results.Conda.Size)                    " -ForegroundColor $(if ($results.Conda.Size -gt 0) { "Cyan" } else { "Gray" })

    # HuggingFace
    Write-Host "  Scanning HuggingFace..." -NoNewline -ForegroundColor Gray
    if (Test-Path $AIToolPaths.HuggingFace.Path) {
        $results.HuggingFace.Available = $true
        $results.HuggingFace.Size = Get-FolderSize -Path $AIToolPaths.HuggingFace.Path
    }
    Write-Host "`r  HuggingFace:     $(Format-FileSize $results.HuggingFace.Size)                    " -ForegroundColor $(if ($results.HuggingFace.Size -gt 0) { "Cyan" } else { "Gray" })

    # PyTorch
    Write-Host "  Scanning PyTorch..." -NoNewline -ForegroundColor Gray
    if (Test-Path $AIToolPaths.PyTorch.Path) {
        $results.PyTorch.Available = $true
        $results.PyTorch.Size = Get-FolderSize -Path $AIToolPaths.PyTorch.Path
    }
    Write-Host "`r  PyTorch:         $(Format-FileSize $results.PyTorch.Size)                    " -ForegroundColor $(if ($results.PyTorch.Size -gt 0) { "Cyan" } else { "Gray" })

    # Calculate total
    $results.TotalReclaimable =
        $results.Ollama.TotalSize +
        $results.ComfyUI.TempSize + $results.ComfyUI.PycacheSize +
        $results.LMStudio.CacheSize +
        $results.NVIDIAShader.Size +
        $results.Docker.Info.TotalReclaimable +
        $results.PipCache.Size +
        $results.NpmCache.Size +
        $results.Conda.Size

    # Note: HuggingFace and PyTorch not included in auto-clean total (risky)

    return $results
}

# ============================================
# DISPLAY FUNCTIONS
# ============================================

function Show-OllamaModels {
    param([array]$Models)

    if ($Models.Count -eq 0) {
        Write-Log "  No Ollama models found" "Gray" $LogFile
        return
    }

    Write-Log "" "White" $LogFile
    Write-Log "  OLLAMA MODELS:" "Yellow" $LogFile
    Write-Log "" "White" $LogFile

    $index = 1
    foreach ($model in $Models) {
        $ageColor = if ($model.DaysSinceAccess -gt 30) { "Yellow" } elseif ($model.DaysSinceAccess -gt 7) { "Cyan" } else { "Green" }
        $ageNote = if ($model.DaysSinceAccess -gt 30) { " [Unused?]" } elseif ($model.DaysSinceAccess -gt 7) { " [Stale]" } else { " [Active]" }

        Write-Log "  [$index] $($model.Name)$ageNote" "White" $LogFile
        Write-Log "      Size: $($model.SizeFormatted) | Last used: $($model.DaysSinceAccess) days ago" $ageColor $LogFile
        $index++
    }
}

function Show-ScanResults {
    param([hashtable]$Results)

    Write-Log "" "White" $LogFile
    Write-Summary "SCAN RESULTS" -LogFile $LogFile

    # Ollama details
    if ($Results.Ollama.Available) {
        Show-OllamaModels -Models $Results.Ollama.Models
    }

    Write-Log "" "White" $LogFile
    Write-Log "  CACHE SUMMARY:" "Yellow" $LogFile
    Write-Log "" "White" $LogFile

    $items = @(
        @{ Name = "NVIDIA Shader Cache"; Size = $Results.NVIDIAShader.Size; Risk = "Safe" }
        @{ Name = "pip cache"; Size = $Results.PipCache.Size; Risk = "Safe" }
        @{ Name = "npm cache"; Size = $Results.NpmCache.Size; Risk = "Safe" }
        @{ Name = "Conda package cache"; Size = $Results.Conda.Size; Risk = "Safe" }
        @{ Name = "ComfyUI temp/pycache"; Size = ($Results.ComfyUI.TempSize + $Results.ComfyUI.PycacheSize); Risk = "Safe" }
        @{ Name = "LM Studio cache"; Size = $Results.LMStudio.CacheSize; Risk = "Review" }
        @{ Name = "HuggingFace cache"; Size = $Results.HuggingFace.Size; Risk = "Review" }
        @{ Name = "PyTorch cache"; Size = $Results.PyTorch.Size; Risk = "Review" }
    )

    foreach ($item in $items | Where-Object { $_.Size -gt 0 }) {
        $riskColor = if ($item.Risk -eq "Safe") { "Green" } else { "Yellow" }
        $riskTag = if ($item.Risk -eq "Review") { " [Review]" } else { "" }
        Write-Log "  - $($item.Name): $(Format-FileSize $item.Size)$riskTag" $riskColor $LogFile
    }

    if ($Results.Docker.Available) {
        Write-Log "" "White" $LogFile
        Write-Log "  DOCKER:" "Yellow" $LogFile
        Write-Log "  - Dangling images: $($Results.Docker.Info.DanglingImages.Count) ($(Format-FileSize $Results.Docker.Info.DanglingImages.Size))" "Cyan" $LogFile
        Write-Log "  - Unused volumes: $($Results.Docker.Info.UnusedVolumes.Count)" "Cyan" $LogFile
        Write-Log "  - Build cache: $(Format-FileSize $Results.Docker.Info.BuildCache.Size)" "Cyan" $LogFile
    }

    Write-Log "" "White" $LogFile
    Write-Log "----------------------------------------" "Gray" $LogFile
    Write-Log "  TOTAL RECLAIMABLE: $(Format-FileSize $Results.TotalReclaimable)" "Green" $LogFile
    Write-Log "  (Excludes HuggingFace/PyTorch - clean manually if needed)" "Gray" $LogFile
    Write-Log "----------------------------------------" "Gray" $LogFile
}

function Show-InteractiveMenu {
    param([hashtable]$Results)

    Write-Log "" "White" $LogFile
    Write-Summary "CLEANING OPTIONS" -LogFile $LogFile
    Write-Log "" "White" $LogFile

    Write-Host "  [A] Clean ALL safe caches (NVIDIA, pip, npm, Conda, ComfyUI)" -ForegroundColor Green
    Write-Host "  [D] Clean Docker (images, volumes, build cache)" -ForegroundColor Cyan
    Write-Host "  [O] Manage Ollama models (choose which to delete)" -ForegroundColor Yellow
    Write-Host "  [C] Choose specific items" -ForegroundColor White
    Write-Host "  [N] Don't clean anything (scan only)" -ForegroundColor Gray
    Write-Host ""

    $choice = Read-Host "Enter your choice (A/D/O/C/N)"

    return $choice.ToUpper()
}

function Show-OllamaMenu {
    param([array]$Models)

    if ($Models.Count -eq 0) {
        Write-Log "No Ollama models to manage" "Gray" $LogFile
        return @()
    }

    Write-Host ""
    Write-Host "Select models to DELETE (e.g., 1 3 5) or 'unused' for >30 days old:" -ForegroundColor Yellow
    $selection = Read-Host "Selection"

    if ($selection.ToLower() -eq "unused") {
        return $Models | Where-Object { $_.DaysSinceAccess -gt 30 }
    }

    $numbers = $selection -split '\s+' | ForEach-Object { [int]$_ }
    $selected = @()

    foreach ($num in $numbers) {
        if ($num -gt 0 -and $num -le $Models.Count) {
            $selected += $Models[$num - 1]
        }
    }

    return $selected
}

# ============================================
# CLEANING FUNCTIONS
# ============================================

function Invoke-SafeCacheCleanup {
    param([hashtable]$Results, [switch]$WhatIf)

    $totalFreed = 0

    # NVIDIA Shader Cache
    if ($Results.NVIDIAShader.Size -gt 0) {
        Write-Log "  Cleaning NVIDIA shader cache..." "Yellow" $LogFile
        foreach ($path in $AIToolPaths.NVIDIAShaderCache.Paths) {
            if (Test-Path $path) {
                $cleanResult = Remove-FolderContents -Path $path -WhatIf:$WhatIf
                $totalFreed += $cleanResult.BytesFreed
            }
        }
        Write-Log "    -> Done" "Green" $LogFile
    }

    # pip cache
    if ($Results.PipCache.Size -gt 0) {
        Write-Log "  Cleaning pip cache..." "Yellow" $LogFile
        $pipResult = Clear-PipCache -WhatIf:$WhatIf
        $totalFreed += $Results.PipCache.Size  # Approximate
        Write-Log "    -> Done" "Green" $LogFile
    }

    # npm cache
    if ($Results.NpmCache.Size -gt 0) {
        Write-Log "  Cleaning npm cache..." "Yellow" $LogFile
        $npmResult = Clear-NpmCache -WhatIf:$WhatIf
        $totalFreed += $Results.NpmCache.Size  # Approximate
        Write-Log "    -> Done" "Green" $LogFile
    }

    # Conda
    if ($Results.Conda.Size -gt 0) {
        Write-Log "  Cleaning Conda package cache..." "Yellow" $LogFile
        $condaResult = Remove-FolderContents -Path $AIToolPaths.Conda.Path -WhatIf:$WhatIf
        $totalFreed += $condaResult.BytesFreed
        Write-Log "    -> Done" "Green" $LogFile
    }

    # ComfyUI
    if ($Results.ComfyUI.TempSize + $Results.ComfyUI.PycacheSize -gt 0) {
        Write-Log "  Cleaning ComfyUI cache..." "Yellow" $LogFile
        if (Test-Path $AIToolPaths.ComfyUI.TempPath) {
            $cleanResult = Remove-FolderContents -Path $AIToolPaths.ComfyUI.TempPath -WhatIf:$WhatIf
            $totalFreed += $cleanResult.BytesFreed
        }
        foreach ($pycachePath in $AIToolPaths.ComfyUI.PycachePaths) {
            if (Test-Path $pycachePath) {
                $cleanResult = Remove-FolderContents -Path $pycachePath -WhatIf:$WhatIf
                $totalFreed += $cleanResult.BytesFreed
            }
        }
        Write-Log "    -> Done" "Green" $LogFile
    }

    return $totalFreed
}

# ============================================
# MAIN EXECUTION
# ============================================

$Mode = "Interactive"
$DryRun = $false

foreach ($arg in $args) {
    switch ($arg.ToLower()) {
        "-scan" { $Mode = "ScanOnly" }
        "-clean" { $Mode = "CleanAll" }
        "-dryrun" { $DryRun = $true }
        "-whatif" { $DryRun = $true }
        "-safe" { $Mode = "CleanSafe" }
        "-docker" { $Mode = "CleanDocker" }
    }
}

# Remove previous logs
Get-ChildItem -Path "$env:USERPROFILE\Desktop" -Filter "AIToolchainCleaner_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "AIToolchainCleaner_$(Get-Date -Format 'yyyy-MM-dd').log" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Display banner
Write-Banner "AI/ML TOOLCHAIN CLEANER" $ScriptVersion -LogFile $LogFile
Write-Log "$(Get-Date -Format 'dddd, MMMM dd, yyyy - HH:mm')" "Gray" $LogFile

if ($DryRun) {
    Write-Log "" "White" $LogFile
    Write-Log "*** DRY RUN MODE - No files will be deleted ***" "Yellow" $LogFile
}

# Run scan
$scanResults = Invoke-AIToolchainScan

# Display results
Show-ScanResults -Results $scanResults

# Save scan data
$scanData = @{
    Type = "AIToolchainCleaner"
    Results = $scanResults
    TotalReclaimable = $scanResults.TotalReclaimable
}
Save-ScanData -ScanData $scanData -FileName "ai-toolchain-scan.json" | Out-Null

# Handle mode
$totalFreed = 0

switch ($Mode) {
    "ScanOnly" {
        Write-Log "" "White" $LogFile
        Write-Log "Scan complete. Use -clean or -safe to clean items." "Cyan" $LogFile
    }
    "CleanAll" {
        Write-Log "" "White" $LogFile
        Write-Section "CLEANING" -Number 2 -LogFile $LogFile
        $totalFreed = Invoke-SafeCacheCleanup -Results $scanResults -WhatIf:$DryRun
        if ($scanResults.Docker.Available) {
            Write-Log "  Cleaning Docker..." "Yellow" $LogFile
            Invoke-DockerCleanup -All -WhatIf:$DryRun | Out-Null
            $totalFreed += $scanResults.Docker.Info.TotalReclaimable
            Write-Log "    -> Done" "Green" $LogFile
        }
    }
    "CleanSafe" {
        Write-Log "" "White" $LogFile
        Write-Section "CLEANING SAFE CACHES" -Number 2 -LogFile $LogFile
        $totalFreed = Invoke-SafeCacheCleanup -Results $scanResults -WhatIf:$DryRun
    }
    "CleanDocker" {
        if ($scanResults.Docker.Available) {
            Write-Log "" "White" $LogFile
            Write-Section "CLEANING DOCKER" -Number 2 -LogFile $LogFile
            Write-Log "  Running docker system prune..." "Yellow" $LogFile
            Invoke-DockerCleanup -All -WhatIf:$DryRun | Out-Null
            $totalFreed = $scanResults.Docker.Info.TotalReclaimable
            Write-Log "    -> Done" "Green" $LogFile
        }
    }
    "Interactive" {
        $choice = Show-InteractiveMenu -Results $scanResults

        switch ($choice) {
            "A" {
                Write-Log "" "White" $LogFile
                Write-Section "CLEANING SAFE CACHES" -Number 2 -LogFile $LogFile
                $totalFreed = Invoke-SafeCacheCleanup -Results $scanResults -WhatIf:$DryRun
            }
            "D" {
                if ($scanResults.Docker.Available) {
                    Write-Log "" "White" $LogFile
                    Write-Section "CLEANING DOCKER" -Number 2 -LogFile $LogFile
                    Write-Log "  Running docker system prune..." "Yellow" $LogFile
                    Invoke-DockerCleanup -All -WhatIf:$DryRun | Out-Null
                    $totalFreed = $scanResults.Docker.Info.TotalReclaimable
                    Write-Log "    -> Done" "Green" $LogFile
                } else {
                    Write-Log "Docker is not available" "Yellow" $LogFile
                }
            }
            "O" {
                $modelsToDelete = Show-OllamaMenu -Models $scanResults.Ollama.Models
                if ($modelsToDelete.Count -gt 0) {
                    Write-Log "" "White" $LogFile
                    Write-Section "REMOVING OLLAMA MODELS" -Number 2 -LogFile $LogFile
                    foreach ($model in $modelsToDelete) {
                        Write-Log "  Removing $($model.Name)..." "Yellow" $LogFile
                        $removeResult = Remove-OllamaModel -Model $model -WhatIf:$DryRun
                        $totalFreed += $removeResult.BytesFreed
                        if ($removeResult.Success) {
                            Write-Log "    -> Removed ($(Format-FileSize $model.Size))" "Green" $LogFile
                        } else {
                            Write-Log "    -> Failed: $($removeResult.Errors -join ', ')" "Red" $LogFile
                        }
                    }
                }
            }
            "C" {
                Write-Host ""
                Write-Host "Enter items to clean (nvidia, pip, npm, conda, comfyui, docker, lmstudio):" -ForegroundColor White
                $items = (Read-Host "Items") -split '\s+|,'

                Write-Log "" "White" $LogFile
                Write-Section "CLEANING SELECTED ITEMS" -Number 2 -LogFile $LogFile

                foreach ($item in $items) {
                    switch ($item.ToLower().Trim()) {
                        "nvidia" {
                            if ($scanResults.NVIDIAShader.Size -gt 0) {
                                Write-Log "  Cleaning NVIDIA shader cache..." "Yellow" $LogFile
                                foreach ($path in $AIToolPaths.NVIDIAShaderCache.Paths) {
                                    if (Test-Path $path) {
                                        $cleanResult = Remove-FolderContents -Path $path -WhatIf:$DryRun
                                        $totalFreed += $cleanResult.BytesFreed
                                    }
                                }
                                Write-Log "    -> Done" "Green" $LogFile
                            }
                        }
                        "pip" {
                            Write-Log "  Cleaning pip cache..." "Yellow" $LogFile
                            Clear-PipCache -WhatIf:$DryRun | Out-Null
                            $totalFreed += $scanResults.PipCache.Size
                            Write-Log "    -> Done" "Green" $LogFile
                        }
                        "npm" {
                            Write-Log "  Cleaning npm cache..." "Yellow" $LogFile
                            Clear-NpmCache -WhatIf:$DryRun | Out-Null
                            $totalFreed += $scanResults.NpmCache.Size
                            Write-Log "    -> Done" "Green" $LogFile
                        }
                        "conda" {
                            if ($scanResults.Conda.Size -gt 0) {
                                Write-Log "  Cleaning Conda cache..." "Yellow" $LogFile
                                $cleanResult = Remove-FolderContents -Path $AIToolPaths.Conda.Path -WhatIf:$DryRun
                                $totalFreed += $cleanResult.BytesFreed
                                Write-Log "    -> Done" "Green" $LogFile
                            }
                        }
                        "comfyui" {
                            Write-Log "  Cleaning ComfyUI cache..." "Yellow" $LogFile
                            if (Test-Path $AIToolPaths.ComfyUI.TempPath) {
                                $cleanResult = Remove-FolderContents -Path $AIToolPaths.ComfyUI.TempPath -WhatIf:$DryRun
                                $totalFreed += $cleanResult.BytesFreed
                            }
                            Write-Log "    -> Done" "Green" $LogFile
                        }
                        "docker" {
                            if ($scanResults.Docker.Available) {
                                Write-Log "  Cleaning Docker..." "Yellow" $LogFile
                                Invoke-DockerCleanup -All -WhatIf:$DryRun | Out-Null
                                $totalFreed += $scanResults.Docker.Info.TotalReclaimable
                                Write-Log "    -> Done" "Green" $LogFile
                            }
                        }
                        "lmstudio" {
                            if ($scanResults.LMStudio.CacheSize -gt 0) {
                                Write-Log "  Cleaning LM Studio cache..." "Yellow" $LogFile
                                $cleanResult = Remove-FolderContents -Path $AIToolPaths.LMStudio.CachePath -WhatIf:$DryRun
                                $totalFreed += $cleanResult.BytesFreed
                                Write-Log "    -> Done" "Green" $LogFile
                            }
                        }
                    }
                }
            }
            "N" {
                Write-Log "" "White" $LogFile
                Write-Log "No items will be cleaned." "Gray" $LogFile
            }
        }
    }
}

# Summary
if ($totalFreed -gt 0 -or $Mode -ne "ScanOnly") {
    Write-Log "" "White" $LogFile
    Write-Summary "CLEANUP COMPLETE" -LogFile $LogFile
    Write-Log "" "White" $LogFile

    if ($DryRun) {
        Write-Log "  Would have freed: $(Format-FileSize $totalFreed)" "Cyan" $LogFile
        Write-Log "" "White" $LogFile
        Write-Log "  Run without -dryrun to actually clean these items." "Yellow" $LogFile
    } else {
        Write-Log "  Space freed: $(Format-FileSize $totalFreed)" "Green" $LogFile

        # Update history
        Add-ScanHistory -ScanSummary @{
            TotalReclaimable = $scanResults.TotalReclaimable
            TotalUpdates = 0
            ItemsCleaned = 1
            SpaceFreed = $totalFreed
        } | Out-Null

        # Toast notification
        Show-ToastNotification -Title "AI Toolchain Cleaner" -Message "Freed $(Format-FileSize $totalFreed) of disk space" -Type "Success" | Out-Null
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
