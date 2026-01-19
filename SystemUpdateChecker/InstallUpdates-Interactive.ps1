$ErrorActionPreference = "Continue"
$ScriptVersion = "1.0.0"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INTERACTIVE APPLICATION UPDATER v$ScriptVersion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for winget
$wingetPath = Get-Command winget -ErrorAction SilentlyContinue
if (-not $wingetPath) {
    Write-Host "ERROR: winget not found!" -ForegroundColor Red
    Write-Host "Install Windows Package Manager from Microsoft Store" -ForegroundColor Yellow
    pause
    exit
}

Write-Host "Checking for application updates..." -ForegroundColor Yellow
Write-Host ""

# Get available updates
$wingetOutput = winget upgrade 2>&1 | Out-String
$lines = $wingetOutput -split "`n"

# Parse the upgrade list
$updates = @()
$headerFound = $false

foreach ($line in $lines) {
    # Skip until we find the header
    if ($line -match "^Name\s+Id\s+Version\s+Available") {
        $headerFound = $true
        continue
    }

    # Skip separator line
    if ($line -match "^-+") {
        continue
    }

    # Parse update lines
    if ($headerFound -and $line.Trim() -and $line -match "\S+\s+\S+\s+[\d\.]+\s+[\d\.]+") {
        $parts = $line -split '\s{2,}'
        if ($parts.Count -ge 4) {
            $updates += [PSCustomObject]@{
                Name = $parts[0].Trim()
                Id = $parts[1].Trim()
                CurrentVersion = $parts[2].Trim()
                NewVersion = $parts[3].Trim()
                Source = if ($parts.Count -ge 5) { $parts[4].Trim() } else { "winget" }
            }
        }
    }
}

if ($updates.Count -eq 0) {
    Write-Host "All applications are up to date!" -ForegroundColor Green
    pause
    exit
}

Write-Host "Found $($updates.Count) application update(s):" -ForegroundColor Green
Write-Host ""

# Categorize updates
$securityApps = @("Chrome", "Firefox", "Edge", "VPN", "Proton")
$devTools = @("Git", "Node.js", "Python", "Visual Studio", "VS Code", "Docker")
$utilities = @("WinRAR", "7-Zip", "HWiNFO", "AIDA64")

$selectedUpdates = @()
$index = 1

foreach ($update in $updates) {
    $category = "Other"
    $recommendation = "Optional"
    $color = "White"

    # Categorize
    foreach ($app in $securityApps) {
        if ($update.Name -match $app) {
            $category = "Security/Browser"
            $recommendation = "RECOMMENDED"
            $color = "Yellow"
            break
        }
    }

    if ($category -eq "Other") {
        foreach ($app in $devTools) {
            if ($update.Name -match $app) {
                $category = "Development Tool"
                $recommendation = "Recommended if you use it"
                $color = "Cyan"
                break
            }
        }
    }

    if ($category -eq "Other") {
        foreach ($app in $utilities) {
            if ($update.Name -match $app) {
                $category = "Utility"
                $recommendation = "Optional"
                $color = "Gray"
                break
            }
        }
    }

    # Display
    Write-Host "[$index] " -NoNewline -ForegroundColor White
    Write-Host "$($update.Name)" -ForegroundColor $color
    Write-Host "    Current: $($update.CurrentVersion) -> New: $($update.NewVersion)" -ForegroundColor Gray
    Write-Host "    Category: $category" -ForegroundColor Gray
    Write-Host "    Recommendation: $recommendation" -ForegroundColor $color
    Write-Host ""

    $index++
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "How would you like to update?" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [A] Update ALL applications" -ForegroundColor White
Write-Host "  [S] Update SECURITY/BROWSER apps only (recommended)" -ForegroundColor Yellow
Write-Host "  [D] Update DEVELOPMENT TOOLS only" -ForegroundColor Cyan
Write-Host "  [C] Choose specific apps" -ForegroundColor White
Write-Host "  [N] Don't update anything" -ForegroundColor Gray
Write-Host ""

$choice = Read-Host "Enter your choice (A/S/D/C/N)"

switch ($choice.ToUpper()) {
    "A" {
        Write-Host ""
        Write-Host "Updating ALL applications..." -ForegroundColor Green
        winget upgrade --all --accept-source-agreements --accept-package-agreements
    }

    "S" {
        Write-Host ""
        Write-Host "Updating security/browser applications..." -ForegroundColor Yellow
        foreach ($update in $updates) {
            foreach ($app in $securityApps) {
                if ($update.Name -match $app) {
                    Write-Host "Updating $($update.Name)..." -ForegroundColor Yellow
                    winget upgrade --id $update.Id --accept-source-agreements --accept-package-agreements
                }
            }
        }
    }

    "D" {
        Write-Host ""
        Write-Host "Updating development tools..." -ForegroundColor Cyan
        foreach ($update in $updates) {
            foreach ($app in $devTools) {
                if ($update.Name -match $app) {
                    Write-Host "Updating $($update.Name)..." -ForegroundColor Cyan
                    winget upgrade --id $update.Id --accept-source-agreements --accept-package-agreements
                }
            }
        }
    }

    "C" {
        Write-Host ""
        Write-Host "Enter the numbers of apps to update (e.g., 1 3 5 7):" -ForegroundColor White
        $selection = Read-Host "Numbers"
        $numbers = $selection -split '\s+' | ForEach-Object { [int]$_ }

        Write-Host ""
        foreach ($num in $numbers) {
            if ($num -gt 0 -and $num -le $updates.Count) {
                $update = $updates[$num - 1]
                Write-Host "Updating $($update.Name)..." -ForegroundColor Green
                winget upgrade --id $update.Id --accept-source-agreements --accept-package-agreements
            }
        }
    }

    "N" {
        Write-Host ""
        Write-Host "No updates will be installed." -ForegroundColor Gray
    }

    default {
        Write-Host ""
        Write-Host "Invalid choice. No updates will be installed." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Update process complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
pause
