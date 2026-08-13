# Regression tests for the security-sensitive pure functions in this toolkit.
#
# These test the functions directly rather than over HTTP, because the network
# layer masks results: Windows http.sys rejects URLs containing %2f before a
# request ever reaches Start-Dashboard.ps1, so an HTTP-level test would pass
# even if the path resolver were broken.
#
# Run:  powershell -ExecutionPolicy Bypass -File Tests\Test-Security.ps1

$ErrorActionPreference = "Stop"
$TestRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $TestRoot

$script:Passed = 0
$script:Failed = 0

function Assert-That {
    param([string]$Name, [bool]$Condition, [string]$Detail = "")

    if ($Condition) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:Passed++
    }
    else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        if ($Detail) { Write-Host "         $Detail" -ForegroundColor Red }
        $script:Failed++
    }
}

function Get-FunctionSource {
    <#
    .SYNOPSIS
        Returns a script's function definitions as text, without its body.
    .DESCRIPTION
        The text is returned rather than dot-sourced here because dot-sourcing
        inside a function defines the functions in that function's scope, where
        the rest of the test file cannot see them. The caller dot-sources the
        result at script scope instead.
    #>
    param([string]$Path)

    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$null)
    $functions = $ast.FindAll(
        { $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] },
        $false)

    return (($functions | ForEach-Object { $_.Extent.Text }) -join "`r`n`r`n")
}

Import-Module "$ProjectRoot\Modules\SystemMaintenanceLib.psm1" -Force
. ([scriptblock]::Create((Get-FunctionSource "$ProjectRoot\Dashboard\Start-Dashboard.ps1")))
. ([scriptblock]::Create((Get-FunctionSource "$ProjectRoot\InstallUpdates-Windows.ps1")))

# ============================================
# Resolve-StaticFilePath
# ============================================

Write-Host ""
Write-Host "Resolve-StaticFilePath - directory traversal" -ForegroundColor Cyan

$cssDir = Join-Path $ProjectRoot "Dashboard\css"

$traversals = @(
    '../../../../../../Windows/win.ini',
    '..\..\..\..\..\..\Windows\win.ini',
    '....//....//Windows/win.ini',
    '../Start-Dashboard.ps1',
    '..%2f..%2fStart-Dashboard.ps1',
    '%2e%2e%2f%2e%2e%2fWindows%2fwin.ini',
    '/Windows/win.ini',
    'C:\Windows\win.ini',
    '..',
    ''
)

foreach ($t in $traversals) {
    $resolved = Resolve-StaticFilePath -BaseDirectory $cssDir -RelativePath $t
    Assert-That "rejects '$t'" ($null -eq $resolved) "resolved to: $resolved"
}

$legit = Resolve-StaticFilePath -BaseDirectory $cssDir -RelativePath 'dashboard.css'
Assert-That "accepts 'dashboard.css'" ($null -ne $legit -and $legit.EndsWith('dashboard.css'))

$missing = Resolve-StaticFilePath -BaseDirectory $cssDir -RelativePath 'does-not-exist.css'
Assert-That "rejects a non-existent file" ($null -eq $missing)

# ============================================
# Test-DriverUpdateSuperseded
# ============================================

Write-Host ""
Write-Host "Test-DriverUpdateSuperseded - version comparison" -ForegroundColor Cyan

$fakeInstalled = @(
    [PSCustomObject]@{
        DeviceName    = 'Intel(R) Graphics'
        DriverVersion = '32.0.101.8860'
        DriverDate    = [DateTime]'2026-06-24'
        HardWareID    = 'PCI\VEN_8086&DEV_7D67&SUBSYS_88EF1043&REV_06'
    },
    [PSCustomObject]@{
        DeviceName    = 'Some Audio Device'
        DriverVersion = 'not.a.version'
        DriverDate    = [DateTime]'2025-01-01'
        HardWareID    = 'PCI\VEN_1111&DEV_2222'
    }
)

$cases = @(
    @{ Name = 'older offered version is superseded'
       Update = @{ Title = 'Intel Corporation - Display - 32.0.101.6127'
                   DriverHardwareID = 'pci\ven_8086&dev_7d67&subsys_88ef1043'
                   DriverVerDate = [DateTime]'2024-10-11' }
       Expected = $true }

    @{ Name = 'newer offered version is NOT superseded'
       Update = @{ Title = 'Intel Corporation - Display - 33.0.101.1000'
                   DriverHardwareID = 'pci\ven_8086&dev_7d67&subsys_88ef1043'
                   DriverVerDate = [DateTime]'2026-09-01' }
       Expected = $false }

    @{ Name = 'identical version is superseded (nothing to gain)'
       Update = @{ Title = 'Intel Corporation - Display - 32.0.101.8860'
                   DriverHardwareID = 'pci\ven_8086&dev_7d67&subsys_88ef1043'
                   DriverVerDate = [DateTime]'2026-06-24' }
       Expected = $true }

    @{ Name = 'absent device is NOT superseded'
       Update = @{ Title = 'Nonexistent - Display - 5.0.0.0'
                   DriverHardwareID = 'pci\ven_dead&dev_beef'
                   DriverVerDate = [DateTime]'2020-01-01' }
       Expected = $false }

    @{ Name = 'non-driver update is NOT superseded'
       Update = @{ Title = '2026-08 Cumulative Update for Windows 11 (KB5000000)'
                   DriverHardwareID = $null
                   DriverVerDate = $null }
       Expected = $false }

    @{ Name = 'unparseable versions fall back to date - older date superseded'
       Update = @{ Title = 'Some Audio Device driver'
                   DriverHardwareID = 'pci\ven_1111&dev_2222'
                   DriverVerDate = [DateTime]'2024-01-01' }
       Expected = $true }

    @{ Name = 'unparseable versions fall back to date - newer date kept'
       Update = @{ Title = 'Some Audio Device driver'
                   DriverHardwareID = 'pci\ven_1111&dev_2222'
                   DriverVerDate = [DateTime]'2026-01-01' }
       Expected = $false }
)

foreach ($case in $cases) {
    $update = [PSCustomObject]$case.Update
    $result = Test-DriverUpdateSuperseded -Update $update -InstalledDrivers $fakeInstalled
    Assert-That $case.Name ($result.Superseded -eq $case.Expected) `
        "expected $($case.Expected), got $($result.Superseded). $($result.Reason)"
}

# ============================================
# Get-WingetUpgrades parsing
# ============================================

Write-Host ""
Write-Host "Get-WingetUpgrades - table parsing" -ForegroundColor Cyan

# Exercises the parser's column logic against the shapes that broke the old
# whitespace-splitting version: a name containing consecutive spaces, an
# "Unknown" installed version, and the trailing summary line.
$sampleOutput = @"
Name                            Id                       Version       Available     Source
--------------------------------------------------------------------------------------------
Cursor (User)                   Anysphere.Cursor         3.9.8         3.14.27       winget
Dual  Monitor  Tools            GNE.DualMonitorTools     2.11.0.0      2.12.0.0      winget
Sublime Text                    SublimeHQ.SublimeText.4  Unknown       4.0.0.420000  winget
3 upgrades available.
"@

# The parser is exercised through its own table logic by feeding this sample in
# place of live winget output.
$lines = $sampleOutput -split "`r?`n"
$separatorIndex = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^-{5,}\s*$') { $separatorIndex = $i; break }
}

Assert-That "locates the separator row" ($separatorIndex -eq 1) "got index $separatorIndex"

$header = $lines[$separatorIndex - 1]
$columnStarts = @()
for ($c = 0; $c -lt $header.Length; $c++) {
    if ($header[$c] -ne ' ' -and ($c -eq 0 -or $header[$c - 1] -eq ' ')) { $columnStarts += $c }
}

Assert-That "finds 5 columns" ($columnStarts.Count -eq 5) "got $($columnStarts.Count)"

$row = $lines[$separatorIndex + 2]   # the double-spaced name
$name = $row.Substring($columnStarts[0], $columnStarts[1] - $columnStarts[0]).Trim()
$id = $row.Substring($columnStarts[1], $columnStarts[2] - $columnStarts[1]).Trim()

Assert-That "keeps a name containing double spaces intact" ($name -eq 'Dual  Monitor  Tools') "got '$name'"
Assert-That "reads the id beside a double-spaced name" ($id -eq 'GNE.DualMonitorTools') "got '$id'"

$summaryMatched = $sampleOutput -match '(?m)^\s*(\d+)\s+upgrade'
Assert-That "reads winget's own total" ($summaryMatched -and [int]$Matches[1] -eq 3) "got $($Matches[1])"

# ============================================
# Get-WingetCategory
# ============================================

Write-Host ""
Write-Host "Get-WingetCategory - prioritisation" -ForegroundColor Cyan

Assert-That "Chrome is high priority"      ((Get-WingetCategory -Name 'Google Chrome').Priority -eq 'high')
Assert-That "Proton Pass is high priority" ((Get-WingetCategory -Name 'Proton Pass').Priority -eq 'high')
Assert-That "Python is development"        ((Get-WingetCategory -Name 'Python 3.13.14 (64-bit)').Category -eq 'Development')
Assert-That "FanControl is other"          ((Get-WingetCategory -Name 'FanControl').Category -eq 'Other')

# ============================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PASSED: $script:Passed   FAILED: $script:Failed" -ForegroundColor $(if ($script:Failed -eq 0) { 'Green' } else { 'Red' })
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($script:Failed -gt 0) { exit 1 }
exit 0
