# System Maintenance Toolkit v2.0.0

A comprehensive Windows system maintenance toolkit that combines update checking, system cleaning, driver management, and startup optimization - all in one place with a modern web dashboard.

## What's New in v2.0.0

- **Web Dashboard** - Beautiful browser-based UI to monitor and manage your system
- **System Cleaner** - Clean temp files, browser caches, and Windows junk
- **AI/ML Toolchain Cleaner** - Specialized cleaning for Ollama, Docker, pip, npm, NVIDIA caches
- **Driver Backup/Restore** - Export and restore third-party drivers with System Restore points
- **Startup Manager** - View, analyze, and optimize startup items
- **HTML Reports** - Generate detailed system reports
- **Unified Launcher** - One menu to access everything

## Quick Start

### Main Launcher (Recommended)
```
Double-click: SystemMaintenance.bat
```

This opens the unified menu with all options:
- Scan for updates
- Clean system junk
- Manage drivers
- Optimize startup
- Generate reports
- Open web dashboard

### Web Dashboard
```
Double-click: Dashboard\Start-Dashboard.bat
```

Opens a modern web interface at `http://localhost:8080` with:
- Real-time system status
- One-click actions
- Historical charts
- Quick access to all tools

## All Tools

| Launcher | Purpose |
|----------|---------|
| `SystemMaintenance.bat` | **Main launcher** - unified menu for everything |
| `Dashboard\Start-Dashboard.bat` | Web-based dashboard UI |
| `RunUpdateChecker.bat` | Scan for Windows, driver, and app updates |
| `InstallUpdates-Interactive.bat` | Choose which apps to update |
| `InstallUpdates-Windows.bat` | Install Windows/driver updates (admin) |
| `SystemCleaner.bat` | Clean temp files and caches |
| `AIToolchainCleaner.bat` | Clean AI/ML toolchain caches |
| `DriverBackup-Admin.bat` | Backup/restore drivers (admin) |
| `StartupManager.bat` | Manage startup items |

## Features

### System Cleaner
- Windows temp files (`%TEMP%`, `Windows\Temp`)
- Windows Prefetch
- Windows Update cache
- Thumbnail cache
- Browser caches (Chrome, Firefox, Edge)
- Recycle Bin analysis
- **Dry-run mode** - see what would be deleted before cleaning
- **Interactive mode** - choose what to clean

### AI/ML Toolchain Cleaner
- **Ollama** - Model inventory, unused model detection, cleanup
- **ComfyUI** - Temp files, `__pycache__` cleanup
- **LM Studio** - Cache cleanup
- **Docker** - Unused images, volumes, build cache
- **NVIDIA** - Shader cache (DXCache, GLCache)
- **pip** - Package cache
- **npm** - Package cache
- **Conda** - Package cache
- **HuggingFace/PyTorch** - Cache analysis (manual cleanup)

### Driver Backup/Restore
- Export all third-party (non-Microsoft) drivers
- Creates System Restore Point before operations
- Timestamped backup folders
- Restore individual or all drivers
- View installed third-party drivers

### Startup Manager
- Registry Run keys (HKCU and HKLM)
- Startup folders (User and All Users)
- Scheduled tasks (logon triggers)
- **Impact analysis** - High/Medium/Low/Essential ratings
- **Category detection** - Security, Cloud Sync, Gaming, etc.
- Disable with backup, re-enable later
- Protects essential/security items

### System Reports
- Beautiful HTML reports with dark theme
- System overview (CPU, RAM, Disk)
- Update summary
- Cleanable space analysis
- Startup item count
- Recommendations

### Web Dashboard
- Real-time system status cards
- One-click scan and clean actions
- Historical charts (space freed over time)
- Quick access to all tools
- Auto-refresh every 60 seconds

## Menu Structure

```
SYSTEM MAINTENANCE TOOLKIT v2.0.0

SCAN / CHECK (read-only)
  [1] Scan for Updates (Windows + Drivers + Apps)
  [2] Scan for Cleanable Items (System + AI/ML)
  [3] Scan Everything (Full System Report)

UPDATE (install updates)
  [4] Update Windows & Drivers
  [5] Update Applications (Interactive)
  [6] Update ALL (Windows + Drivers + Apps)

CLEAN (remove junk)
  [7] Clean System (Temp, Cache, Browser)
  [8] Clean AI/ML Toolchain (Ollama, Docker, pip, npm)
  [9] Clean ALL (System + AI/ML)

FULL MAINTENANCE
  [F] FULL MAINTENANCE (Update ALL + Clean ALL)

TOOLS
  [D] Driver Backup/Restore
  [S] Startup Manager
  [R] Generate HTML Report
  [W] Open Web Dashboard

[Q] Quit
```

## Command Line Options

All scripts support command-line arguments for automation:

```powershell
# System Cleaner
SystemCleaner.ps1 -scan      # Scan only (no changes)
SystemCleaner.ps1 -clean     # Clean all items
SystemCleaner.ps1 -safe      # Clean safe items only
SystemCleaner.ps1 -dryrun    # Show what would be cleaned

# AI Toolchain Cleaner
AIToolchainCleaner.ps1 -scan
AIToolchainCleaner.ps1 -safe
AIToolchainCleaner.ps1 -docker

# Startup Manager
StartupManager.ps1 -list
StartupManager.ps1 -analyze

# System Report
SystemReport.ps1 -generate
SystemReport.ps1 -notify

# Main Launcher
SystemMaintenance.ps1 -scan
SystemMaintenance.ps1 -update
SystemMaintenance.ps1 -clean
SystemMaintenance.ps1 -full
SystemMaintenance.ps1 -dashboard
```

## File Structure

```
SystemUpdateChecker/
├── SystemMaintenance.bat         # Main launcher
├── SystemMaintenance.ps1
├── DailyUpdateChecker.ps1        # Update scanner
├── InstallUpdates-Interactive.ps1
├── InstallUpdates-Windows.ps1
├── SystemCleaner.ps1             # System cleaner
├── SystemCleaner.bat
├── AIToolchainCleaner.ps1        # AI/ML cleaner
├── AIToolchainCleaner.bat
├── DriverBackup.ps1              # Driver backup/restore
├── DriverBackup-Admin.bat
├── StartupManager.ps1            # Startup manager
├── StartupManager.bat
├── SystemReport.ps1              # Report generator
├── Modules/
│   └── SystemMaintenanceLib.psm1 # Shared functions
├── Dashboard/
│   ├── Start-Dashboard.ps1       # Web server
│   ├── Start-Dashboard.bat
│   ├── index.html
│   ├── css/dashboard.css
│   └── js/app.js
├── Config/
│   ├── ai-tools-paths.json       # AI tool paths
│   ├── cleaner-rules.json        # Cleaning rules
│   ├── dashboard-settings.json   # Dashboard config
│   └── startup-backup/           # Disabled startup items
├── Data/
│   ├── last-scan.json            # Latest scan results
│   └── scan-history.json         # Historical data
├── Reports/                      # Generated HTML reports
├── DriverBackups/                # Driver backup folders
└── README.md
```

## Configuration

### AI Tools Paths (`Config/ai-tools-paths.json`)
Customize paths for your AI tools if they're not in default locations:
```json
{
    "Ollama": {
        "ModelsPath": "D:\\AI\\ollama\\models"
    }
}
```

### Cleaner Rules (`Config/cleaner-rules.json`)
Customize what gets cleaned automatically vs. requiring confirmation.

### Dashboard Settings (`Config/dashboard-settings.json`)
Change port, auto-refresh interval, notification settings.

## Requirements

- Windows 10/11
- PowerShell 5.1+ (pre-installed)
- winget (for app updates) - [Install from Microsoft Store](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1)
- Administrator rights for some features (driver backup, Windows updates)

## Tips

1. **First run**: Use `SystemMaintenance.bat` and try option [3] for a full system scan
2. **Regular maintenance**: Run [F] Full Maintenance weekly
3. **AI developers**: Use [8] regularly to clean pip/npm caches
4. **Before driver updates**: Use [D] to backup drivers first
5. **Slow boot?**: Use [S] Startup Manager to analyze and optimize

## Logs

All operations create logs on your Desktop:
- `UpdateCheck_YYYY-MM-DD.log`
- `SystemCleaner_YYYY-MM-DD.log`
- `AIToolchainCleaner_YYYY-MM-DD.log`
- `DriverBackup_YYYY-MM-DD.log`
- `StartupManager_YYYY-MM-DD.log`

## Troubleshooting

**Dashboard won't start:**
- Port 8080 may be in use - it will try ports 8080-8090
- Check firewall settings

**Scripts won't run:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**winget not found:**
- Install from Microsoft Store: [App Installer](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1)

**Antivirus blocks scripts:**
- Add folder to exclusions: `C:\Users\YourName\SystemUpdateChecker`

## Version History

### v2.0.0 (Current)
- Added System Cleaner module
- Added AI/ML Toolchain Cleaner
- Added Driver Backup/Restore
- Added Startup Manager
- Added Web Dashboard
- Added HTML Report Generator
- Added unified launcher menu
- Reorganized file structure

### v1.0.0
- Initial release
- Daily update checker
- Interactive app updater
- Windows update installer

## License

MIT License - Free to use and modify.

---

**System Maintenance Toolkit** - Keep your system clean, updated, and optimized.
