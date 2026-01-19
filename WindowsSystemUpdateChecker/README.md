# Windows System Update Checker

A simple, automated daily checker for Windows updates, driver updates, and application updates with smart recommendations.

## 🎯 What It Does

- **Checks for updates** daily: Windows Updates, Drivers, and Applications (via winget)
- **Explains what matters**: Instead of just listing updates, it tells you what to actually install and what to ignore
- **Interactive installer**: Choose which apps to update with a simple menu
- **Smart recommendations**: Distinguishes between critical security updates and optional updates
- **Daily logs**: Creates a log file on your Desktop each day with findings

## 🚀 Quick Start

1. Download/clone this repository to `C:\Users\YourName\SystemUpdateChecker`
2. Double-click `UpdateTaskLocation-Admin.bat` to set up daily automatic checks (requires admin)
3. Choose what time you want it to run daily (default: 9:00 AM)
4. Done! It will check for updates automatically and create logs on your Desktop

### Manual Usage

Want to check now without scheduling?
- Double-click `RunUpdateChecker.bat` - See what needs updating
- Double-click `InstallUpdates-Interactive.bat` - Choose which apps to update

## 📁 Files Included

| File | Purpose |
|------|---------|
| `DailyUpdateChecker.ps1` | Main script that checks for all updates |
| `RunUpdateChecker.bat` | Quick launcher to check for updates |
| `InstallUpdates-Interactive.bat` | Choose which applications to update |
| `InstallUpdates-Windows.bat` | Install Windows/Driver updates (admin required) |
| `CheckTaskStatus.bat` | See if scheduled task is running |
| `UpdateTaskLocation-Admin.bat` | Set up or change daily schedule time |
| `HOW-TO-READ-UPDATES.txt` | Reference guide for interpreting results |
| `README.txt` | Local documentation |

## ✨ Key Features

### Smart Analysis

Instead of dumping raw update data, the checker explains:
- ✅ **What to install** - Security updates, critical patches
- ⚠️ **What's optional** - Non-critical updates
- ❌ **What to ignore** - Updates you don't need (older versions, Windows built-in drivers)

### Example Output

```
[1] CHECKING WINDOWS UPDATES...
Found 2 Windows update(s):
  Security Update for Windows (KB5xxxxx)
  Driver Update - Display Adapter

[2] CHECKING DRIVER UPDATES...
Found 1 driver update(s):
  Display Adapter - Version X.X.X.X

[3] APPLICATION UPDATES...
Found 15 application updates available:
  Chrome, Git, Node.js, Visual Studio...

========================================
WHAT YOU SHOULD ACTUALLY DO
========================================

WINDOWS/DRIVER UPDATES:
  You have 2 update(s) available
  -> Review the list above for specific recommendations

  Note: Old drivers (2006 dates) for USB/Bluetooth are usually
  Windows built-in drivers and can be ignored unless you have issues

APPLICATIONS TO UPDATE:
  Found 15 application update(s)

  Priority recommendations:
    - Security/Browsers (Chrome, Firefox, Edge): RECOMMENDED
    - Development tools: Update if you use them
    - Other apps: Update when convenient

========================================
BOTTOM LINE
========================================

WHAT TO DO NOW:
  1. Run: InstallUpdates-Interactive.bat
  2. Choose [S] for Security apps (Chrome, VPN)
  3. Or choose [C] to pick specific apps

Everything else can be ignored!
```

## 📊 What It Checks

### Windows Updates
- All available Windows Updates
- Identifies critical vs optional
- Filters out updates you don't need

### Driver Updates
- Available driver updates from Windows Update
- Analyzes driver ages
- Explains which old drivers are normal (Windows built-in from 2006)
- Recommends which drivers actually need attention

### Application Updates (via winget)
- All apps managed by Windows Package Manager
- Categorizes by priority:
  - **Security/Browsers** (Chrome, Firefox, VPN) - Recommended
  - **Development Tools** (Git, Node.js, Visual Studio) - If you use them
  - **Utilities** (WinRAR, etc.) - Optional

### System Health
- Device Manager errors
- Windows Defender signature status
- Explains what errors are normal vs need fixing

## 🛠️ Installation

### Prerequisites
- Windows 10/11
- PowerShell (pre-installed on Windows)
- Windows Package Manager (winget) - [Install from Microsoft Store](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1) if not already installed

### Setup

1. **Clone or download** this repository to your user folder:
   ```
   C:\Users\YourName\SystemUpdateChecker
   ```

2. **Set up daily automatic checks** (optional):
   - Right-click `UpdateTaskLocation-Admin.bat` and "Run as Administrator"
   - Choose what time you want daily checks (e.g., 9:00 AM)
   - Creates a Windows Task Scheduler task

3. **Add antivirus exclusion** (if needed):

   Some antivirus software may flag PowerShell scripts as suspicious (false positive).

   **Option 1 - Windows Security GUI:**
   - Open Windows Security → Virus & threat protection
   - Manage settings → Exclusions → Add exclusion
   - Add folder: `C:\Users\YourName\SystemUpdateChecker`

   **Option 2 - PowerShell (run as admin):**
   ```powershell
   Add-MpPreference -ExclusionPath "C:\Users\YourName\SystemUpdateChecker"
   ```

## 📖 Usage

### Daily Automatic Checks

Once set up, the checker runs automatically each day at your chosen time. You'll know it ran because:
- A log file appears on your Desktop: `UpdateCheck_2025-10-13.log`
- Open the log to see what was found
- Look for the "WHAT YOU SHOULD ACTUALLY DO" section at the bottom

### Manual Checks

**Check for updates anytime:**
```
Double-click: RunUpdateChecker.bat
```

**Install application updates interactively:**
```
Double-click: InstallUpdates-Interactive.bat
```

Choose from:
- `[A]` Update ALL applications
- `[S]` Update SECURITY apps only (recommended)
- `[D]` Update DEVELOPMENT tools only
- `[C]` Choose specific apps by number
- `[N]` Don't update anything

**Install Windows/Driver updates:**
```
Double-click: InstallUpdates-Windows.bat
```
(Requires administrator privileges)

## 🔧 Managing the Scheduled Task

### Check Status
```
Double-click: CheckTaskStatus.bat
```

Shows:
- Is the task enabled?
- When did it last run?
- When will it run next?
- Last run result

### PowerShell Commands

**View task:**
```powershell
Get-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

**Run manually:**
```powershell
Start-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

**Disable (pause automatic checks):**
```powershell
Disable-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

**Enable (resume automatic checks):**
```powershell
Enable-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

**Remove task completely:**
```powershell
Unregister-ScheduledTask -TaskName "DailySystemUpdateChecker" -Confirm:$false
```

**Change schedule time:**
```
Double-click: UpdateTaskLocation-Admin.bat
```

## 🤔 Understanding the Results

### Common Scenarios

**"Drivers from 2006-06-21"**
- ✅ **Normal** - These are Windows inbox drivers (USB, Bluetooth stack, etc.)
- These are maintained automatically through Windows Update
- No action needed

**"Driver update shows older version than installed"**
- Check your current version first in Device Manager
- If you have a newer version already, **ignore** this update
- Windows Update cache can be outdated

**"Error Code 22 - Device disabled"**
- ✅ **Normal** - Not actually an error
- The device is manually disabled by you
- No action needed

**"Error Code 52 - Unsigned driver"**
- Usually antivirus or security software drivers
- If the software works, **ignore** this

**"30+ application updates available"**
- **Priority 1:** Browsers (Chrome, Firefox, Edge) - Security important
- **Priority 2:** Development tools you actually use
- **Priority 3:** Everything else - update when convenient

## 🛡️ Security & Privacy

- **No data collection** - Everything runs locally on your machine
- **Read-only checks** - The checker only reads system information
- **Manual installation** - Updates are never installed automatically without your approval
- **Open source** - You can review all scripts before running

## ❓ FAQ

**Q: Will this install updates automatically?**
A: No. It only checks and reports. You must run the install scripts manually.

**Q: Why does my antivirus flag this?**
A: PowerShell scripts that check system settings can trigger heuristic detection (IDP.Generic). This is a false positive. Add the folder to your antivirus exclusions.

**Q: Can I run this without Administrator privileges?**
A: Yes for checking updates. Admin is only required for creating the scheduled task and installing Windows/Driver updates.

**Q: Does this work on Windows 10?**
A: Yes, tested on Windows 10 and 11.

**Q: What if I don't have winget installed?**
A: The checker will still work for Windows and Driver updates. Install [App Installer](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1) from Microsoft Store to enable application update checking.

**Q: Can I customize the install location?**
A: Yes, but you'll need to update the paths in `Setup-DailyUpdateTask.ps1` and `RunUpdateChecker.bat` to match your location.

**Q: How do I uninstall?**
A:
1. Remove the scheduled task: `Unregister-ScheduledTask -TaskName "DailySystemUpdateChecker"`
2. Delete the folder: `C:\Users\YourName\SystemUpdateChecker`
3. Delete log files from Desktop: `UpdateCheck_*.log`

## 🐛 Troubleshooting

**Task not running:**
- Open Task Scheduler (`Win + R` → `taskschd.msc`)
- Find "DailySystemUpdateChecker"
- Check "Last Run Result" (0x0 = success)
- Verify it's enabled

**No log files appearing:**
- Run `CheckTaskStatus.bat` to verify task status
- Manually run `RunUpdateChecker.bat` to test
- Check antivirus hasn't quarantined the folder

**PowerShell execution policy errors:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"Cannot find winget" error:**
- Install from Microsoft Store: [App Installer](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1)

## 📝 License

MIT License - Feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

Ideas for contributions:
- Support for other package managers (Chocolatey, Scoop)
- Email notifications when critical updates are found
- Web dashboard for viewing update history
- Better driver age analysis
- Integration with other update tools

## 🙏 Acknowledgments

- Uses Windows Update API for Windows/Driver checking
- Uses winget (Windows Package Manager) for application updates
- Built with PowerShell

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions

---

**Note:** This tool provides information about available updates but does not replace official update mechanisms. Always verify critical updates through official channels.
