# System Update Checker - Instructions

## 📍 Location
**C:\Users\YourName\SystemUpdateChecker**

All update checker files should be in your user folder.

---

## ✅ How to Know When It Runs

The checker creates a **log file on your Desktop** every time it runs:
- **File name**: `UpdateCheck_2025-10-12.log` (with current date)
- **Location**: Your Desktop
- **When**: Created daily at 9:05 AM (or your scheduled time)

### Visual Indicators:
1. **Check your Desktop** - You'll see a new log file each day
2. **Open the log file** - Shows all updates found
3. **Task Scheduler** - Shows last run time and next run time

---

## 🛑 How to Stop It Running

### Option 1: Disable Temporarily (Recommended)
Run in PowerShell:
```powershell
Disable-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

To re-enable:
```powershell
Enable-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

### Option 2: Delete Completely
Run in PowerShell:
```powershell
Unregister-ScheduledTask -TaskName "DailySystemUpdateChecker" -Confirm:$false
```

### Option 3: Use Task Scheduler GUI
1. Press `Win + R`
2. Type: `taskschd.msc`
3. Press Enter
4. Find "DailySystemUpdateChecker" in the list
5. Right-click → **Disable** (to pause) or **Delete** (to remove)

---

## 🔧 How to Change When It Runs

### Run the update script as Administrator:
```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\YourName\SystemUpdateChecker\UpdateTaskLocation.ps1"
```

This will let you set a new time.

### Or use Task Scheduler GUI:
1. Press `Win + R`, type `taskschd.msc`
2. Find "DailySystemUpdateChecker"
3. Right-click → **Properties**
4. Go to **Triggers** tab
5. Edit the daily trigger time

---

## ▶️ How to Run Manually Anytime

### Option 1: Double-click
```
C:\Users\YourName\SystemUpdateChecker\RunUpdateChecker.bat
```

### Option 2: PowerShell Command
```powershell
Start-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

### Option 3: Direct PowerShell
```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\YourName\SystemUpdateChecker\DailyUpdateChecker.ps1"
```

---

## 📊 Check Task Status

### See if task is enabled/disabled:
```powershell
Get-ScheduledTask -TaskName "DailySystemUpdateChecker"
```

Look for the **State** field:
- **Ready** = Enabled and will run
- **Disabled** = Won't run until you enable it

### See last run time and result:
```powershell
Get-ScheduledTask -TaskName "DailySystemUpdateChecker" | Get-ScheduledTaskInfo
```

Shows:
- **LastRunTime** = When it last ran
- **LastTaskResult** = 0 means success
- **NextRunTime** = When it will run next

---

## 📁 Files in This Folder

- **DailyUpdateChecker.ps1** - Main script that checks for updates
- **RunUpdateChecker.bat** - Double-click to run manually
- **UpdateTaskLocation.ps1** - Update task schedule/location
- **INSTRUCTIONS.md** - This file

---

## 🗑️ Complete Removal

If you want to completely remove everything:

1. **Delete the scheduled task:**
   ```powershell
   Unregister-ScheduledTask -TaskName "DailySystemUpdateChecker" -Confirm:$false
   ```

2. **Delete this folder:**
   ```
   C:\Users\YourName\SystemUpdateChecker
   ```

3. **Delete old log files from Desktop:**
   ```
   UpdateCheck_*.log
   ```

---

## ❓ FAQ

**Q: How do I know if updates were found?**
A: Check the log file on your Desktop. If updates exist, it will say "ACTION REQUIRED" at the bottom.

**Q: Will it install updates automatically?**
A: No! It only checks and reports. You must install updates manually.

**Q: Can I change where the log file is saved?**
A: Yes, edit line 2 of `DailyUpdateChecker.ps1` to change the `$LogFile` location.

**Q: Does it need admin rights?**
A: The scheduled task runs with admin rights for full system access. But you can run it manually without admin for basic checks.

**Q: What if I don't see a log file?**
A: The task might not have run yet, or it might be disabled. Check Task Scheduler to verify it's enabled and see the last run time.

---

## 📞 Quick Reference Commands

| Action | Command |
|--------|---------|
| View task status | `Get-ScheduledTask -TaskName "DailySystemUpdateChecker"` |
| Run now | `Start-ScheduledTask -TaskName "DailySystemUpdateChecker"` |
| Disable | `Disable-ScheduledTask -TaskName "DailySystemUpdateChecker"` |
| Enable | `Enable-ScheduledTask -TaskName "DailySystemUpdateChecker"` |
| Delete | `Unregister-ScheduledTask -TaskName "DailySystemUpdateChecker"` |
| Check last run | `Get-ScheduledTask -TaskName "DailySystemUpdateChecker" \| Get-ScheduledTaskInfo` |
