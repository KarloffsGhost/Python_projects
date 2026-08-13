# Changelog

All notable changes to the Windows System Update Checker project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-08-12

### Security
- **Dashboard remote code execution.** `/api/action/update` interpolated
  POSTed package ids into a PowerShell `-Command` string; an id containing a
  single quote escaped it and ran arbitrary commands. Combined with a wildcard
  CORS header and an OPTIONS handler that approved every preflight, any page
  open in the browser could execute code as the user. Ids are now validated
  against the live winget upgrade list and passed as discrete arguments.
- **Dashboard cross-site request forgery.** Wildcard CORS removed, preflights
  refused, and all `/api` routes now require a per-session token injected into
  `index.html` at serve time.
- **Dashboard path traversal.** `/css/` and `/js/` concatenated the request
  path onto a folder name, serving files such as `/css/../../../Windows/win.ini`.
  Paths are resolved and confined to their directory.
- **Dashboard cross-site scripting.** Package names and update titles from
  winget and Windows Update were written into `innerHTML` unescaped.

### Fixed
- Windows and driver updates were double-counted: `IsInstalled=0` already
  returns drivers, so the extra `Type='Driver'` search counted each one twice.
  A single pending driver was reported as "2 updates available".
- The application update counter sat inside a `Select-Object -First 10` display
  loop, so the summary could never report more than 10 regardless of the real
  number.
- winget output is parsed positionally from its own header offsets instead of
  by splitting on whitespace runs, which mis-parsed names containing double
  spaces and counted header and summary lines as packages.
- `--include-unknown` is passed, so packages whose installed version cannot be
  determined are no longer omitted.
- Console encoding is set to UTF-8, fixing mangled package names in the log.
- The antivirus check always reported "signatures up to date". Defender leaves
  `AntivirusSignatureLastUpdated` null when a third-party AV is active, the
  subtraction threw, and `$null.TotalDays -lt 2` is `$true`. Security Center is
  now queried for the product actually protecting the machine.
- Driver age analysis excluded: Microsoft inbox drivers carry a fixed
  2006-06-21 placeholder date and always looked stale, producing 40+ lines of
  noise the report itself said to ignore.
- `InstallUpdates-Windows.ps1` skipped one hardcoded Intel driver version by
  string match. Offered drivers are now compared against installed ones by
  hardware ID, version and date, with an opt-in prompt to hide superseded
  updates so Windows stops re-offering them.
- `SystemCleaner.ps1` defined a `Clear-RecycleBin` function that shadowed the
  built-in cmdlet, so its own call resolved back to itself and always failed
  into the fallback path. Renamed, and freed space is now measured rather than
  assumed from the pre-deletion size.
- Recycle Bin sizing read a localised display string and stripped non-digits,
  turning "1.2 MB" into 12 bytes.
- Today's log is no longer deleted at startup, so a second run on the same day
  appends instead of destroying the first run's output.
- `Config\dashboard-settings.json` is read; it had never been loaded and the
  port was hardcoded.
- `Get-WmiObject` replaced with `Get-CimInstance` (the former only resolves in
  PowerShell 7 through the Windows PowerShell compatibility shim).
- Empty `catch {}` blocks that hid Windows Update failures, malformed request
  bodies and save failures now report what went wrong.

### Added
- `Tests\Test-Security.ps1` - 28 assertions covering the path resolver, the
  driver supersede logic and the winget parser.
- Shared `Get-WingetUpgrades` and `Get-WingetCategory` in
  `SystemMaintenanceLib.psm1`, replacing three divergent copies of the same
  parser.
- `-NoBrowser` switch on the dashboard.

## [2.0.0] - 2026-01-25

### Added
- System Cleaner (temp files, browser caches, Windows junk)
- AI/ML Toolchain Cleaner (Ollama, ComfyUI, LM Studio, Docker, NVIDIA, pip,
  npm, Conda)
- Driver Backup/Restore with System Restore point creation
- Startup Manager with impact analysis
- HTML system report generator
- Web dashboard
- Unified launcher menu (`SystemMaintenance.bat`)
- `Modules/SystemMaintenanceLib.psm1` shared function library
- JSON configuration under `Config/`

## [1.0.0] - 2025-01-24

### Added
- Initial release of Windows System Update Checker
- Daily automated checking for Windows Updates, Driver Updates, and Application Updates
- Smart analysis and recommendations system
- Interactive application update installer with categorization
- Windows/Driver update installer scripts
- Scheduled task setup and management
- Comprehensive logging to Desktop
- System health checking (device errors, Windows Defender status)
- Driver age analysis
- Support for winget (Windows Package Manager) integration
- Multiple batch file launchers for ease of use
- Detailed documentation (README.md, INSTRUCTIONS.md, HOW-TO-READ-UPDATES.txt)

### Changed
- Made recommendations data-driven based on actual scan results
- Added version numbers to main scripts
- Improved time input validation with proper range checking
- Enhanced time formatting with zero-padding

### Security
- Read-only scanning by default (no automatic installations)
- Explicit administrator privilege requirements where needed
- Local-only operation with no data collection
- Open source for transparency

## [Unreleased]

### Planned Features
- Support for other package managers (Chocolatey, Scoop)
- Email notifications for critical updates
- Integration with other update tools
