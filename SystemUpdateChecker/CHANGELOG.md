# Changelog

All notable changes to the Windows System Update Checker project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Web dashboard for viewing update history
- Enhanced driver age analysis
- Integration with other update tools
