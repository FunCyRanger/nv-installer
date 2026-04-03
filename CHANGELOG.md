# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.1-rc2] - 2026-04-03

### Fixed

- Critical bug: Installation failed with "Unknown package tool: fedora" because main.py passed distro.id to functions expecting tool names
- Added `tool` property to PackageManager base class for tool name access
- Fixed all CLI call sites to use `pkg_mgr.tool` instead of `distro.id`

### Testing

- Added comprehensive E2E tests for CLI workflows (simulation, CUDA, offline, rollback, revert)
- Added integration tests for full installation flow with tool detection
- Added regression tests for tool name vs distro ID
- All tests run in CI across Ubuntu, Fedora, and Arch containers

## [v0.2.1-rc1] - 2026-04-03

### Added

- Remove dead enterprise module (fleet, monitoring, security)
- Add real package manager integration tests (DNF, APT, Pacman, Zypper)
- Add comprehensive E2E tests covering all CLI workflows
- Add Arch Linux container to CI matrix
- Update Fedora from 40 to 43

### Changed

- CI now runs tests in containers to avoid resource limit issues
- Coverage improved from 57% to 65%

## [v0.2.0] - 2026-04-02

### Added

- Full installation orchestration with safety checks
- Driver state detection (Proprietary/NVIDIA Open/Nouveau)
- Pre-installation validation (disk space, package availability, kernel deps, Secure Boot)
- Post-installation validation (packages, kernel module, nvidia-smi)
- Nouveau handling with automatic re-enable on failure
- Comprehensive unit tests (800+ tests)
- Integration tests with real distro containers

### Features

- Package manager abstraction with tool-based detection
- GPU compatibility matrix with auto-update
- CUDA version locking for EOL/Limited GPUs (Maxwell, Pascal, Volta, Kepler)
- Offline installation support with package caching
- Rollback capability with snapshots
- Hybrid graphics power profile management
- Secure Boot MOK key enrollment

## [v0.1.0-b1] - 2026-03-18

### Added

- Cross-distribution Nvidia driver installation (Ubuntu, Fedora, Arch, Debian, openSUSE)
- GPU detection with compute capability recognition
- Compatibility matrix with driver ranges per GPU generation
- CLI with dry-run mode
- Revert to Nouveau feature
- Automatic matrix updates with online/offline fallback
- Comprehensive unit tests (70 tests)

### Features

- **GPU Support Matrix**:
  - Full support: Turing, Ampere, Ada, Blackwell (590.x branch)
  - Limited support: Maxwell, Pascal, Volta (580.x branch)
  - EOL: Kepler (470.x branch only)

- **Package Managers**:
  - APT (Debian/Ubuntu)
  - DNF (Fedora)
  - Pacman (Arch)
  - Zypper (openSUSE)

- **CLI Options**:
  - `--check` - Check compatibility
  - `--install` - Install driver
  - `--dry-run` - Simulate without changes
  - `--revert-to-nouveau` - Switch to Nouveau driver
  - Hybrid graphics detection (auto-shown with `--check`)
  - `--power-profile` - Set hybrid graphics mode

### Tested Configurations

| Distro | Version | GPU | Status |
|--------|---------|-----|--------|
| Fedora | 43 | Quadro M2200 (Maxwell) | Working |

### Known Limitations

- Only tested on Fedora 43 with Maxwell GPU
- Manual testing on real hardware
- Other distributions: Untested (contributions welcome)

### Testing

- Unit Tests: 70 passing
- Lint: Clean
- Type Check: Clean
- CI/CD: Configured with GitHub Actions

[Unreleased]: https://github.com/FunCyRanger/nv-installer/compare/v0.2.1-rc2...main
[v0.2.1-rc2]: https://github.com/FunCyRanger/nv-installer/releases/tag/v0.2.1-rc2
[v0.2.1-rc1]: https://github.com/FunCyRanger/nv-installer/releases/tag/v0.2.1-rc1
[v0.2.0]: https://github.com/FunCyRanger/nv-installer/releases/tag/v0.2.0
[v0.1.0-b1]: https://github.com/FunCyRanger/nv-installer/releases/tag/v0.1.0-b1
