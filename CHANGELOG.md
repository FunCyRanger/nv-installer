# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - `--matrix-info` - Show compatibility matrix
  - `--update-matrix` - Update matrix from online sources

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

[Unreleased]: https://github.com/FunCyRanger/nv-installer/compare/v0.1.0-b1...main
[v0.1.0-b1]: https://github.com/FunCyRanger/nv-installer/releases/tag/v0.1.0-b1
