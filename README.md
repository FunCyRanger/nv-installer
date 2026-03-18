# nvidia-inst

Cross-distribution Linux script for installing the latest compatible Nvidia driver with CUDA support.

## Features

- **Automatic Detection**: Detects Linux distribution (Ubuntu, Fedora, Arch, Debian, openSUSE) and GPU
- **Compatibility Matrix**: Versioned matrix sourced from official Nvidia documentation with auto-update
- **GPU Compatibility**: Follows Nvidia's official compatibility matrix for driver versions
- **Version Checking**: Verifies available driver versions in repositories against GPU compatibility
- **Branch Locking**: Supports branch-level version locking to prevent incompatible driver updates
- **Safety Checks**: Pre and post-installation validation
- **Multiple Interfaces**: CLI, Tkinter GUI, and Zenity GUI support
- **Dry-Run Mode**: Test installation without making changes
- **Revert to Nouveau**: Switch from proprietary driver to open-source Nouveau

## Supported Distributions

- Fedora/RHEL/Rocky/Alma/CentOS
- Ubuntu/Debian/Linux Mint/Pop!_OS
- Arch Linux/Manjaro/EndeavourOS
- openSUSE

## Compatibility Matrix

nvidia-inst uses a versioned compatibility matrix to determine the correct driver for your GPU. The matrix is generated from official Nvidia sources.

### How It Works

1. **Auto-Update**: On startup, nvidia-inst checks for matrix updates (every 24 hours)
2. **Smart Caching**: Results are cached in `~/.cache/nvidia-inst/matrix_cache.json`
3. **Offline Fallback**: If offline, uses the committed fallback matrix in `src/nvidia_inst/gpu/matrix/compatibility_data.json`
4. **Version Tracking**: Matrix version is tracked for audit purposes

### GPU Support Matrix

| GPU Generation | Examples | Driver Branch | CUDA Support | Status |
|---------------|----------|---------------|--------------|--------|
| **Blackwell** | RTX 5090, 5080, GB200 | 590.x | 12.4+ | Full Support |
| **Ada Lovelace** | RTX 4090, 4080, L40 | 590.x | 11.8+ | Full Support |
| **Ampere** | RTX 3090, 3080, A100 | 590.x | 11.0+ | Full Support |
| **Turing** | RTX 2080, GTX 1650, T4 | 590.x | 10.0+ | Full Support |
| **Volta** | V100, Titan V | 580.x | 9.0+ | Limited Support |
| **Pascal** | GTX 1080, P100 | 580.x | 8.0+ | Limited Support |
| **Maxwell** | GTX 980, 970, M-series | 580.x | 7.5+ | Limited Support |
| **Kepler** | GTX 780, K-series | 470.x | 7.5+ | EOL (security only) |

### Driver Branches

| Branch | Type | Latest Version | EOL Date | Supported GPUs |
|--------|------|---------------|----------|---------------|
| **590** | New Feature | 590.48.01 | Aug 2028 | Turing+ (RTX 20xx, 30xx, 40xx) |
| **580** | Production | 580.142 | Oct 2028 | Maxwell, Pascal, Volta |
| **470** | Legacy | 470.256.02 | Dec 2025 | Kepler (GTX 600/700) |

### CUDA Compatibility

| GPU Generation | Min CUDA | Max CUDA | Recommended |
|---------------|----------|----------|-------------|
| Blackwell | 12.4 | 13.x | 12.6 |
| Ada Lovelace | 11.8 | 12.8 | 12.2 |
| Ampere | 11.0 | 12.8 | 12.2 |
| Turing | 10.0 | 12.8 | 12.2 |
| Volta | 9.0 | 12.8 | 11.8 |
| Pascal | 8.0 | 12.8 | 11.8 |
| Maxwell | 7.5 | 12.8 | 11.8 |
| Kepler | 7.5 | 9.0 | 9.0 |

### Matrix Management

```bash
# Check matrix status
nvidia-inst --matrix-info
make matrix-check

# Update matrix from online sources
nvidia-inst --update-matrix
make matrix-update

# Verify matrix data integrity
make matrix-verify
```

### Official Sources

For detailed CUDA/driver compatibility information, see:
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [Unix Driver Archive](https://www.nvidia.com/en-us/drivers/unix/)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/nvidia-inst.git
cd nvidia-inst

# Run with dry-run to see what would be installed
sudo ./nv-install --dry-run

# Actually install (will prompt for confirmation)
sudo ./nv-install
```

### Options

```bash
./nv-install --help

Options:
  --dry-run, --simulate  Show what would be installed without making changes
  --check                Check prerequisites only
  --gui                  Use GUI mode (Tkinter)
  --zenity               Use Zenity dialogs
  --with-cuda            Install CUDA packages (default: yes)
  --cuda-version VER      Specify CUDA version
  --driver-version VER   Specify driver version
  --skip-confirmation    Skip installation confirmation
  --fix                  Attempt to fix missing repositories
  --revert-to-nouveau    Switch from proprietary to Nouveau (open-source)
  --update-matrix        Force update of compatibility matrix
  --matrix-info          Show compatibility matrix information
  --debug                Enable debug logging
```

## Usage Examples

### Check Prerequisites

```bash
sudo ./nv-install --check
```

### Dry-Run (Simulation)

```bash
sudo ./nv-install --dry-run
```

Output:
```
--- System Information ---
Distribution: Fedora Linux 43
GPU: Nvidia Quadro M2200 (Maxwell)
Driver Range: 450.191.0 - 580.142

--- Prerequisites Check ---
[✓] Package manager: dnf
[✓] RPM Fusion nonfree
[✓] Driver packages: akmod-nvidia, xorg-x11-drv-nvidia-cuda

--- Commands to Execute ---
# Step 1: Update package lists
  sudo dnf makecache

# Step 2: BLOCK wrong driver branch (IMPORTANT!)
  sudo dnf versionlock add '*590.*' || true

# Step 3: Install driver packages
  sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia xorg-x11-drv-nvidia-cuda

# Step 4: Lock driver to branch 580.xx (optional)
  sudo dnf versionlock add 'akmod-nvidia-580.*'

# Step 5: Reboot
  sudo reboot
```

### Actual Installation

```bash
sudo ./nv-install
```

The installer will:
1. Run safety checks (disk space, package availability, kernel deps, Secure Boot, GUI session)
2. Ask to disable Nouveau if loaded
3. Install driver packages
4. Validate installation
5. If validation fails, re-enable Nouveau to ensure bootable system
6. Prompt for reboot

## Reverting to Nouveau

Nouveau is the open-source Nvidia driver included in the Linux kernel. To switch from the proprietary driver to Nouveau:

```bash
sudo nvidia-inst --revert-to-nouveau
```

This will:
1. Remove proprietary Nvidia driver packages
2. Remove Nouveau blacklist
3. Rebuild initramfs
4. Enable Nouveau (open-source) driver

You will need to reboot after reverting.

## Safety Features

### Pre-Installation Checks
- Disk space verification (500MB+ recommended)
- Package availability in repositories
- Kernel development packages
- Secure Boot status
- Running environment (tty vs GUI)

### Post-Installation Validation
- Verify packages installed
- Check kernel module built
- Verify Nouveau blocking
- Test nvidia-smi availability

### Nouveau Handling
- Prompts user before blocking Nouveau
- If validation fails, automatically re-enables Nouveau to ensure bootable system
- Clear instructions for manual fix if needed

## Project Structure

```
nvidia-inst/
├── nv-install              # Shell wrapper script
├── src/nvidia_inst/
│   ├── cli.py             # Main CLI entry point
│   ├── distro/
│   │   ├── detector.py    # Distribution detection
│   │   ├── factory.py     # Package manager factory
│   │   ├── package_manager.py  # Abstract base class
│   │   ├── apt.py        # Debian/Ubuntu
│   │   ├── dnf.py        # Fedora/RHEL
│   │   ├── pacman.py     # Arch Linux
│   │   └── zypper.py     # openSUSE
│   ├── gpu/
│   │   ├── detector.py    # GPU detection
│   │   ├── compatibility.py  # Driver version logic
│   │   └── matrix/
│   │       ├── __init__.py
│   │       ├── data.py        # Dataclasses
│   │       ├── manager.py     # Matrix fetching/caching
│   │       └── compatibility_data.json  # Fallback data
│   ├── installer/
│   │   ├── driver.py      # Driver installation
│   │   ├── cuda.py        # CUDA installation
│   │   ├── uninstaller.py  # Revert to Nouveau
│   │   ├── prerequisites.py  # Pre-install checks
│   │   ├── validation.py   # Post-install validation
│   │   └── version_checker.py  # Version availability
│   ├── gui/
│   │   ├── tkinter_gui.py
│   │   └── zenity_gui.py
│   └── utils/
│       └── logger.py
├── scripts/
│   └── update-matrix.py   # Matrix update script
├── tests/
├── .github/workflows/
│   └── matrix-update.yml  # GitHub Actions workflow
├── AGENTS.md              # Developer guidelines
└── requirements.txt
```

## Requirements

### Python Dependencies
- Python 3.8+
- See requirements.txt for full list

### System Requirements
- Root/sudo access
- Internet connection for downloading packages (for matrix update)
- Supported Linux distribution

## Development

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test
pytest tests/
```

### Linting

```bash
make lint
```

### Matrix Management

```bash
# Check matrix status
make matrix-check

# Update matrix (if new drivers available)
make matrix-update

# Verify matrix data integrity
make matrix-verify
```

### Code Style

Follow AGENTS.md for:
- Python formatting (Black, 88 chars)
- Type hints (strict)
- Docstrings (Google style)
- ShellCheck compliance

## Troubleshooting

### System won't boot after driver installation

Boot to recovery mode and run:

```bash
sudo rm /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u  # or: sudo dracut -f
sudo reboot
```

### Re-enable Nouveau if proprietary driver causes issues

```bash
sudo nvidia-inst --revert-to-nouveau
```

## License

MIT License

## Credits

- Nvidia Driver Compatibility Matrix: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- Linux Driver Archive: https://www.nvidia.com/en-us/drivers/unix/
- CUDA Compatibility Guide: https://docs.nvidia.com/deploy/cuda-compatibility/
