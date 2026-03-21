# nvidia-inst

> **⚠️ DISCLAIMER & WARNING**
>
> **This software is provided for testing and development purposes ONLY.**
>
> - **NOT for production use** - Do not use in production environments
> - **User assumes all risk** - You are solely responsible for any consequences
> - **System damage possible** - Driver installation can render your system unbootable
> - **No warranty** - This software is provided "as is" without any warranty
> - **Backup first** - Always have a working backup before using this script
> - **Self-responsibility** - By using this software, you accept full responsibility
>
> **The authors/contributors are NOT liable for any damages, data loss, or system failures.**

---

> **Beta Status**: This software is in active development. Currently tested on Fedora 43 with Maxwell GPU. Contributions welcome!

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
- **Interactive Driver Selection**: Choose between proprietary, NVIDIA Open, and Nouveau drivers
- **CUDA Awareness**: Shows CUDA support information in driver options
- **Driver State Detection**: Automatically detects current driver state (Proprietary/NVIDIA Open/Nouveau)
- **Non-Free Repository Detection**: Warns if non-free repos need to be enabled
- **Root Privilege Management**: Caches sudo credentials for smooth workflow
- **Graceful Cancellation**: Ctrl+C handling with clean exit

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

The compatibility matrix is automatically updated on startup. For manual control:

```bash
make matrix-check     # Check matrix status
make matrix-update   # Force update from online sources
make matrix-verify   # Verify matrix data integrity
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
  --check                Check prerequisites only (shows hybrid info if detected)
  --gui                  Use GUI mode (Tkinter or Zenity)
  --dry-run, --simulate  Show what would be installed without making changes
  --yes, -y              Skip installation confirmation
  --driver-version VER   Specify driver version
  --cuda-version VER     Specify CUDA version
  --no-cuda              Install driver without CUDA
  --revert-to-nouveau    Switch from proprietary to Nouveau (open-source)
  --power-profile        Set hybrid graphics profile (intel, hybrid, nvidia)
  --debug                Enable debug logging
  --version              Show version information
```

## Driver Options

nvidia-inst supports three driver types with CUDA-aware installation options:

### Driver Types

| Driver | Description | CUDA Support | Package Types |
|--------|-------------|-------------|--------------|
| **Proprietary** | NVIDIA's closed-source driver | Full CUDA support | `nvidia-driver-*`, `akmod-nvidia` |
| **NVIDIA Open** | Open kernel modules (Turing+) | Full CUDA support | `nvidia-driver-*-open`, `nvidia-open` |
| **Nouveau** | Open-source Xorg driver | No CUDA support | `xserver-xorg-video-nouveau` |

### Menu Options

Depending on your current driver state and GPU, you'll see relevant options:

**No driver installed (repos enabled):**
```
Options:
  [1] Install proprietary driver (CUDA 11.0-12.8) [RECOMMENDED]
  [2] Install NVIDIA Open (CUDA 11.0-12.8)
  [3] Install Nouveau (open-source, no CUDA support)
  [4] Cancel
```

**Proprietary driver working optimally:**
```
Options:
  [1] Upgrade to latest [RECOMMENDED]
  [2] Keep current driver
  [3] Switch to NVIDIA Open (CUDA 11.0-12.8)
  [4] Switch to Nouveau (open-source, no CUDA support)
  [5] Cancel
```

**Proprietary driver installed but non-free repos not enabled:**
```
Options:
  [1] Enable non-free repos + install proprietary (CUDA 11.0-12.8) [RECOMMENDED]
  [2] Enable non-free repos + install NVIDIA Open (CUDA 11.0-12.8)
  [3] Install Nouveau (open-source, no CUDA support)
  [4] Cancel
```

### CUDA Support Indication

Options show CUDA version ranges when available:
- `(CUDA 11.0-12.8)` - Full CUDA support
- `(no CUDA support)` - Nouveau only

### Distro-Specific Package Names

| Distro | Proprietary | NVIDIA Open | Nouveau |
|--------|-------------|------------|---------|
| Ubuntu/Debian/Mint | `nvidia-driver-*` | `nvidia-driver-*-open` | `xserver-xorg-video-nouveau` |
| Fedora/RHEL | `akmod-nvidia` | `xorg-x11-drv-nvidia-open` | `xorg-x11-drv-nouveau` |
| Arch/Manjaro | `nvidia` | `nvidia-open` | `xf86-video-nouveau` |
| openSUSE | `x11-video-nvidiaG0*` | `nvidia-open-driver-G06` | `xf86-video-nouveau` |

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
1. Remove proprietary Nvidia driver packages (queries actual installed packages)
2. Remove versionlock entries
3. Remove Nouveau blacklist
4. Rebuild initramfs
5. Enable Nouveau (open-source) driver

You will need to reboot after reverting.

### Driver State Detection

The installer automatically detects your current driver state:

| State | Message | Available Options |
|-------|---------|-------------------|
| Proprietary Optimal | `NVIDIA driver X.Y.Z is working optimally` | Upgrade/Keep/NVIDIA Open/Nouveau |
| Proprietary Wrong Branch | `Driver X.Y.Z may not be optimal` | Install correct/Keep/NVIDIA Open/Nouveau |
| NVIDIA Open Active | `NVIDIA Open driver is active` | Upgrade/Keep/Proprietary/Nouveau |
| Nouveau Active | `Nouveau (open-source) driver is active` | Proprietary/NVIDIA Open/Keep Nouveau |
| No Driver | `No NVIDIA driver installed` | Install (with CUDA info)/Nouveau |
| No Driver (repos missing) | `No NVIDIA driver installed (non-free repos not enabled)` | Enable repos + Install options |

### Root Privilege Management

- Scripts requests sudo access when needed
- Credentials are cached for smooth multi-step operations
- Works without root in `--dry-run` and `--check` modes

## Safety Features

### Pre-Installation Checks
- Disk space verification (500MB+ recommended)
- Package availability in repositories
- Kernel development packages
- Secure Boot status and MOK enrollment
- Running environment (tty vs GUI)
- Non-free repository detection

### Post-Installation Validation
- Verify packages installed
- Check kernel module built
- Verify Nouveau blocking
- Test nvidia-smi availability

### Package Removal Safety
- Queries actual installed packages before removal (no glob pattern issues)
- DKMS cleanup before package removal
- Versionlock cleanup for Fedora
- APT preferences cleanup for Ubuntu/Debian

### Nouveau Handling
- Prompts user before blocking Nouveau
- If validation fails, automatically re-enables Nouveau to ensure bootable system
- Clear instructions for manual fix if needed

### Graceful Cancellation
- Ctrl+C handling exits cleanly without traceback
- Cancel option available in all menus
- Partial operations can be safely interrupted

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

### Slow Boot - akmods.service Takes Minutes (Fedora/RHEL)

If boot time is slow and `systemd-analyze critical-chain` shows akmods.service
taking several minutes, check for these common causes:

#### Cause: Missing /var/lib/akmods/ directory

Akmods uses this directory to track which modules have been built. If missing,
it rebuilds every boot even when kmod-nvidia is installed:

```bash
# Check if directory exists
ls -la /var/lib/akmods/

# If missing, create it
sudo mkdir -p /var/lib/akmods
sudo chown root:akmods /var/lib/akmods
sudo chmod 0755 /var/lib/akmods
```

After creating this directory, akmods should detect existing kmod-nvidia
packages and skip rebuilding on subsequent boots.

#### Cause: Rebooting too quickly after kernel update

Akmods builds in the background after dnf update. Rebooting before completion
forces a rebuild at next boot:

```bash
# After kernel update, wait 5-10 minutes OR run manually:
sudo akmods --force
sudo reboot
```

#### Cause: Build failure not detected

If akmods failed silently, it retries every boot:

```bash
# Check for failed builds
cat /var/cache/akmods/nvidia/*.failed.log 2>/dev/null

# Force rebuild
sudo akmods --force
```

## Fast Boot Options for Fedora/RHEL

There are three approaches to avoid slow boot times with NVIDIA drivers:

### Option A: Fix akmods (Recommended)

If akmods.service runs at every boot, the most common cause is a missing
`/var/lib/akmods/` tracking directory:

```bash
# Create missing directory
sudo mkdir -p /var/lib/akmods
sudo chown root:akmods /var/lib/akmods
sudo chmod 0755 /var/lib/akmods
```

After this fix, akmods should detect existing kmod-nvidia packages and skip
rebuilding on subsequent boots.

### Option B: Disable akmods at Boot

If fixing doesn't help, disable the boot service while keeping akmod-nvidia
for future kernel updates:

```bash
sudo systemctl disable --now akmods.service
```

After kernel updates, manually trigger build:
```bash
sudo akmods --force
sudo reboot
```

### Option C: kmod-only (No Automatic Rebuilds)

For systems where pre-built kmod packages are reliably available:

```bash
# Remove akmod build system
sudo dnf remove akmod-nvidia kmodtool akmods

# Install pre-built kmod
sudo dnf install kmod-nvidia
```

After kernel updates, manually install new kmod when available (usually 1-7 days).

### Trade-offs

| Approach | Boot Time | After Kernel Update |
|----------|-----------|-------------------|
| Fix akmods | Fast | Automatic |
| Disable at boot | Fast | Manual trigger |
| kmod-only | Fast | Manual install |

### Verification Commands

```bash
# Check if akmods is causing delays
systemd-analyze critical-chain | grep akmods

# Check installed packages
rpm -qa | grep -E "akmod|kmod-nvidia"

# Check kmod is loaded
lsmod | grep nvidia
```

### Other Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Secure Boot block | Module not loading | `mokutil --sb-state`, enroll MOK key |
| Build failure | nvidia-smi not found | Check `/var/cache/akmods/nvidia/*.log` |
| /boot full | initramfs errors | Clean old kernels: `dnf remove $(rpm -q kernel | grep -v $(uname -r))` |
| Repo conflict | "package filtered" errors | Remove all nvidia, reinstall from RPM Fusion only |

## Tested Configurations

| Distro | Version | GPU | Driver | Status |
|--------|---------|-----|--------|--------|
| Fedora | 43 | Quadro M2200 Mobile (Maxwell) | 580.126.18 | Working |

### Contributing Test Results

If you test on other distributions or GPUs, please report your results:

- **Open an Issue**: https://github.com/FunCyRanger/nv-installer/issues
- **Discussions**: https://github.com/FunCyRanger/nv-installer/discussions

### Untested (Help Wanted)

- Ubuntu, Arch, Debian, openSUSE
- Ampere, Ada, Turing, Blackwell GPUs
- Kepler GPUs (EOL)

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

SPDX-License-Identifier: AGPL-3.0-or-later

See [LICENSE](LICENSE) file for full license text.

## Trademarks

NVIDIA, CUDA, GeForce, RTX, and related marks are trademarks of NVIDIA Corporation. This project is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.

## Third-Party Repositories

This project uses distribution package repositories to install drivers:

- **RPM Fusion** (Fedora/RHEL): Community-maintained repository, not affiliated with NVIDIA Corporation
- **Ubuntu Graphics Drivers PPA**: Community-maintained, not affiliated with NVIDIA Corporation
- **Arch Linux Extra**: Official community repository, not affiliated with NVIDIA Corporation
- **openSUSE NVIDIA Repository**: Provided by NVIDIA for openSUSE users

## Credits

- Nvidia Driver Compatibility Matrix: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- Linux Driver Archive: https://www.nvidia.com/en-us/drivers/unix/
- CUDA Compatibility Guide: https://docs.nvidia.com/deploy/cuda-compatibility/
