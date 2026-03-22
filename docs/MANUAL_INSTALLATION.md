# NVIDIA Driver & CUDA Installation Guide

This document describes the automated installation process and provides manual steps for reproducing each step.

## Overview

The `nvidia-inst` tool performs the following major operations:

1. GPU Detection
2. Driver Selection (based on GPU generation)
3. CUDA Repository Configuration
4. Version Locking (pattern-based)
5. Package Installation
6. Environment Setup

---

## Automated Installation

### Quick Start

```bash
cd /home/felix/nvidia-inst
./nv-install --yes  # Non-interactive installation
./nv-install        # Interactive installation
./nv-install --gui # GUI mode
./nv-install --dry-run  # Preview only
```

### With CUDA (Default)

```bash
./nv-install --yes --cuda-version 12.9
```

### Without CUDA

```bash
./nv-install --yes --no-cuda
```

---

## Manual Installation Steps

### Step 1: Detect GPU

```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia

# Get detailed GPU info
nvidia-smi -L
```

Example output:
```
GPU 0: NVIDIA Quadro M2200 (UUID: GPU-...)
```

### Step 2: Determine Compatible Driver & CUDA Versions

Based on GPU architecture:

| GPU Generation | Codename | Driver Branch | CUDA Support |
|----------------|----------|---------------|--------------|
| Kepler (GTX 600/700) | GKxxx | 470 (EOL) | 9.x - 11.x |
| Maxwell (GTX 900, M2000) | GMxxx | 470 (EOL) | 9.x - 11.x |
| Pascal (GTX 1000, P100) | GPxxx | 535 | 9.x - 12.x |
| Volta (V100) | GVxxx | 535 | 9.x - 12.x |
| Turing (RTX 2000) | TUxxx | 535 | 10.x - 12.x |
| Ampere (RTX 3000, A100) | GAxxx | 535/550+ | 11.x - 13.x |
| Ada (RTX 4000) | ADxxx | 550+ | 12.x - 13.x |
| Hopper (H100) | GHxxx | 550+ | 12.x - 13.x |

### Step 3: Add NVIDIA Driver Repository

#### Fedora/RHEL/CentOS

```bash
# Install RPM Fusion (if not already installed)
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Enable NVIDIA driver repository
sudo dnf config-manager --enable rpmfusion-nonfree-nvidia-driver
sudo dnf makecache
```

#### Ubuntu/Debian

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### Step 4: Add CUDA Repository

**Important**: NVIDIA repos only contain the latest CUDA version per distribution. To install older CUDA versions:

| CUDA Version | Repository |
|--------------|------------|
| CUDA 12.x | Fedora 41 / Ubuntu 22.04 |
| CUDA 13.x | Fedora 42 / Ubuntu 24.04 |

#### Fedora/RHEL/CentOS

```bash
# For CUDA 12.x (on Fedora 41+)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
sudo dnf makecache

# Verify repo
dnf repolist | grep cuda
```

### Step 5: Remove Incompatible CUDA (if upgrading)

```bash
# Check installed CUDA version
rpm -qa | grep -E 'cuda-toolkit-[0-9]+-[0-9]+' | head -3

# Remove incompatible CUDA
sudo dnf remove -y cuda* libnv* libcuda* nvidia-libcuda*
```

### Step 6: Install Driver

#### Fedora/RHEL/CentOS

```bash
# For standard driver
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia

# For driver with CUDA support
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia xorg-x11-drv-nvidia-cuda
```

#### Ubuntu/Debian

```bash
sudo apt install -y nvidia-driver-535
```

### Step 7: Install CUDA Toolkit

#### Fedora/RHEL/CentOS

```bash
# Install specific version (e.g., CUDA 12.8)
sudo dnf install -y cuda-toolkit-12-8

# Or install specific minor version
sudo dnf install -y cuda-toolkit-12-8-12.8.1
```

#### Ubuntu/Debian

```bash
sudo apt install -y cuda-toolkit-12-8
```

### Step 8: Setup Environment Variables

```bash
# Create profile script
sudo tee /etc/profile.d/cuda.sh << 'EOF'
# CUDA environment setup - managed by nvidia-inst
if [ -d /usr/local/cuda ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi
EOF

sudo chmod 644 /etc/profile.d/cuda.sh

# Source immediately
source /etc/profile.d/cuda.sh

# Verify
nvcc --version
```

### Step 9: Version Lock (Prevent Unintended Upgrades)

#### Fedora/RHEL (dnf5 versionlock)

```bash
# Install versionlock plugin
sudo dnf install -y dnf5-plugins

# Lock driver to branch (e.g., 580.x)
sudo dnf5 versionlock add akmod-nvidia-580.*

# Lock CUDA to major version (e.g., 12.x)
sudo dnf5 versionlock add cuda-toolkit-12.*

# View locks
dnf5 versionlock list

# Remove lock
sudo dnf5 versionlock delete cuda-toolkit-12.*
```

#### Ubuntu/Debian (apt pinning)

```bash
# Create apt preferences
sudo tee /etc/apt/preferences.d/cuda-12 << 'EOF'
Package: cuda-*
Pin: version 12.*
Pin-Priority: 1000
EOF
```

### Step 10: Rebuild Initramfs

```bash
# Fedora/RHEL
sudo dracut -f

# Ubuntu/Debian
sudo update-initramfs -u
```

### Step 11: Reboot

```bash
sudo reboot
```

### Step 12: Verify Installation

```bash
# Check driver
nvidia-smi

# Check CUDA
nvcc --version

# Check libraries
ldconfig -p | grep nvidia

# Check versionlock
cat /etc/dnf/versionlock.toml  # Fedora
```

---

## Pattern-Based Version Locking

### How It Works

The versionlock TOML file supports pattern-based locks using conditions:

```toml
version = "1.0"

[[packages]]
name = "cuda-toolkit-12.*"
[[packages.conditions]]
key = "evr"
comparator = ">="
value = "12"
[[packages.conditions]]
key = "evr"
comparator = "<"
value = "13"
```

This locks:
- ✅ `cuda-toolkit-12-6-12.6.3-1.x86_64` (12.6)
- ✅ `cuda-toolkit-12-8-12.8.1-1.x86_64` (12.8)
- ❌ `cuda-toolkit-13-2-13.2.0-1.x86_64` (13.2)

### Manual TOML Edit

```bash
sudo nano /etc/dnf/versionlock.toml
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check kernel modules
lsmod | grep nvidia

# Load module manually
sudo modprobe nvidia

# Check dmesg
dmesg | grep -i nvidia
```

### CUDA Version Wrong

```bash
# Check which CUDA is active
/usr/local/cuda/bin/nvcc --version

# Check alternatives
update-alternatives --display cuda

# Check symlink
ls -la /usr/local/cuda
```

### Versionlock Not Working

```bash
# Check versionlock is enabled
dnf5 versionlock status

# Verify TOML syntax
cat /etc/dnf/versionlock.toml

# Clear and re-add
sudo dnf5 versionlock clear
sudo dnf5 versionlock add cuda-toolkit-12.*
```

### Driver Not Loading

```bash
# Check Secure Boot
mokutil --sb-state

# If Secure Boot enabled, sign the module
sudo modprobe -v nvidia

# Or disable Secure Boot in BIOS
```

---

## Cleanup

### Remove NVIDIA Driver

```bash
# Fedora/RHEL
sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia '*nvidia*'
sudo dnf autoremove

# Remove versionlock
sudo dnf5 versionlock delete akmod-nvidia-*

# Ubuntu/Debian
sudo apt remove -y nvidia-* --purge
sudo apt autoremove
```

### Remove CUDA

```bash
# Fedora/RHEL
sudo dnf remove -y 'cuda-*' 'libnv*'
sudo dnf5 versionlock delete cuda-toolkit-*

# Ubuntu/Debian
sudo apt remove -y 'cuda-*' --purge

# Remove environment
sudo rm /etc/profile.d/cuda.sh
```

---

## Reference

- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-downloads-archive)
- [RPM Fusion NVIDIA Guide](https://rpmfusion.org/Howto/NVIDIA)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
