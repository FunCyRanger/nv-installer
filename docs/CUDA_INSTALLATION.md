# CUDA Installation Guide

Comprehensive guide for installing NVIDIA CUDA toolkit on Linux systems, with compatibility matrices and distro-specific instructions.

## Introduction

This guide provides detailed information about CUDA installation for NVIDIA GPUs on Linux. It covers:

- **GPU compatibility** - Compute capability and CUDA version support
- **Installation methods** - Official NVIDIA repositories, distro packages, and runfile installers
- **Distro-specific instructions** - Fedora, Ubuntu, Arch, openSUSE, and more
- **Integration with nvidia-inst** - How the tool handles CUDA installation
- **Developer information** - Code examples and implementation details

**Audience:** Both end users installing CUDA and developers working on nvidia-inst.

## Comprehensive Hardware/CUDA Support Matrix

This matrix combines GPU generation, compute capability, driver branches, and CUDA compatibility.

### GPU Compute Capability & CUDA Compatibility

| GPU Generation | Compute Capability | Examples | Driver Branch | CUDA Min | CUDA Max | Support Status |
|----------------|-------------------|----------|---------------|----------|----------|----------------|
| **Kepler** | 3.0-3.7 | GTX 6xx, 7xx, K-series | 470.x | 7.5 | 9.0 | EOL (security only) |
| **Maxwell** | 5.0-5.2 | GTX 9xx, M-series, Quadro M | 580.x | 7.5 | 12.8 | Limited (580 branch) |
| **Pascal** | 6.0-6.1 | GTX 10xx, P100 | 580.x | 8.0 | 12.8 | Limited (580 branch) |
| **Volta** | 7.0 | V100, Titan V | 580.x | 9.0 | 12.8 | Limited (580 branch) |
| **Turing** | 7.5 | RTX 20xx, GTX 16xx, T4 | 590.x | 10.0 | 12.8 | Full (latest drivers) |
| **Ampere** | 8.0-8.6 | RTX 30xx, A100, A30 | 590.x | 11.0 | 12.8 | Full (latest drivers) |
| **Ada Lovelace** | 8.9 | RTX 40xx, L40, L10 | 590.x | 11.8 | 12.8 | Full (latest drivers) |
| **Blackwell** | 9.0 | RTX 50xx, GB200 | 590.x | 12.4 | 13.x | Full (latest drivers) |

### Driver Branch Details

| Branch | Type | Latest Version | EOL Date | Supported GPUs |
|--------|------|---------------|----------|---------------|
| **590** | New Feature | 590.48.01 | Aug 2028 | Turing+ (RTX 20xx, 30xx, 40xx, 50xx) |
| **580** | Production | 580.142 | Oct 2028 | Maxwell, Pascal, Volta |
| **470** | Legacy | 470.256.02 | Dec 2025 | Kepler (GTX 600/700) |

### CUDA Version Recommendations

| GPU Generation | Recommended CUDA | Notes |
|---------------|------------------|-------|
| Blackwell | 12.6 | Latest features, optimal performance |
| Ada Lovelace | 12.2 | Good balance of features and stability |
| Ampere | 12.2 | Widely supported |
| Turing | 12.2 | Stable compatibility |
| Volta | 11.8 | Last fully supported version |
| Pascal | 11.8 | Last fully supported version |
| Maxwell | 11.8 | Last fully supported version |
| Kepler | 9.0 | Only CUDA 9.0 compatible |

## GPU Compute Capability Details

### What is Compute Capability?

Compute capability defines the features supported by a GPU's hardware. Different CUDA versions require specific compute capabilities.

### How to Check Your GPU's Compute Capability

**Method 1: Using nvidia-smi (if driver installed)**
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Method 2: Using lspci**
```bash
lspci -vnn | grep -i nvidia
# Look for device ID, then check NVIDIA's documentation
```

**Method 3: Online lookup**
- Visit: https://developer.nvidia.com/cuda-gpus
- Search for your GPU model

### Example: Quadro M2200 (Maxwell)
- Compute capability: 5.2
- Compatible CUDA versions: 7.5 - 12.8
- Recommended CUDA: 11.8 (Maxwell's last fully supported version)
- Driver branch: 580.x

## CUDA Version Compatibility Rules

### General Rules

1. **Minimum CUDA version**: Determined by compute capability
2. **Maximum CUDA version**: Determined by driver branch
3. **Forward compatibility**: CUDA toolkit can be newer than driver, but limited
4. **Backward compatibility**: Newer drivers support older CUDA toolkits

### Compatibility Scenarios

**Scenario 1: Fresh Installation**
- Install latest compatible driver branch
- Install CUDA toolkit within supported range
- Example: Ampere GPU → driver 590.x, CUDA 12.2

**Scenario 2: Existing Driver**
- Check current driver version
- Install CUDA toolkit compatible with existing driver
- Example: Driver 550.54 → CUDA 12.4 compatible

**Scenario 3: EOL GPU (Kepler)**
- Must use driver branch 470.x
- Limited to CUDA 9.0
- Security updates only

## Installation Methods

### Method 1: Official NVIDIA Repository (Recommended)

Advantages:
- Latest CUDA versions
- Official support
- Easy updates

**Fedora Example (Quadro M2200):**

*Note: If you use `nvidia-inst`, it will automatically add the CUDA repository and install CUDA toolkit (unless you use `--no-cuda`). The manual steps are shown below for reference.*

```bash
# Add NVIDIA CUDA repository for Fedora 43 (dnf5 syntax)
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo --overwrite

# For older Fedora with dnf4, use:
# sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo

# Install CUDA toolkit (runtime + development)
sudo dnf install cuda-toolkit-13-0

# Or install specific version
sudo dnf install cuda-toolkit-12-2
```

**Ubuntu Example:**

*Note: `nvidia-inst` will automatically add the CUDA repository and install CUDA toolkit (unless you use `--no-cuda`). The manual steps are shown below for reference.*

```bash
# Download and install CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install CUDA
sudo apt update
sudo apt install cuda-toolkit-12-2
```

### Method 2: Distro Repository

Advantages:
- Tested with distro
- Integrated package management
- May be older versions

**Fedora (RPM Fusion):**
```bash
# Install NVIDIA driver first
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda

# CUDA components included with driver
```

**Ubuntu (Graphics Drivers PPA):**
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-535
```

### Method 3: Runfile Installer

Advantages:
- Most control
- Custom installation paths
- Offline installation

**Steps:**
1. Download runfile from NVIDIA website
2. Disable existing NVIDIA packages
3. Run installer with options
4. Set environment variables

```bash
# Download runfile
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run

# Run installer
sudo sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit --samples
```

## Distro-Specific Instructions

### Fedora 43 (Quadro M2200 Example)

**Complete installation steps:**
```bash
# 1. Enable RPM Fusion repositories
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

# 2. Install NVIDIA driver
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda

# 3. Add NVIDIA CUDA repository (dnf5 syntax)
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo --overwrite
# For older Fedora with dnf4: sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo

# 4. Install CUDA toolkit
sudo dnf install cuda-toolkit-11-8  # For Maxwell (compute 5.2)

# 5. Set environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# 6. Verify installation
nvidia-smi
nvcc --version
```

### Ubuntu 22.04

```bash
# 1. Install prerequisites
sudo apt update
sudo apt install build-essential dkms

# 2. Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 3. Install driver and CUDA
sudo apt update
sudo apt install cuda-toolkit-12-2

# 4. Post-installation
sudo usermod -aG video $USER
```

### Arch Linux

```bash
# 1. Enable multilib repository
sudo pacman -Syy
sudo pacman -S multilib-devel

# 2. Install NVIDIA driver
sudo pacman -S nvidia nvidia-utils

# 3. Install CUDA from AUR
yay -S cuda

# Or install specific version
yay -S cuda12.2
```

### openSUSE Tumbleweed

```bash
# 1. Add NVIDIA repository
sudo zypper addrepo --refresh https://download.nvidia.com/opensuse/tumbleweed NVIDIA

# 2. Install driver
sudo zypper install x11-video-nvidiaG06

# 3. Install CUDA
sudo zypper install cuda
```

## Verification

### Basic Verification

```bash
# Check driver
nvidia-smi

# Check CUDA compiler
nvcc --version

# Check CUDA samples (if installed)
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

### Environment Variables

Ensure these are set:
```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

Expected output:
```
/usr/local/cuda-12.2/bin
/usr/local/cuda-12.2/lib64
```

### Test CUDA Compilation

Create a test file `test.cu`:
```cuda
#include <stdio.h>

__global__ void helloKernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

Compile and run:
```bash
nvcc test.cu -o test
./test
```

## Integration with nvidia-inst

### How nvidia-inst Handles CUDA

The nvidia-inst tool automatically:

1. **Detects GPU compute capability**
2. **Determines compatible CUDA versions**
3. **Adds NVIDIA CUDA repository if missing** (for Fedora, Ubuntu, etc.)
4. **Installs appropriate driver + CUDA packages**
5. **Handles repository setup**
6. **Validates CUDA/driver compatibility** (warns if incompatible versions specified)

### Compatibility Validation

When you specify a specific CUDA version (`--cuda-version`) or driver version (`--driver-version`), nvidia-inst checks if the requested version is compatible with your GPU generation. If incompatible:

- A warning is displayed with the reason
- Installation continues unless you choose to abort
- Use `--ignore-compatibility` to suppress these warnings (not recommended)

Example warnings:
```
WARNING: CUDA 13.x exceeds maximum supported version 12.8 for pascal
WARNING: Driver 470.256.02 is below minimum supported version 520.56.06 for ampere
```

### Code Example: CUDA Compatibility Check

```python
def get_cuda_range(generation: str) -> tuple[str, str]:
    """Get compatible CUDA version range for GPU generation."""
    cuda_ranges = {
        "kepler": ("7.5", "9.0"),
        "maxwell": ("7.5", "12.8"),
        "pascal": ("8.0", "12.8"),
        "volta": ("9.0", "12.8"),
        "turing": ("10.0", "12.8"),
        "ampere": ("11.0", "12.8"),
        "ada_lovelace": ("11.8", "12.8"),
        "blackwell": ("12.4", "13.x"),
    }
    return cuda_ranges.get(generation, ("unknown", "unknown"))
```

### Installation Flow

1. GPU detection → compute capability
2. Driver selection → branch 580/590/470
3. CUDA compatibility → version range
4. Package installation → driver + CUDA (repository added automatically if missing)
5. Environment setup → PATH, LD_LIBRARY_PATH
6. Verification → nvidia-smi, nvcc

## Troubleshooting

### Common Issues

**Issue 1: CUDA toolkit not found**
```
nvcc: command not found
```
**Solution:** Set environment variables
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Issue 2: Driver/CUDA version mismatch**
```
CUDA driver version is insufficient for CUDA runtime version
```
**Solution:** Update driver or install compatible CUDA version

**Issue 3: Secure Boot blocking driver**
```
NVIDIA kernel module not loading
```
**Solution:** Enroll MOK key
```bash
sudo mokutil --import /usr/src/kmod-nvidia-*/mok-nvidia.der
sudo reboot
# Select "Enroll MOK" in boot menu
```

**Issue 4: Distro repository has old CUDA**
```
Available CUDA version: 11.7
Required CUDA version: 12.2
```
**Solution:** Use official NVIDIA repository

### Debug Commands

```bash
# Check installed packages
dpkg -l | grep -i cuda  # Debian/Ubuntu
rpm -qa | grep -i cuda  # Fedora/RHEL

# Check driver modules
lsmod | grep nvidia

# Check kernel module build
dkms status

# Check CUDA installation
ls -la /usr/local/cuda*
```

## References

### Official Documentation
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [Linux Driver Archive](https://www.nvidia.com/en-us/drivers/unix/)

### nvidia-inst Resources
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Detailed system design
- [AGENTS.md](../AGENTS.md) - Development guidelines
- [README.md](../README.md) - Main documentation

### Community Resources
- [RPM Fusion NVIDIA Guide](https://rpmfusion.org/Howto/NVIDIA)
- [Ubuntu NVIDIA Guide](https://help.ubuntu.com/drivers/NVIDIA)
- [Arch Linux NVIDIA](https://wiki.archlinux.org/title/NVIDIA)
