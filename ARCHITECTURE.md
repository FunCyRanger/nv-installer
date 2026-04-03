# ARCHITECTURE.md - Detailed System Design

This file contains detailed architecture documentation for nvidia-inst.
See AGENTS.md for core guidelines and PATTERNS.md for code examples.

---

## Compatibility & Driver Versioning

### Nvidia CUDA Compatibility Matrix
- Reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- CUDA Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/
- Always check official Nvidia docs for latest compatibility info
- Linux Driver Archive: https://www.nvidia.com/en-us/drivers/unix/

### GPU Compute Capability Mapping
| GPU Generation | Examples | Compute Capability | Driver Lock | CUDA Lock | Support Status |
|----------------|----------|---------------------|-------------|-----------|----------------|
| Kepler | GTX 6xx, 7xx, K-series | 3.0-3.7 | 470.256.02 | 11.* | EOL (security only) |
| Maxwell | GTX 9xx, M-series, Quadro M | 5.0-5.2 | 580.* | 12.* | Limited (Oct 2028) |
| Pascal | GTX 10xx, P100 | 6.0-6.1 | 580.* | 12.* | Limited (Oct 2028) |
| Volta | V100, Titan V | 7.0 | 580.* | 12.* | Limited (Oct 2028) |
| Turing | RTX 20xx, GTX 16xx, T4 | 7.5 | No lock | No lock | Full (latest) |
| Ampere | RTX 30xx, A100, A30 | 8.0-8.6 | No lock | No lock | Full (latest) |
| Ada Lovelace | RTX 40xx, L40, L10 | 8.9 | No lock | No lock | Full (latest) |
| Blackwell | RTX 50xx | 9.0 | No lock | No lock | Full (latest) |

### Latest Driver Versions (March 2026)
- **Production Branch (580.x)**: 580.142 - Recommended for Maxwell/Pascal/Volta
- **New Feature Branch (590.x)**: 590.48.01 - For Turing+
- **Beta Branch**: 590.44.01
- **Legacy (470.x)**: 470.256.02 - Final for Kepler

### Driver Version Selection Logic
```python
GPU_DRIVER_MAX_VERSIONS: dict[str, str] = {
    "kepler": "470.256.02",      # EOL - security updates only
    "maxwell": "580.142",        # Limited - 580 branch
    "pascal": "580.142",         # Limited - 580 branch
    "volta": "580.142",          # Limited - 580 branch
    "turing": "590.48.01",       # Full support - latest
    # Ampere+: use latest (590.x)
}

def is_driver_eol(generation: str) -> bool:
    """Check if GPU generation is truly EOL."""
    return generation == "kepler"  # Only Kepler is EOL

def get_driver_branch(generation: str) -> str:
    """Get driver branch for GPU generation."""
    if generation in ("maxwell", "pascal", "volta"):
        return "580"
    return "590"  # Turing+
```

### Kernel Version Compatibility
- Check distro kernel version: `uname -r`
- Nvidia drivers require specific kernel versions
- Warn user if kernel is too old/new for driver
- Some distros lock kernel with driver (e.g., Ubuntu HWE)

### Version Locking for EOL GPUs
- Detect EOL GPUs automatically via lspci/nvidia-smi
- Lock package manager to max version (e.g., apt pin, dnf versionlock)
- Show clear warning to user about limited support
- Offer to install compatible driver version

### CUDA Version Compatibility Matrix
Based on official NVIDIA CUDA Toolkit and Architecture Matrix:

| GPU Generation | CUDA Range | CUDA Lock | Lock Type | Last CUDA Support |
|----------------|------------|-----------|-----------|-------------------|
| Kepler | 7.5 - 11.8 | `11.*` | Major (EOL) | CUDA 11.x |
| Maxwell | 7.5 - 12.8 | `12.*` | Major (Limited) | CUDA 12.x |
| Pascal | 8.0 - 12.8 | `12.*` | Major (Limited) | CUDA 12.x |
| Volta | 9.0 - 12.8 | `12.*` | Major (Limited) | CUDA 12.x |
| Turing | 10.0 - 13.x | None | N/A | Ongoing |
| Ampere | 11.0 - 13.x | None | N/A | Ongoing |
| Ada Lovelace | 11.8 - 13.x | None | N/A | Ongoing |
| Blackwell | 12.4 - 13.x | None | N/A | Ongoing |

**Notes:**
- Maxwell, Pascal, Volta: CUDA support frozen at 12.x (feature-complete as of CUDA 12.9)
- CUDA 13.0+ drops support for Maxwell, Pascal, and Volta
- Kepler: End-of-life, locked to CUDA 11.x branch
- Modern GPUs (Turing+): Full CUDA support, no version lock

### CUDA Version Locking Implementation
```python
# CUDARange dataclass includes locking fields
@dataclass(frozen=True)
class CUDARange:
    min_version: str
    max_version: str | None = None
    recommended: str = "12.2"
    locked_major: str | None = None  # Lock to major (e.g., "12")
    is_locked: bool = False          # Whether CUDA is locked

# DriverRange includes CUDA lock info
@dataclass
class DriverRange:
    # ... driver fields ...
    cuda_locked_major: str | None = None  # e.g., "12" for 12.*
    cuda_is_locked: bool = False          # True for EOL/Limited GPUs

# Validation respects locks
def validate_cuda_version_with_lock(cuda_version: str, gpu: GPUInfo) -> tuple[bool, str]:
    driver_range = get_driver_range(gpu)
    if driver_range.cuda_is_locked:
        cuda_major = cuda_version.split(".")[0]
        if cuda_major != driver_range.cuda_locked_major:
            return False, f"CUDA locked to {driver_range.cuda_locked_major}.x"
    return validate_cuda_version(cuda_version, gpu)
```

### CUDA Version Pinning by Distro
```python
# APT: Pattern-based pinning
Pin: version 12.*
Pin-Priority: 1001

# DNF: Versionlock with wildcard
dnf versionlock add --raw 'cuda-toolkit-12.*'

# Zypper: Version lock
zypper addlock 'cuda-toolkit >= 12.0, < 13.0'

# Pacman: Manual lock
pacman -D --lock cuda
```

---

## Distro Detection & Package Manager

### Tool-Based Package Manager Detection

Instead of mapping distro IDs to package managers, nvidia-inst detects the actual package manager tool available on the system. This approach supports any Linux distribution using supported tools.

### How It Works

```python
def detect_package_tool() -> str | None:
    """Detect available package management tool.

    Checks for tools in order of preference:
    1. apt (Debian/Ubuntu)
    2. dnf5 (Fedora latest)
    3. dnf (Fedora)
    4. pacman (Arch)
    5. zypper (openSUSE)

    Returns:
        Tool name (apt, dnf5, dnf, pacman, zypper) or None.
    """
    tools = ["apt", "dnf5", "dnf", "yum", "pacman", "pamac", "paru", "yay", "zypper"]
    for tool in tools:
        if shutil.which(tool):
            return tool
    return None

def get_package_manager() -> PackageManager:
    """Get appropriate package manager by detecting available tools."""
    tool = detect_package_tool()
    manager_class = _TOOL_MANAGERS.get(tool)
    return manager_class()
```

### Tool Property

Each package manager class has a `tool` property that returns the actual tool name:

```python
class DnfManager(PackageManager):
    @property
    def tool(self) -> str:
        """Get the tool name (dnf or dnf5)."""
        return self._dnf_version  # Detects dnf4 vs dnf5
```

### Distro-Specific Setup
- Fedora: Enable RPM Fusion (`dnf install https://.../rpmfusion-free-release-*.noarch.rpm`)
- Ubuntu: Add PPA or use proprietary driver repository
- Arch: Enable multilib for 32-bit CUDA
- Debian: Add contrib/non-free repos

---

## GUI Requirements

### Tkinter Implementation
- Use `ttk` widgets for native look
- Handle DPI scaling (`root.tk.call('tk', 'scaling', factor)`)
- Graceful fallback to CLI if DISPLAY not available
- Use `ttk.Progressbar` for progress indication
- Use `tkinter.scrolledtext` for log viewer with auto-scroll

### Zenity Implementation
- Check availability: `command -v zenity`
- Progress: `zenity --progress --percentage=0`
- Info dialog: `zenity --info --text="..."`
- Error dialog: `zenity --error --text="..."`
- Question: `zenity --question --text="..."`
- Parse output, strip whitespace

### Required GUI Elements
| Element | Description |
|---------|-------------|
| Distro Details | Show OS name, version, kernel version |
| GPU Details | Show GPU model, compute capability, VRAM |
| Driver Range | Show min/max compatible driver versions |
| Current Status | Show installed driver version (if any) |
| CUDA Version | Show available CUDA version |
| Install Button | Start installation with confirmation |
| Progress Bar | Show installation progress |
| Log Viewer | Scrollable text area, auto-scroll to bottom |
| Tooltips | Hover help on all interactive elements |

### GUI Layout (Tkinter example)
```
┌─────────────────────────────────────────────┐
│  nvidia-inst                               │
├─────────────────────────────────────────────┤
│  ┌─ Distribution ───────────────────────┐  │
│  │  Ubuntu 22.04, Kernel 5.15.0-91      │  │
│  └───────────────────────────────────────┘  │
│  ┌─ GPU ─────────────────────────────────┐  │
│  │  NVIDIA GeForce RTX 3080             │  │
│  │  Compute Capability: 8.6            │  │
│  │  VRAM: 10GB                          │  │
│  └───────────────────────────────────────┘  │
│  ┌─ Compatibility ──────────────────────┐  │
│  │  Driver: 535.x - 550.x              │  │
│  │  CUDA: 11.8 - 12.x                  │  │
│  │  Status: ✓ Compatible               │  │
│  └───────────────────────────────────────┘  │
│  [  Install Driver  ]   [  Cancel  ]       │
│  ┌─ Logs ───────────────────────────────┐  │
│  │  10:00:00 Starting installation...   │  │
│  │  10:00:01 Downloading driver...      │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## Error Handling

### Common Error Scenarios
| Error | Detection | Handling |
|-------|-----------|----------|
| Nouveau loaded | `lsmod | grep nouveau` | Blacklist and reboot prompt |
| Secure Boot | `mokutil --sb-state` | Ask to sign module or disable SB |
| Kernel mismatch | Compare `uname -r` with driver | Show warning, offer alternatives |
| Dependency missing | Package manager error | Auto-install or show missing list |
| Driver conflict | Multiple nvidia packages | Remove old packages first |
| GPU not detected | lspci/nvidia-smi empty | Show no-GPU message |

### Recovery Actions
- Always provide clear error messages with resolution steps
- Offer "Try Again" option after fixing errors
- Log all errors to `/var/log/nvidia-inst/`
- Save installer log before exiting on error

---

## Logging

### Log Directory
- Primary: `/var/log/nvidia-inst/`
- Fallback: `~/.local/share/nvidia-inst/logs/` (if no root)
- Create directory with `mkdir -p` if not exists

### Log File Naming
- Main log: `install.log`
- Error log: `error.log`
- Rotate: Keep last 5 logs (install.log.1, install.log.2, ...)

### Log Levels
- `DEBUG`: Detailed debug info (only with --debug flag)
- `INFO`: Normal operation messages
- `WARNING`: Non-critical issues
- `ERROR`: Operation failed

---

## Compatibility Matrix System

### Overview
The compatibility matrix provides GPU/driver compatibility data sourced from official Nvidia documentation, with support for online updates and offline fallback.

### Directory Structure
```
src/nvidia_inst/gpu/matrix/
├── __init__.py               # Package exports
├── data.py                  # Dataclasses for matrix structures
├── manager.py               # Matrix fetching, caching, smart update
└── compatibility_data.json   # Versioned fallback data
```

### Data Structures (data.py)
- `SupportStatus`: Enum for GPU support status (FULL, LIMITED, EOL)
- `CUDARange`: CUDA version range for a GPU generation
- `ComputeCapability`: GPU compute capability range
- `GPUGenerationInfo`: Compatibility info for a GPU generation
- `DriverBranchInfo`: Information about a driver branch
- `MatrixMeta`: Metadata for the matrix (version, sources, timestamps)

### Matrix Manager (manager.py)
- `MatrixManager`: Main class for managing matrix data
  - `check_for_updates()`: Check for updates (non-blocking)
  - `get_generation_info()`: Get GPU compatibility info
  - `get_branch_info()`: Get driver branch info
  - `is_online_data`: Property indicating if using online/fallback data

### Cache Behavior
- Cache location: `~/.cache/nvidia-inst/matrix_cache.json`
- Cache TTL: 24 hours
- Automatic refresh on startup if cache is stale
- Fallback to committed JSON if offline

### Update Script (scripts/update-matrix.py)
```bash
python scripts/update-matrix.py --dry-run    # Preview changes
python scripts/update-matrix.py --check      # Check status only
python scripts/update-matrix.py --verify    # Verify data integrity
python scripts/update-matrix.py              # Update if changed
```

### GitHub Actions Workflow
- Runs daily at midnight (cron: '0 0 * * *')
- Can be triggered manually via workflow_dispatch
- Creates PR if updates available

### Sources
- CUDA Toolkit Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- Nvidia Driver Archive: https://download.nvidia.com/XFree86/Linux-x86_64/
- Unix Driver Archive: https://www.nvidia.com/en-us/drivers/unix/

---

## Revert to Nouveau Feature

### Purpose
The revert feature allows switching from the proprietary Nvidia driver to the open-source Nouveau driver.

### Implementation (installer/uninstaller.py)
- `revert_to_nouveau(distro_id)`: Main function
  - Removes proprietary packages (distro-specific)
  - Removes Nouveau blacklist
  - Rebuilds initramfs
  - Returns `RevertResult` with status

### CLI Integration
- `--revert-to-nouveau`: Switch to Nouveau
- Always prompts for confirmation (no `-y` bypass)
- Shows list of packages to be removed
- Provides clear instructions for reboot

### Distro-Specific Package Removal
| Distro | Packages Removed |
|--------|----------------|
| Ubuntu/Debian | `nvidia-driver-*`, `nvidia-dkms-*`, `libnvidia-*` |
| Fedora | `akmod-nvidia`, `xorg-x11-drv-nvidia*` |
| Arch | `nvidia`, `nvidia-580xx-dkms`, `nvidia-open` |
| openSUSE | `x11-video-nvidia*`, `nvidia-compute*` |

### Safety
- Always requires user confirmation
- Can be run without GPU installed
- Gracefully handles missing packages

---

## Distribution Support Priority

1. Ubuntu (most common, well-tested)
2. Fedora (RPM Fusion complexity)
3. Debian (older packages)
4. Arch (rolling release, latest drivers)
