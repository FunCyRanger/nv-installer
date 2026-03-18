# AGENTS.md - Development Guidelines for nvidia-inst

## Project Overview
Cross-distribution Linux script for installing the latest compatible Nvidia driver with CUDA support.
Supports Ubuntu, Fedora, Arch, Debian and detects GPU/distro automatically using Python/Bash.

---

## Build/Lint/Test Commands

### Python
```bash
# Install dependencies
pip install -r requirements.txt

# Linting
ruff check .

# Formatting
black .

# Type checking
mypy .

# Run all tests
pytest

# Run single test
pytest tests/ -k "test_name"
python -m pytest path/to/test.py::TestClass::test_method
```

### Bash
```bash
# Linting
shellcheck scripts/*.sh
```

### Combined (if using Makefile)
```bash
make lint    # Run ruff + shellcheck
make test    # Run pytest
make format  # Run black
```

---

## Code Style - Python

### Formatting
- Line length: 88 characters (Black default)
- Indentation: 4 spaces
- No trailing whitespace
- Use implicit string concatenation for related strings

### Imports (per PEP 8, sorted alphabetically)
```python
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests  # third-party

from nvidia_inst import utils  # local
```

### Type Hints
- Use type hints for all function arguments and return values
- Use `Any` sparingly, prefer Union/Optional
```python
def get_driver_version(gpu_model: str) -> str | None:
    ...
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Docstrings (Google style)
```python
def install_driver(version: str) -> bool:
    """Install the specified Nvidia driver version.

    Args:
        version: Driver version string (e.g., "535.154.05")

    Returns:
        True if installation succeeded, False otherwise.

    Raises:
        DriverInstallError: If installation fails.
    """
```

### Error Handling
- Use specific exception classes
- Never catch bare `Exception` unless re-raising
- Always log errors with context
```python
class DriverInstallError(Exception):
    """Raised when driver installation fails."""
    pass

try:
    install_driver(version)
except DriverInstallError as e:
    logger.error(f"Failed to install driver: {e}")
    raise
```

---

## Code Style - Bash

### ShellCheck Compliance
- All shell scripts must pass `shellcheck`
- Use `set -euo pipefail` at script top
- Use `[[ ]]` for tests, not `[ ]`

### Naming and Structure
- Scripts: `kebab-case.sh`
- Variables: `UPPER_SNAKE_CASE` for constants, `snake_case` for locals
- Functions: `snake_case()`
- Always quote variables: `"$VAR"`

### Error Handling
```bash
#!/bin/bash
set -euo pipefail

function install_driver() {
    local version="$1"
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi not found" >&2
        return 1
    fi
}
```

---

## Project Structure
```
nvidia-inst/
├── src/
│   ├── __init__.py
│   ├── cli.py              # Main entry point
│   ├── distro/
│   │   ├── __init__.py
│   │   ├── detector.py     # Distro detection logic
│   │   └── package_manager.py  # Abstract + implementations
│   ├── gpu/
│   │   ├── __init__.py
│   │   ├── detector.py     # GPU detection
│   │   └── compatibility.py  # Nvidia compatibility matrix
│   ├── installer/
│   │   ├── __init__.py
│   │   ├── driver.py       # Driver installation logic
│   │   └── cuda.py         # CUDA installation
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── tkinter_gui.py
│   │   └── zenity_gui.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py       # Logging utilities
├── scripts/
│   └── install-*.sh
├── tests/
│   ├── test_distro.py
│   ├── test_gpu.py
│   ├── test_installer.py
│   └── test_compatibility.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Compatibility & Driver Versioning

### Nvidia CUDA Compatibility Matrix
- Reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- CUDA Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/
- Always check official Nvidia docs for latest compatibility info
- Linux Driver Archive: https://www.nvidia.com/en-us/drivers/unix/

### GPU Compute Capability Mapping
| GPU Generation | Examples | Compute Capability | Max Driver | Support Status |
|----------------|----------|---------------------|------------|----------------|
| Kepler | GTX 6xx, 7xx, K系列 | 3.0-3.7 | 470.x | EOL (security only) |
| Maxwell | GTX 9xx, M系列, Quadro M | 5.0-5.2 | 580.x | Limited (580 branch) |
| Pascal | GTX 10xx, P100 | 6.0-6.1 | 580.x | Limited (580 branch) |
| Volta | V100, Titan V | 7.0 | 580.x | Limited (580 branch) |
| Turing | RTX 20xx, GTX 16xx, T4 | 7.5 | 590.x | Full (latest drivers) |
| Ampere | RTX 30xx, A100, A30 | 8.0-8.6 | 590.x | Full (latest drivers) |
| Ada Lovelace | RTX 40xx, L40, L10 | 8.9 | 590.x | Full (latest drivers) |
| Blackwell | RTX 50xx | 9.0 | 590.x | Full (latest drivers) |

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

---

## Distro Detection & Package Manager

### Distro Detection
```python
def detect_distro() -> dict[str, str]:
    """Detect Linux distribution from /etc/os-release or lsb_release."""
    if Path("/etc/os-release").exists():
        # Parse ID, VERSION_ID, NAME, PRETTY_NAME
        ...
    elif command_exists("lsb_release"):
        # Use lsb_release -is, -rs
        ...
    raise DistroDetectionError("Cannot detect distribution")
```

### Package Manager Abstraction
```python
class PackageManager(ABC):
    """Abstract base class for package managers."""

    @abstractmethod
    def update(self) -> bool:
        """Update package lists."""
        ...

    @abstractmethod
    def upgrade(self) -> bool:
        """Upgrade all packages."""
        ...

    @abstractmethod
    def install(self, packages: list[str]) -> bool:
        """Install packages."""
        ...

    @abstractmethod
    def remove(self, packages: list[str]) -> bool:
        """Remove packages."""
        ...

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """Search for packages."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if package manager is available."""
        ...
```

### Implementations
- **Apt**: Debian/Ubuntu - uses `apt`, `apt-get`
- **Dnf**: Fedora/RHEL - uses `dnf`, handles RPM Fusion
- **Pacman**: Arch/Manjaro - uses `pacman`
- **Zypper**: openSUSE - uses `zypper`

### Distro-Specific Setup
- Fedora: Enable RPM Fusion (`dnf install https://.../rpmfusion-free-release-*.noarch.rpm`)
- Ubuntu: Add PPA or use proprietary driver repository
- Arch: Enable multilib for 32-bit CUDA
- Debian: Add contrib/non-free repos

---

## GPU Detection

### Primary Detection: nvidia-smi
```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap,cuda_version --format=csv
```
Returns: GPU name, driver version, compute capability, CUDA version

### Fallback Detection: lspci
```bash
lspci | grep -i nvidia
```
Parse VGA compatible controller for GPU model

### GPU Model Parsing
```python
GPU_PATTERNS = [
    (r"RTX\s*50\d{2}", "blackwell", 9.0),
    (r"RTX\s*40\d{2}", "ada", 8.9),
    (r"RTX\s*30\d{2}", "ampere", 8.6),
    (r"RTX\s*20\d{2}", "turing", 7.5),
    (r"GTX\s*16\d{2}", "turing", 7.5),
    (r"GTX\s*10\d{2}", "pascal", 6.1),
    (r"GTX\s*9\d{3}", "maxwell", 5.2),
    (r"GTX\s*[67]\d{2}", "kepler", 3.7),
]

def parse_gpu_model(model_string: str) -> tuple[str, str, float]:
    """Return (marketing_name, generation, compute_capability)."""
    for pattern, gen, cc in GPU_PATTERNS:
        if re.search(pattern, model_string, re.IGNORECASE):
            return model_string, gen, cc
    return model_string, "unknown", 0.0
```

### No GPU Detected
- Handle systems without Nvidia GPU gracefully
- Show appropriate message
- Exit with code 0 (not an error, just nothing to do)

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
| Nouveau loaded | `lsmod \| grep nouveau` | Blacklist and reboot prompt |
| Secure Boot | `mokutil --sb-state` | Ask to sign module or disable SB |
| Kernel mismatch | Compare `uname -r` with driver | Show warning, offer alternatives |
| Dependency missing | Package manager error | Auto-install or show missing list |
| Driver conflict | Multiple nvidia packages | Remove old packages first |
| GPU not detected | lspci/nvidia-smi empty | Show no-GPU message |

### Error Classes
```python
class NouveauLoadedError(Exception):
    """Nouveau kernel module is loaded and must be disabled."""
    pass

class SecureBootError(Exception):
    """Secure Boot is enabled and preventing driver installation."""
    pass

class KernelIncompatibleError(Exception):
    """Kernel version is incompatible with driver."""
    pass

class EOLGPUError(Exception):
    """GPU is end-of-life and has limited driver support."""
    pass
```

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

### Log Format
```
2024-01-15 10:23:45 [INFO] Starting nvidia-inst v1.0.0
2024-01-15 10:23:45 [INFO] Detected distro: Ubuntu 22.04
2024-01-15 10:23:46 [INFO] Detected GPU: NVIDIA GeForce RTX 3080
2024-01-15 10:23:46 [INFO] Compatible driver range: 535.x - 550.x
2024-01-15 10:23:47 [WARNING] Current driver: 525.x (outdated)
2024-01-15 10:24:01 [INFO] Installing nvidia-driver-535...
2024-01-15 10:25:30 [INFO] Installation complete
```

### Log Levels
- `DEBUG`: Detailed debug info (only with --debug flag)
- `INFO`: Normal operation messages
- `WARNING`: Non-critical issues
- `ERROR`: Operation failed

### Python Logging Setup
```python
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("/var/log/nvidia-inst")
LOG_FILE = LOG_DIR / "install.log"

def setup_logging(debug: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO

    handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5_242_880, backupCount=5  # 5MB, 5 files
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logging.basicConfig(level=level, handlers=[handler])
```

---

## Testing Guidelines

### Unit Tests
- Use `pytest` with `pytest-mock`
- Test one thing per test function
- Use descriptive names: `test_detect_ubuntu_returns_22_04()`
- Mock system calls (lspci, apt, dnf, pacman, nvidia-smi)

### Test Fixtures
```python
@pytest.fixture
def mock_gpu():
    return {
        "model": "NVIDIA GeForce RTX 3080",
        "compute_capability": 8.6,
        "driver_version": "535.154.05",
        "cuda_version": "12.2",
    }

@pytest.fixture
def mock_distro():
    return {
        "id": "ubuntu",
        "version_id": "22.04",
        "name": "Ubuntu 22.04.3 LTS",
        "kernel": "5.15.0-91-generic",
    }
```

### Mocking Examples
```python
@patch("subprocess.run")
def test_detect_gpu_rtx_3080(mock_run):
    mock_run.return_value = Mock(
        stdout="NVIDIA GeForce RTX 3080\n10.0 GB\n",
        returncode=0
    )
    gpu = detect_gpu()
    assert gpu["model"] == "NVIDIA GeForce RTX 3080"

@pytest.mark.parametrize("distro,expected", [
    ("Ubuntu", "apt"),
    ("Fedora", "dnf"),
    ("Arch", "pacman"),
])
def test_detect_package_manager(distro, expected):
    assert detect_package_manager(distro) == expected
```

### Integration Tests
- Mark with `@pytest.mark.integration`
- Skip in CI unless explicitly required
- Use Docker/containers for testing
- Test on clean installations of each distro

---

## Important Notes

### Safety
- NEVER install drivers without user confirmation
- Always create backup/recovery options (Timeshift snapshot, etc.)
- Log all operations to `/var/log/nvidia-inst/`
- Verify driver signature before installation
- Require root/sudo for installation

### No Self-Updates
- Only check for distro updates (apt update/upgrade, etc.)
- Do NOT auto-update the script itself without explicit user consent
- Version check is allowed, but user must approve updates

### End-of-Life GPU Handling
- Detect older GPU generations automatically
- Lock to maximum supported driver version
- Show clear warning about limited support
- Offer best-effort installation

### Distribution Support Priority
1. Ubuntu (most common, well-tested)
2. Fedora (RPM Fusion complexity)
3. Debian (older packages)
4. Arch (rolling release, latest drivers)

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
