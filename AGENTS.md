# AGENTS.md - Development Guidelines for nvidia-inst

Cross-distribution Linux script for installing Nvidia drivers with CUDA support.
Supports Ubuntu, Fedora, Arch, Debian and detects GPU/distro automatically.

---

## Build/Lint/Test Commands

```bash
pip install -r requirements.txt          # Install dependencies
ruff check . && shellcheck scripts/*.sh  # Linting
black .                                  # Formatting (line-length: 88)
mypy .                                   # Type checking
pytest                                   # Run all tests
pytest tests/ -k "test_name"             # Run single test
pytest tests/test_gpu.py::TestClass::test_method  # Specific method
make lint, make test, make format        # Make targets
```

---

## CUDA Support by GPU Generation

### CUDA Support by Driver Type

| Driver | CUDA Support | GPU Generations |
|--------|-------------|-----------------|
| **Proprietary** | Full CUDA | All (Kepler → Blackwell) |
| **NVIDIA Open** | Full CUDA | Turing+ ONLY (Turing → Blackwell) |
| **Nouveau** | No CUDA | All (no CUDA support ever) |

**Important**: NVIDIA Open is only available for Turing+ GPUs. Older generations
(Maxwell, Pascal, Volta) must use proprietary drivers for CUDA support.

### CUDA Version Ranges by GPU Generation

| Generation | Examples | CUDA Min | CUDA Max | Driver Branch |
|------------|----------|----------|----------|---------------|
| Blackwell | RTX 5090, 5080, GB200 | 12.4 | 13.x | 590/595 |
| Ada Lovelace | RTX 4090, 4080, L40 | 11.8 | 12.8 | 590 |
| Ampere | RTX 3090, 3080, A100 | 11.0 | 12.8 | 590 |
| Turing | RTX 2080, GTX 1650, T4 | 10.0 | 12.8 | 590 |
| Volta | V100, Titan V | 9.0 | 12.8 | 580 (Limited) |
| Pascal | GTX 1080, P100 | 8.0 | 12.8 | 580 (Limited) |
| Maxwell | GTX 980, 970, M-series | 7.5 | 12.8 | 580 (Limited) |
| Kepler | GTX 780, K-series | 7.5 | 9.0 | 470 (EOL) |

### When CUDA Is NOT Possible

1. **Nouveau driver**: Open-source Xorg driver has NO CUDA support
2. **Kepler GPUs**: EOL status, limited to CUDA 9.0 max (security updates only)

---

## Driver Recommendations

Both proprietary and NVIDIA Open are valid configurations. The choice depends on user needs.

### Proprietary Driver (Recommended for most users)
- **Best performance and feature support**
- Full CUDA compatibility for all GPU generations
- Proprietary blob with closed-source kernel modules
- Available in non-free repositories

### NVIDIA Open Driver (For open-source enthusiasts)
- **Open kernel modules** while maintaining CUDA support
- Turing+ GPUs only (availability depends on repository support)
- Same CUDA capabilities as proprietary driver
- Use when distro supports `nvidia-open` packages

### Nouveau Driver (For non-CUDA workloads)
- **Open-source Xorg driver** built into Linux kernel
- NO CUDA support whatsoever
- Use for basic display only
- Default fallback when proprietary drivers unavailable

---

## Driver State Detection

The CLI automatically detects current driver state and builds appropriate options:

```python
# Driver states (from cli.py)
class DriverStatus(Enum):
    OPTIMAL = "optimal"           # Proprietary driver working
    WRONG_BRANCH = "wrong_branch" # Driver may not be optimal
    NVIDIA_OPEN_ACTIVE = "nvidia_open_active"
    NOUVEAU_ACTIVE = "nouveau_active"
    BROKEN_INSTALL = "broken_install"
    NOTHING = "nothing"           # No driver installed

# CLI option building based on state
def _build_nothing_options(cuda_range, nvidia_open_available, nonfree_available):
    # Builds: [Proprietary (RECOMMENDED), NVIDIA Open, Nouveau, Cancel]
```

---

## Hybrid Graphics Support

nvidia-inst supports hybrid graphics systems (Intel/AMD iGPU + NVIDIA dGPU) with automatic detection and power profile management.

### Detection

```python
from nvidia_inst.gpu.hybrid import (
    detect_hybrid,           # Full hybrid info
    is_hybrid_system,        # Boolean check
    detect_system_type,      # 'laptop' or 'desktop'
    get_native_tool,         # Native tool per distro
)

hybrid_info = detect_hybrid(distro_id)
if hybrid_info:
    print(f"System: {hybrid_info.system_type}")
    print(f"iGPU: {hybrid_info.igpu_type}")
    print(f"dGPU: {hybrid_info.dgpu_model}")
    print(f"Native Tool: {hybrid_info.native_tool}")
```

### Native Tools by Distro (Built-in First)

| Distro | Native Tool | Packages | Power Profiles |
|--------|-------------|----------|----------------|
| Ubuntu/Debian | `nvidia-prime` | Included with driver | intel, on-demand, nvidia |
| Fedora | `switcherooctl` | `switcheroo-control` | **intel, nvidia** |
| Pop!_OS | `system76-power` | Built-in | integrated, hybrid, nvidia, compute |
| openSUSE | `switcherooctl` | Built-in (Tumbleweed) | **intel, nvidia** |
| CachyOS | `cachyos-settings` | Built-in | **intel, nvidia** |
| Arch | PRIME env vars | None | hybrid, intel, nvidia |

**Note**: `switcherooctl` does not support hybrid mode. Use DE's right-click menu
("Launch using Dedicated GPU") for per-app GPU selection.

### Power Profiles by Tool

| Tool | Supported Modes |
|------|----------------|
| `nvidia-prime` | `intel`, `hybrid` (on-demand), `nvidia` |
| `switcherooctl` | `intel`, `nvidia` (config file: `/etc/switcherooctl.conf`) |
| `system76-power` | `integrated`, `hybrid`, `nvidia`, `compute` |
| PRIME env vars | `hybrid` (default), `intel`, `nvidia` |

### CLI Commands

```bash
nvidia-inst --show-hybrid-info       # Show hybrid detection results
nvidia-inst --power-profile intel   # iGPU only
nvidia-inst --power-profile hybrid  # iGPU + dGPU on-demand
nvidia-inst --power-profile nvidia  # dGPU always
```

### Running Apps on dGPU

```bash
# With native tool (Ubuntu)
prime-run <app>

# With environment variables (all distros)
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia <app>
```

---

## Python Code Style

### Imports (PEP 8, sorted alphabetically)
```python
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests  # third-party
from nvidia_inst import utils  # local
```

### Type Hints - Use `str | None` over `Optional[str]`
```python
def get_driver_version(gpu_model: str) -> str | None: ...
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions/variables | `snake_case` | `detect_gpu()` |
| Classes | `PascalCase` | `PackageManager` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_DRIVER_VERSION` |
| Private methods | `_leading_underscore` | `_validate_args()` |

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
- Use specific exception classes, never catch bare `Exception` unless re-raising
- Log errors with context
```python
class DriverInstallError(Exception):
    """Raised when driver installation fails."""

try:
    install_driver(version)
except DriverInstallError as e:
    logger.error(f"Failed to install driver: {e}")
    raise
```

---

## Bash Code Style

- All scripts must pass `shellcheck`
- Use `set -euo pipefail` at script top, `[[ ]]` for tests, not `[ ]`
- Scripts: `kebab-case.sh`, Functions: `snake_case()`, always quote `"$VAR"`

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
src/nvidia_inst/
├── cli.py              # Main entry point
├── distro/             # Package manager abstraction (apt, dnf, pacman, zypper)
├── gpu/                # GPU detection and compatibility matrix
│   ├── hybrid.py       # Hybrid graphics detection & management
│   └── matrix/          # GPU compatibility matrix data
├── installer/          # Driver, CUDA installation, uninstaller
│   └── hybrid.py       # Hybrid package installation & configuration
├── gui/                # Tkinter and Zenity implementations
└── utils/logger.py     # Logging utilities

tests/
├── test_hybrid.py      # Hybrid graphics tests
└── conftest.py         # Test fixtures

scripts/                # install-*.sh, update-matrix.py
```

---

## Safety Guidelines

- **NEVER** install drivers without user confirmation
- **NEVER** auto-update the script without explicit user consent
- Log all operations to `/var/log/nvidia-inst/install.log`
- Require root/sudo for installation
- Exit code 0 when no GPU detected (not an error)

---

## Testing

```python
# Mocking subprocess calls
@patch("subprocess.run")
def test_nouveau_loaded(mock_run):
    mock_run.return_value = MagicMock(stdout="nouveau  1638400  0\n", returncode=0)
    assert check_nouveau() is True

# Testing CLI arguments
def test_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--check", "--debug"])
    from nvidia_inst.cli import parse_args
    args = parse_args()
    assert args.check is True
    assert args.debug is True

# Using fixtures from conftest.py
def test_with_mock_gpu(mock_gpu):
    assert mock_gpu["model"] == "NVIDIA GeForce RTX 3080"
```

### Test Fixtures (in `tests/conftest.py`)
- `mock_gpu` - Mock GPU info dict
- `mock_distro` - Mock distro info dict
- `mock_driver_range` - Mock driver range for Ampere
- `mock_driver_range_eol` - Mock driver range for Kepler (EOL)
- `mock_user_yes` / `mock_user_no` - Mock user input
- `mock_has_nvidia_gpu_true` / `mock_has_nvidia_gpu_false`
- `mock_distro_ubuntu` / `mock_distro_fedora`
- `mock_nouveau_loaded` / `mock_nouveau_not_loaded`
- `mock_secure_boot_enabled` / `mock_secure_boot_disabled`

---

## Key Documentation

- **PATTERNS.md** - Code patterns and implementation examples
- **ARCHITECTURE.md** - Detailed system design, GPU matrix, GUI specs
