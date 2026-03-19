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
├── installer/          # Driver, CUDA installation, uninstaller
├── gui/                # Tkinter and Zenity implementations
└── utils/logger.py     # Logging utilities

tests/                  # pytest with pytest-mock
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
