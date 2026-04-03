# AGENTS.md - Development Guidelines for nvidia-inst

Cross-distribution Linux script for installing Nvidia drivers with CUDA support.
Supports Ubuntu, Fedora, Arch, Debian and detects GPU/distro automatically.

## Build/Lint/Test Commands

```bash
pip install -r requirements.txt          # Install dependencies
ruff check . && shellcheck nv-install scripts/*.sh  # Linting (Python + Bash)
black .                                  # Formatting (line-length: 88)
mypy src/                                # Type checking (targets src/ directory)
pytest                                   # Run all tests
pytest tests/ -k "test_name"             # Run single test by name
pytest tests/test_gpu.py::TestClass::test_method  # Specific method
make lint, make test, make format        # Make targets
```

## Coverage Commands

```bash
pytest tests/ -v --cov=nvidia_inst --cov-report=term-missing  # Run tests with coverage
make coverage                           # Run tests with coverage (Makefile)
make coverage-xml                       # Generate XML report for CI
```

**Coverage Requirements:**
- Minimum coverage: 70% (configured in pyproject.toml)
- GUI modules (tkinter_gui.py, zenity_gui.py) are excluded from coverage
- CI fails if coverage drops below threshold

## Code Style Guidelines

### Python Code Style

**Imports (PEP 8, sorted alphabetically)**
```python
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests  # third-party
from nvidia_inst import utils  # local
```

**Type Hints** - Use `str | None` over `Optional[str]`

**Naming Conventions**
| Type | Convention | Example |
|------|------------|---------|
| Functions/variables | `snake_case` | `detect_gpu()` |
| Classes | `PascalCase` | `PackageManager` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_DRIVER_VERSION` |
| Private methods | `_leading_underscore` | `_validate_args()` |

**Docstrings (Google style)**
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

**Error Handling**
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

### Bash Code Style

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

## Ruff Rules (from pyproject.toml)

- **E**: pycodestyle errors
- **F**: pyflakes
- **W**: pycodestyle warnings
- **I**: isort (import sorting)
- **N**: pep8-naming
- **UP**: pyupgrade (Python 3.10+ syntax)
- **B**: flake8-bugbear
- **C4**: flake8-comprehensions
- **SIM**: flake8-simplify

## Safety Guidelines

- **NEVER** install drivers without user confirmation
- **NEVER** auto-update the script without explicit user consent
- Log all operations to `/var/log/nvidia-inst/install.log`
- Require root/sudo for installation
- Exit code 0 when no GPU detected (not an error)

## Project Structure

```
src/nvidia_inst/
├── cli/                # CLI package
│   ├── main.py         # Main CLI entry point
│   ├── parser.py       # Argument parsing
│   ├── installer.py    # Installation orchestration
│   ├── commands.py     # CLI commands
│   ├── simulate.py     # Simulation/dry-run logic
│   └── __main__.py    # CLI entry point
├── distro/             # Package manager abstraction
│   ├── detector.py     # Distribution detection
│   ├── factory.py      # Package manager factory
│   ├── tools.py        # Tool-based command generation
│   ├── package_manager.py  # Abstract base class
│   ├── apt.py          # Debian/Ubuntu (APT)
│   ├── dnf.py          # Fedora/RHEL (DNF)
│   ├── pacman.py       # Arch Linux (Pacman)
│   └── zypper.py       # openSUSE (Zypper)
├── gpu/               # GPU detection and compatibility
│   ├── detector.py     # GPU detection
│   ├── compatibility.py # Driver version logic
│   └── matrix/         # GPU compatibility matrix
├── installer/         # Driver, CUDA, uninstaller
│   ├── driver.py       # Driver installation
│   ├── cuda.py         # CUDA installation
│   ├── uninstaller.py  # Revert to Nouveau
│   ├── prerequisites.py # Pre-install checks
│   ├── validation.py   # Post-install validation
│   ├── rollback.py     # Rollback capability
│   ├── offline.py      # Offline installation
│   └── secureboot.py   # Secure Boot handling
├── gui/               # Tkinter and Zenity implementations
└── utils/
    └── logger.py      # Logging utilities

tests/
├── conftest.py        # Test fixtures
├── test_installation_flow.py  # Integration tests
├── test_apt_real.py   # Real APT tests
├── test_dnf_real.py    # Real DNF tests
├── test_pacman_real.py # Real Pacman tests
├── test_zypper_real.py # Real Zypper tests
├── test_cli_workflows_e2e.py  # CLI workflow tests
├── test_simulation_accuracy.py # Simulation tests
├── test_cuda_e2e.py    # CUDA installation tests
├── test_offline_e2e.py # Offline installation tests
├── test_rollback_e2e.py # Rollback tests
├── test_revert_e2e.py  # Revert to Nouveau tests
└── test_e2e_installation.py # E2E installation tests

scripts/
├── update-matrix.py    # Matrix update script
└── install-*.sh        # Distribution install scripts
```

## Testing

**Mocking subprocess calls**
```python
@patch("subprocess.run")
def test_nouveau_loaded(mock_run):
    mock_run.return_value = MagicMock(stdout="nouveau  1638400  0\n", returncode=0)
    assert check_nouveau() is True
```

**Testing CLI arguments**
```python
def test_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--check", "--debug"])
    from nvidia_inst.cli import parse_args
    args = parse_args()
    assert args.check is True
    assert args.debug is True
```

**Using fixtures from conftest.py**
```python
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

## Key Documentation

- **PATTERNS.md** - Code patterns and implementation examples
- **ARCHITECTURE.md** - Detailed system design, GPU matrix, GUI specs
- **docs/CUDA_INSTALLATION.md** - Comprehensive CUDA installation guide with hardware compatibility matrices
