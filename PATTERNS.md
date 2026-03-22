# PATTERNS.md - Code Patterns and Implementation Examples

This file contains reusable code patterns for nvidia-inst development.

---

## Package Manager Abstraction

### Abstract Base Class
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

### Concrete Implementation (Apt)
```python
class Apt(PackageManager):
    """Debian/Ubuntu package manager."""

    def install(self, packages: list[str]) -> bool:
        cmd = ["apt-get", "install", "-y", *packages]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def update(self) -> bool:
        result = subprocess.run(
            ["apt-get", "update"], capture_output=True, text=True
        )
        return result.returncode == 0

    def is_available(self) -> bool:
        return shutil.which("apt-get") is not None
```

### Factory Pattern
```python
def get_package_manager(distro_id: str) -> PackageManager:
    """Get appropriate package manager for distribution."""
    managers: dict[str, type[PackageManager]] = {
        "ubuntu": Apt,
        "debian": Apt,
        "fedora": Dnf,
        "arch": Pacman,
        "opensuse": Zypper,
    }
    manager_class = managers.get(distro_id)
    if manager_class is None:
        raise UnsupportedDistroError(distro_id)
    return manager_class()
```

---

## GPU Detection

### Primary Detection: nvidia-smi
```python
def detect_gpu() -> dict[str, Any] | None:
    """Detect GPU using nvidia-smi (primary method)."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap,cuda_version",
        "--format=csv,noheader",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    
    parts = result.stdout.strip().split(",")
    if len(parts) < 3:
        return None
    
    return {
        "model": parts[0].strip(),
        "driver_version": parts[1].strip(),
        "compute_capability": float(parts[2].strip()),
        "cuda_version": parts[3].strip() if len(parts) > 3 else None,
    }
```

### Fallback Detection: lspci
```python
def detect_via_lspci() -> dict[str, Any] | None:
    """Fallback GPU detection using lspci."""
    result = subprocess.run(
        ["lspci"], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "nvidia" in line.lower() and "vga" in line.lower():
            return parse_gpu_model(line)
    return None

def parse_gpu_model(pci_line: str) -> dict[str, Any]:
    """Parse GPU model from lspci output."""
    model_match = re.search(r"NVIDIA.*?(?=\[|$)", pci_line, re.IGNORECASE)
    if model_match:
        model = model_match.group().strip()
        generation, compute_cap = get_generation_info(model)
        return {"model": model, "generation": generation, "compute_capability": compute_cap}
    return None
```

---

## Testing Patterns

### Fixtures
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

@pytest.fixture
def mock_package_manager():
    manager = MagicMock(spec=PackageManager)
    manager.install.return_value = True
    manager.update.return_value = True
    return manager
```

### Mocking Subprocess
```python
@patch("subprocess.run")
def test_detect_gpu_rtx_3080(mock_run):
    mock_run.return_value = Mock(
        stdout="NVIDIA GeForce RTX 3080,535.154.05,8.6,12.2\n",
        returncode=0
    )
    gpu = detect_gpu()
    assert gpu["model"] == "NVIDIA GeForce RTX 3080"
    assert gpu["compute_capability"] == 8.6
```

### Parametrized Tests
```python
@pytest.mark.parametrize("distro,expected", [
    ("ubuntu", "apt"),
    ("debian", "apt"),
    ("fedora", "dnf"),
    ("arch", "pacman"),
    ("opensuse", "zypper"),
])
def test_get_package_manager(distro, expected):
    manager = get_package_manager(distro)
    assert isinstance(manager, eval(expected.capitalize()))
```

---

## Logging Setup

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

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
    logger = logging.getLogger(__name__)
    return logger
```

### Log Format
```
2024-01-15 10:23:45 [INFO] Starting nvidia-inst v1.0.0
2024-01-15 10:23:45 [INFO] Detected distro: Ubuntu 22.04
2024-01-15 10:23:46 [INFO] Detected GPU: NVIDIA GeForce RTX 3080
2024-01-15 10:23:47 [WARNING] Current driver: 525.x (outdated)
2024-01-15 10:24:01 [INFO] Installing nvidia-driver-535...
```

---

## Error Handling Patterns

### Custom Exception Classes
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

class DriverInstallError(Exception):
    """Raised when driver installation fails."""
    pass
```

### Error Recovery Pattern
```python
try:
    install_driver(version)
except NouveauLoadedError as e:
    logger.error(f"Nouveau must be disabled: {e}")
    offer_disable_nouveau()
except SecureBootError as e:
    logger.error(f"Secure Boot blocking installation: {e}")
    offer_disable_secure_boot()
except DriverInstallError as e:
    logger.error(f"Driver installation failed: {e}")
    save_log_and_exit()
```

---

## CUDA Version Locking

### Locking Strategy by GPU Status

| GPU Status | Driver Lock | CUDA Lock | Reason |
|------------|-------------|-----------|--------|
| EOL (Kepler) | `470.256.02` (exact) | `11.*` (major) | No more CUDA updates |
| Limited (Maxwell/Pascal/Volta) | `580.*` (branch) | `12.*` (major) | Frozen at CUDA 12.x |
| Full (Turing+) | None | None | Latest CUDA supported |

### Validating CUDA Version with Lock
```python
def validate_cuda_version_with_lock(cuda_version: str, gpu: GPUInfo) -> tuple[bool, str]:
    """Validate CUDA version respecting locked major version."""
    driver_range = get_driver_range(gpu)

    if not driver_range.cuda_is_locked:
        return validate_cuda_version(cuda_version, gpu)

    # Check major version matches lock
    cuda_major = cuda_version.split(".")[0]
    if cuda_major != driver_range.cuda_locked_major:
        return (
            False,
            f"CUDA for {gpu.generation} GPUs is locked to {driver_range.cuda_locked_major}.x",
        )

    return validate_cuda_version(cuda_version, gpu)
```

### Pinning CUDA by Major Version
```python
def pin_cuda_to_major_version(
    distro_id: str,
    major_version: str,
    pkg_manager: PackageManager,
) -> bool:
    """Pin CUDA packages to a major version (e.g., '12' → 12.*)"""
    pattern = f"{major_version}.*"
    packages = _get_cuda_packages_for_pinning(distro_id, major_version)

    for pkg in packages:
        if not pkg_manager.pin_version(pkg, pattern):
            logger.warning(f"Failed to pin {pkg} to {pattern}")
            return False
        logger.info(f"Pinned {pkg} to {pattern}")

    return True
```

### Auto-Selecting CUDA Version Based on Lock
```python
# In install_driver() - auto-select CUDA if locked
if with_cuda and cuda_version is None and driver_range and driver_range.cuda_is_locked:
    if driver_range.cuda_locked_major:
        # Limited: use locked major (e.g., "12.0")
        cuda_version = f"{driver_range.cuda_locked_major}.0"
        logger.info(f"Auto-selected CUDA {cuda_version} (locked to {driver_range.cuda_locked_major}.x)")
    elif driver_range.cuda_max:
        # EOL: use max version
        cuda_version = driver_range.cuda_max
        logger.info(f"Auto-selected CUDA {cuda_version} (locked for EOL GPU)")
```

### CLI Display with Lock Info
```python
def print_compatibility_info(distro, gpu, driver_range):
    # ... existing code ...

    # CUDA line with lock info
    if driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            print(f"CUDA: {driver_range.cuda_locked_major}.x (locked to major version)")
        else:
            print(f"CUDA: {driver_range.cuda_max or driver_range.cuda_min} (locked)")
    elif driver_range.cuda_min:
        if driver_range.cuda_max:
            print(f"CUDA: {driver_range.cuda_min} - {driver_range.cuda_max}")
        else:
            print(f"CUDA: {driver_range.cuda_min} or later")
```
