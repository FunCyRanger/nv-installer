"""Distribution detection for Linux systems."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DistroInfo:
    """Information about the detected Linux distribution."""

    id: str
    version_id: str
    name: str
    pretty_name: str
    kernel: str

    def __str__(self) -> str:
        return f"{self.pretty_name}, Kernel {self.kernel}"


class DistroDetectionError(Exception):
    """Raised when distribution cannot be detected."""
    pass


def detect_distro() -> DistroInfo:
    """Detect Linux distribution from /etc/os-release or lsb_release.

    Returns:
        DistroInfo: Information about the detected distribution.

    Raises:
        DistroDetectionError: If distribution cannot be detected.
    """
    if Path("/etc/os-release").exists():
        return _detect_from_os_release()

    try:
        result = subprocess.run(
            ["lsb_release", "-is"],
            capture_output=True,
            text=True,
            check=True
        )
        distro_id = result.stdout.strip()

        result = subprocess.run(
            ["lsb_release", "-rs"],
            capture_output=True,
            text=True,
            check=True
        )
        version_id = result.stdout.strip()

        result = subprocess.run(
            ["lsb_release", "-ds"],
            capture_output=True,
            text=True,
            check=True
        )
        pretty_name = result.stdout.strip().strip('"')

        return DistroInfo(
            id=distro_id.lower(),
            version_id=version_id,
            name=distro_id,
            pretty_name=pretty_name,
            kernel=_get_kernel_version(),
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise DistroDetectionError(f"Cannot detect distribution: {e}") from e


def _detect_from_os_release() -> DistroInfo:
    """Parse /etc/os-release for distribution info."""
    os_release_path = Path("/etc/os-release")

    data: dict[str, str] = {}
    content = os_release_path.read_text()

    for line in content.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            data[key] = value.strip('"')

    kernel = _get_kernel_version()

    return DistroInfo(
        id=data.get("ID", "unknown"),
        version_id=data.get("VERSION_ID", "unknown"),
        name=data.get("NAME", "Unknown"),
        pretty_name=data.get("PRETTY_NAME", data.get("NAME", "Unknown")),
        kernel=kernel,
    )


def _get_kernel_version() -> str:
    """Get the current kernel version."""
    try:
        result = subprocess.run(
            ["uname", "-r"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_package_manager() -> str:
    """Detect which package manager is available.

    Returns:
        Package manager name: 'apt', 'dnf', 'pacman', 'zypper', or 'unknown'.
    """
    managers = [
        ("apt", ["/usr/bin/apt", "/usr/bin/apt-get"]),
        ("dnf", ["/usr/bin/dnf"]),
        ("pacman", ["/usr/bin/pacman"]),
        ("zypper", ["/usr/bin/zypper"]),
    ]

    for name, paths in managers:
        if any(Path(p).exists() for p in paths):
            logger.info(f"Detected package manager: {name}")
            return name

    logger.warning("Could not detect package manager")
    return "unknown"


def is_ubuntu() -> bool:
    """Check if running on Ubuntu."""
    try:
        distro = detect_distro()
        return distro.id in ("ubuntu", "linuxmint", "pop")
    except DistroDetectionError:
        return False


def is_fedora() -> bool:
    """Check if running on Fedora."""
    try:
        distro = detect_distro()
        return distro.id in ("fedora", "rhel", "centos", "rocky", "alma")
    except DistroDetectionError:
        return False


def is_arch() -> bool:
    """Check if running on Arch Linux or derivatives."""
    try:
        distro = detect_distro()
        return distro.id in ("arch", "manjaro", "endeavouros")
    except DistroDetectionError:
        return False


def is_debian() -> bool:
    """Check if running on Debian."""
    try:
        distro = detect_distro()
        return distro.id == "debian"
    except DistroDetectionError:
        return False


def is_opensuse() -> bool:
    """Check if running on openSUSE."""
    try:
        distro = detect_distro()
        return distro.id in ("opensuse", "sles")
    except DistroDetectionError:
        return False
