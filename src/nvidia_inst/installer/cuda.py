"""CUDA toolkit installation."""

from abc import ABC, abstractmethod

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class CUDAInstaller(ABC):
    """Abstract base class for CUDA installation."""

    @abstractmethod
    def get_cuda_packages(self, version: str | None = None) -> list[str]:
        """Get CUDA packages for installation.

        Args:
            version: CUDA version (optional).

        Returns:
            List of package names.
        """
        ...

    @abstractmethod
    def is_cuda_installed(self) -> bool:
        """Check if CUDA is already installed.

        Returns:
            True if installed, False otherwise.
        """
        ...

    @abstractmethod
    def get_installed_cuda_version(self) -> str | None:
        """Get currently installed CUDA version.

        Returns:
            Version string if installed, None otherwise.
        """
        ...


class UbuntuCUDAInstaller(CUDAInstaller):
    """CUDA installer for Ubuntu/Debian."""

    def get_cuda_packages(self, version: str | None = None) -> list[str]:
        """Get CUDA packages for Ubuntu."""
        if version:
            return [
                f"cuda-{version}",
                f"cuda-toolkit-{version}",
            ]

        return [
            "cuda",
            "cuda-toolkit",
        ]

    def is_cuda_installed(self) -> bool:
        """Check if CUDA is installed."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_installed_cuda_version(self) -> str | None:
        """Get installed CUDA version."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "release" in line:
                    import re

                    match = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        return match.group(1)
            return None
        except FileNotFoundError:
            return None


class FedoraCUDAInstaller(CUDAInstaller):
    """CUDA installer for Fedora/RHEL."""

    def get_cuda_packages(self, version: str | None = None) -> list[str]:
        """Get CUDA packages for Fedora."""
        if version:
            return [
                f"cuda-runtime-{version}",
                f"cuda-devel-{version}",
            ]

        return ["cuda-toolkit"]

    def is_cuda_installed(self) -> bool:
        """Check if CUDA is installed."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_installed_cuda_version(self) -> str | None:
        """Get installed CUDA version."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "release" in line:
                    import re

                    match = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        return match.group(1)
            return None
        except FileNotFoundError:
            return None


class ArchCUDAInstaller(CUDAInstaller):
    """CUDA installer for Arch Linux."""

    def get_cuda_packages(self, version: str | None = None) -> list[str]:
        """Get CUDA packages for Arch."""
        if version:
            return [f"cuda-{version}"]

        return ["cuda"]

    def is_cuda_installed(self) -> bool:
        """Check if CUDA is installed."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_installed_cuda_version(self) -> str | None:
        """Get installed CUDA version."""
        import subprocess

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "release" in line:
                    import re

                    match = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        return match.group(1)
            return None
        except FileNotFoundError:
            return None


def get_cuda_installer(distro_id: str) -> CUDAInstaller:
    """Get appropriate CUDA installer for distribution.

    Args:
        distro_id: Distribution ID.

    Returns:
        CUDAInstaller instance.
    """
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return UbuntuCUDAInstaller()
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        return FedoraCUDAInstaller()
    elif distro_id in ("arch", "manjaro", "endeavouros"):
        return ArchCUDAInstaller()

    return UbuntuCUDAInstaller()


# ============================================================================
# CUDA Detection and Management Functions
# ============================================================================


def detect_installed_cuda_version() -> str | None:
    """Detect installed CUDA version via nvcc.

    Returns:
        CUDA version string (e.g., "12.2") or None if not installed.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            if "release" in line:
                import re

                match = re.search(r"release (\d+\.\d+)", line)
                if match:
                    return match.group(1)
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_cuda_packages_for_version(distro_id: str, cuda_version: str) -> list[str]:
    """Get CUDA packages for a specific version.

    Args:
        distro_id: Distribution ID
        cuda_version: CUDA version (e.g., "12.2")

    Returns:
        List of package names to install
    """
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return [f"cuda-{cuda_version}", f"cuda-toolkit-{cuda_version}"]
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        return [f"cuda-toolkit-{cuda_version}"]
    elif distro_id in ("arch", "manjaro"):
        return [f"cuda-{cuda_version}"]
    elif distro_id in ("opensuse", "sles"):
        return [f"cuda-{cuda_version}"]
    return []


def get_uninstall_cuda_packages(
    distro_id: str, cuda_version: str | None = None
) -> list[str]:
    """Get packages to remove for CUDA uninstallation.

    Args:
        distro_id: Distribution ID
        cuda_version: Specific CUDA version to remove, or None for all

    Returns:
        List of package patterns to remove
    """
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        if cuda_version:
            return [f"cuda-{cuda_version}*", f"cuda-toolkit-{cuda_version}*"]
        return ["cuda-*", "cuda-toolkit-*", "nvidia-cuda*"]
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        if cuda_version:
            return [f"cuda-toolkit-{cuda_version}*", f"cuda-devel-{cuda_version}*"]
        return ["cuda-toolkit*", "cuda-runtime*", "cuda-devel*"]
    elif distro_id in ("arch", "manjaro"):
        if cuda_version:
            return [f"cuda-{cuda_version}*"]
        return ["cuda*"]
    elif distro_id in ("opensuse", "sles"):
        if cuda_version:
            return [f"cuda-{cuda_version}*"]
        return ["cuda*"]
    return []


def pin_cuda_to_major_version(
    distro_id: str,
    major_version: str,
    pkg_manager: "PackageManager",
) -> bool:
    """Pin CUDA packages to a major version.

    Uses branch-style locking: "12.*" allows 12.0, 12.1, 12.2, etc.

    Args:
        distro_id: Distribution ID
        major_version: Major version to pin (e.g., "12")
        pkg_manager: Package manager instance

    Returns:
        True if pinning succeeded
    """
    pattern = f"{major_version}.*"
    packages = _get_cuda_packages_for_pinning(distro_id, major_version)

    pinned = True
    for pkg in packages:
        try:
            if not pkg_manager.pin_version(pkg, pattern):
                logger.warning(f"Failed to pin {pkg} to {pattern}")
                pinned = False
            else:
                logger.info(f"Pinned {pkg} to {pattern}")
        except Exception as e:
            logger.warning(f"Error pinning {pkg}: {e}")
            pinned = False

    return pinned


def pin_cuda_to_exact_version(
    distro_id: str,
    cuda_version: str,
    pkg_manager: "PackageManager",
) -> bool:
    """Pin CUDA packages to exact version.

    Args:
        distro_id: Distribution ID
        cuda_version: Exact CUDA version (e.g., "11.8")
        pkg_manager: Package manager instance

    Returns:
        True if pinning succeeded
    """
    packages = get_cuda_packages_for_version(distro_id, cuda_version)

    pinned = True
    for pkg in packages:
        try:
            if not pkg_manager.pin_version(pkg, cuda_version):
                logger.warning(f"Failed to pin {pkg} to {cuda_version}")
                pinned = False
            else:
                logger.info(f"Pinned {pkg} to {cuda_version}")
        except Exception as e:
            logger.warning(f"Error pinning {pkg}: {e}")
            pinned = False

    return pinned


def _get_cuda_packages_for_pinning(distro_id: str, major_version: str) -> list[str]:
    """Get CUDA package names for pinning by major version.

    Args:
        distro_id: Distribution ID
        major_version: Major version (e.g., "12")

    Returns:
        List of package patterns for pinning
    """
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return [f"cuda-{major_version}*", f"cuda-toolkit-{major_version}*"]
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        return [f"cuda-toolkit-{major_version}*", f"cuda-runtime-{major_version}*"]
    elif distro_id in ("arch", "manjaro"):
        return [f"cuda-{major_version}*"]
    elif distro_id in ("opensuse", "sles"):
        return [f"cuda-{major_version}*"]
    return []


def check_cuda_driver_compatibility(
    cuda_version: str, driver_version: str
) -> tuple[bool, str]:
    """Check if CUDA version is compatible with driver version.

    CUDA Toolkit requires minimum driver version:
    - CUDA 12.x: Driver 525+
    - CUDA 11.x: Driver 450+

    Args:
        cuda_version: CUDA version (e.g., "12.2")
        driver_version: Driver version (e.g., "535.154.05")

    Returns:
        Tuple of (is_compatible, message)
    """
    try:
        cuda_major = int(cuda_version.split(".")[0])
        driver_major = int(driver_version.split(".")[0])

        if cuda_major >= 12:
            if driver_major < 525:
                return (
                    False,
                    f"CUDA {cuda_version} requires driver 525+, found {driver_version}",
                )
        elif cuda_major == 11:
            if driver_major < 450:
                return (
                    False,
                    f"CUDA {cuda_version} requires driver 450+, found {driver_version}",
                )
        elif cuda_major <= 10:
            if driver_major < 410:
                return (
                    False,
                    f"CUDA {cuda_version} requires driver 410+, found {driver_version}",
                )

        return True, f"CUDA {cuda_version} is compatible with driver {driver_version}"
    except (ValueError, IndexError):
        return True, "Unable to validate compatibility"


# Type import for PackageManager (avoiding circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nvidia_inst.distro.package_manager import PackageManager
