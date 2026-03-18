"""CUDA toolkit installation."""

from abc import ABC, abstractmethod
from typing import Optional

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class CUDAInstaller(ABC):
    """Abstract base class for CUDA installation."""

    @abstractmethod
    def get_cuda_packages(self, version: Optional[str] = None) -> list[str]:
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
    def get_installed_cuda_version(self) -> Optional[str]:
        """Get currently installed CUDA version.

        Returns:
            Version string if installed, None otherwise.
        """
        ...


class UbuntuCUDAInstaller(CUDAInstaller):
    """CUDA installer for Ubuntu/Debian."""

    def get_cuda_packages(self, version: Optional[str] = None) -> list[str]:
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

    def get_installed_cuda_version(self) -> Optional[str]:
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

    def get_cuda_packages(self, version: Optional[str] = None) -> list[str]:
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

    def get_installed_cuda_version(self) -> Optional[str]:
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

    def get_cuda_packages(self, version: Optional[str] = None) -> list[str]:
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

    def get_installed_cuda_version(self) -> Optional[str]:
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
