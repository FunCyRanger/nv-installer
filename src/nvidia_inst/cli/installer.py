"""Installation orchestration for nvidia-inst CLI.

This module provides installation orchestration functions for
driver and CUDA installation.
"""

import subprocess
from dataclasses import dataclass

from nvidia_inst.cli.commands import get_initramfs_command
from nvidia_inst.distro.tools import (
    PackageContext,
    get_install_command,
    get_remove_command,
)
from nvidia_inst.utils.logger import get_logger
from nvidia_inst.utils.permissions import is_root

logger = get_logger(__name__)


@dataclass
class InstallResult:
    """Result of installation operation."""

    success: bool
    message: str
    packages_installed: list[str] | None = None

    def __post_init__(self) -> None:
        if self.packages_installed is None:
            self.packages_installed = []


def get_packages_to_remove(tool: str) -> list[str]:
    """Get list of NVIDIA packages to remove based on tool.

    Args:
        tool: Package manager tool name

    Returns:
        List of package patterns to remove
    """
    packages = {
        "apt": [
            "nvidia-driver-*",
            "nvidia-dkms-*",
            "nvidia-kernel-common-*",
            "nvidia-kernel-source-*",
            "nvidia-settings",
            "nvidia-utils-*",
            "libnvidia-*",
            "xserver-xorg-video-nvidia",
        ],
        "apt-get": [
            "nvidia-driver-*",
            "nvidia-dkms-*",
            "nvidia-kernel-common-*",
            "nvidia-kernel-source-*",
            "nvidia-settings",
            "nvidia-utils-*",
            "libnvidia-*",
            "xserver-xorg-video-nvidia",
        ],
        "dnf": [
            "akmod-nvidia",
            "xorg-x11-drv-nvidia",
            "xorg-x11-drv-nvidia-cuda",
            "xorg-x11-drv-nvidia-drm",
            "xorg-x11-drv-nvidia-kmodsrc",
            "nvidia-persistenced",
            "nvidia-settings",
        ],
        "dnf5": [
            "akmod-nvidia",
            "xorg-x11-drv-nvidia",
            "xorg-x11-drv-nvidia-cuda",
            "xorg-x11-drv-nvidia-drm",
            "xorg-x11-drv-nvidia-kmodsrc",
            "nvidia-persistenced",
            "nvidia-settings",
        ],
        "pacman": [
            "nvidia",
            "nvidia-open",
            "nvidia-580xx-dkms",
            "nvidia-470xx-dkms",
            "nvidia-utils",
            "nvidia-settings",
            "lib32-nvidia-utils",
        ],
        "zypper": [
            "x11-video-nvidiaG05",
            "x11-video-nvidiaG04",
            "nvidia-computeG05",
            "nvidia-computeG04",
        ],
    }
    return packages.get(tool, [])


def remove_packages(tool: str, packages: list[str]) -> list[str]:
    """Remove packages using the package manager.

    Args:
        tool: Package manager tool name
        packages: List of package patterns to remove

    Returns:
        List of successfully removed packages
    """
    removed = []
    remove_cmd = get_remove_command(tool)

    for pkg_pattern in packages:
        try:
            cmd = remove_cmd + [pkg_pattern]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"Removed: {pkg_pattern}")
                removed.append(pkg_pattern)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Failed to remove {pkg_pattern}: {e}")

    return removed


def rebuild_initramfs(tool: str) -> bool:
    """Rebuild initramfs for the distribution.

    Args:
        tool: Package manager tool name

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = get_initramfs_command(tool)

        if not is_root():
            cmd = ["sudo"] + cmd

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            logger.warning(f"Initramfs rebuild failed: {result.stderr}")
            return False

        logger.info("Initramfs rebuilt successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Initramfs rebuild timed out")
        return False
    except Exception as e:
        logger.error(f"Initramfs rebuild error: {e}")
        return False


def install_driver_packages(tool: str, packages: list[str]) -> InstallResult:
    """Install driver packages.

    Args:
        tool: Package manager tool name
        packages: List of package names to install

    Returns:
        InstallResult with success status and message
    """
    if not packages:
        return InstallResult(success=True, message="No packages to install")

    try:
        install_cmd = get_install_command(tool)
        cmd = install_cmd + packages

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return InstallResult(
                success=True,
                message=f"Installed: {' '.join(packages)}",
                packages_installed=packages,
            )
        else:
            return InstallResult(
                success=False,
                message=f"Installation failed: {result.stderr}",
            )
    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="Installation timed out",
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"Installation error: {e}",
        )


def install_cuda_packages(tool: str, packages: list[str]) -> InstallResult:
    """Install CUDA packages.

    Args:
        tool: Package manager tool name
        packages: List of CUDA package names to install

    Returns:
        InstallResult with success status and message
    """
    if not packages:
        return InstallResult(success=True, message="No CUDA packages to install")

    try:
        install_cmd = get_install_command(tool)
        cmd = install_cmd + packages

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return InstallResult(
                success=True,
                message=f"Installed CUDA: {' '.join(packages)}",
                packages_installed=packages,
            )
        else:
            return InstallResult(
                success=False,
                message=f"CUDA installation failed: {result.stderr}",
            )
    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            message="CUDA installation timed out",
        )
    except Exception as e:
        return InstallResult(
            success=False,
            message=f"CUDA installation error: {e}",
        )


def prompt_reboot() -> None:
    """Prompt user to reboot."""
    print("\nPlease reboot your system for changes to take effect.")
    response = input("Reboot now? [y/N]: ")
    if response.lower() in ("y", "yes"):
        try:
            subprocess.run(["sudo", "reboot"])
        except Exception:
            print("Reboot command failed. Please reboot manually.")
