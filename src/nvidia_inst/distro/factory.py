"""Factory function to get the appropriate package manager."""

from nvidia_inst.distro.apt import AptManager
from nvidia_inst.distro.dnf import DnfManager
from nvidia_inst.distro.package_manager import PackageManager
from nvidia_inst.distro.pacman import PacmanManager
from nvidia_inst.distro.zypper import ZypperManager
from nvidia_inst.distro.detector import (
    detect_distro,
    is_ubuntu,
    is_fedora,
    is_arch,
    is_debian,
    is_opensuse,
)
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def get_package_manager() -> PackageManager:
    """Get the appropriate package manager for the current distribution.

    Returns:
        PackageManager: Instance of the appropriate package manager.

    Raises:
        ValueError: If no supported package manager is found.
    """
    if is_ubuntu() or is_debian():
        logger.info("Using APT package manager")
        return AptManager()

    if is_fedora():
        logger.info("Using DNF package manager")
        return DnfManager()

    if is_arch():
        logger.info("Using Pacman package manager")
        return PacmanManager()

    if is_opensuse():
        logger.info("Using Zypper package manager")
        return ZypperManager()

    raise ValueError("Unsupported distribution")
