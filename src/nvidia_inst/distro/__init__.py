"""Distribution detection and package management.

This package provides:
- Distro detection via /etc/os-release
- Tool-based package manager detection (apt, dnf, pacman, etc.)
- Package name maps for cross-distro compatibility
"""

from nvidia_inst.distro.detector import DistroInfo, detect_distro
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.distro.tools import (
    PackageContext,
    detect_package_context,
    detect_package_tool,
    get_install_command,
    get_remove_command,
    get_update_command,
)

__all__ = [
    # Distro detection
    "DistroInfo",
    "detect_distro",
    # Package manager
    "get_package_manager",
    # Tool detection
    "PackageContext",
    "detect_package_context",
    "detect_package_tool",
    "get_install_command",
    "get_remove_command",
    "get_update_command",
]
