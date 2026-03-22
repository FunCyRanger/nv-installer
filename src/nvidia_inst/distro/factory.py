"""Factory function to get the appropriate package manager.

This module provides tool-based package manager selection that works
with any distro using supported package management tools.
"""

from nvidia_inst.distro.apt import AptManager
from nvidia_inst.distro.dnf import DnfManager
from nvidia_inst.distro.package_manager import PackageManager
from nvidia_inst.distro.pacman import PacmanManager
from nvidia_inst.distro.tools import detect_package_tool, get_tool_family
from nvidia_inst.distro.zypper import ZypperManager
from nvidia_inst.utils.logger import get_logger

# Re-export distro detection functions for backward compatibility
from nvidia_inst.distro.detector import (
    is_arch,
    is_debian,
    is_fedora,
    is_opensuse,
    is_ubuntu,
)

logger = get_logger(__name__)


# Tool to manager class mapping
_TOOL_MANAGERS: dict[str, type[PackageManager]] = {
    "apt": AptManager,
    "apt-get": AptManager,
    "dnf": DnfManager,
    "dnf5": DnfManager,
    "yum": DnfManager,
    "pacman": PacmanManager,
    "pamac": PacmanManager,
    "paru": PacmanManager,
    "yay": PacmanManager,
    "trizen": PacmanManager,
    "zypper": ZypperManager,
}


def get_package_manager() -> PackageManager:
    """Get the appropriate package manager by detecting available tools.

    This function detects which package management tool is available
    on the system and returns the appropriate manager instance.
    Works with any distro using supported tools, not just known distros.

    Returns:
        PackageManager: Instance of the appropriate package manager.

    Raises:
        RuntimeError: If no supported package manager is found.
    """
    tool = detect_package_tool()
    if tool is None:
        raise RuntimeError(
            "No supported package manager found. "
            "Tried: apt, dnf, dnf5, pacman, pamac, paru, yay, zypper"
        )

    manager_class = _TOOL_MANAGERS.get(tool)
    if manager_class is None:
        # Fallback: try to use family-based selection
        family = get_tool_family(tool)
        family_manager = _get_manager_for_family(family)
        if family_manager is not None:
            logger.info(f"Using {family_manager.__name__} for unknown tool {tool}")
            return family_manager()
        raise RuntimeError(f"No manager implementation for tool: {tool}")

    logger.info(f"Using {manager_class.__name__} for tool {tool}")
    return manager_class()


def get_manager_for_tool(tool: str) -> PackageManager | None:
    """Get package manager for a specific tool.

    Args:
        tool: Package manager tool name (apt, dnf, pacman, etc.)

    Returns:
        PackageManager instance or None if tool not supported.
    """
    manager_class = _TOOL_MANAGERS.get(tool)
    if manager_class:
        return manager_class()
    return None


def _get_manager_for_family(family: str) -> type[PackageManager] | None:
    """Get manager class for a family.

    Args:
        family: Tool family (debian, fedora, arch, suse)

    Returns:
        Manager class or None.
    """
    family_managers: dict[str, type[PackageManager]] = {
        "debian": AptManager,
        "fedora": DnfManager,
        "arch": PacmanManager,
        "suse": ZypperManager,
    }
    return family_managers.get(family)


def is_tool_supported(tool: str) -> bool:
    """Check if a tool is supported.

    Args:
        tool: Package manager tool name

    Returns:
        True if the tool has a manager implementation.
    """
    return tool in _TOOL_MANAGERS


def get_supported_tools() -> list[str]:
    """Get list of supported package manager tools.

    Returns:
        List of supported tool names.
    """
    return list(_TOOL_MANAGERS.keys())
