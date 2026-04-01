"""Command generation for nvidia-inst CLI.

This module provides command generation functions for different
package managers. Uses the tool-based approach from tools.py.
"""

import shutil

from nvidia_inst.distro.tools import (
    get_install_command,
    get_remove_command,
    get_update_command,
)
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def detect_dnf_path() -> str:
    """Detect the correct dnf executable path (dnf5 vs dnf).

    Returns:
        Path to dnf executable (dnf5 or dnf)
    """
    # Try dnf5 first if available
    if shutil.which("dnf5"):
        return "dnf5"

    # Try dnf
    if shutil.which("dnf"):
        return "dnf"

    # Default to dnf
    return "dnf"


def sudo_path() -> str:
    """Get path to sudo.

    Returns:
        Path to sudo executable
    """
    return shutil.which("sudo") or "sudo"


def get_nouveau_remove_command(tool: str) -> list[str]:
    """Get command to remove nouveau.

    Args:
        tool: Package manager tool name

    Returns:
        List of command arguments
    """
    commands = {
        "apt": ["apt-get", "remove", "-y", "xserver-xorg-video-nouveau"],
        "apt-get": ["apt-get", "remove", "-y", "xserver-xorg-video-nouveau"],
        "dnf": ["dnf", "remove", "-y", "xorg-x11-drv-nouveau"],
        "dnf5": ["dnf5", "remove", "-y", "xorg-x11-drv-nouveau"],
        "pacman": ["pacman", "-Rns", "--noconfirm", "xf86-video-nouveau"],
        "zypper": ["zypper", "remove", "-y", "xf86-video-nouveau"],
    }
    return commands.get(tool, ["apt-get", "remove", "-y", "xserver-xorg-video-nouveau"])


def get_initramfs_command(tool: str) -> list[str]:
    """Get initramfs rebuild command for a tool.

    Args:
        tool: Package manager tool name

    Returns:
        List of command arguments for rebuilding initramfs
    """
    commands = {
        "apt": ["update-initramfs", "-u"],
        "apt-get": ["update-initramfs", "-u"],
        "dnf": ["dracut", "-f", "--regenerate-all"],
        "dnf5": ["dracut", "-f", "--regenerate-all"],
        "yum": ["dracut", "-f", "--regenerate-all"],
        "pacman": ["mkinitcpio", "-P"],
        "pamac": ["mkinitcpio", "-P"],
        "paru": ["mkinitcpio", "-P"],
        "yay": ["mkinitcpio", "-P"],
        "zypper": ["dracut", "-f", "--regenerate-all"],
    }
    return commands.get(tool, ["update-initramfs", "-u"])


def get_driver_lock_command(tool: str, branch: str) -> list[str]:
    """Get command to lock driver to branch.

    Args:
        tool: Package manager tool name
        branch: Driver branch (e.g., '580')

    Returns:
        List of command arguments
    """
    if tool in ("apt", "apt-get"):
        return [
            "bash",
            "-c",
            f"echo 'Pin: version {branch}.*' | sudo tee -a /etc/apt/preferences.d/nvidia",
        ]
    elif tool in ("dnf", "dnf5"):
        dnf_path = detect_dnf_path()
        package = "akmod-nvidia"
        return [sudo_path(), dnf_path, "versionlock", "add", package]
    elif tool in ("pacman", "pamac", "paru", "yay"):
        return [sudo_path(), "pacman", "-D", "--lock", f"nvidia-{branch}xx"]
    elif tool == "zypper":
        return [sudo_path(), "zypper", "addlock", "x11-video-nvidiaG05"]
    return ["bash", "-c", f"# Lock to branch {branch}"]


def get_driver_unlock_command(tool: str, branch: str) -> list[str] | None:
    """Get command to unlock driver from branch.

    Args:
        tool: Package manager tool name
        branch: Driver branch (e.g., '580')

    Returns:
        List of command arguments or None if not supported
    """
    if tool in ("dnf", "dnf5"):
        dnf_path = detect_dnf_path()
        package = "akmod-nvidia"
        return [sudo_path(), dnf_path, "versionlock", "delete", package]
    return None


def get_cuda_lock_command(tool: str, major: str) -> list[str]:
    """Get command to lock CUDA to major version.

    Args:
        tool: Package manager tool name
        major: CUDA major version (e.g., '12')

    Returns:
        List of command arguments
    """
    if tool in ("apt", "apt-get"):
        return [
            "bash",
            "-c",
            f"echo 'Pin: version {major}.*' | sudo tee -a /etc/apt/preferences.d/cuda",
        ]
    elif tool in ("dnf", "dnf5"):
        dnf_path = detect_dnf_path()
        package = "cuda-toolkit"
        return [sudo_path(), dnf_path, "versionlock", "add", package]
    elif tool in ("pacman", "pamac", "paru", "yay"):
        return [sudo_path(), "pacman", "-D", "--lock", f"cuda-{major}*"]
    elif tool == "zypper":
        return [sudo_path(), "zypper", "addlock", f"cuda-toolkit-{major}-*"]
    return ["bash", "-c", f"# Lock CUDA to {major}.*"]


def get_cuda_unlock_command(tool: str, major: str) -> list[str] | None:
    """Get command to unlock CUDA from major version.

    Args:
        tool: Package manager tool name
        major: CUDA major version (e.g., '12')

    Returns:
        List of command arguments or None if not supported
    """
    if tool in ("dnf", "dnf5"):
        dnf_path = detect_dnf_path()
        pattern = f"cuda-toolkit-{major}-*"
        return [sudo_path(), dnf_path, "versionlock", "delete", pattern]
    return None


def format_install_command(tool: str, packages: list[str]) -> str:
    """Format install command for display.

    Args:
        tool: Package manager tool name
        packages: List of package names

    Returns:
        Formatted command string
    """
    pkg_str = " ".join(packages[:3])
    if len(packages) > 3:
        pkg_str += " ..."

    base_cmd = get_install_command(tool)
    return f"sudo {' '.join(base_cmd)} {pkg_str}"


def format_update_command(tool: str) -> str:
    """Format update command for display.

    Args:
        tool: Package manager tool name

    Returns:
        Formatted command string
    """
    base_cmd = get_update_command(tool)
    return f"sudo {' '.join(base_cmd)}"


def format_remove_command(tool: str) -> str:
    """Format remove command for display.

    Args:
        tool: Package manager tool name

    Returns:
        Formatted command string
    """
    base_cmd = get_remove_command(tool)
    return f"sudo {' '.join(base_cmd)}"
