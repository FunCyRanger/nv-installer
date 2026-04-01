"""Tool-based package manager detection.

This module provides detection of available package management tools
(apt, dnf, pacman, pamac, paru, yay, zypper, dnf5) independent of
specific distro names. This allows the installer to support new
derivatives automatically.
"""

import shutil
from dataclasses import dataclass

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PackageContext:
    """Unified context for package operations.

    Attributes:
        tool: Package manager tool name (apt, dnf, pacman, etc.)
        distro_id: Original distro ID from /etc/os-release
        distro_family: Family for package name variations (debian, fedora, arch, suse)
        version_id: Distro version for repository URLs
    """

    tool: str
    distro_id: str
    distro_family: str
    version_id: str


# Tool definitions: binary name -> family
TOOL_FAMILIES: dict[str, str] = {
    "apt": "debian",
    "apt-get": "debian",
    "dnf": "fedora",
    "dnf5": "fedora",
    "yum": "fedora",
    "pacman": "arch",
    "pamac": "arch",
    "paru": "arch",
    "yay": "arch",
    "trizen": "arch",
    "yay-bin": "arch",
    "yaourt": "arch",
    "zypper": "suse",
    "dnfdragora": "fedora",
}

# Priority order - prefer modern tools over legacy
TOOL_PRIORITY: list[str] = [
    "apt",
    "dnf5",
    "dnf",
    "pacman",
    "pamac",
    "paru",
    "yay",
    "zypper",
]

# Distro ID to family mapping for package name variations
DISTRO_FAMILIES: dict[str, str] = {
    # Debian family
    "ubuntu": "debian",
    "debian": "debian",
    "linuxmint": "debian",
    "pop": "debian",
    "zorin": "debian",
    "elementary": "debian",
    "kali": "debian",
    "parrot": "debian",
    "mx": "debian",
    "antiX": "debian",
    "devuan": "debian",
    "pureos": "debian",
    # Fedora family
    "fedora": "fedora",
    "rhel": "fedora",
    "centos": "fedora",
    "rocky": "fedora",
    "alma": "fedora",
    "nobara": "fedora",
    "ultramarine": "fedora",
    "boros": "fedora",
    # Arch family
    "arch": "arch",
    "manjaro": "arch",
    "endeavouros": "arch",
    "garuda": "arch",
    "cachyos": "arch",
    "arcolinux": "arch",
    "artix": "arch",
    "obarun": "arch",
    "hyperbola": "arch",
    "parabola": "arch",
    # SUSE family
    "opensuse": "suse",
    "sles": "suse",
    "opensuse-leap": "suse",
    "opensuse-tumbleweed": "suse",
    # Unknown - will use tool family
}

# Fallback family if neither tool nor distro is recognized
DEFAULT_FAMILY = "unknown"


def detect_package_tool() -> str | None:
    """Detect available package management tool.

    Checks for tools in priority order, returning the first found.

    Returns:
        Tool name (apt, dnf, pacman, etc.) or None if no supported tool found.
    """
    for tool in TOOL_PRIORITY:
        if shutil.which(tool):
            logger.info(f"Detected package tool: {tool}")
            return tool

    # Fallback: check any known tool
    for tool in TOOL_FAMILIES:
        if shutil.which(tool):
            logger.info(f"Detected package tool (fallback): {tool}")
            return tool

    logger.warning("No supported package manager tool found")
    return None


def get_tool_family(tool: str) -> str:
    """Get the family for a package tool.

    Args:
        tool: Tool name (apt, dnf, pacman, etc.)

    Returns:
        Family name (debian, fedora, arch, suse) or 'unknown'.
    """
    return TOOL_FAMILIES.get(tool, "unknown")


def get_distro_family(distro_id: str) -> str:
    """Get the family for a distro ID.

    Args:
        distro_id: Distro ID from /etc/os-release

    Returns:
        Family name (debian, fedora, arch, suse) or distro_id if unknown.
    """
    return DISTRO_FAMILIES.get(distro_id, distro_id)


def detect_package_context(
    distro_id: str = "unknown",
    version_id: str = "unknown",
) -> PackageContext:
    """Detect package context with tool and family information.

    Args:
        distro_id: Distro ID from /etc/os-release
        version_id: Distro version from /etc/os-release

    Returns:
        PackageContext with detected tool and family information.

    Raises:
        RuntimeError: If no supported package manager is found.
    """
    tool = detect_package_tool()
    if tool is None:
        raise RuntimeError(
            "No supported package manager found (tried: apt, dnf, pacman, zypper)"
        )

    # Get family from tool first, then check if distro provides better info
    tool_family = get_tool_family(tool)
    distro_family = get_distro_family(distro_id)

    # Use distro family if it's more specific (not just the distro_id echoed back)
    if distro_family != distro_id and distro_family != "unknown":
        family = distro_family
    else:
        family = tool_family

    logger.debug(f"Package context: tool={tool}, family={family}, distro={distro_id}")

    return PackageContext(
        tool=tool,
        distro_id=distro_id,
        distro_family=family,
        version_id=version_id,
    )


def get_install_command(tool: str, upgrade: bool = False) -> list[str]:
    """Get the install command for a tool.

    Args:
        tool: Package manager tool name
        upgrade: If True, include upgrade flag

    Returns:
        List of command arguments (without package names).
    """
    commands: dict[str, list[str]] = {
        "apt": ["apt-get", "install", "-y"],
        "apt-get": ["apt-get", "install", "-y"],
        "dnf": ["dnf", "install", "-y"],
        "dnf5": ["dnf5", "install", "-y"],
        "yum": ["yum", "install", "-y"],
        "pacman": ["pacman", "-S", "--noconfirm"],
        "pamac": ["pamac", "install", "--no-confirm"],
        "paru": ["paru", "-S", "--noconfirm"],
        "yay": ["yay", "-S", "--noconfirm"],
        "trizen": ["trizen", "-S", "--noconfirm"],
        "zypper": ["zypper", "install", "-y"],
    }

    cmd = commands.get(tool)
    if cmd is None:
        raise ValueError(f"Unknown package tool: {tool}")

    if upgrade and tool in ("apt", "apt-get"):
        cmd = cmd[:1] + ["upgrade", "-y"]  # apt upgrade instead of install
    elif upgrade and tool in ("dnf", "dnf5", "yum"):
        cmd = cmd[:1] + ["upgrade", "-y"]
    elif upgrade and tool in ("zypper",):
        cmd = cmd[:1] + ["update", "-y"]
    elif upgrade and tool in ("pacman",):
        cmd = ["pacman", "-Syu", "--noconfirm"]

    return cmd


def get_remove_command(tool: str) -> list[str]:
    """Get the remove command for a tool.

    Args:
        tool: Package manager tool name

    Returns:
        List of command arguments (without package names).
    """
    commands: dict[str, list[str]] = {
        "apt": ["apt-get", "remove", "-y", "--purge"],
        "apt-get": ["apt-get", "remove", "-y", "--purge"],
        "dnf": ["dnf", "remove", "-y"],
        "dnf5": ["dnf5", "remove", "-y"],
        "yum": ["yum", "remove", "-y"],
        "pacman": ["pacman", "-Rns", "--noconfirm"],
        "pamac": ["pamac", "remove", "--no-confirm"],
        "paru": ["paru", "-Rns", "--noconfirm"],
        "yay": ["yay", "-Rns", "--noconfirm"],
        "trizen": ["trizen", "-Rns", "--noconfirm"],
        "zypper": ["zypper", "remove", "-y"],
    }

    cmd = commands.get(tool)
    if cmd is None:
        raise ValueError(f"Unknown package tool: {tool}")

    return cmd


def get_update_command(tool: str) -> list[str]:
    """Get the update/refresh command for a tool.

    Args:
        tool: Package manager tool name

    Returns:
        List of command arguments.
    """
    commands: dict[str, list[str]] = {
        "apt": ["apt-get", "update"],
        "apt-get": ["apt-get", "update"],
        "dnf": ["dnf", "makecache"],
        "dnf5": ["dnf5", "makecache"],
        "yum": ["yum", "makecache"],
        "pacman": ["pacman", "-Sy"],
        "pamac": ["pamac", "update", "--force-refresh"],
        "paru": ["paru", "-Sy"],
        "yay": ["yay", "-Sy"],
        "trizen": ["trizen", "-Sy"],
        "zypper": ["zypper", "refresh"],
    }

    cmd = commands.get(tool)
    if cmd is None:
        raise ValueError(f"Unknown package tool: {tool}")

    return cmd


def is_aur_tool(tool: str) -> bool:
    """Check if a tool is an AUR helper.

    Args:
        tool: Package manager tool name

    Returns:
        True if the tool is an AUR helper.
    """
    return tool in ("paru", "yay", "trizen", "yaourt")


def is_gui_tool(tool: str) -> bool:
    """Check if a tool is a GUI package manager.

    Args:
        tool: Package manager tool name

    Returns:
        True if the tool is a GUI manager.
    """
    return tool in ("pamac", "dnfdragora")


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
