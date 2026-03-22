"""Package name maps for different tools and distro families.

This module provides package name mappings that allow the installer to
work with any distro that uses the same package management tool, while
still supporting distro-specific package name variations.
"""

from nvidia_inst.distro.tools import PackageContext
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for package map: distro_id -> package names
PackageMap = dict[str, list[str]]


# =============================================================================
# Driver Packages
# =============================================================================

# NVIDIA driver packages by tool and distro family
DRIVER_PACKAGES: dict[str, PackageMap] = {
    "apt": {
        "ubuntu": ["nvidia-driver-{branch}", "nvidia-dkms-{branch}"],
        "debian": ["nvidia-driver", "nvidia-dkms"],
        "_default": ["nvidia-driver", "nvidia-dkms"],
    },
    "apt-get": {
        "_default": ["nvidia-driver", "nvidia-dkms"],
    },
    "dnf": {
        "fedora": ["akmod-nvidia", "xorg-x11-drv-nvidia"],
        "_default": ["akmod-nvidia", "xorg-x11-drv-nvidia"],
    },
    "dnf5": {
        "_default": ["akmod-nvidia", "xorg-x11-drv-nvidia"],
    },
    "yum": {
        "_default": ["akmod-nvidia", "xorg-x11-drv-nvidia"],
    },
    "pacman": {
        "arch": ["nvidia", "nvidia-utils"],
        "_default": ["nvidia", "nvidia-utils"],
    },
    "pamac": {
        "_default": ["nvidia", "nvidia-utils"],
    },
    "paru": {
        "_default": ["nvidia", "nvidia-utils"],
    },
    "yay": {
        "_default": ["nvidia", "nvidia-utils"],
    },
    "trizen": {
        "_default": ["nvidia", "nvidia-utils"],
    },
    "zypper": {
        "opensuse": ["x11-video-nvidiaG05", "nvidia-computeG05"],
        "_default": ["x11-video-nvidiaG05", "nvidia-computeG05"],
    },
}

# Open kernel driver packages (nvidia-open)
DRIVER_OPEN_PACKAGES: dict[str, PackageMap] = {
    "apt": {
        "ubuntu": ["nvidia-driver-{branch}", "nvidia-dkms-{branch}-open"],
        "_default": ["nvidia-driver-open", "nvidia-dkms-open"],
    },
    "dnf": {
        "fedora": ["akmod-nvidia", "xorg-x11-drv-nvidia-open"],
        "_default": ["akmod-nvidia", "xorg-x11-drv-nvidia-open"],
    },
    "pacman": {
        "arch": ["nvidia-open", "nvidia-utils"],
        "_default": ["nvidia-open", "nvidia-utils"],
    },
    "zypper": {
        "_default": ["x11-video-nvidiaG05", "nvidia-computeG05"],
    },
}


# =============================================================================
# CUDA Packages
# =============================================================================

# CUDA toolkit packages by tool and distro family
# Uses meta-packages by default - package manager resolves to latest compatible
CUDA_PACKAGES: dict[str, PackageMap] = {
    "apt": {
        "_default": ["cuda-toolkit"],
    },
    "apt-get": {
        "_default": ["cuda-toolkit"],
    },
    "dnf": {
        "_default": ["cuda-toolkit"],
    },
    "dnf5": {
        "_default": ["cuda-toolkit"],
    },
    "yum": {
        "_default": ["cuda-toolkit"],
    },
    "pacman": {
        "_default": ["cuda"],
    },
    "pamac": {
        "_default": ["cuda"],
    },
    "paru": {
        "_default": ["cuda"],
    },
    "yay": {
        "_default": ["cuda"],
    },
    "zypper": {
        "_default": ["cuda"],
    },
}

# CUDA major version lock packages (for versionlock/pinning)
# Fedora packages use hyphen format: cuda-toolkit-{major}-{minor}
# e.g., cuda-toolkit-13-2 (not cuda-toolkit-13.2)
CUDA_MAJOR_PACKAGES: dict[str, PackageMap] = {
    "apt": {
        "_default": ["cuda-toolkit-{major}*"],
    },
    "apt-get": {
        "_default": ["cuda-toolkit-{major}*"],
    },
    "dnf": {
        "_default": ["cuda-toolkit-{major}-*"],  # Hyphen pattern for Fedora
    },
    "dnf5": {
        "_default": ["cuda-toolkit-{major}-*"],  # Hyphen pattern for Fedora
    },
    "yum": {
        "_default": ["cuda-toolkit-{major}-*"],  # Hyphen pattern for Fedora
    },
    "pacman": {
        "_default": ["cuda-{major}*"],
    },
    "pamac": {
        "_default": ["cuda-{major}*"],
    },
    "paru": {
        "_default": ["cuda-{major}*"],
    },
    "yay": {
        "_default": ["cuda-{major}*"],
    },
    "zypper": {
        "_default": ["cuda-{major}*"],
    },
}


# =============================================================================
# Nouveau Removal Packages
# =============================================================================

# Packages to remove when blacklisting nouveau
NOUVEAU_REMOVE_PACKAGES: dict[str, PackageMap] = {
    "apt": {
        "_default": ["xserver-xorg-video-nouveau", "libdrm-nouveau2"],
    },
    "dnf": {
        "_default": ["xorg-x11-drv-nouveau"],
    },
    "pacman": {
        "_default": ["xf86-video-nouveau"],
    },
    "zypper": {
        "_default": ["xf86-video-nouveau"],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def _get_package_from_map(
    pkg_map: PackageMap,
    distro_id: str,
    **kwargs,
) -> list[str]:
    """Get packages from a package map.

    Args:
        pkg_map: Package map (distro_id -> package names)
        distro_id: Distro ID to look up
        **kwargs: Format arguments for package names (version, branch, major)

    Returns:
        List of package names.
    """
    # Try exact distro match first
    packages = pkg_map.get(distro_id)

    # Fall back to default
    if packages is None:
        packages = pkg_map.get("_default", [])

    if not packages:
        logger.warning(f"No packages found for distro {distro_id}")
        return []

    # Format package names with provided kwargs
    result = []
    for pkg in packages:
        try:
            result.append(pkg.format(**kwargs))
        except KeyError:
            # Missing format argument, use as-is
            result.append(pkg)

    return result


def get_driver_packages(ctx: PackageContext, branch: str | None = None) -> list[str]:
    """Get driver package names for the current context.

    Args:
        ctx: Package context
        branch: Driver branch (e.g., "535") for distros that need it

    Returns:
        List of driver package names.
    """
    pkg_map = DRIVER_PACKAGES.get(ctx.tool, {})
    kwargs = {"branch": branch} if branch else {}
    return _get_package_from_map(pkg_map, ctx.distro_id, **kwargs)


def get_driver_open_packages(
    ctx: PackageContext, branch: str | None = None
) -> list[str]:
    """Get open kernel driver package names.

    Args:
        ctx: Package context
        branch: Driver branch

    Returns:
        List of package names.
    """
    pkg_map = DRIVER_OPEN_PACKAGES.get(ctx.tool, {})
    kwargs = {"branch": branch} if branch else {}
    return _get_package_from_map(pkg_map, ctx.distro_id, **kwargs)


def get_cuda_packages(ctx: PackageContext, version: str) -> list[str]:
    """Get CUDA package names for a specific version.

    Args:
        ctx: Package context
        version: CUDA version (e.g., "12.2")

    Returns:
        List of CUDA package names.
    """
    pkg_map = CUDA_PACKAGES.get(ctx.tool, {})
    return _get_package_from_map(pkg_map, ctx.distro_id, version=version)


def get_cuda_major_packages(ctx: PackageContext, major: str) -> list[str]:
    """Get CUDA package patterns for major version pinning.

    Args:
        ctx: Package context
        major: Major version (e.g., "12")

    Returns:
        List of package patterns for pinning.
    """
    pkg_map = CUDA_MAJOR_PACKAGES.get(ctx.tool, {})
    return _get_package_from_map(pkg_map, ctx.distro_id, major=major)


def get_nouveau_remove_packages(ctx: PackageContext) -> list[str]:
    """Get packages to remove when disabling nouveau.

    Args:
        ctx: Package context

    Returns:
        List of package names.
    """
    pkg_map = NOUVEAU_REMOVE_PACKAGES.get(ctx.tool, {})
    return _get_package_from_map(pkg_map, ctx.distro_id)


def format_package_name(template: str, **kwargs) -> str:
    """Format a package name template.

    Args:
        template: Package name template with {placeholders}
        **kwargs: Format arguments

    Returns:
        Formatted package name.
    """
    try:
        return template.format(**kwargs)
    except KeyError:
        return template
