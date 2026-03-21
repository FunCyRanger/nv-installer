"""Revert from proprietary Nvidia driver to Nouveau."""

import subprocess
from dataclasses import dataclass

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RevertResult:
    """Result of reverting to Nouveau."""

    success: bool
    packages_removed: list[str]
    versionlock_removed: list[str]
    apt_preferences_removed: list[str]
    errors: list[str]
    message: str


def revert_to_nouveau(distro_id: str) -> RevertResult:
    """Revert from proprietary Nvidia driver to Nouveau open-source driver.

    This function:
    1. Removes proprietary Nvidia driver packages
    2. Removes versionlock entries (Fedora) and apt preferences (Ubuntu)
    3. Removes Nouveau blacklist if present
    4. Rebuilds initramfs
    5. Provides instructions for reboot

    Args:
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora', 'arch').

    Returns:
        RevertResult with success status and details.
    """
    from nvidia_inst.utils.permissions import is_root

    result = RevertResult(
        success=False,
        packages_removed=[],
        versionlock_removed=[],
        apt_preferences_removed=[],
        errors=[],
        message="",
    )

    if not is_root():
        result.errors.append("Root privileges required")
        result.message = "Root privileges required. Run with sudo."
        return result

    if distro_id not in (
        "ubuntu",
        "debian",
        "linuxmint",
        "pop",
        "fedora",
        "rhel",
        "centos",
        "rocky",
        "alma",
        "arch",
        "manjaro",
        "opensuse",
        "sles",
    ):
        result.errors.append(f"Unsupported distribution: {distro_id}")
        result.message = f"Unsupported distribution: {distro_id}"
        return result

    _cleanup_dkms(distro_id)

    installed_packages = _query_installed_nvidia_packages(distro_id)
    if installed_packages:
        logger.info(f"Found installed Nvidia packages: {installed_packages}")
        packages = installed_packages
    else:
        packages = _get_packages_to_remove(distro_id)

    if not packages:
        result.errors.append("No Nvidia packages found to remove")
        result.message = "No Nvidia packages found"
        return result

    removed = _remove_packages(distro_id, packages)
    result.packages_removed = removed

    if not removed:
        result.errors.append("Failed to remove packages")
        result.message = "Failed to remove Nvidia packages"
        return result

    if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        vl_removed = _remove_versionlock_entries(packages)
        result.versionlock_removed = vl_removed

    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        apt_removed = _remove_apt_preferences(packages)
        result.apt_preferences_removed = apt_removed

    _remove_blacklist()

    initramfs_result = _rebuild_initramfs(distro_id)
    if not initramfs_result:
        result.errors.append("Failed to rebuild initramfs")

    result.success = len(result.errors) == 0
    result.message = _build_message(result, distro_id)

    return result


def _query_installed_nvidia_packages(distro_id: str) -> list[str]:
    """Query actually installed NVIDIA packages on the system.

    Args:
        distro_id: Distribution ID.

    Returns:
        List of installed NVIDIA package names.
    """
    installed = []
    patterns = ["nvidia", "xorg-x11-drv-nvidia", "x11-video-nvidia"]
    exclude = [
        "nvidia-gpu-firmware",
        "akmods",
        "nvidia-modprobe",
        "nvidia-settings",
    ]

    try:
        if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
            result = subprocess.run(
                ["dpkg", "-l"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line_lower = line.lower()
                    if any(p in line_lower for p in patterns):
                        parts = line.split()
                        if len(parts) >= 2 and parts[0] == "ii":
                            pkg_full = parts[1]
                            idx = 0
                            for i, c in enumerate(pkg_full):
                                if c.isdigit():
                                    idx = i
                                    break
                            pkg_name = pkg_full[:idx].rstrip("-")
                            if (
                                pkg_name
                                and pkg_name not in installed
                                and pkg_name not in exclude
                            ):
                                installed.append(pkg_name)

        elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
            result = subprocess.run(
                ["rpm", "-qa", "--queryformat", "%{NAME}\n"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line_lower = line.lower()
                    if any(p in line_lower for p in patterns) and line not in installed:
                        pkg_name = line.strip()
                        if pkg_name not in exclude:
                            installed.append(pkg_name)

        elif distro_id in ("arch", "manjaro"):
            result = subprocess.run(
                ["pacman", "-Qq"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line_lower = line.lower()
                    if (
                        any(p in line_lower for p in ["nvidia", "akmod"])
                        and line not in installed
                    ):
                        installed.append(line)

        elif distro_id in ("opensuse", "sles"):
            result = subprocess.run(
                ["rpm", "-qa", "--queryformat", "%{NAME}\n"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line_lower = line.lower()
                    if (
                        any(p in line_lower for p in ["nvidia", "x11-video"])
                        and line not in installed
                    ):
                        pkg_name = line.strip()
                        if pkg_name not in exclude:
                            installed.append(pkg_name)

    except Exception as e:
        logger.warning(f"Failed to query installed packages: {e}")

    return installed


def _cleanup_dkms(distro_id: str) -> None:
    """Clean up DKMS modules before package removal.

    Args:
        distro_id: Distribution ID.
    """
    try:
        if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
            subprocess.run(
                ["dkms", "status", "-m", "nvidia"],
                capture_output=True,
                timeout=30,
            )
            modules = subprocess.run(
                ["ls", "/var/lib/dkms/"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if modules.returncode == 0:
                for line in modules.stdout.splitlines():
                    if "nvidia" in line.lower():
                        parts = line.split("/")
                        if len(parts) >= 1:
                            module = parts[0]
                            logger.info(f"Removing DKMS module: {module}")
                            subprocess.run(
                                ["dkms", "remove", module, "--all"],
                                capture_output=True,
                                timeout=60,
                            )
    except Exception as e:
        logger.warning(f"DKMS cleanup warning: {e}")


def _get_packages_to_remove(distro_id: str) -> list[str]:
    """Get list of Nvidia packages to remove for distribution."""
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return [
            "nvidia-driver-*",
            "nvidia-dkms-*",
            "nvidia-kernel-common-*",
            "nvidia-kernel-source-*",
            "nvidia-settings",
            "nvidia-utils-*",
            "libnvidia-*",
            "xserver-xorg-video-nvidia",
        ]

    if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        return [
            "akmod-nvidia",
            "xorg-x11-drv-nvidia",
            "xorg-x11-drv-nvidia-cuda",
            "xorg-x11-drv-nvidia-drm",
            "xorg-x11-drv-nvidia-kmodsrc",
            "nvidia-persistenced",
            "nvidia-settings",
            "libnvidia-persistenced",
            "nvidia-driver-NVML",
            "nvidia-driver-cuda",
            "nvidia-driver-cuda-libs",
        ]

    if distro_id in ("arch", "manjaro"):
        return [
            "nvidia",
            "nvidia-open",
            "nvidia-580xx-dkms",
            "nvidia-470xx-dkms",
            "nvidia-utils",
            "nvidia-settings",
            "lib32-nvidia-utils",
            "lib32-nvidia-580xx-utils",
            "lib32-nvidia-470xx-utils",
        ]

    if distro_id in ("opensuse", "sles"):
        return [
            "x11-video-nvidiaG05",
            "x11-video-nvidiaG04",
            "nvidia-computeG05",
            "nvidia-computeG04",
            "nvidia-gfxG05-kmp-default",
            "nvidia-gfxG04-kmp-default",
        ]

    return []


def _remove_packages(distro_id: str, packages: list[str]) -> list[str]:
    """Remove packages using the distribution's package manager."""
    removed = []

    for pkg_pattern in packages:
        try:
            if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
                cmd = ["apt-get", "remove", "--purge", "-y", pkg_pattern]
            elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
                cmd = ["dnf", "remove", "-y", pkg_pattern]
            elif distro_id in ("arch", "manjaro"):
                cmd = ["pacman", "-Rns", "--noconfirm", pkg_pattern]
            elif distro_id in ("opensuse", "sles"):
                cmd = ["zypper", "remove", "-y", "--", pkg_pattern]
            else:
                continue

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                logger.info(f"Removed package: {pkg_pattern}")
                removed.append(pkg_pattern)
            elif result.returncode not in (1, 2):
                logger.warning(f"Failed to remove {pkg_pattern}: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout removing package: {pkg_pattern}")
        except Exception as e:
            logger.warning(f"Error removing {pkg_pattern}: {e}")

    return removed


def _remove_versionlock_entries(packages: list[str]) -> list[str]:
    """Remove versionlock entries for packages on Fedora/RHEL."""
    removed = []
    for pkg_pattern in packages:
        try:
            result = subprocess.run(
                ["dnf", "versionlock", "delete", "--", pkg_pattern],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                removed.append(pkg_pattern)
                logger.info(f"Removed versionlock: {pkg_pattern}")
            elif "No matches" not in result.stderr:
                logger.warning(f"Versionlock removal warning: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to remove versionlock for {pkg_pattern}: {e}")
    return removed


def _remove_apt_preferences(packages: list[str]) -> list[str]:
    """Remove apt preferences files created for packages."""
    removed = []
    from pathlib import Path

    for pkg_pattern in packages:
        base_name = pkg_pattern.replace("-*", "").replace("*", "")
        pref_file = Path(f"/etc/apt/preferences.d/{base_name}")
        try:
            if pref_file.exists():
                pref_file.unlink()
                removed.append(str(pref_file))
                logger.info(f"Removed apt preferences: {pref_file}")
        except Exception as e:
            logger.warning(f"Failed to remove {pref_file}: {e}")
    return removed


def _remove_blacklist() -> bool:
    """Remove Nouveau blacklist file if present."""
    blacklist = "/etc/modprobe.d/blacklist-nouveau.conf"

    try:
        from pathlib import Path

        if Path(blacklist).exists():
            Path(blacklist).unlink()
            logger.info("Removed Nouveau blacklist")
            return True
    except Exception as e:
        logger.warning(f"Failed to remove blacklist: {e}")

    return False


def _rebuild_initramfs(distro_id: str) -> bool:
    """Rebuild initramfs to enable Nouveau."""
    try:
        if distro_id in (
            "fedora",
            "rhel",
            "centos",
            "rocky",
            "alma",
            "opensuse",
            "sles",
        ):
            cmd = ["dracut", "-f", "--regenerate-all"]
        elif distro_id in ("arch", "manjaro"):
            cmd = ["mkinitcpio", "-P"]
        else:
            cmd = ["update-initramfs", "-u"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode == 0:
            logger.info("Initramfs rebuilt successfully")
            return True
        else:
            logger.warning(f"Initramfs rebuild had issues: {result.stderr}")
            return True

    except subprocess.TimeoutExpired:
        logger.error("Timeout during initramfs rebuild")
        return False
    except Exception as e:
        logger.warning(f"Failed to rebuild initramfs: {e}")
        return False


def _build_message(result: RevertResult, distro_id: str) -> str:
    """Build user-friendly result message."""
    parts = []

    if result.success:
        parts.append("Successfully reverted to Nouveau driver!")

    if result.packages_removed:
        parts.append(f"Removed {len(result.packages_removed)} package(s)")

    if result.versionlock_removed:
        parts.append(
            f"Removed {len(result.versionlock_removed)} versionlock entry(ies)"
        )

    if result.apt_preferences_removed:
        parts.append(f"Removed {len(result.apt_preferences_removed)} apt preference(s)")

    if result.errors:
        parts.append(f"{len(result.errors)} error(s) occurred")

    parts.append("\nPlease reboot your system to complete the transition.")
    parts.append("Nouveau (open-source) driver will be used after reboot.")

    return "\n".join(parts)


def check_nvidia_packages_installed(distro_id: str) -> list[str]:
    """Check which Nvidia packages are currently installed.

    Args:
        distro_id: Distribution ID.

    Returns:
        List of installed Nvidia package names.
    """
    installed = []

    patterns = {
        "ubuntu": ["dpkg", "-l"],
        "debian": ["dpkg", "-l"],
        "fedora": ["dnf", "list", "--installed"],
        "centos": ["dnf", "list", "--installed"],
        "arch": ["pacman", "-Qq"],
        "opensuse": ["rpm", "-qa"],
    }

    try:
        if distro_id in patterns:
            cmd = patterns[distro_id]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                nvidia_patterns = ["nvidia", "xorg-x11-drv-nvidia", "x11-video-nvidia"]
                for line in result.stdout.splitlines():
                    for pattern in nvidia_patterns:
                        if pattern in line.lower():
                            pkg_name = line.split()[0] if " " in line else line
                            if pkg_name not in installed:
                                installed.append(pkg_name)
                            break

    except Exception as e:
        logger.warning(f"Failed to check installed packages: {e}")

    return installed
