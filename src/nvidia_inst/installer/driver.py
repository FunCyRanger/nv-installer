"""Driver installation for different distributions."""

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass

from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class DriverInstallError(Exception):
    """Raised when driver installation fails."""

    pass


class NouveauLoadedError(Exception):
    """Raised when Nouveau kernel module is loaded."""

    pass


class SecureBootError(Exception):
    """Raised when Secure Boot is enabled."""

    pass


class KernelIncompatibleError(Exception):
    """Raised when kernel is incompatible with driver."""

    pass


@dataclass
class InstallResult:
    """Result of driver installation."""

    success: bool
    message: str
    packages_installed: list[str] | None = None

    def __post_init__(self) -> None:
        if self.packages_installed is None:
            self.packages_installed = []


class DistroInstaller(ABC):
    """Abstract base class for distro-specific installers."""

    @abstractmethod
    def get_driver_packages(self, driver_version: str | None = None) -> list[str]:
        """Get list of driver packages to install.

        Args:
            driver_version: Specific driver version (optional).

        Returns:
            List of package names.
        """
        ...

    @abstractmethod
    def get_cuda_packages(self, cuda_version: str | None = None) -> list[str]:
        """Get list of CUDA packages to install.

        Args:
            cuda_version: Specific CUDA version (optional).

        Returns:
            List of package names.
        """
        ...

    @abstractmethod
    def install(self, packages: list[str]) -> None:
        """Install packages.

        Args:
            packages: List of package names to install.
        """
        ...

    @abstractmethod
    def pre_install_check(self) -> bool:
        """Perform pre-installation checks.

        Returns:
            True if ready to install, False otherwise.
        """
        ...

    @abstractmethod
    def post_install(self) -> bool:
        """Perform post-installation steps.

        Returns:
            True if successful, False otherwise.
        """
        ...


def check_nouveau() -> bool:
    """Check if Nouveau kernel module is loaded.

    Returns:
        True if Nouveau is loaded, False otherwise.
    """
    try:
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "nouveau" in result.stdout
    except subprocess.CalledProcessError:
        return False


def check_secure_boot() -> bool:
    """Check if Secure Boot is enabled.

    Returns:
        True if Secure Boot is enabled, False otherwise.
    """
    try:
        result = subprocess.run(
            ["mokutil", "--sb-state"],
            capture_output=True,
            text=True,
        )
        return "enabled" in result.stdout.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_current_driver_type() -> str:
    """Detect current driver type.

    Returns:
        'proprietary' - NVIDIA proprietary driver loaded
        'nouveau'     - Nouveau open-source driver loaded
        'none'        - No NVIDIA driver detected
    """
    from nvidia_inst.installer.validation import is_nvidia_working

    if is_nvidia_working().is_working:
        return "proprietary"
    if check_nouveau():
        return "nouveau"
    return "none"


def disable_nouveau() -> bool:
    """Disable Nouveau kernel module.

    Returns:
        True if successful, False otherwise.
    """
    from nvidia_inst.distro.detector import detect_distro

    nouveau_blacklist = "/etc/modprobe.d/blacklist-nouveau.conf"

    if os.geteuid() != 0:
        logger.error("Root privileges required to disable Nouveau")
        return False

    try:
        with open(nouveau_blacklist, "w") as f:
            f.write("blacklist nouveau\n")
            f.write("options nouveau modeset=0\n")

        try:
            distro = detect_distro()
        except Exception:
            distro = None

        if distro and distro.id in (
            "fedora",
            "rhel",
            "centos",
            "rocky",
            "alma",
            "opensuse",
            "sles",
        ):
            cmd = ["dracut", "-f"]
        elif distro and distro.id in ("arch", "manjaro", "endeavouros"):
            cmd = ["mkinitcpio", "-P"]
        else:
            cmd = ["update-initramfs", "-u"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logger.warning(f"Initramfs rebuild command failed: {result.stderr}")
            if cmd[0] == "dracut":
                logger.info(
                    "Nouveau has been blacklisted. Run 'sudo dracut -f' manually if needed."
                )
            elif cmd[0] == "mkinitcpio":
                logger.info(
                    "Nouveau has been blacklisted. Run 'sudo mkinitcpio -P' manually if needed."
                )
            else:
                logger.info(
                    "Nouveau has been blacklisted. Run 'sudo update-initramfs -u' manually if needed."
                )
            return False

        logger.info("Nouveau disabled. Reboot required.")
        return True

    except OSError as e:
        logger.error(f"Failed to write blacklist: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to disable Nouveau: {e}")
        return False


def install_driver(
    installer: DistroInstaller,
    driver_version: str | None = None,
    with_cuda: bool = True,
    cuda_version: str | None = None,
    pkg_manager=None,
) -> InstallResult:
    """Install Nvidia driver.

    Args:
        installer: Distribution-specific installer.
        driver_version: Specific driver version (optional).
        with_cuda: Install CUDA packages.
        cuda_version: Specific CUDA version (optional).
        pkg_manager: Package manager for version pinning.

    Returns:
        InstallResult with success status and message.
    """
    if not installer.pre_install_check():
        return InstallResult(
            success=False,
            message="Pre-installation check failed",
        )

    driver_pkgs = installer.get_driver_packages(driver_version)

    try:
        logger.info(f"Installing driver packages: {driver_pkgs}")
        installer.install(driver_pkgs)

    except Exception as e:
        logger.error(f"Failed to install driver: {e}")
        return InstallResult(
            success=False,
            message=f"Installation failed: {e}",
        )

    if driver_version and pkg_manager:
        for pkg in driver_pkgs:
            if pkg_manager.pin_version(pkg, driver_version):
                logger.info(f"Pinned {pkg} to version {driver_version}")

    if with_cuda:
        cuda_pkgs = installer.get_cuda_packages(cuda_version)
        if cuda_pkgs:
            try:
                logger.info(f"Installing CUDA packages: {cuda_pkgs}")
                installer.install(cuda_pkgs)
            except Exception as e:
                logger.warning(f"CUDA installation failed: {e}")

    installer.post_install()

    return InstallResult(
        success=True,
        message="Driver installed successfully. Reboot required.",
        packages_installed=driver_pkgs,
    )


def get_compatible_driver_packages(
    distro_id: str,
    driver_range: DriverRange,
) -> list[str]:
    """Get list of compatible driver packages for distribution.

    Args:
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora').
        driver_range: Compatible driver version range.

    Returns:
        List of package names.
    """
    if distro_id in ("ubuntu", "linuxmint", "pop"):
        return _get_ubuntu_packages(driver_range)
    elif distro_id in ("fedora", "rhel", "centos"):
        return _get_fedora_packages(driver_range)
    elif distro_id in ("arch", "manjaro"):
        return _get_arch_packages(driver_range)
    elif distro_id in ("debian"):
        return _get_debian_packages(driver_range)
    elif distro_id in ("opensuse"):
        return _get_opensuse_packages(driver_range)

    return []


def _get_ubuntu_packages(driver_range: DriverRange) -> list[str]:
    """Get Ubuntu driver packages."""
    if driver_range.is_eol and driver_range.max_version:
        major = driver_range.max_version.split(".")[0]
        return [f"nvidia-driver-{major}", f"nvidia-dkms-{major}"]

    if driver_range.max_branch:
        branch = driver_range.max_branch
        if branch == "580":
            return ["nvidia-driver-580", "nvidia-dkms-580"]
        elif branch == "590":
            return ["nvidia-driver-535", "nvidia-dkms-535"]

    return ["nvidia-driver-535", "nvidia-dkms-535"]


def _get_fedora_packages(driver_range: DriverRange) -> list[str]:
    """Get Fedora driver packages.

    Note: Fedora uses base package names (akmod-nvidia, etc.) - the available
    version is determined by what's in the repo. Branch locking is handled via
    dnf versionlock after installation (shown in Step 4 of dry-run output).
    """
    packages = [
        "akmod-nvidia",
        "xorg-x11-drv-nvidia",
        "xorg-x11-drv-nvidia-cuda",
    ]
    return packages


def _get_arch_packages(driver_range: DriverRange) -> list[str]:
    """Get Arch Linux driver packages."""
    if driver_range.is_eol:
        return ["nvidia-470xx-dkms", "nvidia-470xx-utils"]

    if driver_range.max_branch == "580":
        return ["nvidia-580xx-dkms", "nvidia-580xx-utils"]

    if driver_range.max_branch == "590":
        return ["nvidia-open", "nvidia-utils"]

    return ["nvidia-open", "nvidia-utils"]


def _get_debian_packages(driver_range: DriverRange) -> list[str]:
    """Get Debian driver packages."""
    if driver_range.is_eol and driver_range.max_version:
        version_parts = driver_range.max_version.split(".")
        if len(version_parts) >= 2:
            major = version_parts[0]
            return [f"nvidia-driver-{major}"]

    return ["nvidia-driver"]


def _get_opensuse_packages(driver_range: DriverRange) -> list[str]:
    """Get openSUSE driver packages."""
    if driver_range.is_eol:
        return ["x11-video-nvidiaG04"]

    return ["x11-video-nvidiaG05", "nvidia-computeG05"]
