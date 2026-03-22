"""Driver installation for different distributions."""

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.utils.logger import get_logger

if TYPE_CHECKING:
    from nvidia_inst.gpu.detector import GPUInfo

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
    """Check if Nouveau kernel module is loaded or nouveau packages installed.

    Returns:
        True if Nouveau is detected, False otherwise.
    """
    try:
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "nouveau" in result.stdout:
            return True
    except subprocess.CalledProcessError:
        pass

    return _check_nouveau_packages()


def _check_nouveau_packages() -> bool:
    """Check for nouveau-related packages on the system.

    Returns:
        True if nouveau packages are installed, False otherwise.
    """
    nouveau_pkgs = {
        "ubuntu": ["xserver-xorg-video-nouveau"],
        "debian": ["xserver-xorg-video-nouveau"],
        "fedora": ["xorg-x11-drv-nouveau"],
        "centos": ["xorg-x11-drv-nouveau"],
        "rhel": ["xorg-x11-drv-nouveau"],
        "rocky": ["xorg-x11-drv-nouveau"],
        "alma": ["xorg-x11-drv-nouveau"],
        "arch": ["xf86-video-nouveau"],
        "manjaro": ["xf86-video-nouveau"],
        "opensuse": ["xf86-video-nouveau"],
        "sles": ["xf86-video-nouveau"],
    }

    from nvidia_inst.distro.detector import detect_distro

    distro = detect_distro()
    distro_id = distro.id
    packages = nouveau_pkgs.get(distro_id, [])

    if not packages:
        return False

    try:
        if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
            result = subprocess.run(
                ["dpkg", "-l"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == "ii":
                        return True
        elif distro_id in ("fedora", "centos", "rhel", "rocky", "alma"):
            result = subprocess.run(
                ["rpm", "-q"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
        elif distro_id in ("arch", "manjaro", "opensuse", "sles"):
            result = subprocess.run(
                ["pacman", "-Q"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
    except Exception:
        pass

    return False


def check_nvidia_open_installed() -> bool:
    """Check if NVIDIA Open (nvidia-open) kernel module is installed.

    Returns:
        True if nvidia-open is installed, False otherwise.
    """
    nvidia_open_pkgs = {
        "ubuntu": [
            "nvidia-driver-535-open",
            "nvidia-driver-550-open",
            "nvidia-driver-580-open",
        ],
        "debian": ["nvidia-open", "nvidia-open-kernel-dkms"],
        "linuxmint": [
            "nvidia-driver-535-open",
            "nvidia-driver-550-open",
            "nvidia-driver-580-open",
        ],
        "pop": [
            "nvidia-driver-535-open",
            "nvidia-driver-550-open",
            "nvidia-driver-580-open",
        ],
        "fedora": ["xorg-x11-drv-nvidia-open"],
        "centos": ["xorg-x11-drv-nvidia-open"],
        "rhel": ["xorg-x11-drv-nvidia-open"],
        "rocky": ["xorg-x11-drv-nvidia-open"],
        "alma": ["xorg-x11-drv-nvidia-open"],
        "arch": ["nvidia-open"],
        "manjaro": ["nvidia-open"],
        "opensuse": ["nvidia-open-driver-G06"],
        "sles": ["nvidia-open-driver-G06"],
    }

    from nvidia_inst.distro.detector import detect_distro

    distro = detect_distro()
    distro_id = distro.id
    packages = nvidia_open_pkgs.get(distro_id, [])

    if not packages:
        return False

    try:
        if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
            result = subprocess.run(
                ["dpkg", "-l"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == "ii":
                        return True
        elif distro_id in ("fedora", "centos", "rhel", "rocky", "alma"):
            result = subprocess.run(
                ["rpm", "-q"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
        elif distro_id in ("arch", "manjaro"):
            result = subprocess.run(
                ["pacman", "-Q"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
        elif distro_id in ("opensuse", "sles"):
            result = subprocess.run(
                ["rpm", "-q"] + packages,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
    except Exception:
        pass

    return False


def check_nvidia_open_available() -> bool:
    """Check if NVIDIA Open (nvidia-open) is available in repos.

    Returns:
        True if nvidia-open packages are available, False otherwise.
    """
    from nvidia_inst.distro.detector import detect_distro

    distro = detect_distro()
    distro_id = distro.id

    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return _check_apt_nvidia_open()
    elif distro_id in ("fedora", "centos", "rhel", "rocky", "alma"):
        return _check_dnf_nvidia_open()
    elif distro_id in ("arch", "manjaro"):
        return _check_pacman_nvidia_open()
    elif distro_id in ("opensuse", "sles"):
        return True

    return False


def _check_apt_nvidia_open() -> bool:
    """Check if nvidia-open packages are available in APT repos."""
    try:
        result = subprocess.run(
            ["apt-cache", "policy", "nvidia-driver-535-open"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return "nvidia-driver-535-open" in result.stdout
    except Exception:
        pass
    return False


def _check_dnf_nvidia_open() -> bool:
    """Check if nvidia-open packages are available in DNF repos."""
    try:
        result = subprocess.run(
            ["dnf", "search", "xorg-x11-drv-nvidia-open"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return "xorg-x11-drv-nvidia-open" in result.stdout
    except Exception:
        pass
    return False


def _check_pacman_nvidia_open() -> bool:
    """Check if nvidia-open is available in pacman repos."""
    try:
        result = subprocess.run(
            ["pacman", "-Ss", "nvidia-open"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return "nvidia-open" in result.stdout
    except Exception:
        pass
    return False


def check_nonfree_available() -> bool:
    """Check if proprietary (nonfree) driver is available in repos.

    Returns:
        True if nonfree driver packages are available, False otherwise.
    """
    from nvidia_inst.distro.detector import detect_distro

    distro = detect_distro()
    distro_id = distro.id

    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return _check_apt_nonfree()
    elif distro_id in ("fedora", "centos", "rhel", "rocky", "alma"):
        return _check_dnf_nonfree()
    elif distro_id in ("arch", "manjaro"):
        return _check_pacman_nvidia()
    elif distro_id in ("opensuse", "sles"):
        return True

    return False


def _check_apt_nonfree() -> bool:
    """Check if non-free repository is enabled for APT."""
    try:
        result = subprocess.run(
            ["apt-cache", "policy", "nvidia-driver-535"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "nvidia-driver-535" in line and "500%3a" not in line:
                    return True
    except Exception:
        pass
    return False


def _check_dnf_nonfree() -> bool:
    """Check if RPMFusion nonfree is enabled for DNF."""
    try:
        result = subprocess.run(
            ["dnf", "repolist", "enabled"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            repos = result.stdout.lower()
            return "rpmfusion-nonfree" in repos
    except Exception:
        pass
    return False


def _check_pacman_nvidia() -> bool:
    """Check if nvidia packages are available in pacman."""
    try:
        result = subprocess.run(
            ["pacman", "-Ss", "nvidia"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return "nvidia" in result.stdout.lower()
    except Exception:
        pass
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
        'nvidia_open' - NVIDIA Open driver loaded
        'nouveau'     - Nouveau open-source driver loaded
        'none'        - No NVIDIA driver detected
    """
    from nvidia_inst.installer.validation import is_nvidia_working

    if is_nvidia_working().is_working:
        if check_nvidia_open_installed():
            return "nvidia_open"
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
    from nvidia_inst.utils.permissions import is_root

    nouveau_blacklist = "/etc/modprobe.d/blacklist-nouveau.conf"

    try:
        blacklist_content = "blacklist nouveau\noptions nouveau modeset=0\n"

        if is_root():
            with open(nouveau_blacklist, "w") as f:
                f.write(blacklist_content)
        else:
            # Use tee with sudo to write the blacklist file
            subprocess.run(
                ["sudo", "tee", nouveau_blacklist],
                input=blacklist_content,
                capture_output=True,
                text=True,
                check=True,
            )

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

        if not is_root():
            cmd = ["sudo"] + cmd

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logger.warning(f"Initramfs rebuild command failed: {result.stderr}")
            if cmd[-2] == "dracut" if not is_root() else cmd[0] == "dracut":
                logger.info(
                    "Nouveau has been blacklisted. Run 'sudo dracut -f' manually if needed."
                )
            elif cmd[-2] == "mkinitcpio" if not is_root() else cmd[0] == "mkinitcpio":
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
    driver_range: DriverRange | None = None,
    gpu_info: "GPUInfo | None" = None,
) -> InstallResult:
    """Install Nvidia driver.

    Args:
        installer: Distribution-specific installer.
        driver_version: Specific driver version (optional).
        with_cuda: Install CUDA packages.
        cuda_version: Specific CUDA version (optional).
        pkg_manager: Package manager for version pinning.
        driver_range: Compatible driver version range for GPU.
        gpu_info: GPU information for CUDA compatibility check.

    Returns:
        InstallResult with success status and message.
    """
    from nvidia_inst.gpu.compatibility import validate_cuda_version_with_lock
    from nvidia_inst.installer.cuda import (
        pin_cuda_to_exact_version,
        pin_cuda_to_major_version,
    )

    if not installer.pre_install_check():
        return InstallResult(
            success=False,
            message="Pre-installation check failed",
        )

    # Auto-select CUDA version based on lock if not specified
    if (
        with_cuda
        and cuda_version is None
        and driver_range
        and driver_range.cuda_is_locked
    ):
        if driver_range.cuda_locked_major:
            # Limited: use locked major version (e.g., "12.0")
            cuda_version = f"{driver_range.cuda_locked_major}.0"
            logger.info(
                f"Auto-selected CUDA {cuda_version} (locked to {driver_range.cuda_locked_major}.x for {gpu_info.generation if gpu_info else 'GPU'})"
            )
        elif driver_range.cuda_max:
            # EOL: use max version
            cuda_version = driver_range.cuda_max
            logger.info(f"Auto-selected CUDA {cuda_version} (locked for EOL GPU)")

    # Validate CUDA with lock if GPU info available
    if with_cuda and cuda_version and gpu_info:
        valid, message = validate_cuda_version_with_lock(cuda_version, gpu_info)
        if not valid:
            logger.error(f"CUDA validation failed: {message}")
            return InstallResult(
                success=False,
                message=message,
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

                # Pin CUDA version if locked
                if driver_range and driver_range.cuda_is_locked and pkg_manager:
                    if driver_range.cuda_locked_major:
                        pin_cuda_to_major_version(
                            _get_distro_id_from_installer(installer),
                            driver_range.cuda_locked_major,
                            pkg_manager,
                        )
                    elif cuda_version:
                        pin_cuda_to_exact_version(
                            _get_distro_id_from_installer(installer),
                            cuda_version,
                            pkg_manager,
                        )

            except Exception as e:
                logger.warning(f"CUDA installation failed: {e}")

    installer.post_install()

    return InstallResult(
        success=True,
        message="Driver installed successfully. Reboot required.",
        packages_installed=driver_pkgs,
    )


def _get_distro_id_from_installer(installer: DistroInstaller) -> str:
    """Get distribution ID from installer class name."""
    class_name = installer.__class__.__name__.lower()
    if "ubuntu" in class_name or "debian" in class_name:
        return "ubuntu"
    elif "fedora" in class_name or "rhel" in class_name:
        return "fedora"
    elif "arch" in class_name:
        return "arch"
    elif "suse" in class_name:
        return "opensuse"
    return "ubuntu"  # Default


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


def get_nvidia_open_packages(
    distro_id: str,
    driver_range: DriverRange,
) -> list[str]:
    """Get NVIDIA Open packages for distribution.

    Args:
        distro_id: Distribution ID.
        driver_range: Compatible driver version range.

    Returns:
        List of nvidia-open package names.
    """
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return _get_apt_nvidia_open_packages(driver_range)
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        return _get_fedora_nvidia_open_packages()
    elif distro_id in ("arch", "manjaro"):
        return _get_arch_nvidia_open_packages()
    elif distro_id in ("opensuse", "sles"):
        return _get_opensuse_nvidia_open_packages()

    return []


def _get_apt_nvidia_open_packages(driver_range: DriverRange) -> list[str]:
    """Get APT-based NVIDIA Open packages (Ubuntu/Debian/Mint/Pop)."""
    if driver_range.max_branch == "580":
        return [
            "nvidia-driver-550-open",
            "nvidia-dkms-550-open",
            "nvidia-settings",
        ]
    if driver_range.max_branch == "590":
        return [
            "nvidia-driver-535-open",
            "nvidia-dkms-535-open",
            "nvidia-settings",
        ]

    return [
        "nvidia-driver-535-open",
        "nvidia-dkms-535-open",
        "nvidia-settings",
    ]


def _get_arch_nvidia_open_packages() -> list[str]:
    """Get Arch Linux NVIDIA Open packages."""
    return ["nvidia-open"]


def _get_opensuse_nvidia_open_packages() -> list[str]:
    """Get openSUSE NVIDIA Open packages."""
    return [
        "nvidia-open-driver-G06",
        "nvidia-compute-G06",
    ]


def _get_fedora_nvidia_open_packages() -> list[str]:
    """Get Fedora NVIDIA Open packages."""
    return [
        "xorg-x11-drv-nvidia-open",
        "xorg-x11-drv-nvidia-open-cuda",
    ]


def get_nouveau_packages(distro_id: str) -> list[str]:
    """Get Nouveau packages for distribution.

    Args:
        distro_id: Distribution ID.

    Returns:
        List of nouveau package names.
    """
    nouveau_pkg_map = {
        "ubuntu": ["xserver-xorg-video-nouveau"],
        "debian": ["xserver-xorg-video-nouveau"],
        "linuxmint": ["xserver-xorg-video-nouveau"],
        "pop": ["xserver-xorg-video-nouveau"],
        "fedora": ["xorg-x11-drv-nouveau"],
        "centos": ["xorg-x11-drv-nouveau"],
        "rhel": ["xorg-x11-drv-nouveau"],
        "rocky": ["xorg-x11-drv-nouveau"],
        "alma": ["xorg-x11-drv-nouveau"],
        "arch": ["xf86-video-nouveau"],
        "manjaro": ["xf86-video-nouveau"],
        "opensuse": ["xf86-video-nouveau"],
        "sles": ["xf86-video-nouveau"],
    }
    return nouveau_pkg_map.get(distro_id, [])
