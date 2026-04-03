"""Driver installation for different distributions."""

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nvidia_inst.distro.packages import (
    get_driver_open_packages,
    get_driver_packages,
    get_nouveau_remove_packages,
)
from nvidia_inst.distro.tools import PackageContext
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

    # Set version locks BEFORE installation to prevent pulling wrong versions
    lock_errors = []
    if driver_version and pkg_manager:
        for pkg in driver_pkgs:
            if not pkg_manager.pin_version(pkg, driver_version):
                lock_errors.append(pkg)
                logger.error(f"Failed to pin {pkg} to version {driver_version}")
            else:
                logger.info(f"Pinned {pkg} to version {driver_version}")

    if lock_errors:
        return InstallResult(
            success=False,
            message=f"Failed to set version locks for: {', '.join(lock_errors)}",
        )

    # Install driver packages (constrained by version locks)
    try:
        logger.info(f"Installing driver packages: {driver_pkgs}")
        installer.install(driver_pkgs)

    except Exception as e:
        logger.error(f"Failed to install driver: {e}")
        return InstallResult(
            success=False,
            message=f"Installation failed: {e}",
        )

    # Install CUDA packages (constrained by version locks)
    if with_cuda:
        cuda_pkgs = installer.get_cuda_packages(cuda_version)
        if cuda_pkgs:
            try:
                logger.info(f"Installing CUDA packages: {cuda_pkgs}")
                installer.install(cuda_pkgs)

                # Pin CUDA version if locked
                if driver_range and driver_range.cuda_is_locked and pkg_manager:
                    cuda_lock_errors = []
                    if driver_range.cuda_locked_major and not pin_cuda_to_major_version(
                        _get_distro_id_from_installer(installer),
                        driver_range.cuda_locked_major,
                        pkg_manager,
                    ):
                        cuda_lock_errors.append("cuda-toolkit")
                    elif cuda_version and not pin_cuda_to_exact_version(
                        _get_distro_id_from_installer(installer),
                        cuda_version,
                        pkg_manager,
                    ):
                        cuda_lock_errors.append("cuda-toolkit")

                    if cuda_lock_errors:
                        logger.warning(
                            f"Failed to set CUDA version locks for: {', '.join(cuda_lock_errors)}"
                        )

            except Exception as e:
                logger.warning(f"CUDA installation failed: {e}")

    # Verify version locks are active
    if driver_range and driver_range.max_branch and pkg_manager:
        from nvidia_inst.distro.versionlock import verify_versionlock_pattern_active

        for pkg in driver_pkgs:
            success, msg = verify_versionlock_pattern_active(
                pkg, driver_range.max_branch
            )
            if success:
                logger.info(f"Verified lock for {pkg}: {msg}")
            else:
                logger.warning(f"Lock verification failed for {pkg}: {msg}")

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


def _get_distro_tool(distro_id: str) -> str:
    """Get the package manager tool for a distribution.

    Args:
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora').

    Returns:
        Package manager tool name (e.g., 'apt', 'dnf', 'pacman').
    """
    apt_distros = ("ubuntu", "linuxmint", "pop", "debian")
    dnf_distros = ("fedora", "rhel", "centos", "rocky", "alma")
    pacman_distros = ("arch", "manjaro", "endeavouros")
    zypper_distros = ("opensuse", "sles")

    if distro_id in apt_distros:
        return "apt"
    elif distro_id in dnf_distros:
        return "dnf"
    elif distro_id in pacman_distros:
        return "pacman"
    elif distro_id in zypper_distros:
        return "zypper"
    else:
        # Default to apt for unknown distros
        logger.warning(f"Unknown distro {distro_id}, defaulting to apt")
        return "apt"


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
    tool = _get_distro_tool(distro_id)
    branch = driver_range.max_branch
    is_eol = driver_range.is_eol

    # Create a PackageContext for the template-based system
    ctx = PackageContext(
        tool=tool,
        distro_id=distro_id,
        distro_family=_get_distro_family(distro_id),
        version_id="",  # Not needed for package selection
    )

    packages = get_driver_packages(ctx, branch=branch, is_eol=is_eol)

    # If no packages found, try fallback
    if not packages:
        logger.warning(f"No packages found for {distro_id} branch {branch}")
        # For Fedora, use generic packages
        if tool == "dnf":
            packages = [
                "akmod-nvidia",
                "xorg-x11-drv-nvidia",
                "xorg-x11-drv-nvidia-cuda",
            ]

    return packages


def _get_distro_family(distro_id: str) -> str:
    """Get the distro family for a distribution.

    Args:
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora').

    Returns:
        Distro family name (e.g., 'debian', 'redhat', 'arch').
    """
    debian_distros = ("ubuntu", "linuxmint", "pop", "debian")
    redhat_distros = ("fedora", "rhel", "centos", "rocky", "alma")
    arch_distros = ("arch", "manjaro", "endeavouros")
    suse_distros = ("opensuse", "sles")

    if distro_id in debian_distros:
        return "debian"
    elif distro_id in redhat_distros:
        return "redhat"
    elif distro_id in arch_distros:
        return "arch"
    elif distro_id in suse_distros:
        return "suse"
    else:
        return "unknown"


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
    tool = _get_distro_tool(distro_id)
    branch = driver_range.max_branch

    # Create a PackageContext for the template-based system
    ctx = PackageContext(
        tool=tool,
        distro_id=distro_id,
        distro_family=_get_distro_family(distro_id),
        version_id="",  # Not needed for package selection
    )

    packages = get_driver_open_packages(ctx, branch=branch)

    # If no packages found, try fallback
    if not packages:
        logger.warning(f"No open packages found for {distro_id} branch {branch}")
        # For Fedora, use generic packages
        if tool == "dnf":
            packages = ["xorg-x11-drv-nvidia-open", "xorg-x11-drv-nvidia-open-cuda"]
        # For Arch, use generic open packages
        elif tool == "pacman":
            packages = ["nvidia-open", "nvidia-utils"]

    return packages


def get_nouveau_packages(distro_id: str) -> list[str]:
    """Get Nouveau packages for distribution.

    Args:
        distro_id: Distribution ID.

    Returns:
        List of nouveau package names.
    """
    tool = _get_distro_tool(distro_id)

    # Create a PackageContext for the template-based system
    ctx = PackageContext(
        tool=tool,
        distro_id=distro_id,
        distro_family=_get_distro_family(distro_id),
        version_id="",  # Not needed for package selection
    )

    return get_nouveau_remove_packages(ctx)
