"""Prerequisite checking for driver installation."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.installer.version_checker import VersionChecker, VersionCheckResult
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PrerequisitesResult:
    """Result of prerequisite checks."""

    success: bool
    package_manager_available: bool = False
    package_manager: str = ""
    repos_configured: list[str] = field(default_factory=list)
    repos_missing: list[str] = field(default_factory=list)
    driver_packages_available: bool = False
    driver_packages: list[str] = field(default_factory=list)
    version_check: VersionCheckResult | None = None
    fix_commands: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PrerequisitesChecker:
    """Check system prerequisites for driver installation."""

    def __init__(self):
        self._pm = None

    def check_all(
        self,
        distro_id: str,
        distro_version: str = "",
        driver_range: DriverRange | None = None,
    ) -> PrerequisitesResult:
        """Run all prerequisite checks.

        Args:
            distro_id: Distribution ID (e.g., 'fedora', 'ubuntu').
            distro_version: Distribution version (e.g., '43', '22.04').
            driver_range: Compatible driver range for the detected GPU.

        Returns:
            PrerequisitesResult with all check results.
        """
        result = PrerequisitesResult(success=False)

        pm_available, pm_name = self._check_package_manager()
        result.package_manager_available = pm_available
        result.package_manager = pm_name

        if not pm_available:
            result.errors.append("Package manager not available")
            return result

        repos_status = self._check_repositories(distro_id, distro_version)
        result.repos_configured = repos_status["configured"]
        result.repos_missing = repos_status["missing"]
        result.fix_commands = repos_status["fix_commands"]

        packages_status = self._check_driver_packages(distro_id)
        result.driver_packages_available = packages_status["available"]
        result.driver_packages = packages_status["packages"]

        if driver_range:
            version_checker = VersionChecker()
            result.version_check = version_checker.check_compatibility(distro_id, driver_range)

            if not result.version_check.compatible:
                result.success = False
                result.errors.extend(result.version_check.errors)
            else:
                result.warnings.extend(result.version_check.warnings)
        else:
            result.version_check = None

        result.success = (
            pm_available
            and len(result.repos_missing) == 0
            and result.driver_packages_available
            and (result.version_check is None or result.version_check.compatible)
        )

        return result

    def _check_package_manager(self) -> tuple[bool, str]:
        """Check if package manager is available."""
        managers = {
            "apt": ["/usr/bin/apt", "/usr/bin/apt-get"],
            "dnf": ["/usr/bin/dnf"],
            "pacman": ["/usr/bin/pacman"],
            "zypper": ["/usr/bin/zypper"],
        }

        for name, paths in managers.items():
            if any(Path(p).exists() for p in paths):
                logger.info(f"Found package manager: {name}")
                return True, name

        return False, ""

    def _check_repositories(self, distro_id: str, distro_version: str) -> dict:
        """Check if required repositories are configured."""
        repos_status = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
            return self._check_fedora_repos(distro_id, distro_version)
        elif distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
            return self._check_ubuntu_repos(distro_id, distro_version)
        elif distro_id in ("arch", "manjaro", "endeavouros"):
            return self._check_arch_repos()
        elif distro_id in ("opensuse", "sles"):
            return self._check_opensuse_repos()

        return repos_status

    def _check_fedora_repos(self, distro_id: str, distro_version: str) -> dict:
        """Check Fedora/RHEL repositories."""
        repos_status = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        if not distro_version:
            distro_version = "43"

        rpmfusion_nonfree = self._repo_exists("rpmfusion-nonfree")

        if rpmfusion_nonfree:
            repos_status["configured"].append("RPM Fusion nonfree")
        else:
            repos_status["missing"].append("RPM Fusion nonfree (required for proprietary drivers)")
            repos_status["fix_commands"].append(
                f"sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/"
                f"rpmfusion-nonfree-release-{distro_version}.noarch.rpm"
            )

        return repos_status

    def _check_ubuntu_repos(self, distro_id: str, distro_version: str) -> dict:
        """Check Ubuntu/Debian repositories."""
        repos_status = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        if distro_id == "ubuntu":
            repos_status["configured"].append("Ubuntu main repositories")
            repos_status["fix_commands"].append(
                "sudo add-apt-repository ppa:graphics-drivers/ppa"
            )
        else:
            repos_status["configured"].append("Debian contrib, non-free")

        return repos_status

    def _check_arch_repos(self) -> dict:
        """Check Arch Linux repositories."""
        repos_status = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        repos_status["configured"].append("Core repositories")

        repos_status["fix_commands"].append(
            "sudo pacman -Syu --needed multilib-devel"
        )

        return repos_status

    def _check_opensuse_repos(self) -> dict:
        """Check openSUSE repositories."""
        repos_status = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        repos_status["configured"].append("openSUSE OSS")
        repos_status["fix_commands"].append(
            "sudo zypper addrepo --refresh https://download.nvidia.com/opensuse/leap NVIDIA"
        )

        return repos_status

    def _repo_exists(self, repo_pattern: str) -> bool:
        """Check if a repository is configured."""
        try:
            result = subprocess.run(
                ["dnf", "repolist", "-q"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return repo_pattern in result.stdout.lower()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_driver_packages(self, distro_id: str) -> dict:
        """Check if driver packages are available."""
        packages_status = {
            "available": False,
            "packages": [],
        }

        try:
            if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
                result = subprocess.run(
                    ["dnf", "search", "akmod-nvidia", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and "akmod-nvidia" in result.stdout:
                    packages_status["available"] = True
                    packages_status["packages"] = ["akmod-nvidia", "xorg-x11-drv-nvidia-cuda"]

            elif distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
                result = subprocess.run(
                    ["apt-cache", "search", "nvidia-driver"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    packages_status["available"] = True
                    packages_status["packages"] = ["nvidia-driver-535", "nvidia-dkms-535"]

            elif distro_id in ("arch", "manjaro"):
                packages_status["available"] = True
                packages_status["packages"] = ["nvidia", "nvidia-utils"]

            elif distro_id in ("opensuse", "sles"):
                packages_status["available"] = True
                packages_status["packages"] = ["x11-video-nvidiaG05"]

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not check packages: {e}")

        return packages_status

    def fix_repositories(self, fix_commands: list[str]) -> tuple[bool, str]:
        """Attempt to fix missing repositories.

        Args:
            fix_commands: List of commands to run.

        Returns:
            Tuple of (success, message).
        """
        logger.info("Attempting to fix repositories...")

        for cmd in fix_commands:
            logger.info(f"Running: {cmd}")
            try:
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    logger.error(f"Command failed: {result.stderr}")
                    return False, f"Failed: {cmd}"
            except subprocess.TimeoutExpired:
                logger.error("Command timed out")
                return False, "Command timed out"
            except Exception as e:
                logger.error(f"Error running command: {e}")
                return False, str(e)

        return True, "Repositories configured successfully"
