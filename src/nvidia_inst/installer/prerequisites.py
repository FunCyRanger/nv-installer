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
            result.version_check = version_checker.check_compatibility(
                distro_id, driver_range
            )

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

    @staticmethod
    def get_cuda_repo_version(
        distro_version: str, cuda_major: str | None = None
    ) -> str:
        """Get the appropriate CUDA repo version for a given CUDA major version.

        NVIDIA repos only keep the latest CUDA version for each distro. To get older
        CUDA versions, we need to use an older distro repo that had that CUDA version.

        Args:
            distro_version: Distribution version (e.g., "43")
            cuda_major: Desired CUDA major version (e.g., "12"). If None, uses current distro.

        Returns:
            Repo version string to use (e.g., "43" or "41")
        """
        if cuda_major is None:
            return distro_version

        cuda_major_int = int(cuda_major) if cuda_major else 0

        cuda_to_distro = {
            12: "41",
            13: "42",
        }

        if cuda_major_int in cuda_to_distro:
            target_distro = cuda_to_distro[cuda_major_int]
            if int(distro_version) >= int(target_distro):
                return target_distro

        return distro_version

    def get_cuda_repo_fix_commands(
        self,
        distro_id: str,
        distro_version: str,
        cuda_major: str | None = None,
    ) -> list[str]:
        """Get fix commands to add missing CUDA repository.

        Args:
            distro_id: Distribution ID
            distro_version: Distribution version
            cuda_major: Desired CUDA major version for version-locked installs

        Returns:
            List of commands to run to add CUDA repository.
        """
        fix_commands = []
        if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
            if not distro_version:
                distro_version = "43"

            repo_version = self.get_cuda_repo_version(distro_version, cuda_major)
            current_repo_version = self._get_cuda_repo_version_from_file()

            cuda_repo = self._repo_exists("cuda-fedora")
            cuda_repo_name: str | None = None
            if isinstance(cuda_repo, str):
                cuda_repo_name = cuda_repo

            needs_update = (
                cuda_major is not None
                and current_repo_version is not None
                and current_repo_version != repo_version
            )

            if not cuda_repo_name or needs_update:
                if needs_update:
                    logger.info(
                        f"Updating CUDA repo from fedora{current_repo_version} to "
                        f"fedora{repo_version} for CUDA {cuda_major}.x"
                    )
                    fix_commands.append("sudo rm -f /etc/yum.repos.d/cuda-fedora*.repo")
                fix_commands.append(
                    f"sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora{repo_version}/x86_64/cuda-fedora{repo_version}.repo --overwrite 2>/dev/null || "
                    f"sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora{repo_version}/x86_64/cuda-fedora{repo_version}.repo"
                )
        elif distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
            cuda_keyring_installed = self._package_installed("cuda-keyring", distro_id)
            if not cuda_keyring_installed:
                if distro_id == "ubuntu":
                    version_short = distro_version.replace(".", "")
                    fix_commands.append(
                        f"wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{version_short}/x86_64/cuda-keyring_1.1-1_all.deb && "
                        f"sudo dpkg -i cuda-keyring_1.1-1_all.deb"
                    )
                else:
                    version_short = distro_version.replace(".", "")
                    fix_commands.append(
                        f"wget https://developer.download.nvidia.com/compute/cuda/repos/debian{version_short}/x86_64/cuda-keyring_1.1-1_all.deb && "
                        f"sudo dpkg -i cuda-keyring_1.1-1_all.deb"
                    )
        elif distro_id in ("arch", "manjaro", "endeavouros"):
            # CUDA is available via AUR, no repository needed
            pass
        elif distro_id in ("opensuse", "sles"):
            # CUDA repository is the same as NVIDIA driver repository
            # Already added via openSUSE repos check
            pass
        return fix_commands

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
        repos_status: dict[str, list[str]] = {
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
        repos_status: dict[str, list[str]] = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        if not distro_version:
            distro_version = "43"

        rpmfusion_nonfree = self._repo_exists("rpmfusion-nonfree")
        cuda_repo = self._repo_exists("cuda-fedora")
        # Also check for cuda-fedora{version} pattern
        if not cuda_repo:
            cuda_repo = self._repo_exists(f"cuda-fedora{distro_version}")

        if rpmfusion_nonfree:
            repos_status["configured"].append("RPM Fusion nonfree")
        else:
            repos_status["missing"].append(
                "RPM Fusion nonfree (required for proprietary drivers)"
            )
            repos_status["fix_commands"].append(
                f"sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/"
                f"rpmfusion-nonfree-release-{distro_version}.noarch.rpm"
            )

        if cuda_repo:
            repos_status["configured"].append("NVIDIA CUDA repository")
        else:
            repos_status["missing"].append(
                "NVIDIA CUDA repository (required for CUDA toolkit)"
            )
            repos_status["fix_commands"].append(
                f"sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora{distro_version}/x86_64/cuda-fedora{distro_version}.repo --overwrite 2>/dev/null || "
                f"sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora{distro_version}/x86_64/cuda-fedora{distro_version}.repo"
            )

        return repos_status

    def _check_ubuntu_repos(self, distro_id: str, distro_version: str) -> dict:
        """Check Ubuntu/Debian repositories."""
        repos_status: dict[str, list[str]] = {
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

        # Check for CUDA repository (via cuda-keyring package)
        cuda_keyring_installed = self._package_installed("cuda-keyring", distro_id)
        if cuda_keyring_installed:
            repos_status["configured"].append("NVIDIA CUDA repository")
        else:
            repos_status["missing"].append(
                "NVIDIA CUDA repository (required for CUDA toolkit)"
            )
            if distro_id == "ubuntu":
                # Convert version like "22.04" to "2204"
                version_short = distro_version.replace(".", "")
                fix_cmd = (
                    f"wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{version_short}/x86_64/cuda-keyring_1.1-1_all.deb && "
                    f"sudo dpkg -i cuda-keyring_1.1-1_all.deb"
                )
            else:  # Debian
                # Assuming Debian 11 (bullseye) or 12 (bookworm)
                version_short = distro_version.replace(".", "")
                fix_cmd = (
                    f"wget https://developer.download.nvidia.com/compute/cuda/repos/debian{version_short}/x86_64/cuda-keyring_1.1-1_all.deb && "
                    f"sudo dpkg -i cuda-keyring_1.1-1_all.deb"
                )
            repos_status["fix_commands"].append(fix_cmd)

        return repos_status

    def _check_arch_repos(self) -> dict:
        """Check Arch Linux repositories."""
        repos_status: dict[str, list[str]] = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        repos_status["configured"].append("Core repositories")
        repos_status["configured"].append("CUDA available via AUR (not a repository)")

        repos_status["fix_commands"].append("sudo pacman -Syu --needed multilib-devel")

        return repos_status

    def _check_opensuse_repos(self) -> dict:
        """Check openSUSE repositories."""
        repos_status: dict[str, list[str]] = {
            "configured": [],
            "missing": [],
            "fix_commands": [],
        }

        repos_status["configured"].append("openSUSE OSS")
        repos_status["configured"].append("NVIDIA CUDA repository (provided by NVIDIA)")
        repos_status["fix_commands"].append(
            "sudo zypper addrepo --refresh https://download.nvidia.com/opensuse/leap NVIDIA"
        )

        return repos_status

    def _repo_exists(self, repo_pattern: str) -> str | bool:
        """Check if a repository is configured.

        Returns:
            Repo name if found, False otherwise.
        """
        try:
            result = subprocess.run(
                ["dnf", "repolist", "-q"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.lower().splitlines()
            for line in lines:
                if repo_pattern in line:
                    return line.split()[0]
            return False
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            return False

    def _get_cuda_repo_version(self, repo_name: str) -> str | None:
        """Get the fedora version from a CUDA repo URL.

        Args:
            repo_name: Name of the repo (e.g., 'cuda-fedora43-x86_64')

        Returns:
            Fedora version string (e.g., '43') or None if not parseable.
        """
        import re

        try:
            result = subprocess.run(
                ["dnf", "reponame", repo_name, "-v"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if "baseurl" in line.lower():
                    match = re.search(r"fedora(\d+)", line)
                    if match:
                        return match.group(1)
        except Exception:
            pass
        return None

    def _get_cuda_repo_version_from_file(self) -> str | None:
        """Get the fedora version from existing CUDA repo file.

        Parses /etc/yum.repos.d/cuda-*.repo files directly to get the
        fedora version from the baseurl.

        Returns:
            Fedora version string (e.g., '43') or None if not found.
        """
        import glob
        import re

        for repo_file in glob.glob("/etc/yum.repos.d/cuda-fedora*.repo"):
            try:
                with open(repo_file) as f:
                    content = f.read().lower()
                    match = re.search(r"baseurl.*fedora(\d+)", content)
                    if match:
                        return match.group(1)
            except Exception:
                pass
        return None

    def _package_installed(self, package_name: str, distro_id: str) -> bool:
        """Check if a package is installed."""
        try:
            if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
                result = subprocess.run(
                    ["dpkg", "-l", package_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return package_name in result.stdout and "ii" in result.stdout
            elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
                result = subprocess.run(
                    ["rpm", "-q", package_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0
            elif distro_id in ("arch", "manjaro", "endeavouros"):
                result = subprocess.run(
                    ["pacman", "-Qi", package_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0
            elif distro_id in ("opensuse", "sles"):
                result = subprocess.run(
                    ["rpm", "-q", package_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
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
                    packages_status["packages"] = [
                        "akmod-nvidia",
                        "xorg-x11-drv-nvidia-cuda",
                    ]

            elif distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
                result = subprocess.run(
                    ["apt-cache", "search", "nvidia-driver"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    packages_status["available"] = True
                    packages_status["packages"] = [
                        "nvidia-driver-535",
                        "nvidia-dkms-535",
                    ]

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
