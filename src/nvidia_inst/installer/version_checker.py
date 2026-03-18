"""Version checking for driver availability and compatibility."""

import re
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


NVIDIA_ARCHIVE_URL = "https://download.nvidia.com/XFree86/Linux-x86_64/"


@dataclass
class VersionCheckResult:
    """Result of version availability checking."""

    success: bool = False
    repo_versions: list[str] = field(default_factory=list)
    official_versions: list[str] = field(default_factory=list)
    installed_driver_version: Optional[str] = None
    compatible: bool = False
    compatible_versions: list[str] = field(default_factory=list)
    incompatible_versions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class VersionChecker:
    """Check driver version availability and compatibility."""

    def fetch_official_versions(self, branch: Optional[str] = None) -> list[str]:
        """Fetch available versions from Nvidia driver archive.

        Args:
            branch: Optional branch to filter (e.g., "580", "590").

        Returns:
            List of version strings from official archive.
        """
        try:
            request = urllib.request.Request(
                NVIDIA_ARCHIVE_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8")

            versions = []
            pattern = r'href="(\d+\.\d+\.\d+)/"'
            matches = re.findall(pattern, html)

            for version in matches:
                major = version.split(".")[0]
                if branch is None or major == branch:
                    versions.append(version)

            return sorted(versions, key=self._version_sort_key, reverse=True)

        except Exception as e:
            logger.warning(f"Failed to fetch official versions: {e}")
            return []

    def get_repo_versions(self, distro_id: str, package: str) -> list[str]:
        """Get available versions from distro repos.

        Args:
            distro_id: Distribution ID.
            package: Package name to query.

        Returns:
            List of version strings available in repos.
        """
        try:
            pm = get_package_manager()
            return pm.get_all_versions(package)
        except Exception as e:
            logger.warning(f"Failed to get repo versions: {e}")
            return []

    def check_installed_driver(self, distro_id: str) -> Optional[str]:
        """Check what driver version is currently installed.

        Args:
            distro_id: Distribution ID.

        Returns:
            Installed driver version string, or None if not installed.
        """
        try:
            pm = get_package_manager()

            if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
                return self._check_dnf_installed(pm)
            elif distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
                return self._check_apt_installed(pm)
            elif distro_id in ("arch", "manjaro", "endeavouros"):
                return self._check_pacman_installed(pm)
            elif distro_id in ("opensuse", "sles"):
                return self._check_zypper_installed(pm)

        except Exception as e:
            logger.warning(f"Failed to check installed driver: {e}")
        return None

    def _check_dnf_installed(self, pm) -> Optional[str]:
        """Check installed driver via DNF."""
        import subprocess
        try:
            result = subprocess.run(
                ["dnf", "list", "installed"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            for line in result.stdout.splitlines():
                if "xorg-x11-drv-nvidia" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        full_version = parts[1]
                        if ":" in full_version:
                            full_version = full_version.split(":", 1)[1]
                        version = full_version.split("-")[0]
                        return version
            for line in result.stdout.splitlines():
                if "akmod-nvidia" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        full_version = parts[1]
                        if ":" in full_version:
                            full_version = full_version.split(":", 1)[1]
                        version = full_version.split("-")[0]
                        return version
        except Exception:
            pass
        return None

    def _check_apt_installed(self, pm) -> Optional[str]:
        """Check installed driver via APT."""
        import subprocess
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Version}", "nvidia-driver-535"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("-")[0]
        except Exception:
            pass
        return None

    def _check_pacman_installed(self, pm) -> Optional[str]:
        """Check installed driver via Pacman."""
        installed = pm.get_installed_version("nvidia")
        if installed:
            return "nvidia"
        installed_470xx = pm.get_installed_version("nvidia-470xx")
        if installed_470xx:
            return "nvidia-470xx"
        return None

    def _check_zypper_installed(self, pm) -> Optional[str]:
        """Check installed driver via Zypper."""
        version = pm.get_installed_version("x11-video-nvidiaG05")
        if version:
            return "G05"
        version = pm.get_installed_version("x11-video-nvidiaG04")
        if version:
            return "G04"
        return None

    def _extract_branch(self, version: str) -> Optional[str]:
        """Extract major version number from driver version."""
        match = re.match(r"(\d+)\.", version)
        return match.group(1) if match else None

    def _version_sort_key(self, version: str) -> tuple:
        """Sort key for version strings."""
        nums = re.findall(r"\d+", version)
        return tuple(int(n) for n in nums[:3]) if nums else (0, 0, 0)

    def check_compatibility(
        self,
        distro_id: str,
        driver_range: DriverRange,
    ) -> VersionCheckResult:
        """Check version availability and compatibility.

        This is the main method that:
        1. Gets versions from distro repos
        2. Gets versions from Nvidia official archive
        3. Checks currently installed driver
        4. Compares available versions against GPU compatibility

        Args:
            distro_id: Distribution ID.
            driver_range: Compatible driver range for the GPU.

        Returns:
            VersionCheckResult with compatibility status.
        """
        result = VersionCheckResult()

        package = self._get_driver_package(distro_id)
        if not package:
            result.errors.append(f"Unknown distro: {distro_id}")
            return result

        result.repo_versions = self.get_repo_versions(distro_id, package)

        result.official_versions = self.fetch_official_versions()

        result.installed_driver_version = self.check_installed_driver(distro_id)

        if result.repo_versions:
            result.compatible_versions = []
            result.incompatible_versions = []

            for version in result.repo_versions:
                if self._is_version_compatible(version, driver_range):
                    result.compatible_versions.append(version)
                else:
                    result.incompatible_versions.append(version)

            if result.compatible_versions:
                result.compatible = True
                result.success = True
            else:
                result.compatible = False
                result.success = False
                result.errors.append(
                    f"No compatible driver version in repos. "
                    f"Available: {result.incompatible_versions}"
                )
        else:
            result.success = False
            result.errors.append("No driver packages found in repositories")

        if result.installed_driver_version:
            installed_branch = self._extract_branch(result.installed_driver_version)
            if installed_branch:
                if driver_range.max_branch and installed_branch > driver_range.max_branch:
                    result.warnings.append(
                        f"Installed driver {result.installed_driver_version} may be "
                        f"incompatible with this GPU. "
                        f"GPU requires: {driver_range.max_branch}.xx"
                    )
            if driver_range.is_limited and driver_range.max_branch:
                if not self._is_branch_compatible(
                    result.installed_driver_version, driver_range.max_branch
                ):
                    result.warnings.append(
                        f"WARNING: Installed driver ({result.installed_driver_version}) "
                        f"is incompatible with this GPU! "
                        f"Your GPU requires {driver_range.max_branch}.xx drivers. "
                        f"Installing {result.installed_driver_version} may fail or not work properly."
                    )

        return result

    def _get_driver_package(self, distro_id: str) -> Optional[str]:
        """Get the primary driver package name for the distro."""
        package_map = {
            "fedora": "akmod-nvidia",
            "rhel": "akmod-nvidia",
            "centos": "akmod-nvidia",
            "rocky": "akmod-nvidia",
            "alma": "akmod-nvidia",
            "ubuntu": "nvidia-driver-535",
            "debian": "nvidia-driver",
            "linuxmint": "nvidia-driver-535",
            "pop": "nvidia-driver-535",
            "arch": "nvidia",
            "manjaro": "nvidia",
            "endeavouros": "nvidia",
            "opensuse": "x11-video-nvidiaG05",
            "sles": "x11-video-nvidiaG05",
        }
        return package_map.get(distro_id)

    def _is_version_compatible(self, version: str, driver_range: DriverRange) -> bool:
        """Check if a version is compatible with the GPU."""
        if driver_range.is_eol:
            return True

        branch = self._extract_branch(version)
        if not branch:
            return False

        if driver_range.max_branch:
            return branch <= driver_range.max_branch

        if driver_range.min_version:
            min_branch = self._extract_branch(driver_range.min_version)
            if min_branch and branch < min_branch:
                return False

        return True

    def _is_branch_compatible(self, version: str, max_branch: str) -> bool:
        """Check if driver version branch is within max allowed branch."""
        branch = self._extract_branch(version)
        if not branch:
            return True
        return branch <= max_branch


def check_driver_versions(
    distro_id: str,
    driver_range: DriverRange,
) -> VersionCheckResult:
    """Convenience function to check driver version compatibility.

    Args:
        distro_id: Distribution ID.
        driver_range: Compatible driver range for the GPU.

    Returns:
        VersionCheckResult with compatibility status.
    """
    checker = VersionChecker()
    return checker.check_compatibility(distro_id, driver_range)
