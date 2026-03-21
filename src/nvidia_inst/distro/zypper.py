"""Zypper package manager implementation for openSUSE."""

import subprocess

from nvidia_inst.distro.package_manager import PackageManager, PackageManagerError
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class ZypperManager(PackageManager):
    """Zypper package manager for openSUSE."""

    def __init__(self) -> None:
        self._zypper_path = "/usr/bin/zypper"

    def update(self) -> bool:
        """Update package lists using zypper."""
        try:
            subprocess.run(
                [self._zypper_path, "refresh"],
                check=True,
                capture_output=True,
            )
            logger.info("Package lists updated")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update package lists: {e.stderr}")
            return False

    def upgrade(self) -> bool:
        """Upgrade all packages using zypper."""
        try:
            subprocess.run(
                [self._zypper_path, "update", "-y"],
                check=True,
                capture_output=True,
            )
            logger.info("Packages upgraded")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade packages: {e.stderr}")
            return False

    def install(self, packages: list[str]) -> bool:
        """Install packages using zypper."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._zypper_path, "install", "-y"] + packages
            if not is_root():
                cmd = ["sudo"] + cmd
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Installed packages: {', '.join(packages)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e.stderr}")
            raise PackageManagerError(
                f"Failed to install: {', '.join(packages)}"
            ) from e

    def remove(self, packages: list[str]) -> bool:
        """Remove packages using zypper."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._zypper_path, "remove", "-y"] + packages
            if not is_root():
                cmd = ["sudo"] + cmd
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Removed packages: {', '.join(packages)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove packages: {e.stderr}")
            return False

    def search(self, query: str) -> list[str]:
        """Search for packages using zypper."""
        try:
            result = subprocess.run(
                [self._zypper_path, "search", query],
                check=True,
                capture_output=True,
                text=True,
            )
            packages = []
            for line in result.stdout.splitlines():
                if "nvidia" in line.lower():
                    parts = line.split("|")
                    if parts:
                        pkg_name = parts[0].strip()
                        if pkg_name not in packages:
                            packages.append(pkg_name)
            return packages
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to search packages: {e.stderr}")
            return []

    def is_available(self) -> bool:
        """Check if zypper is available."""
        import shutil

        return shutil.which(self._zypper_path) is not None

    def get_installed_version(self, package: str) -> str | None:
        """Get installed version of a package."""
        try:
            result = subprocess.run(
                [self._zypper_path, "info", "-i", package],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
            return None
        except subprocess.CalledProcessError:
            return None

    def get_available_version(self, package: str) -> str | None:
        """Get available version of a package."""
        try:
            result = subprocess.run(
                [self._zypper_path, "info", package],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
            return None
        except subprocess.CalledProcessError:
            return None

    def pin_version(self, package: str, version: str = "*") -> bool:
        """Pin package to version using zypper.

        Args:
            package: Package name to lock.
            version: Version pattern. Defaults to '*' for package lock.
                     Use 'package=version' for specific version lock.

        Returns:
            True if successful, False otherwise.
        """
        try:
            lock_name = package if version == "*" else f"{package}={version}"
            subprocess.run(
                [self._zypper_path, "addlock", lock_name],
                check=True,
                capture_output=True,
            )
            logger.info(f"Locked {lock_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to lock package: {e.stderr}")
            return False

    def get_all_versions(self, package: str) -> list[str]:
        """Get all available versions of a package using zypper."""
        try:
            result = subprocess.run(
                [
                    self._zypper_path,
                    "packages",
                    "-s",
                    "version",
                    "--match-substring",
                    package,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            versions = []
            seen = set()
            for line in result.stdout.splitlines():
                if "nvidia" in line.lower() and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        version = parts[2].strip().split("-")[0]
                        if version and version not in seen:
                            versions.append(version)
                            seen.add(version)
            return sorted(versions, key=self._version_sort_key, reverse=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to get versions for {package}: {e.stderr}")
            return []

    def _version_sort_key(self, version: str) -> tuple:
        """Sort key for version strings."""
        import re

        nums = re.findall(r"\d+", version)
        return tuple(int(n) for n in nums[:3]) if nums else (0, 0, 0)
