"""DNF package manager implementation for Fedora/RHEL."""

import subprocess

from nvidia_inst.distro.package_manager import PackageManager, PackageManagerError
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class DnfManager(PackageManager):
    """DNF package manager for Fedora/RHEL."""

    def __init__(self) -> None:
        self._dnf_path = "/usr/bin/dnf"

    def update(self) -> bool:
        """Update package lists using dnf."""
        try:
            subprocess.run(
                [self._dnf_path, "makecache"],
                check=True,
                capture_output=True,
            )
            logger.info("Package lists updated")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update package lists: {e.stderr}")
            return False

    def upgrade(self) -> bool:
        """Upgrade all packages using dnf."""
        try:
            subprocess.run(
                [self._dnf_path, "upgrade", "-y"],
                check=True,
                capture_output=True,
            )
            logger.info("Packages upgraded")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade packages: {e.stderr}")
            return False

    def install(self, packages: list[str]) -> bool:
        """Install packages using dnf with progress spinner."""
        import sys
        import time

        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._dnf_path, "install", "-y"] + packages
            if not is_root():
                cmd = ["sudo"] + cmd

            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            i = 0
            while proc.poll() is None:
                sys.stdout.write(f"\r{spinner[i % len(spinner)]} Installing driver...")
                sys.stdout.flush()
                time.sleep(0.3)
                i += 1

            sys.stdout.write("\r" + " " * 25 + "\r")
            sys.stdout.flush()

            result = proc.stdout.read() if proc.stdout else ""

            if proc.returncode != 0:
                logger.error(f"Failed to install packages: {result}")
                raise PackageManagerError(
                    f"Failed to install: {', '.join(packages)}\n{result}"
                )

            logger.info(f"Installed packages: {', '.join(packages)}")
            return True

        except PackageManagerError:
            raise
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            raise PackageManagerError(
                f"Failed to install: {', '.join(packages)}"
            ) from e

    def remove(self, packages: list[str]) -> bool:
        """Remove packages using dnf."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._dnf_path, "remove", "-y"] + packages
            if not is_root():
                cmd = ["sudo"] + cmd
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Removed packages: {', '.join(packages)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove packages: {e.stderr}")
            return False

    def search(self, query: str) -> list[str]:
        """Search for packages using dnf."""
        try:
            result = subprocess.run(
                [self._dnf_path, "search", query],
                check=True,
                capture_output=True,
                text=True,
            )
            packages = []
            for line in result.stdout.splitlines():
                if line.startswith("nvidia"):
                    pkg_name = line.split(".")[0].strip()
                    if pkg_name not in packages:
                        packages.append(pkg_name)
            return packages
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to search packages: {e.stderr}")
            return []

    def is_available(self) -> bool:
        """Check if dnf is available."""
        import shutil

        return shutil.which(self._dnf_path) is not None

    def get_installed_version(self, package: str) -> str | None:
        """Get installed version of a package."""
        try:
            result = subprocess.run(
                [self._dnf_path, "info", "--installed", package],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version"):
                    return line.split(":")[1].strip()
            return None
        except subprocess.CalledProcessError:
            return None

    def get_available_version(self, package: str) -> str | None:
        """Get available version of a package."""
        try:
            result = subprocess.run(
                [self._dnf_path, "info", package],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version"):
                    return line.split(":")[1].strip()
            return None
        except subprocess.CalledProcessError:
            return None

    def pin_version(self, package: str, version: str = "*") -> bool:
        """Pin package to version using dnf versionlock.

        Args:
            package: Package name (e.g., 'akmod-nvidia').
            version: Version pattern. Defaults to '*' for branch locking.
                     Use '580.*' to lock to 580 branch, 'exact.version' for exact.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cmd = [
                self._dnf_path,
                "versionlock",
                "add",
                "--raw",
                f"{package}-{version}",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Locked {package} to pattern {version}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to lock version: {e.stderr}")
            return False

    def get_all_versions(self, package: str) -> list[str]:
        """Get all available versions of a package using dnf."""

        def clean_version(ver: str) -> str:
            if ":" in ver:
                ver = ver.split(":", 1)[1]
            return ver.split("-")[0]

        try:
            result = subprocess.run(
                [self._dnf_path, "list", "--showduplicates", package],
                capture_output=True,
                text=True,
                timeout=60,
            )
            versions = []
            seen = set()
            for line in result.stdout.splitlines():
                if (
                    line.strip()
                    and package in line
                    and not line.startswith("Installed")
                ):
                    parts = line.split()
                    if len(parts) >= 2:
                        version = clean_version(parts[1])
                        if version not in seen:
                            versions.append(version)
                            seen.add(version)
            if versions:
                return sorted(versions, key=self._version_sort_key, reverse=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to get versions for {package}: {e}")

        try:
            result = subprocess.run(
                [self._dnf_path, "info", package],
                capture_output=True,
                text=True,
                timeout=30,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version"):
                    version = clean_version(line.split(":")[1].strip())
                    return [version]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        return []

    def _version_sort_key(self, version: str) -> tuple:
        """Sort key for version strings."""
        import re

        nums = re.findall(r"\d+", version)
        return tuple(int(n) for n in nums[:3]) if nums else (0, 0, 0)
