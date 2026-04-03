"""DNF package manager implementation for Fedora/RHEL."""

import subprocess

from nvidia_inst.distro.package_manager import PackageManager, PackageManagerError
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class DnfManager(PackageManager):
    """DNF package manager for Fedora/RHEL."""

    def __init__(self) -> None:
        self._dnf_path = "/usr/bin/dnf"
        self._dnf_version = self._detect_dnf_version()

    def _detect_dnf_version(self) -> str:
        """Detect if running dnf4 or dnf5.

        Returns:
            'dnf4' or 'dnf5'
        """
        try:
            result = subprocess.run(
                [self._dnf_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "dnf5" in result.stdout.lower() or "dnf5" in result.stderr.lower():
                logger.info("Detected DNF5")
                return "dnf5"
            logger.info("Detected DNF4")
            return "dnf4"
        except Exception as e:
            logger.warning(f"Could not detect DNF version, defaulting to dnf4: {e}")
            return "dnf4"

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
        """Pin package to version using versionlock.toml.

        Uses direct TOML file editing for pattern-based branch locking.
        Allows minor/bugfix updates within the branch while blocking major version changes.

        Args:
            package: Package name (e.g., 'akmod-nvidia', 'cuda-toolkit').
            version: Version pattern. Use '580.*' to lock to 580 branch,
                     '12.*' to lock to CUDA 12.x, 'exact.version' for exact.

        Returns:
            True if successful, False otherwise.
        """
        from nvidia_inst.distro.versionlock import add_pattern_versionlock_entry

        # Extract major version from pattern (e.g., "580.*" -> "580", "12.*" -> "12")
        major_version = version.split(".")[0]

        # Validate major version is a number
        if not major_version.isdigit():
            logger.error(
                f"Invalid version pattern '{version}' for {package}: "
                f"major version '{major_version}' is not numeric"
            )
            return False

        success, msg = add_pattern_versionlock_entry(
            package_name=package,
            major_version=major_version,
            comment=f"nvidia-inst: Lock {package} to {major_version}.x",
        )

        if success:
            logger.info(f"Locked {package} to {major_version}.x via versionlock.toml")
        else:
            logger.error(f"Failed to lock {package}: {msg}")

        return success

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
