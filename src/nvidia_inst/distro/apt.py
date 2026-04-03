"""APT package manager implementation for Debian/Ubuntu."""

import os
import subprocess

from nvidia_inst.distro.package_manager import PackageManager, PackageManagerError
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class AptManager(PackageManager):
    """APT package manager for Debian/Ubuntu."""

    def __init__(self) -> None:
        self._apt_path = "/usr/bin/apt"
        self._apt_get_path = "/usr/bin/apt-get"

    def update(self) -> bool:
        """Update package lists using apt."""
        try:
            subprocess.run(
                [self._apt_path, "update"],
                check=True,
                capture_output=True,
            )
            logger.info("Package lists updated")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update package lists: {e.stderr}")
            return False

    def upgrade(self) -> bool:
        """Upgrade all packages using apt."""
        try:
            subprocess.run(
                [self._apt_path, "upgrade", "-y"],
                check=True,
                capture_output=True,
            )
            logger.info("Packages upgraded")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade packages: {e.stderr}")
            return False

    def install(self, packages: list[str]) -> bool:
        """Install packages using apt."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._apt_path, "install", "-y"] + packages
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
        """Remove packages using apt."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._apt_path, "remove", "-y"] + packages
            if not is_root():
                cmd = ["sudo"] + cmd
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Removed packages: {', '.join(packages)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove packages: {e.stderr}")
            return False

    def search(self, query: str) -> list[str]:
        """Search for packages using apt-cache."""
        try:
            result = subprocess.run(
                ["apt-cache", "search", query],
                check=True,
                capture_output=True,
                text=True,
            )
            packages = []
            for line in result.stdout.splitlines():
                if line:
                    pkg_name = line.split(" - ")[0].strip()
                    packages.append(pkg_name)
            return packages
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to search packages: {e.stderr}")
            return []

    def is_available(self) -> bool:
        """Check if apt is available."""
        import shutil

        return shutil.which(self._apt_path) is not None

    def get_installed_version(self, package: str) -> str | None:
        """Get installed version of a package."""
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Version}", package],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def get_available_version(self, package: str) -> str | None:
        """Get available version of a package."""
        try:
            result = subprocess.run(
                ["apt-cache", "policy", package],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.splitlines()
            for line in lines:
                if "Candidate:" in line:
                    version = line.split(":")[1].strip()
                    if version != "(none)":
                        return version
            return None
        except subprocess.CalledProcessError:
            return None

    def pin_version(self, package: str, version: str = "*") -> bool:
        """Pin package to version using apt preferences.

        Args:
            package: Package name (e.g., 'nvidia-driver-580*', 'cuda-toolkit-12*').
            version: Version pattern. Use '580.*' to lock to 580 branch,
                     '12.*' to lock to CUDA 12.x, 'exact.version' for exact version.

        Returns:
            True if successful, False otherwise.
        """
        import re
        import tempfile

        # Sanitize filename (remove * and special chars)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", package)
        pin_file = f"/etc/apt/preferences.d/nvidia-inst-{safe_name}"

        content = f"""# nvidia-inst: Pin {package} to {version}
Package: {package}
Pin: version {version}
Pin-Priority: 1001
"""

        try:
            # Write to temp file first
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".pref"
            ) as f:
                f.write(content)
                temp_path = f.name

            # Copy to destination with sudo
            result = subprocess.run(
                ["sudo", "cp", temp_path, pin_file],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Clean up temp file
            os.unlink(temp_path)

            if result.returncode == 0:
                logger.info(f"Pinned {package} to pattern {version} in {pin_file}")
                return True
            else:
                logger.error(f"Failed to create pin file: {result.stderr}")
                return False

        except PermissionError:
            logger.error(f"Permission denied to create {pin_file}")
            return False
        except OSError as e:
            logger.error(f"Failed to pin version: {e}")
            return False

    def get_all_versions(self, package: str) -> list[str]:
        """Get all available versions of a package using apt-cache madison."""
        try:
            result = subprocess.run(
                ["apt-cache", "madison", package],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            versions = []
            seen = set()
            for line in result.stdout.splitlines():
                if line.strip():
                    parts = line.split("|")
                    if len(parts) >= 2:
                        version = parts[1].strip()
                        if version not in seen:
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

    @property
    def tool(self) -> str:
        """Get the tool name."""
        return "apt"
