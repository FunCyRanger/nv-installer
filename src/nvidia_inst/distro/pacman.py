"""Pacman package manager implementation for Arch Linux."""

import subprocess

from nvidia_inst.distro.package_manager import PackageManager, PackageManagerError
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class PacmanManager(PackageManager):
    """Pacman package manager for Arch Linux."""

    def __init__(self) -> None:
        self._pacman_path = "/usr/bin/pacman"

    def update(self) -> bool:
        """Update package lists using pacman."""
        try:
            subprocess.run(
                [self._pacman_path, "-Sy"],
                check=True,
                capture_output=True,
            )
            logger.info("Package lists updated")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update package lists: {e.stderr}")
            return False

    def upgrade(self) -> bool:
        """Upgrade all packages using pacman."""
        try:
            subprocess.run(
                [self._pacman_path, "-Syu", "--noconfirm"],
                check=True,
                capture_output=True,
            )
            logger.info("Packages upgraded")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade packages: {e.stderr}")
            return False

    def install(self, packages: list[str]) -> bool:
        """Install packages using pacman."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._pacman_path, "-S", "--noconfirm"] + packages
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
        """Remove packages using pacman."""
        from nvidia_inst.utils.permissions import is_root

        try:
            cmd = [self._pacman_path, "-R", "--noconfirm"] + packages
            if not is_root():
                cmd = ["sudo"] + cmd
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Removed packages: {', '.join(packages)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove packages: {e.stderr}")
            return False

    def search(self, query: str) -> list[str]:
        """Search for packages using pacman."""
        try:
            result = subprocess.run(
                [self._pacman_path, "-Ss", query],
                check=True,
                capture_output=True,
                text=True,
            )
            packages = []
            for line in result.stdout.splitlines():
                if line.startswith("community/") or line.startswith("extra/"):
                    pkg_name = line.split("/")[1].split(" ")[0]
                    if pkg_name not in packages:
                        packages.append(pkg_name)
            return packages
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to search packages: {e.stderr}")
            return []

    def is_available(self) -> bool:
        """Check if pacman is available."""
        import shutil

        return shutil.which(self._pacman_path) is not None

    def get_installed_version(self, package: str) -> str | None:
        """Get installed version of a package."""
        try:
            result = subprocess.run(
                [self._pacman_path, "-Q", package],
                check=True,
                capture_output=True,
                text=True,
            )
            parts = result.stdout.split()
            if len(parts) >= 2:
                return parts[1]
            return None
        except subprocess.CalledProcessError:
            return None

    def get_available_version(self, package: str) -> str | None:
        """Get available version of a package."""
        try:
            result = subprocess.run(
                [self._pacman_path, "-Si", package],
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
        """Pin package to version using pacman.

        Arch Linux uses branch-specific packages instead of version locking:
        - nvidia-open (latest, for Turing+ GPUs)
        - nvidia-580xx-dkms (for Maxwell/Pascal/Volta - AUR)
        - nvidia-470xx-dkms (for Kepler - AUR)

        No version locking is needed because branch packages automatically
        stay on the correct branch.

        Args:
            package: Package name (should be branch-specific, e.g., "nvidia-580xx-dkms")
            version: Not used for Arch (branch is in package name)

        Returns:
            True (no locking needed for branch packages)
        """
        logger.info(f"Arch: Using branch package {package} (no version lock needed)")
        return True

    def get_branch_package(self, branch: str) -> str:
        """Get the correct branch-specific package name for Arch Linux.

        Args:
            branch: Driver branch (e.g., "580", "470")

        Returns:
            Branch-specific package name
        """
        # For latest hardware (Turing+), use rolling package
        # For older hardware, use branch-specific packages
        branch_packages = {
            "595": "nvidia-open",  # Latest (Blackwell)
            "590": "nvidia-open",  # Latest (Turing+)
            "580": "nvidia-580xx-dkms",  # Maxwell/Pascal/Volta
            "470": "nvidia-470xx-dkms",  # Kepler
        }
        return branch_packages.get(branch, "nvidia-open")

    def get_all_versions(self, package: str) -> list[str]:
        """Get available nvidia driver branch packages for Arch Linux.

        For Arch, nvidia drivers are delivered as branch packages, not versioned.
        This returns the available branch packages that match the base package.
        """
        branch_packages = {
            "nvidia": ["nvidia", "nvidia-utils"],
            "nvidia-470xx": ["nvidia-470xx", "nvidia-470xx-utils"],
            "nvidia-535xx": ["nvidia-535xx", "nvidia-535xx-utils"],
            "nvidia-550xx": ["nvidia-550xx", "nvidia-550xx-utils"],
            "akmod-nvidia": ["akmod-nvidia"],
            "akmod-nvidia-470xx": ["akmod-nvidia-470xx"],
            "akmod-nvidia-535xx": ["akmod-nvidia-535xx"],
        }

        available_branches = []

        for branch, pkgs in branch_packages.items():
            for pkg in pkgs:
                if self._package_exists(pkg):
                    available_branches.append(branch)
                    break

        return sorted(set(available_branches))

    def _package_exists(self, package: str) -> bool:
        """Check if a package exists in the repos."""
        try:
            result = subprocess.run(
                [self._pacman_path, "-Si", package],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
