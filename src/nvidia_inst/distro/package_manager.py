"""Package manager abstraction for different Linux distributions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class PackageManagerError(Exception):
    """Raised when package manager operation fails."""
    pass


@dataclass
class PackageInfo:
    """Information about a package."""

    name: str
    version: str
    available_version: Optional[str] = None
    installed: bool = False


class PackageManager(ABC):
    """Abstract base class for package managers."""

    @abstractmethod
    def update(self) -> bool:
        """Update package lists.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    def upgrade(self) -> bool:
        """Upgrade all packages.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    def install(self, packages: list[str]) -> bool:
        """Install packages.

        Args:
            packages: List of package names to install.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    def remove(self, packages: list[str]) -> bool:
        """Remove packages.

        Args:
            packages: List of package names to remove.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """Search for packages.

        Args:
            query: Search query.

        Returns:
            List of matching package names.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if package manager is available.

        Returns:
            True if available, False otherwise.
        """
        ...

    @abstractmethod
    def get_installed_version(self, package: str) -> Optional[str]:
        """Get installed version of a package.

        Args:
            package: Package name.

        Returns:
            Version string if installed, None otherwise.
        """
        ...

    @abstractmethod
    def get_available_version(self, package: str) -> Optional[str]:
        """Get available version of a package.

        Args:
            package: Package name.

        Returns:
            Version string if available, None otherwise.
        """
        ...

    @abstractmethod
    def pin_version(self, package: str, version: str) -> bool:
        """Pin package to specific version.

        Args:
            package: Package name.
            version: Version to pin to.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    def get_all_versions(self, package: str) -> list[str]:
        """Get all available versions of a package.

        Args:
            package: Package name.

        Returns:
            List of all available version strings.
        """
        ...
