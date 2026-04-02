"""Offline installation support for air-gapped environments.

This module provides functionality for creating and using offline package caches
for environments without internet connectivity.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nvidia_inst.distro.detector import detect_distro
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CachedPackage:
    """Information about a cached package."""

    name: str
    version: str
    filename: str
    checksum: str
    size: int
    distro_id: str
    architecture: str


@dataclass
class OfflineManifest:
    """Manifest for offline package cache."""

    version: str = "1.0.0"
    created_at: str = ""
    distro_id: str = ""
    distro_version: str = ""
    architecture: str = ""
    packages: list[CachedPackage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class OfflineInstaller:
    """Installer for air-gapped environments."""

    def __init__(self, cache_dir: str = "/var/cache/nvidia-inst"):
        """Initialize offline installer.

        Args:
            cache_dir: Directory for cached packages
        """
        self.cache_dir = Path(cache_dir)
        self.manifest_file = self.cache_dir / "manifest.json"
        self.packages_dir = self.cache_dir / "packages"

    def create_cache(
        self,
        packages: list[str],
        distro_id: str | None = None,
    ) -> bool:
        """Download and cache packages for offline installation.

        Args:
            packages: List of package names to cache
            distro_id: Distribution ID (auto-detected if None)

        Returns:
            True if cache created successfully
        """
        try:
            if distro_id is None:
                distro = detect_distro()
                distro_id = distro.id

            # Create cache directories
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.packages_dir.mkdir(parents=True, exist_ok=True)

            # Get package manager
            pkg_manager = get_package_manager()

            # Update package lists
            logger.info("Updating package lists...")
            if not pkg_manager.update():
                logger.error("Failed to update package lists")
                return False

            # Download packages
            manifest = OfflineManifest(
                distro_id=distro_id,
                distro_version=(
                    distro.version_id if hasattr(distro, "version_id") else ""
                ),
                architecture=self._get_architecture(),
            )

            for pkg_name in packages:
                logger.info(f"Downloading {pkg_name}...")

                # Get package info
                version = pkg_manager.get_available_version(pkg_name)
                if not version:
                    logger.warning(f"Package {pkg_name} not found, skipping")
                    continue

                # Download package
                pkg_file = self._download_package(pkg_name, version, pkg_manager)
                if not pkg_file:
                    logger.error(f"Failed to download {pkg_name}")
                    continue

                # Calculate checksum
                checksum = self._calculate_checksum(pkg_file)

                # Add to manifest
                cached_pkg = CachedPackage(
                    name=pkg_name,
                    version=version,
                    filename=pkg_file.name,
                    checksum=checksum,
                    size=pkg_file.stat().st_size,
                    distro_id=distro_id,
                    architecture=self._get_architecture(),
                )
                manifest.packages.append(cached_pkg)
                logger.info(f"  Cached {pkg_name} version {version}")

            # Save manifest
            self._save_manifest(manifest)
            logger.info(f"Offline cache created with {len(manifest.packages)} packages")

            return len(manifest.packages) > 0

        except Exception as e:
            logger.error(f"Failed to create offline cache: {e}")
            return False

    def install_from_cache(
        self,
        packages: list[str],
        verify_checksums: bool = True,
    ) -> tuple[bool, list[str]]:
        """Install packages from local cache.

        Args:
            packages: List of package names to install
            verify_checksums: Verify package checksums before installing

        Returns:
            Tuple of (success, list_of_installed_packages)
        """
        try:
            manifest = self._load_manifest()
            if not manifest:
                logger.error("No offline cache found")
                return False, []

            if not self.verify_cache_integrity():
                logger.error("Cache integrity check failed")
                return False, []

            # Find requested packages in manifest
            packages_to_install = []
            installed_packages = []

            for pkg_name in packages:
                cached_pkg = self._find_cached_package(manifest, pkg_name)
                if not cached_pkg:
                    logger.error(f"Package {pkg_name} not found in cache")
                    continue

                # Verify checksum if requested
                if verify_checksums:
                    pkg_file = self.packages_dir / cached_pkg.filename
                    if not self._verify_package_checksum(pkg_file, cached_pkg.checksum):
                        logger.error(f"Checksum mismatch for {pkg_name}")
                        continue

                packages_to_install.append(cached_pkg)

            if not packages_to_install:
                logger.error("No valid packages to install")
                return False, []

            # Install packages
            pkg_manager = get_package_manager()

            for cached_pkg in packages_to_install:
                pkg_file = self.packages_dir / cached_pkg.filename
                logger.info(
                    f"Installing {cached_pkg.name} version {cached_pkg.version}..."
                )

                if self._install_package_file(pkg_file, pkg_manager):
                    installed_packages.append(cached_pkg.name)
                    logger.info(f"  Installed {cached_pkg.name}")
                else:
                    logger.error(f"  Failed to install {cached_pkg.name}")

            success = len(installed_packages) == len(packages)
            return success, installed_packages

        except Exception as e:
            logger.error(f"Failed to install from cache: {e}")
            return False, installed_packages

    def verify_cache_integrity(self) -> bool:
        """Verify cached packages have valid checksums.

        Returns:
            True if all cached packages are valid
        """
        try:
            manifest = self._load_manifest()
            if not manifest:
                logger.error("No manifest found")
                return False

            logger.info(f"Verifying {len(manifest.packages)} cached packages...")

            for cached_pkg in manifest.packages:
                pkg_file = self.packages_dir / cached_pkg.filename

                if not pkg_file.exists():
                    logger.error(f"Missing package file: {cached_pkg.filename}")
                    return False

                if not self._verify_package_checksum(pkg_file, cached_pkg.checksum):
                    logger.error(f"Checksum mismatch: {cached_pkg.filename}")
                    return False

            logger.info("Cache integrity verified successfully")
            return True

        except Exception as e:
            logger.error(f"Cache verification failed: {e}")
            return False

    def get_cached_packages(self) -> list[str]:
        """Get list of packages in cache.

        Returns:
            List of cached package names
        """
        manifest = self._load_manifest()
        if not manifest:
            return []
        return [pkg.name for pkg in manifest.packages]

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the offline cache.

        Returns:
            Dictionary with cache information
        """
        manifest = self._load_manifest()
        if not manifest:
            return {"exists": False}

        total_size = sum(pkg.size for pkg in manifest.packages)

        return {
            "exists": True,
            "created_at": manifest.created_at,
            "distro_id": manifest.distro_id,
            "distro_version": manifest.distro_version,
            "architecture": manifest.architecture,
            "package_count": len(manifest.packages),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "packages": [pkg.name for pkg in manifest.packages],
        }

    def _load_manifest(self) -> OfflineManifest | None:
        """Load manifest from cache directory."""
        if not self.manifest_file.exists():
            return None

        try:
            with open(self.manifest_file) as f:
                data = json.load(f)

            packages = [
                CachedPackage(**pkg_data) for pkg_data in data.get("packages", [])
            ]

            return OfflineManifest(
                version=data.get("version", "1.0.0"),
                created_at=data.get("created_at", ""),
                distro_id=data.get("distro_id", ""),
                distro_version=data.get("distro_version", ""),
                architecture=data.get("architecture", ""),
                packages=packages,
            )
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return None

    def _save_manifest(self, manifest: OfflineManifest) -> None:
        """Save manifest to cache directory."""
        data = {
            "version": manifest.version,
            "created_at": manifest.created_at,
            "distro_id": manifest.distro_id,
            "distro_version": manifest.distro_version,
            "architecture": manifest.architecture,
            "packages": [asdict(pkg) for pkg in manifest.packages],
        }

        with open(self.manifest_file, "w") as f:
            json.dump(data, f, indent=2)

    def _find_cached_package(
        self,
        manifest: OfflineManifest,
        pkg_name: str,
    ) -> CachedPackage | None:
        """Find a package in the manifest."""
        for pkg in manifest.packages:
            if pkg.name == pkg_name:
                return pkg
        return None

    def _download_package(
        self,
        pkg_name: str,
        version: str,
        pkg_manager: Any,
    ) -> Path | None:
        """Download a package to the cache."""
        # Implementation depends on package manager type
        # This is a simplified version
        try:
            # Use apt-get download or dnf download or similar
            import subprocess

            # For APT-based systems
            result = subprocess.run(
                ["apt-get", "download", pkg_name],
                cwd=str(self.packages_dir),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Find downloaded file
                for f in self.packages_dir.glob("*.deb"):
                    if pkg_name in f.name:
                        return f

            return None

        except Exception as e:
            logger.error(f"Failed to download {pkg_name}: {e}")
            return None

    def _install_package_file(
        self,
        pkg_file: Path,
        pkg_manager: Any,
    ) -> bool:
        """Install a package from file."""
        try:
            import subprocess

            # For APT-based systems
            result = subprocess.run(
                ["dpkg", "-i", str(pkg_file)],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to install package file: {e}")
            return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _verify_package_checksum(self, pkg_file: Path, expected_checksum: str) -> bool:
        """Verify package checksum matches expected value."""
        if not pkg_file.exists():
            return False
        actual_checksum = self._calculate_checksum(pkg_file)
        return actual_checksum == expected_checksum

    def _get_architecture(self) -> str:
        """Get system architecture."""
        import subprocess

        try:
            result = subprocess.run(
                ["dpkg", "--print-architecture"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Fallback
        import platform

        return platform.machine()
