"""E2E offline installation tests.

These tests verify offline cache operations with real filesystem
operations, including cache creation, verification, and installation
logic.
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.installer.offline import (
    CachedPackage,
    OfflineInstaller,
    OfflineManifest,
)


def is_fedora_container() -> bool:
    """Check if running in a Fedora container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            return "fedora" in f.read().lower()
    return False


def is_ubuntu_container() -> bool:
    """Check if running in an Ubuntu container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            content = f.read().lower()
            return "ubuntu" in content
    return False


def has_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0


# ---------------------------------------------------------------------------
# OfflineManifest tests
# ---------------------------------------------------------------------------


class TestOfflineManifestE2E:
    """E2E tests for OfflineManifest with real filesystem."""

    def test_manifest_auto_timestamp(self, tmp_path):
        """Test that manifest gets auto-generated timestamp."""
        manifest = OfflineManifest(
            distro_id="ubuntu",
            distro_version="24.04",
            architecture="amd64",
        )
        assert manifest.created_at != ""
        assert "T" in manifest.created_at  # ISO format

    def test_manifest_serialization_roundtrip(self, tmp_path):
        """Test manifest can be serialized and deserialized."""
        packages = [
            CachedPackage(
                name="nvidia-driver-590",
                version="590.48.01",
                filename="nvidia-driver-590_590.48.01.deb",
                checksum="abc123",
                size=1024,
                distro_id="ubuntu",
                architecture="amd64",
            )
        ]

        manifest = OfflineManifest(
            distro_id="ubuntu",
            distro_version="24.04",
            architecture="amd64",
            packages=packages,
        )

        # Serialize
        data = {
            "version": manifest.version,
            "created_at": manifest.created_at,
            "distro_id": manifest.distro_id,
            "distro_version": manifest.distro_version,
            "architecture": manifest.architecture,
            "packages": [
                {
                    "name": p.name,
                    "version": p.version,
                    "filename": p.filename,
                    "checksum": p.checksum,
                    "size": p.size,
                    "distro_id": p.distro_id,
                    "architecture": p.architecture,
                }
                for p in manifest.packages
            ],
        }

        # Write to file
        manifest_file = tmp_path / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(data, f)

        # Read back
        with open(manifest_file) as f:
            loaded = json.load(f)

        assert loaded["distro_id"] == "ubuntu"
        assert len(loaded["packages"]) == 1
        assert loaded["packages"][0]["name"] == "nvidia-driver-590"


# ---------------------------------------------------------------------------
# OfflineInstaller cache directory tests
# ---------------------------------------------------------------------------


class TestOfflineInstallerCacheDir:
    """E2E tests for OfflineInstaller cache directory operations."""

    def test_cache_dir_created_on_cache_creation(self, tmp_path):
        """Test that cache directories are created during create_cache."""
        cache_dir = tmp_path / "cache"
        installer = OfflineInstaller(cache_dir=str(cache_dir))

        # Directories should not exist yet
        assert not cache_dir.exists()

        # create_cache will create them (even if it fails to download)
        with patch.object(installer, "_download_package", return_value=None):
            with patch("nvidia_inst.installer.offline.get_package_manager") as mock_pm:
                mock_pm.return_value.update.return_value = True
                installer.create_cache(
                    packages=["test-package"],
                    distro_id="ubuntu",
                )

        # Directories should now exist
        assert cache_dir.exists()
        assert (cache_dir / "packages").exists()

    def test_cache_dir_custom_path(self, tmp_path):
        """Test custom cache directory path."""
        custom_dir = tmp_path / "custom" / "nvidia-cache"
        installer = OfflineInstaller(cache_dir=str(custom_dir))
        assert installer.cache_dir == custom_dir
        assert installer.manifest_file == custom_dir / "manifest.json"
        assert installer.packages_dir == custom_dir / "packages"

    @pytest.mark.skipif("not has_root()")
    def test_cache_dir_default_path(self):
        """Test default cache directory path (/var/cache/nvidia-inst)."""
        installer = OfflineInstaller()
        assert installer.cache_dir == Path("/var/cache/nvidia-inst")


# ---------------------------------------------------------------------------
# create_cache() tests
# ---------------------------------------------------------------------------


class TestCreateCache:
    """E2E tests for create_cache() with real package managers."""

    def test_create_cache_no_packages(self, tmp_path):
        """Test create_cache with empty package list."""
        cache_dir = tmp_path / "cache"
        installer = OfflineInstaller(cache_dir=str(cache_dir))

        result = installer.create_cache(
            packages=[],
            distro_id="ubuntu",
        )

        # No packages means no cached packages
        assert result is False

    @pytest.mark.skipif("not is_ubuntu_container()")
    def test_create_cache_ubuntu_real_pkg_mgr(self, tmp_path):
        """Test create_cache on Ubuntu with real package manager."""
        cache_dir = tmp_path / "cache"
        installer = OfflineInstaller(cache_dir=str(cache_dir))

        # Try to cache a package that should exist on Ubuntu
        # This will fail to download in a container without repos,
        # but verifies the workflow doesn't crash
        result = installer.create_cache(
            packages=["curl"],
            distro_id="ubuntu",
        )

        # Result depends on whether repos are configured
        assert isinstance(result, bool)

    @pytest.mark.skipif("not is_fedora_container()")
    def test_create_cache_fedora_real_pkg_mgr(self, tmp_path):
        """Test create_cache on Fedora with real package manager."""
        cache_dir = tmp_path / "cache"
        installer = OfflineInstaller(cache_dir=str(cache_dir))

        result = installer.create_cache(
            packages=["curl"],
            distro_id="fedora",
        )

        assert isinstance(result, bool)

    def test_create_cache_with_mocked_pkg_mgr(self, tmp_path):
        """Test create_cache with mocked package manager."""
        cache_dir = tmp_path / "cache"
        installer = OfflineInstaller(cache_dir=str(cache_dir))

        mock_pkg_mgr = MagicMock()
        mock_pkg_mgr.update.return_value = True
        mock_pkg_mgr.get_available_version.return_value = "1.0.0"

        # Create a dummy package file
        packages_dir = cache_dir / "packages"
        packages_dir.mkdir(parents=True)
        pkg_file = packages_dir / "test-package_1.0.0.deb"
        pkg_file.write_bytes(b"dummy package")

        with patch(
            "nvidia_inst.installer.offline.get_package_manager",
            return_value=mock_pkg_mgr,
        ):
            with patch("nvidia_inst.installer.offline.detect_distro") as mock_detect:
                mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
                with patch.object(
                    installer,
                    "_download_package",
                    return_value=pkg_file,
                ):
                    # Don't pass distro_id so detect_distro is called
                    result = installer.create_cache(
                        packages=["test-package"],
                    )

        assert result is True
        # Verify manifest was created
        assert installer.manifest_file.exists()


# ---------------------------------------------------------------------------
# verify_cache() tests
# ---------------------------------------------------------------------------


class TestVerifyCache:
    """E2E tests for verify_cache_integrity() with real filesystem."""

    def test_verify_empty_cache(self, tmp_path):
        """Test verify_cache_integrity on empty cache."""
        installer = OfflineInstaller(cache_dir=str(tmp_path))
        result = installer.verify_cache_integrity()
        assert result is False

    def test_verify_valid_cache(self, tmp_path):
        """Test verify_cache_integrity with valid cache."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        content = b"test package content"
        pkg_file = packages_dir / "test_1.0.deb"
        pkg_file.write_bytes(content)

        checksum = hashlib.sha256(content).hexdigest()

        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "test",
                    "version": "1.0",
                    "filename": "test_1.0.deb",
                    "checksum": checksum,
                    "size": len(content),
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                }
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        result = installer.verify_cache_integrity()
        assert result is True

    def test_verify_missing_package_file(self, tmp_path):
        """Test verify_cache_integrity with missing package file."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "test",
                    "version": "1.0",
                    "filename": "missing_1.0.deb",
                    "checksum": "abc123",
                    "size": 100,
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                }
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        result = installer.verify_cache_integrity()
        assert result is False

    def test_verify_corrupted_package(self, tmp_path):
        """Test verify_cache_integrity with corrupted package."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        # Write different content than what checksum expects
        pkg_file = packages_dir / "test_1.0.deb"
        pkg_file.write_bytes(b"corrupted content")

        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "test",
                    "version": "1.0",
                    "filename": "test_1.0.deb",
                    "checksum": "wrong_checksum",
                    "size": 100,
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                }
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        result = installer.verify_cache_integrity()
        assert result is False


# ---------------------------------------------------------------------------
# install_from_cache() tests
# ---------------------------------------------------------------------------


class TestInstallFromCache:
    """E2E tests for install_from_cache() logic."""

    def test_install_from_cache_no_manifest(self, tmp_path):
        """Test install_from_cache when no manifest exists."""
        installer = OfflineInstaller(cache_dir=str(tmp_path))
        success, installed = installer.install_from_cache(
            packages=["test-package"],
        )
        assert success is False
        assert installed == []

    def test_install_from_cache_package_not_in_cache(self, tmp_path):
        """Test install_from_cache when package not in cache."""
        # Create manifest with different package
        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "other-package",
                    "version": "1.0",
                    "filename": "other_1.0.deb",
                    "checksum": "abc123",
                    "size": 100,
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                }
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        success, installed = installer.install_from_cache(
            packages=["test-package"],
        )
        assert success is False
        assert installed == []

    def test_install_from_cache_integrity_check_fails(self, tmp_path):
        """Test install_from_cache when integrity check fails."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        # Create manifest with wrong checksum
        pkg_file = packages_dir / "test_1.0.deb"
        pkg_file.write_bytes(b"test content")

        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "test",
                    "version": "1.0",
                    "filename": "test_1.0.deb",
                    "checksum": "wrong_checksum",
                    "size": len(b"test content"),
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                }
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        success, installed = installer.install_from_cache(
            packages=["test"],
        )
        assert success is False


# ---------------------------------------------------------------------------
# get_cache_info() tests
# ---------------------------------------------------------------------------


class TestGetCacheInfo:
    """E2E tests for get_cache_info()."""

    def test_cache_info_no_cache(self, tmp_path):
        """Test get_cache_info when no cache exists."""
        installer = OfflineInstaller(cache_dir=str(tmp_path))
        info = installer.get_cache_info()
        assert info == {"exists": False}

    def test_cache_info_with_manifest(self, tmp_path):
        """Test get_cache_info with valid manifest."""
        content = b"package data"
        checksum = hashlib.sha256(content).hexdigest()

        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "nvidia-driver-590",
                    "version": "590.48.01",
                    "filename": "nvidia-driver-590_590.48.01.deb",
                    "checksum": checksum,
                    "size": len(content),
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                }
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        info = installer.get_cache_info()

        assert info["exists"] is True
        assert info["distro_id"] == "ubuntu"
        assert info["package_count"] == 1
        assert "total_size_mb" in info
        assert "created_at" in info


# ---------------------------------------------------------------------------
# get_cached_packages() tests
# ---------------------------------------------------------------------------


class TestGetCachedPackages:
    """E2E tests for get_cached_packages()."""

    def test_cached_packages_empty(self, tmp_path):
        """Test get_cached_packages when no cache exists."""
        installer = OfflineInstaller(cache_dir=str(tmp_path))
        packages = installer.get_cached_packages()
        assert packages == []

    def test_cached_packages_with_manifest(self, tmp_path):
        """Test get_cached_packages with manifest."""
        manifest_data = {
            "version": "1.0.0",
            "created_at": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "distro_version": "24.04",
            "architecture": "amd64",
            "packages": [
                {
                    "name": "nvidia-driver-590",
                    "version": "590.48.01",
                    "filename": "nvidia-driver-590_590.48.01.deb",
                    "checksum": "abc123",
                    "size": 1024,
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                },
                {
                    "name": "nvidia-dkms-590",
                    "version": "590.48.01",
                    "filename": "nvidia-dkms-590_590.48.01.deb",
                    "checksum": "def456",
                    "size": 2048,
                    "distro_id": "ubuntu",
                    "architecture": "amd64",
                },
            ],
        }

        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        installer = OfflineInstaller(cache_dir=str(tmp_path))
        packages = installer.get_cached_packages()

        assert len(packages) == 2
        assert "nvidia-driver-590" in packages
        assert "nvidia-dkms-590" in packages
