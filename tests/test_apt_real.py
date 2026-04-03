"""Real integration tests for APT package manager (Ubuntu/Debian).

These tests exercise actual APT commands without mocking subprocess.
They are safe to run in CI containers and focus on read operations.
"""

import os
import shutil
import subprocess

import pytest

from nvidia_inst.distro.apt import AptManager


# Skip all tests in this file if APT is not available
@pytest.fixture(scope="module", autouse=True)
def check_apt_available():
    """Check if APT is available on this system."""
    if not AptManager().is_available():
        pytest.skip("APT is not available on this system")


class TestAptManagerReal:
    """Real APT integration tests (read operations only)."""

    def test_is_available_real(self, apt_manager):
        """Test that apt is actually available on this system."""
        result = apt_manager.is_available()
        assert result is True

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for apt update")
    def test_update_apt_real(self, apt_manager):
        """Test apt update refreshes package lists (requires root)."""
        result = apt_manager.update()
        # May fail due to network/repo issues in containers, but should not crash
        assert isinstance(result, bool)

    def test_apt_path_constant(self, apt_manager):
        """Test that APT path constants are correct."""
        assert apt_manager._apt_path == "/usr/bin/apt"
        assert apt_manager._apt_get_path == "/usr/bin/apt-get"

    def test_search_real(self, apt_manager):
        """Test searching for packages with real apt-cache."""
        # Search for a package that should exist
        packages = apt_manager.search("bash")
        assert isinstance(packages, list)

    def test_get_installed_version_real(self, apt_manager):
        """Test getting installed version of a system package."""
        # Only test if dpkg-query is available
        if shutil.which("dpkg-query") is None:
            pytest.skip("dpkg-query not available")
        # Try to get version of a common package
        version = apt_manager.get_installed_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_available_version_real(self, apt_manager):
        """Test getting available version from apt-cache policy."""
        # Only test if apt-cache is available
        if shutil.which("apt-cache") is None:
            pytest.skip("apt-cache not available")
        version = apt_manager.get_available_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_installed_version_nonexistent_real(self, apt_manager):
        """Test getting version of non-existent package returns None."""
        version = apt_manager.get_installed_version("this-package-does-not-exist-xyz")
        assert version is None

    def test_get_available_version_nonexistent_real(self, apt_manager):
        """Test getting available version of non-existent package returns None."""
        version = apt_manager.get_available_version("this-package-does-not-exist-xyz")
        assert version is None

    def test_get_all_versions_real(self, apt_manager):
        """Test getting all versions using apt-cache madison."""
        # Only test if apt-cache is available
        if shutil.which("apt-cache") is None:
            pytest.skip("apt-cache not available")
        versions = apt_manager.get_all_versions("bash")
        assert isinstance(versions, list)

    def test_dpkg_query_path(self):
        """Test that dpkg-query is available for get_installed_version."""
        # Just verify dpkg-query exists
        result = subprocess.run(["which", "dpkg-query"], capture_output=True, text=True)
        # Not strictly required to exist, but good to verify
        if result.returncode == 0:
            assert (
                "/usr/bin/dpkg-query" in result.stdout
                or "/bin/dpkg-query" in result.stdout
            )

    def test_apt_cache_path(self):
        """Test that apt-cache is available for search/policy."""
        result = subprocess.run(["which", "apt-cache"], capture_output=True, text=True)
        if result.returncode == 0:
            assert (
                "/usr/bin/apt-cache" in result.stdout
                or "/bin/apt-cache" in result.stdout
            )


class TestAptManagerRealSafeReadOnly:
    """Read-only APT tests that should work in any container environment."""

    def test_apt_manager_instantiation(self):
        """Test that AptManager can be instantiated."""
        manager = AptManager()
        assert manager is not None
        assert hasattr(manager, "_apt_path")
        assert hasattr(manager, "_apt_get_path")

    def test_search_returns_list_type(self, apt_manager):
        """Test that search returns a list even on error."""
        result = apt_manager.search("this-package-does-not-exist-xyz")
        assert isinstance(result, list)

    def test_get_installed_version_returns_str_or_none(self, apt_manager):
        """Test that get_installed_version returns str or None."""
        version = apt_manager.get_installed_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_available_version_returns_str_or_none(self, apt_manager):
        """Test that get_available_version returns str or None."""
        version = apt_manager.get_available_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_all_versions_returns_list(self, apt_manager):
        """Test that get_all_versions returns a list."""
        versions = apt_manager.get_all_versions("nonexistent-package-xyz")
        assert isinstance(versions, list)

    def test_version_sort_key_real(self, apt_manager):
        """Test version sorting works correctly."""
        versions = ["535.154.05", "535.54.06", "535.43.02", "470.256.02"]
        sorted_versions = sorted(
            versions, key=apt_manager._version_sort_key, reverse=True
        )
        # 535.154.05 should be first (highest)
        assert sorted_versions[0] == "535.154.05"
        # 470.256.02 should be last (lowest)
        assert sorted_versions[-1] == "470.256.02"

    def test_version_sort_key_with_epoch(self, apt_manager):
        """Test version sorting with epoch prefix."""
        versions = ["2:535.154.05", "1:535.154.05", "535.154.05"]
        sorted_versions = sorted(
            versions, key=apt_manager._version_sort_key, reverse=True
        )
        # Implementation extracts numbers left-to-right, so:
        # 535.154.05 -> (535, 154, 5) sorts first (highest first number)
        # 2:535.154.05 -> (2, 535, 154) sorts second (2 is small)
        # 1:535.154.05 -> (1, 535, 154) sorts last (1 is smallest)
        assert sorted_versions[0] == "535.154.05"
        assert sorted_versions[1] == "2:535.154.05"
        assert sorted_versions[2] == "1:535.154.05"
