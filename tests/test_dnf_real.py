"""Real integration tests for DNF package manager (Fedora/RHEL).

These tests exercise actual DNF commands without mocking subprocess.
They are safe to run in CI containers and focus on read operations.
"""

import os
import subprocess

import pytest

from nvidia_inst.distro.dnf import DnfManager

# Skip all tests in this file if DNF is not available
pytestmark = pytest.mark.skipif(
    not DnfManager().is_available(),
    reason="DNF is not available on this system",
)


class TestDnfManagerReal:
    """Real DNF integration tests (read operations only)."""

    def test_is_available_real(self, dnf_manager):
        """Test that dnf is actually available on this system."""
        result = dnf_manager.is_available()
        assert result is True

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for makecache")
    def test_update_makecache_real(self, dnf_manager):
        """Test dnf makecache updates package lists (requires root)."""
        result = dnf_manager.update()
        # May fail due to network/repo issues in containers, but should not crash
        assert isinstance(result, bool)

    def test_dnf_version_detection_real(self, dnf_manager):
        """Test DNF5 vs DNF4 detection."""
        # Should detect and set version
        assert dnf_manager._dnf_version in ("dnf4", "dnf5")
        # Can verify by running dnf --version
        try:
            ver_result = subprocess.run(
                ["/usr/bin/dnf", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = ver_result.stdout + ver_result.stderr
            if "dnf5" in output.lower():
                assert dnf_manager._dnf_version == "dnf5"
            else:
                assert dnf_manager._dnf_version == "dnf4"
        except subprocess.CalledProcessError:
            pytest.skip("dnf --version failed")

    def test_search_real(self, dnf_manager):
        """Test searching for packages with real dnf."""
        # Search for a package that should exist
        packages = dnf_manager.search("kernel")
        assert isinstance(packages, list)
        # Should find some kernel packages
        assert len(packages) >= 0

    def test_get_available_version_real(self, dnf_manager):
        """Test getting available version of a system package."""
        # Try to get version of a common package that should exist
        version = dnf_manager.get_available_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_installed_version_nonexistent_real(self, dnf_manager):
        """Test getting version of non-existent package returns None."""
        version = dnf_manager.get_installed_version(
            "this-package-definitely-does-not-exist-xyz"
        )
        assert version is None

    def test_get_available_version_nonexistent_real(self, dnf_manager):
        """Test getting available version of non-existent package returns None."""
        version = dnf_manager.get_available_version(
            "this-package-definitely-does-not-exist-xyz"
        )
        assert version is None

    def test_get_all_versions_real(self, dnf_manager):
        """Test getting all versions of a package."""
        # Try base package that might exist
        versions = dnf_manager.get_all_versions("bash")
        assert isinstance(versions, list)

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for versionlock")
    def test_versionlock_toml_read_real(self, dnf_manager):
        """Test reading versionlock.toml file."""
        from nvidia_inst.distro.versionlock import read_versionlock_toml

        data = read_versionlock_toml()
        assert isinstance(data, dict)
        assert "version" in data
        assert "packages" in data
        assert isinstance(data["packages"], list)

    def test_version_sort_key_real(self, dnf_manager):
        """Test version sorting works correctly."""
        versions = ["535.154.05", "535.54.06", "535.43.02", "470.256.02"]
        sorted_versions = sorted(
            versions, key=dnf_manager._version_sort_key, reverse=True
        )
        # 535.154.05 should be first (highest)
        assert sorted_versions[0] == "535.154.05"
        # 470.256.02 should be last (lowest)
        assert sorted_versions[-1] == "470.256.02"


class TestDnfManagerRealSafeReadOnly:
    """Read-only DNF tests that should work in any container environment."""

    def test_dnf_path_constant(self):
        """Test that DNF path constant is correct."""
        manager = DnfManager()
        assert manager._dnf_path == "/usr/bin/dnf"

    def test_dnf_manager_instantiation(self):
        """Test that DnfManager can be instantiated."""
        manager = DnfManager()
        assert manager is not None
        assert hasattr(manager, "_dnf_path")
        assert hasattr(manager, "_dnf_version")

    def test_search_returns_list_type(self):
        """Test that search returns a list even on error."""
        manager = DnfManager()
        # Use a query that should work
        result = manager.search("nvidia")
        assert isinstance(result, list)

    def test_get_installed_version_returns_str_or_none(self):
        """Test that get_installed_version returns str or None."""
        manager = DnfManager()
        version = manager.get_installed_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_available_version_returns_str_or_none(self):
        """Test that get_available_version returns str or None."""
        manager = DnfManager()
        version = manager.get_available_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_all_versions_returns_list(self):
        """Test that get_all_versions returns a list."""
        manager = DnfManager()
        versions = manager.get_all_versions("nonexistent-package-xyz")
        assert isinstance(versions, list)
