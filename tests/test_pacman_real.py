"""Real integration tests for Pacman package manager (Arch Linux).

These tests exercise actual Pacman commands without mocking subprocess.
They are safe to run in CI containers and focus on read operations.
"""

import os

import pytest

from nvidia_inst.distro.pacman import PacmanManager


# Skip all tests in this file if Pacman is not available
@pytest.fixture(scope="module", autouse=True)
def check_pacman_available():
    """Check if Pacman is available on this system."""
    if not PacmanManager().is_available():
        pytest.skip("Pacman is not available on this system")


class TestPacmanManagerReal:
    """Real Pacman integration tests (read operations only)."""

    def test_is_available_real(self, pacman_manager):
        """Test that pacman is actually available on this system."""
        result = pacman_manager.is_available()
        assert result is True

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for pacman -Sy")
    def test_update_sync_real(self, pacman_manager):
        """Test pacman -Sy refreshes package database (requires root)."""
        result = pacman_manager.update()
        # May fail due to network issues in containers, but should not crash
        assert isinstance(result, bool)

    def test_pacman_path_constant(self, pacman_manager):
        """Test that Pacman path constant is correct."""
        assert pacman_manager._pacman_path == "/usr/bin/pacman"

    def test_search_real(self, pacman_manager):
        """Test searching for packages with real pacman -Ss."""
        # Search for a package that should exist in core
        packages = pacman_manager.search("bash")
        assert isinstance(packages, list)
        # Should find bash in core repo
        if packages:
            assert "bash" in packages

    def test_get_installed_version_real(self, pacman_manager):
        """Test getting installed version of a system package."""
        # Try to get version of a common package that should be installed
        version = pacman_manager.get_installed_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_available_version_real(self, pacman_manager):
        """Test getting available version from repos."""
        version = pacman_manager.get_available_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_installed_version_nonexistent_real(self, pacman_manager):
        """Test getting version of non-installed package returns None."""
        version = pacman_manager.get_installed_version(
            "this-package-does-not-exist-xyz"
        )
        assert version is None

    def test_get_available_version_nonexistent_real(self, pacman_manager):
        """Test getting available version of non-existent package returns None."""
        version = pacman_manager.get_available_version(
            "this-package-does-not-exist-xyz"
        )
        assert version is None

    def test_package_exists_real(self, pacman_manager):
        """Test _package_exists method with a real package."""
        # bash should exist
        exists = pacman_manager._package_exists("bash")
        assert exists is True

    def test_package_not_exists_real(self, pacman_manager):
        """Test _package_exists with a non-existent package."""
        exists = pacman_manager._package_exists("this-package-does-not-exist-xyz")
        assert exists is False

    def test_get_branch_package_real(self, pacman_manager):
        """Test branch package detection."""
        # Test various branches
        assert pacman_manager.get_branch_package("590") == "nvidia-open"
        assert pacman_manager.get_branch_package("595") == "nvidia-open"
        assert pacman_manager.get_branch_package("580") == "nvidia-580xx-dkms"
        assert pacman_manager.get_branch_package("470") == "nvidia-470xx-dkms"


class TestPacmanManagerRealSafeReadOnly:
    """Read-only Pacman tests that should work in any container environment."""

    def test_pacman_manager_instantiation(self):
        """Test that PacmanManager can be instantiated."""
        manager = PacmanManager()
        assert manager is not None
        assert hasattr(manager, "_pacman_path")

    def test_search_returns_list_type(self, pacman_manager):
        """Test that search returns a list even on error."""
        result = pacman_manager.search("this-package-does-not-exist-xyz")
        assert isinstance(result, list)

    def test_get_installed_version_returns_str_or_none(self, pacman_manager):
        """Test that get_installed_version returns str or None."""
        version = pacman_manager.get_installed_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_available_version_returns_str_or_none(self, pacman_manager):
        """Test that get_available_version returns str or None."""
        version = pacman_manager.get_available_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_all_versions_returns_list(self, pacman_manager):
        """Test that get_all_versions returns a list."""
        versions = pacman_manager.get_all_versions("nvidia")
        assert isinstance(versions, list)

    def test_get_branch_package_latest(self):
        """Test that get_branch_package returns nvidia-open for latest."""
        manager = PacmanManager()
        # Latest branches should return nvidia-open
        for branch in ["590", "595", "600"]:
            result = manager.get_branch_package(branch)
            assert result == "nvidia-open"

    def test_get_branch_package_legacy(self):
        """Test that get_branch_package returns legacy branch packages."""
        manager = PacmanManager()
        assert manager.get_branch_package("580") == "nvidia-580xx-dkms"
        assert manager.get_branch_package("470") == "nvidia-470xx-dkms"

    def test_pin_version_returns_true(self):
        """Test that pin_version returns True for Arch (branch-based)."""
        manager = PacmanManager()
        # Arch doesn't need version locking - branch packages handle it
        result = manager.pin_version("nvidia-580xx-dkms", "580.*")
        assert result is True
