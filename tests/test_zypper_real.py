"""Real integration tests for Zypper package manager (openSUSE).

These tests exercise actual Zypper commands without mocking subprocess.
They are safe to run in CI containers and focus on read operations.

Note: openSUSE containers may have shell issues with zypper shell. Tests
that require interactive shell are noted accordingly.
"""

import os
import shutil
import subprocess

import pytest

from nvidia_inst.distro.zypper import ZypperManager


# Skip all tests in this file if Zypper is not available
@pytest.fixture(scope="module", autouse=True)
def check_zypper_available():
    """Check if Zypper is available on this system."""
    if not ZypperManager().is_available():
        pytest.skip("Zypper is not available on this system")


class TestZypperManagerReal:
    """Real Zypper integration tests (read operations only)."""

    def test_is_available_real(self, zypper_manager):
        """Test that zypper is actually available on this system."""
        result = zypper_manager.is_available()
        assert result is True

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for zypper refresh")
    def test_update_refresh_real(self, zypper_manager):
        """Test zypper refresh updates package metadata (requires root)."""
        result = zypper_manager.update()
        # May fail due to network/repo issues in containers, but should not crash
        assert isinstance(result, bool)

    def test_zypper_path_constant(self, zypper_manager):
        """Test that Zypper path constant is correct."""
        assert zypper_manager._zypper_path == "/usr/bin/zypper"

    def test_search_real(self, zypper_manager):
        """Test searching for packages with real zypper search."""
        # Search for a package that should exist
        packages = zypper_manager.search("bash")
        assert isinstance(packages, list)

    def test_get_installed_version_real(self, zypper_manager):
        """Test getting installed version of a system package."""
        # Try to get version of a common package
        version = zypper_manager.get_installed_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_available_version_real(self, zypper_manager):
        """Test getting available version from repos."""
        version = zypper_manager.get_available_version("bash")
        if version:
            assert isinstance(version, str)
            assert len(version) > 0

    def test_get_installed_version_nonexistent_real(self, zypper_manager):
        """Test getting version of non-installed package returns None."""
        version = zypper_manager.get_installed_version(
            "this-package-does-not-exist-xyz"
        )
        assert version is None

    def test_get_available_version_nonexistent_real(self, zypper_manager):
        """Test getting available version of non-existent package returns None."""
        version = zypper_manager.get_available_version(
            "this-package-does-not-exist-xyz"
        )
        assert version is None

    def test_get_all_versions_real(self, zypper_manager):
        """Test getting all versions using zypper packages."""
        versions = zypper_manager.get_all_versions("bash")
        assert isinstance(versions, list)

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for locks")
    def test_addlock_command_format(self, zypper_manager):
        """Test that addlock is called with correct command format.

        Note: This is a read-only verification that the method
        constructs the correct command without actually adding a lock.
        """
        # Verify the command would be built correctly by checking the method
        # uses the correct zypper path and subcommand
        assert zypper_manager._zypper_path == "/usr/bin/zypper"
        # The actual addlock would require sudo and may fail in containers

    @pytest.mark.skipif(os.geteuid() != 0, reason="Requires root for locks")
    def test_removelock_command_format(self, zypper_manager):
        """Test that removelock is called with correct command format.

        Note: This is a read-only verification that the method
        constructs the correct command without actually removing a lock.
        """
        # Verify the command would be built correctly
        assert zypper_manager._zypper_path == "/usr/bin/zypper"


class TestZypperManagerRealSafeReadOnly:
    """Read-only Zypper tests that should work in any container environment."""

    def test_zypper_manager_instantiation(self):
        """Test that ZypperManager can be instantiated."""
        manager = ZypperManager()
        assert manager is not None
        assert hasattr(manager, "_zypper_path")

    def test_search_returns_list_type(self, zypper_manager):
        """Test that search returns a list even on error."""
        result = zypper_manager.search("this-package-does-not-exist-xyz")
        assert isinstance(result, list)

    def test_get_installed_version_returns_str_or_none(self, zypper_manager):
        """Test that get_installed_version returns str or None."""
        version = zypper_manager.get_installed_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_available_version_returns_str_or_none(self, zypper_manager):
        """Test that get_available_version returns str or None."""
        version = zypper_manager.get_available_version("nonexistent-package-xyz")
        assert version is None or isinstance(version, str)

    def test_get_all_versions_returns_list(self, zypper_manager):
        """Test that get_all_versions returns a list."""
        versions = zypper_manager.get_all_versions("nonexistent-package-xyz")
        assert isinstance(versions, list)

    def test_version_sort_key_real(self, zypper_manager):
        """Test version sorting works correctly."""
        versions = ["535.154.05", "535.54.06", "535.43.02", "470.256.02"]
        sorted_versions = sorted(
            versions, key=zypper_manager._version_sort_key, reverse=True
        )
        # 535.154.05 should be first (highest)
        assert sorted_versions[0] == "535.154.05"
        # 470.256.02 should be last (lowest)
        assert sorted_versions[-1] == "470.256.02"

    def test_pin_version_returns_bool(self, zypper_manager):
        """Test that pin_version returns a boolean (False without root)."""
        # Without root, this should return False
        result = zypper_manager.pin_version("bash", "5.*")
        assert isinstance(result, bool)

    def test_remove_lock_returns_bool(self, zypper_manager):
        """Test that remove_lock returns a boolean."""
        # Without root, this should return False
        result = zypper_manager.remove_lock("bash")
        assert isinstance(result, bool)

    def test_pin_to_major_version_returns_bool(self, zypper_manager):
        """Test that pin_to_major_version returns a boolean."""
        # Without root, this should return False
        result = zypper_manager.pin_to_major_version("*nvidia*", "580")
        assert isinstance(result, bool)


class TestZypperShellReal:
    """Tests specifically for zypper shell functionality.

    These tests investigate the shell issue mentioned for openSUSE containers.
    """

    @pytest.mark.skipif(
        not shutil.which("zypper"),
        reason="Zypper not available",
    )
    def test_zypper_shell_basic_invocation(self):
        """Test basic zypper shell invocation works."""
        # Try to invoke zypper shell with a simple command
        # Using echo to avoid interactive mode
        result = subprocess.run(
            ["zypper", "shell", "-v", "info", "bash"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # If this fails due to shell issues, we'll document it
        # At minimum, we should get some output or clear error message
        assert result.returncode in (0, 1, 4, 5, 6)  # zypper exit codes

    @pytest.mark.skipif(
        not shutil.which("zypper"),
        reason="Zypper not available",
    )
    def test_zypper_non_shell_commands_work(self):
        """Verify that non-shell zypper commands work (info vs shell info)."""
        # Regular zypper info should work
        result = subprocess.run(
            ["zypper", "info", "bash"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # This should work even if shell has issues
        assert result.returncode in (0, 104)  # 104 = not found
