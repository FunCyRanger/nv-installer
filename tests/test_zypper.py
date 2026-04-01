"""Tests for Zypper package manager (openSUSE)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.distro.package_manager import PackageManagerError
from nvidia_inst.distro.zypper import ZypperManager


class TestZypperManager:
    """Test Zypper package manager."""

    @pytest.fixture
    def zypper_manager(self):
        """Create ZypperManager instance."""
        return ZypperManager()

    def test_is_available(self, zypper_manager, mock_shutil_which):
        """Test zypper availability check."""
        result = zypper_manager.is_available()
        assert result is True
        mock_shutil_which.assert_called_with("/usr/bin/zypper")

    def test_is_available_not_found(self, zypper_manager):
        """Test zypper availability when not installed."""
        with patch("shutil.which", return_value=None):
            result = zypper_manager.is_available()
            assert result is False

    def test_update_success(self, zypper_manager, mock_subprocess_run):
        """Test successful zypper refresh."""
        result = zypper_manager.update()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/zypper" in str(call_args) or call_args[0] == "/usr/bin/zypper"

    def test_update_failure(self, zypper_manager, mock_subprocess_run):
        """Test zypper refresh failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper refresh", stderr="Repository error"
        )
        result = zypper_manager.update()
        assert result is False

    def test_upgrade_success(self, zypper_manager, mock_subprocess_run):
        """Test successful zypper update."""
        result = zypper_manager.upgrade()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/zypper" in str(call_args) or call_args[0] == "/usr/bin/zypper"
        assert "-y" in call_args

    def test_upgrade_failure(self, zypper_manager, mock_subprocess_run):
        """Test zypper update failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper update", stderr="Dependency error"
        )
        result = zypper_manager.upgrade()
        assert result is False

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_install_success(self, mock_root, zypper_manager, mock_subprocess_run):
        """Test successful package installation."""
        result = zypper_manager.install(["x11-video-nvidiaG05", "nvidia-computeG05"])
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "install" in call_args
        assert "-y" in call_args

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_install_failure(self, mock_root, zypper_manager, mock_subprocess_run):
        """Test package installation failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper install", stderr="Package not found"
        )
        with pytest.raises(PackageManagerError) as exc_info:
            zypper_manager.install(["nonexistent-package"])
        assert "Failed to install" in str(exc_info.value)

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_remove_success(self, mock_root, zypper_manager, mock_subprocess_run):
        """Test successful package removal."""
        result = zypper_manager.remove(["nvidia-driver"])
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "remove" in call_args
        assert "-y" in call_args

    def test_remove_failure(self, zypper_manager, mock_subprocess_run):
        """Test package removal failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper remove", stderr="Package not installed"
        )
        result = zypper_manager.remove(["nonexistent-package"])
        assert result is False

    def test_search_found(self, zypper_manager, mock_subprocess_run):
        """Test package search with results."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="S | Name                | Type   | Version       | Repository\n"
            "i | x11-video-nvidiaG05 | package | 535.154.05-1 | rpmfusion",
            stderr="",
        )
        packages = zypper_manager.search("nvidia")
        assert len(packages) >= 1

    def test_search_not_found(self, zypper_manager, mock_subprocess_run):
        """Test package search with no results."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="No matching items.", stderr=""
        )
        packages = zypper_manager.search("nonexistent")
        assert packages == []

    def test_search_failure(self, zypper_manager, mock_subprocess_run):
        """Test package search failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper search", stderr="Error"
        )
        packages = zypper_manager.search("nvidia")
        assert packages == []

    def test_get_installed_version(
        self, zypper_manager, mock_subprocess_run, zypper_info_output
    ):
        """Test getting installed package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=zypper_info_output, stderr=""
        )
        version = zypper_manager.get_installed_version("x11-video-nvidiaG05")
        assert version == "535.154.05"

    def test_get_installed_version_not_installed(
        self, zypper_manager, mock_subprocess_run
    ):
        """Test getting version of uninstalled package."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper info", stderr="not installed"
        )
        version = zypper_manager.get_installed_version("nonexistent")
        assert version is None

    def test_get_available_version(
        self, zypper_manager, mock_subprocess_run, zypper_info_output
    ):
        """Test getting available package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=zypper_info_output, stderr=""
        )
        version = zypper_manager.get_available_version("x11-video-nvidiaG05")
        assert version == "535.154.05"

    def test_get_available_version_not_found(self, zypper_manager, mock_subprocess_run):
        """Test getting available version when package not found."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper info", stderr="not found"
        )
        version = zypper_manager.get_available_version("nvidia-driver")
        assert version is None

    def test_pin_version_success(self, zypper_manager, mock_subprocess_run):
        """Test successful version pinning via addlock."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = zypper_manager.pin_version("x11-video-nvidiaG05", "535.154.05")
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "addlock" in call_args
        assert "x11-video-nvidiaG05=535.154.05" in call_args

    def test_pin_version_default_star(self, zypper_manager, mock_subprocess_run):
        """Test version pinning with default * (just package name)."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = zypper_manager.pin_version("x11-video-nvidiaG05")
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "addlock" in call_args
        assert "x11-video-nvidiaG05" in call_args
        assert "=" not in call_args[-1]

    def test_pin_version_failure(self, zypper_manager, mock_subprocess_run):
        """Test version pinning failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper addlock", stderr="Error"
        )
        result = zypper_manager.pin_version("x11-video-nvidiaG05", "535.154.05")
        assert result is False

    def test_pin_to_major_version_success(self, zypper_manager, mock_subprocess_run):
        """Test successful major version locking via negative lock."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = zypper_manager.pin_to_major_version("*nvidia*", "580")
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "addlock" in call_args
        # Should add lock for >= 581 (next major)
        assert "*nvidia* >= 581" in call_args

    def test_pin_to_major_version_failure(self, zypper_manager, mock_subprocess_run):
        """Test major version locking failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper addlock", stderr="Error"
        )
        result = zypper_manager.pin_to_major_version("*nvidia*", "580")
        assert result is False

    def test_remove_lock_success(self, zypper_manager, mock_subprocess_run):
        """Test successful lock removal."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = zypper_manager.remove_lock("x11-video-nvidiaG05")
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "removelock" in call_args
        assert "x11-video-nvidiaG05" in call_args

    def test_remove_lock_failure(self, zypper_manager, mock_subprocess_run):
        """Test lock removal failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper removelock", stderr="Error"
        )
        result = zypper_manager.remove_lock("x11-video-nvidiaG05")
        assert result is False

    def test_get_all_versions(
        self, zypper_manager, mock_subprocess_run, zypper_packages_output
    ):
        """Test getting all available versions."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=zypper_packages_output, stderr=""
        )
        versions = zypper_manager.get_all_versions("x11-video-nvidiaG05")
        assert len(versions) >= 1
        assert "535.154.05" in versions
        assert "535.54.06" in versions

    def test_get_all_versions_failure(self, zypper_manager, mock_subprocess_run):
        """Test getting versions when command fails."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "zypper packages", stderr="Error"
        )
        versions = zypper_manager.get_all_versions("nvidia-driver")
        assert versions == []

    def test_get_all_versions_returns_branch_versions(
        self, zypper_manager, mock_subprocess_run
    ):
        """Test that get_all_versions returns versions from the correct branch."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="""| nvidia-compute-G06 | 580.82.07-1 | x86_64 | cuda
| nvidia-compute-G06 | 580.65.06-1 | x86_64 | cuda
| nvidia-compute-G06 | 575.57.08-1 | x86_64 | cuda
| nvidia-compute-G06 | 570.148.08-1 | x86_64 | cuda
""",
            stderr="",
        )
        versions = zypper_manager.get_all_versions("nvidia-compute-G06")
        # Should return versions sorted by newest first
        # Note: zypper parses version from column index 1 and splits on "-"
        assert len(versions) == 4
        assert "580.82.07" in versions[0]  # Latest version
        assert any("580.65" in v for v in versions)
        assert any("575.57" in v for v in versions)
        assert any("570.148" in v for v in versions)

    def test_version_sort_key(self, zypper_manager):
        """Test version sorting."""
        versions = ["535.154.05", "535.54.06", "535.43.02"]
        sorted_versions = sorted(
            versions, key=zypper_manager._version_sort_key, reverse=True
        )
        assert sorted_versions == ["535.154.05", "535.54.06", "535.43.02"]


class TestZypperManagerIntegration:
    """Integration-style tests for Zypper manager."""

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_full_install_workflow(
        self, mock_root, mock_subprocess_run, mock_shutil_which
    ):
        """Test full install workflow."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_shutil_which.return_value = "/usr/bin/zypper"

        zypper = ZypperManager()

        assert zypper.is_available() is True
        assert zypper.update() is True
        assert zypper.install(["x11-video-nvidiaG05", "nvidia-computeG05"]) is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_search_and_install_workflow(self, mock_root, mock_subprocess_run):
        """Test search then install workflow."""
        mock_subprocess_run.side_effect = [
            MagicMock(
                returncode=0,
                stdout="S | Name                  | Version\n"
                "  | x11-video-nvidiaG05 | 535.154.05",
                stderr="",
            ),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        zypper = ZypperManager()
        packages = zypper.search("nvidia")
        assert len(packages) >= 1
        assert zypper.install(packages) is True
