"""Tests for APT package manager (Ubuntu/Debian)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.distro.apt import AptManager
from nvidia_inst.distro.package_manager import PackageManagerError


class TestAptManager:
    """Test APT package manager."""

    @pytest.fixture
    def apt_manager(self):
        """Create AptManager instance."""
        return AptManager()

    def test_is_available(self, apt_manager, mock_shutil_which):
        """Test apt availability check."""
        result = apt_manager.is_available()
        assert result is True
        mock_shutil_which.assert_called_with("/usr/bin/apt")

    def test_is_available_not_found(self, apt_manager):
        """Test apt availability when not installed."""
        with patch("shutil.which", return_value=None):
            result = apt_manager.is_available()
            assert result is False

    def test_update_success(self, apt_manager, mock_subprocess_run):
        """Test successful apt update."""
        result = apt_manager.update()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/apt" in str(call_args) or call_args[0] == "/usr/bin/apt"

    def test_upgrade_success(self, apt_manager, mock_subprocess_run):
        """Test successful apt upgrade."""
        result = apt_manager.upgrade()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/apt" in str(call_args) or call_args[0] == "/usr/bin/apt"
        assert "-y" in call_args

    def test_upgrade_failure(self, apt_manager, mock_subprocess_run):
        """Test apt upgrade failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "apt upgrade", stderr="Dependency error"
        )
        result = apt_manager.upgrade()
        assert result is False

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_install_success(self, mock_root, apt_manager, mock_subprocess_run):
        """Test successful package installation."""
        result = apt_manager.install(["nvidia-driver-535"])
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "nvidia-driver-535" in call_args

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_install_failure(self, mock_root, apt_manager, mock_subprocess_run):
        """Test package installation failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "apt install", stderr="Package not found"
        )
        with pytest.raises(PackageManagerError) as exc_info:
            apt_manager.install(["nonexistent-package"])
        assert "nonexistent-package" in str(exc_info.value)

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_remove_success(self, mock_root, apt_manager, mock_subprocess_run):
        """Test successful package removal."""
        result = apt_manager.remove(["nvidia-driver-535"])
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "remove" in call_args

    def test_remove_failure(self, apt_manager, mock_subprocess_run):
        """Test package removal failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "apt remove", stderr="Package not installed"
        )
        result = apt_manager.remove(["nonexistent-package"])
        assert result is False

    def test_search_found(
        self, apt_manager, mock_subprocess_run, apt_package_search_output
    ):
        """Test package search with results."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=apt_package_search_output, stderr=""
        )
        packages = apt_manager.search("nvidia-driver")
        assert len(packages) == 3
        assert "nvidia-driver-535" in packages
        assert "nvidia-driver-540" in packages
        assert "nvidia-driver-550" in packages

    def test_search_not_found(self, apt_manager, mock_subprocess_run):
        """Test package search with no results."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        packages = apt_manager.search("nonexistent")
        assert packages == []

    def test_search_failure(self, apt_manager, mock_subprocess_run):
        """Test package search failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "apt-cache search", stderr="Error"
        )
        packages = apt_manager.search("nvidia")
        assert packages == []

    def test_get_installed_version(self, apt_manager, mock_subprocess_run):
        """Test getting installed package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="535.154.05-0ubuntu1\n", stderr=""
        )
        version = apt_manager.get_installed_version("nvidia-driver-535")
        assert version == "535.154.05-0ubuntu1"
        mock_subprocess_run.assert_called_with(
            ["dpkg-query", "-W", "-f=${Version}", "nvidia-driver-535"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_get_installed_version_not_installed(
        self, apt_manager, mock_subprocess_run
    ):
        """Test getting version of uninstalled package."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dpkg-query", stderr="Package not installed"
        )
        version = apt_manager.get_installed_version("nonexistent")
        assert version is None

    def test_get_available_version(
        self, apt_manager, mock_subprocess_run, apt_policy_output
    ):
        """Test getting available package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=apt_policy_output, stderr=""
        )
        version = apt_manager.get_available_version("nvidia-driver-535")
        assert version == "535.154.05-0ubuntu2"

    def test_get_available_version_none(self, apt_manager, mock_subprocess_run):
        """Test getting available version when package not found."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="nvidia-driver-535:\n  Version table:", stderr=""
        )
        version = apt_manager.get_available_version("nvidia-driver-535")
        assert version is None

    def test_pin_version_success(self, apt_manager, mock_open):
        """Test successful version pinning with wildcard."""
        result = apt_manager.pin_version("nvidia-driver-*", "580.*")
        assert result is True
        mock_open.assert_called_once()
        written_content = mock_open.return_value.__enter__.return_value.write.call_args[
            0
        ][0]
        assert "nvidia-driver-*" in written_content
        assert "580.*" in written_content
        assert "Pin-Priority: 1001" in written_content

    def test_pin_version_with_exact_version(self, apt_manager, mock_open):
        """Test version pinning with exact version."""
        result = apt_manager.pin_version("nvidia-driver-535", "535.154.05")
        assert result is True
        written_content = mock_open.return_value.__enter__.return_value.write.call_args[
            0
        ][0]
        assert "535.154.05" in written_content

    def test_pin_version_permission_denied(self, apt_manager, mock_open):
        """Test version pinning with permission denied."""
        mock_open.side_effect = PermissionError("Permission denied")
        result = apt_manager.pin_version("nvidia-driver-535", "535.154.05")
        assert result is False

    def test_pin_version_os_error(self, apt_manager, mock_open):
        """Test version pinning with OS error."""
        mock_open.side_effect = OSError("Disk full")
        result = apt_manager.pin_version("nvidia-driver-535", "535.154.05")
        assert result is False

    def test_get_all_versions(
        self, apt_manager, mock_subprocess_run, apt_madison_output
    ):
        """Test getting all available versions."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=apt_madison_output, stderr=""
        )
        versions = apt_manager.get_all_versions("nvidia-driver-535")
        assert len(versions) >= 1

    def test_get_all_versions_failure(self, apt_manager, mock_subprocess_run):
        """Test getting versions when command fails."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "apt-cache madison", stderr="Error"
        )
        versions = apt_manager.get_all_versions("nvidia-driver-535")
        assert versions == []

    def test_version_sort_key(self, apt_manager):
        """Test version sorting."""
        versions = ["535.154.05", "535.54.06", "535.43.02"]
        sorted_versions = sorted(
            versions, key=apt_manager._version_sort_key, reverse=True
        )
        assert sorted_versions == ["535.154.05", "535.54.06", "535.43.02"]

    def test_version_sort_key_with_prefix(self, apt_manager):
        """Test version sorting with epoch prefix."""
        versions = ["2:535.154.05", "1:535.154.05", "535.154.05"]
        sorted_versions = sorted(
            versions, key=apt_manager._version_sort_key, reverse=True
        )
        assert "2:535.154.05" in sorted_versions


class TestAptManagerIntegration:
    """Integration-style tests for APT manager with real mock objects."""

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_full_install_workflow(
        self, mock_root, mock_subprocess_run, mock_shutil_which
    ):
        """Test full install workflow."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_shutil_which.return_value = "/usr/bin/apt"

        apt = AptManager()

        assert apt.is_available() is True
        assert apt.update() is True
        assert apt.install(["nvidia-driver-535", "nvidia-dkms-535"]) is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_full_search_to_install_workflow(
        self, mock_root, mock_subprocess_run, apt_package_search_output
    ):
        """Test search then install workflow."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout=apt_package_search_output, stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        apt = AptManager()
        packages = apt.search("nvidia-driver")
        assert apt.install([packages[0]]) is True
