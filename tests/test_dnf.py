"""Tests for DNF package manager (Fedora/RHEL)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.distro.dnf import DnfManager
from nvidia_inst.distro.package_manager import PackageManagerError


class TestDnfManager:
    """Test DNF package manager."""

    @pytest.fixture
    def dnf_manager(self):
        """Create DnfManager instance."""
        return DnfManager()

    def test_is_available(self, dnf_manager, mock_shutil_which):
        """Test dnf availability check."""
        result = dnf_manager.is_available()
        assert result is True
        mock_shutil_which.assert_called_with("/usr/bin/dnf")

    def test_is_available_not_found(self, dnf_manager):
        """Test dnf availability when not installed."""
        with patch("shutil.which", return_value=None):
            result = dnf_manager.is_available()
            assert result is False

    def test_update_success(self, dnf_manager, mock_subprocess_run):
        """Test successful dnf makecache."""
        result = dnf_manager.update()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/dnf" in str(call_args) or call_args[0] == "/usr/bin/dnf"

    def test_update_failure(self, dnf_manager, mock_subprocess_run):
        """Test dnf makecache failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf makecache", stderr="Network error"
        )
        result = dnf_manager.update()
        assert result is False

    def test_upgrade_success(self, dnf_manager, mock_subprocess_run):
        """Test successful dnf upgrade."""
        result = dnf_manager.upgrade()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/dnf" in str(call_args) or call_args[0] == "/usr/bin/dnf"
        assert "-y" in call_args

    def test_upgrade_failure(self, dnf_manager, mock_subprocess_run):
        """Test dnf upgrade failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf upgrade", stderr="Dependency error"
        )
        result = dnf_manager.upgrade()
        assert result is False

    def test_install_success(self, dnf_manager, mock_subprocess_popen):
        """Test successful package installation via Popen."""
        result = dnf_manager.install(["akmod-nvidia", "xorg-x11-drv-nvidia"])
        assert result is True

    def test_install_failure(self, dnf_manager, mock_subprocess_popen):
        """Test package installation failure."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 1
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "Error: Package not found"
        mock_subprocess_popen.return_value = mock_proc

        with pytest.raises(PackageManagerError):
            dnf_manager.install(["nonexistent-package"])

    def test_install_exception(self, dnf_manager, mock_subprocess_popen):
        """Test package installation with exception."""
        mock_subprocess_popen.side_effect = OSError("Failed to start")

        with pytest.raises(PackageManagerError):
            dnf_manager.install(["nvidia-driver"])

    def test_remove_success(self, dnf_manager, mock_subprocess_run):
        """Test successful package removal."""
        result = dnf_manager.remove(["akmod-nvidia"])
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "remove" in call_args
        assert "-y" in call_args

    def test_remove_failure(self, dnf_manager, mock_subprocess_run):
        """Test package removal failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf remove", stderr="Package not installed"
        )
        result = dnf_manager.remove(["nonexistent-package"])
        assert result is False

    def test_search_found(self, dnf_manager, mock_subprocess_run):
        """Test package search with results."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="akmod-nvidia.x86_64 : Libraries for nvidia driver\n"
                   "nvidia-driver.x86_64 : NVIDIA driver metapackage",
            stderr=""
        )
        packages = dnf_manager.search("nvidia")
        assert len(packages) >= 1

    def test_search_not_found(self, dnf_manager, mock_subprocess_run):
        """Test package search with no results."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="No matches found.", stderr=""
        )
        packages = dnf_manager.search("nonexistent")
        assert packages == []

    def test_search_failure(self, dnf_manager, mock_subprocess_run):
        """Test package search failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf search", stderr="Error"
        )
        packages = dnf_manager.search("nvidia")
        assert packages == []

    def test_get_installed_version(self, dnf_manager, mock_subprocess_run, dnf_info_output):
        """Test getting installed package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=dnf_info_output, stderr=""
        )
        version = dnf_manager.get_installed_version("akmod-nvidia")
        assert version == "535.154.05"

    def test_get_installed_version_not_installed(self, dnf_manager, mock_subprocess_run):
        """Test getting version of uninstalled package."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf info", stderr="No matching"
        )
        version = dnf_manager.get_installed_version("nonexistent")
        assert version is None

    def test_get_available_version(self, dnf_manager, mock_subprocess_run, dnf_info_output):
        """Test getting available package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=dnf_info_output, stderr=""
        )
        version = dnf_manager.get_available_version("akmod-nvidia")
        assert version == "535.154.05"

    def test_get_available_version_none(self, dnf_manager, mock_subprocess_run):
        """Test getting available version when package not found."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="Error: No matching", stderr=""
        )
        version = dnf_manager.get_available_version("nvidia-driver")
        assert version is None

    def test_pin_version_success(self, dnf_manager, mock_subprocess_run):
        """Test successful version pinning via versionlock."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = dnf_manager.pin_version("akmod-nvidia", "535.154.05")
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "versionlock" in call_args
        assert "add" in call_args

    def test_pin_version_failure(self, dnf_manager, mock_subprocess_run):
        """Test version pinning failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf versionlock", stderr="Error"
        )
        result = dnf_manager.pin_version("akmod-nvidia", "535.154.05")
        assert result is False

    def test_get_all_versions(
        self, dnf_manager, mock_subprocess_run, dnf_list_output
    ):
        """Test getting all available versions."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=dnf_list_output, stderr=""
        )
        versions = dnf_manager.get_all_versions("akmod-nvidia")
        assert len(versions) == 3
        assert "535.154.05" in versions
        assert "535.54.06" in versions
        assert "535.43.02" in versions

    def test_get_all_versions_with_epoch(self, dnf_manager, mock_subprocess_run):
        """Test getting versions with epoch prefix."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="akmod-nvidia.x86_64        3:535.154.05-1.fc38        @rpmfusion\n"
                   "akmod-nvidia.x86_64        2:535.54.06-1.fc38        rpmfusion",
            stderr=""
        )
        versions = dnf_manager.get_all_versions("akmod-nvidia")
        assert "535.154.05" in versions
        assert "535.54.06" in versions

    def test_get_all_versions_fallback(self, dnf_manager, mock_subprocess_run, dnf_info_output):
        """Test getting versions with fallback to dnf info."""
        mock_subprocess_run.side_effect = [
            subprocess.CalledProcessError(1, "dnf list", stderr="Error"),
            MagicMock(returncode=0, stdout=dnf_info_output, stderr=""),
        ]
        versions = dnf_manager.get_all_versions("akmod-nvidia")
        assert versions == ["535.154.05"]

    def test_get_all_versions_complete_failure(self, dnf_manager, mock_subprocess_run):
        """Test getting versions when all methods fail."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "dnf list", stderr="Error"
        )
        versions = dnf_manager.get_all_versions("akmod-nvidia")
        assert versions == []

    def test_version_sort_key(self, dnf_manager):
        """Test version sorting."""
        versions = ["535.154.05", "535.54.06", "535.43.02"]
        sorted_versions = sorted(versions, key=dnf_manager._version_sort_key, reverse=True)
        assert sorted_versions == ["535.154.05", "535.54.06", "535.43.02"]


class TestDnfManagerIntegration:
    """Integration-style tests for DNF manager."""

    def test_full_install_workflow(self, mock_subprocess_run, mock_subprocess_popen, mock_shutil_which):
        """Test full install workflow."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "Complete!"
        mock_subprocess_popen.return_value = mock_proc
        mock_shutil_which.return_value = "/usr/bin/dnf"

        dnf = DnfManager()

        assert dnf.is_available() is True
        assert dnf.update() is True
        assert dnf.install(["akmod-nvidia", "xorg-x11-drv-nvidia"]) is True

    def test_search_and_install_workflow(self, mock_subprocess_run, mock_subprocess_popen):
        """Test search then install workflow."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="akmod-nvidia.x86_64 : Libraries",
            stderr=""
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "Done"
        mock_subprocess_popen.return_value = mock_proc

        dnf = DnfManager()
        assert dnf.install(["akmod-nvidia"]) is True
