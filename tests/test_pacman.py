"""Tests for Pacman package manager (Arch Linux)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.distro.package_manager import PackageManagerError
from nvidia_inst.distro.pacman import PacmanManager


class TestPacmanManager:
    """Test Pacman package manager."""

    @pytest.fixture
    def pacman_manager(self):
        """Create PacmanManager instance."""
        return PacmanManager()

    def test_is_available(self, pacman_manager, mock_shutil_which):
        """Test pacman availability check."""
        result = pacman_manager.is_available()
        assert result is True
        mock_shutil_which.assert_called_with("/usr/bin/pacman")

    def test_is_available_not_found(self, pacman_manager):
        """Test pacman availability when not installed."""
        with patch("shutil.which", return_value=None):
            result = pacman_manager.is_available()
            assert result is False

    def test_update_success(self, pacman_manager, mock_subprocess_run):
        """Test successful pacman sync."""
        result = pacman_manager.update()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "/usr/bin/pacman" in str(call_args) or call_args[0] == "/usr/bin/pacman"

    def test_update_failure(self, pacman_manager, mock_subprocess_run):
        """Test pacman sync failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Sy", stderr="Failed to sync"
        )
        result = pacman_manager.update()
        assert result is False

    def test_upgrade_success(self, pacman_manager, mock_subprocess_run):
        """Test successful pacman upgrade."""
        result = pacman_manager.upgrade()
        assert result is True
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert "-Syu" in call_args
        assert "--noconfirm" in call_args

    def test_upgrade_failure(self, pacman_manager, mock_subprocess_run):
        """Test pacman upgrade failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Syu", stderr="Dependency error"
        )
        result = pacman_manager.upgrade()
        assert result is False

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_install_success(self, mock_root, pacman_manager, mock_subprocess_run):
        """Test successful package installation."""
        result = pacman_manager.install(["nvidia-open", "nvidia-utils"])
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "nvidia-open" in call_args
        assert "--noconfirm" in call_args

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_install_failure(self, mock_root, pacman_manager, mock_subprocess_run):
        """Test package installation failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -S", stderr="Package not found"
        )
        with pytest.raises(PackageManagerError) as exc_info:
            pacman_manager.install(["nonexistent-package"])
        assert "nonexistent-package" in str(exc_info.value)

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_remove_success(self, mock_root, pacman_manager, mock_subprocess_run):
        """Test successful package removal."""
        result = pacman_manager.remove(["nvidia-open"])
        assert result is True
        call_args = mock_subprocess_run.call_args[0][0]
        assert "-R" in call_args

    def test_remove_failure(self, pacman_manager, mock_subprocess_run):
        """Test package removal failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -R", stderr="Package not installed"
        )
        result = pacman_manager.remove(["nonexistent-package"])
        assert result is False

    def test_search_found(self, pacman_manager, mock_subprocess_run, pacman_ss_output):
        """Test package search with results."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=pacman_ss_output, stderr=""
        )
        packages = pacman_manager.search("nvidia")
        assert "nvidia" in packages
        assert "nvidia-470xx" in packages
        assert "nvidia-535xx" in packages

    def test_search_not_found(self, pacman_manager, mock_subprocess_run):
        """Test package search with no results."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        packages = pacman_manager.search("nonexistent")
        assert packages == []

    def test_search_failure(self, pacman_manager, mock_subprocess_run):
        """Test package search failure."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Ss", stderr="Error"
        )
        packages = pacman_manager.search("nvidia")
        assert packages == []

    def test_get_installed_version(self, pacman_manager, mock_subprocess_run):
        """Test getting installed package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="nvidia 535.154.05-14\n", stderr=""
        )
        version = pacman_manager.get_installed_version("nvidia")
        assert version == "535.154.05-14"

    def test_get_installed_version_not_installed(
        self, pacman_manager, mock_subprocess_run
    ):
        """Test getting version of uninstalled package."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Q", stderr="package not installed"
        )
        version = pacman_manager.get_installed_version("nonexistent")
        assert version is None

    def test_get_available_version(
        self, pacman_manager, mock_subprocess_run, pacman_si_output
    ):
        """Test getting available package version."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout=pacman_si_output, stderr=""
        )
        version = pacman_manager.get_available_version("nvidia")
        assert version == "535.154.05-14"

    def test_get_available_version_not_found(self, pacman_manager, mock_subprocess_run):
        """Test getting available version when package not found."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Si", stderr="not found"
        )
        version = pacman_manager.get_available_version("nvidia-driver")
        assert version is None

    def test_pin_version_not_supported(self, pacman_manager):
        """Test version pinning returns False (not supported)."""
        result = pacman_manager.pin_version("nvidia", "535.154.05")
        assert result is False

    def test_pin_version_default_star(self, pacman_manager):
        """Test version pinning with default * parameter."""
        result = pacman_manager.pin_version("nvidia")
        assert result is False

    def test_package_exists_true(self, pacman_manager, mock_subprocess_run):
        """Test package exists check - true case."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = pacman_manager._package_exists("nvidia")
        assert result is True

    def test_package_exists_false(self, pacman_manager, mock_subprocess_run):
        """Test package exists check - false case."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Si", stderr="not found"
        )
        result = pacman_manager._package_exists("nonexistent-package")
        assert result is False

    def test_get_all_versions_with_nvidia(self, pacman_manager, mock_subprocess_run):
        """Test getting all versions returns branches for nvidia."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        versions = pacman_manager.get_all_versions("nvidia")
        assert "nvidia" in versions

    def test_get_all_versions_empty_result(self, pacman_manager, mock_subprocess_run):
        """Test getting all versions when no packages exist."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pacman -Si", stderr="not found"
        )

        versions = pacman_manager.get_all_versions("nvidia")
        assert versions == []


class TestPacmanManagerBranches:
    """Test Pacman branch detection."""

    @pytest.fixture
    def pacman_manager(self):
        """Create PacmanManager instance."""
        return PacmanManager()

    def test_detect_nvidia_open_branch(self, pacman_manager, mock_subprocess_run):
        """Test detection of nvidia-open branch."""
        call_count = 0

        def mock_side_effect(*args, **kwargs):
            nonlocal call_count
            cmd = args[0] if args else kwargs.get("args", [])
            if (
                "-Si" in cmd
                and "nvidia" in cmd
                and ("nvidia-utils" in str(cmd) or call_count == 0)
            ):
                call_count += 1
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        mock_subprocess_run.side_effect = mock_side_effect

        versions = pacman_manager.get_all_versions("nvidia")
        assert "nvidia" in versions

    def test_detect_470xx_branch(self, pacman_manager, mock_subprocess_run):
        """Test detection of 470xx branch for Kepler."""
        call_count = 0

        def mock_side_effect(*args, **kwargs):
            nonlocal call_count
            cmd = args[0] if args else kwargs.get("args", [])
            if "-Si" in cmd and "470xx" in str(cmd):
                call_count += 1
                if call_count == 1:
                    return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        mock_subprocess_run.side_effect = mock_side_effect

        versions = pacman_manager.get_all_versions("nvidia-470xx")
        assert "nvidia-470xx" in versions


class TestPacmanManagerIntegration:
    """Integration-style tests for Pacman manager."""

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_full_install_workflow(
        self, mock_root, mock_subprocess_run, mock_shutil_which
    ):
        """Test full install workflow."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_shutil_which.return_value = "/usr/bin/pacman"

        pacman = PacmanManager()

        assert pacman.is_available() is True
        assert pacman.update() is True
        assert pacman.install(["nvidia-open", "nvidia-utils"]) is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_search_and_install_workflow(self, mock_root, mock_subprocess_run):
        """Test search then install workflow."""
        outputs = [
            MagicMock(returncode=0, stdout="extra/nvidia 535.154.05", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        mock_subprocess_run.side_effect = outputs

        pacman = PacmanManager()
        packages = pacman.search("nvidia-open")
        assert len(packages) >= 1
        assert pacman.install([packages[0]]) is True
