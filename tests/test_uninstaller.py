"""Tests for the uninstaller module (revert to Nouveau)."""

import subprocess
from unittest.mock import MagicMock, patch

from nvidia_inst.installer.uninstaller import (
    RevertResult,
    _get_packages_to_remove,
    _rebuild_initramfs,
    _remove_blacklist,
    check_nvidia_packages_installed,
    revert_to_nouveau,
)


class TestGetPackagesToRemove:
    """Test package list generation for each distribution."""

    def test_ubuntu_packages(self):
        """Test Ubuntu/Debian package list."""
        packages = _get_packages_to_remove("ubuntu")
        assert "nvidia-driver-*" in packages
        assert "nvidia-dkms-*" in packages
        assert "libnvidia-*" in packages

    def test_fedora_packages(self):
        """Test Fedora package list."""
        packages = _get_packages_to_remove("fedora")
        assert "akmod-nvidia" in packages
        assert "xorg-x11-drv-nvidia" in packages
        assert "nvidia-persistenced" in packages

    def test_arch_packages(self):
        """Test Arch package list."""
        packages = _get_packages_to_remove("arch")
        assert "nvidia" in packages
        assert "nvidia-open" in packages
        assert "nvidia-580xx-dkms" in packages
        assert "nvidia-470xx-dkms" in packages

    def test_opensuse_packages(self):
        """Test openSUSE package list."""
        packages = _get_packages_to_remove("opensuse")
        assert "x11-video-nvidiaG05" in packages
        assert "nvidia-computeG05" in packages

    def test_unsupported_distro(self):
        """Test unsupported distribution returns empty list."""
        packages = _get_packages_to_remove("unknown")
        assert packages == []


class TestRevertToNouveau:
    """Test revert_to_nouveau function."""

    def test_unsupported_distro(self, mock_is_root):
        """Test revert fails on unsupported distribution."""
        result = revert_to_nouveau("unknown")
        assert result.success is False
        assert "Unsupported distribution" in result.errors[0]

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    def test_requires_root(self, mock_is_root):
        """Test revert fails when not running as root."""
        result = revert_to_nouveau("ubuntu")
        assert result.success is False
        assert "Root privileges required" in result.errors[0]

    @patch("nvidia_inst.installer.uninstaller._remove_packages")
    @patch("nvidia_inst.installer.uninstaller._rebuild_initramfs")
    @patch("nvidia_inst.installer.uninstaller._remove_blacklist")
    @patch("nvidia_inst.installer.uninstaller._get_packages_to_remove")
    def test_no_nvidia_packages(
        self, mock_get_pkgs, mock_remove_bl, mock_rebuild, mock_remove_pkgs
    ):
        """Test revert handles no Nvidia packages."""
        mock_get_pkgs.return_value = ["nvidia-driver-535"]
        mock_remove_pkgs.return_value = []
        mock_remove_bl.return_value = True
        mock_rebuild.return_value = True

        result = revert_to_nouveau("ubuntu")
        assert result.success is False

    @patch("subprocess.run")
    @patch("nvidia_inst.installer.uninstaller._remove_blacklist")
    @patch("nvidia_inst.installer.uninstaller._rebuild_initramfs")
    @patch("nvidia_inst.installer.uninstaller._get_packages_to_remove")
    @patch("nvidia_inst.installer.uninstaller._remove_packages")
    def test_successful_revert(
        self,
        mock_remove_pkgs,
        mock_get_pkgs,
        mock_rebuild,
        mock_remove_bl,
        mock_run,
        mock_is_root,
    ):
        """Test successful revert to Nouveau."""
        mock_get_pkgs.return_value = ["nvidia-driver-535"]
        mock_remove_pkgs.return_value = ["nvidia-driver-535"]
        mock_remove_bl.return_value = True
        mock_rebuild.return_value = True

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = revert_to_nouveau("ubuntu")

        assert result.success is True
        assert len(result.packages_removed) == 1


class TestRemoveBlacklist:
    """Test _remove_blacklist function."""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.unlink")
    def test_blacklist_removed(self, mock_unlink, mock_exists):
        """Test blacklist is removed when present."""
        mock_exists.return_value = True

        result = _remove_blacklist()

        assert result is True
        mock_unlink.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_blacklist_not_present(self, mock_exists):
        """Test blacklist removal when not present."""
        mock_exists.return_value = False

        result = _remove_blacklist()

        assert result is False

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.unlink")
    def test_blacklist_removal_error(self, mock_unlink, mock_exists):
        """Test blacklist removal error handling."""
        mock_exists.return_value = True
        mock_unlink.side_effect = OSError("Permission denied")

        result = _remove_blacklist()

        assert result is False


class TestRebuildInitramfs:
    """Test _rebuild_initramfs function."""

    @patch("subprocess.run")
    def test_fedora_drast(self, mock_run):
        """Test Fedora uses dracut."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = _rebuild_initramfs("fedora")

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "dracut" in call_args

    @patch("subprocess.run")
    def test_arch_mkinitcpio(self, mock_run):
        """Test Arch uses mkinitcpio."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = _rebuild_initramfs("arch")

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "mkinitcpio" in call_args

    @patch("subprocess.run")
    def test_ubuntu_update_initramfs(self, mock_run):
        """Test Ubuntu uses update-initramfs."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = _rebuild_initramfs("ubuntu")

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "update-initramfs" in call_args

    @patch("subprocess.run")
    def test_timeout_handling(self, mock_run):
        """Test timeout is handled."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 180)

        result = _rebuild_initramfs("ubuntu")

        assert result is False

    @patch("subprocess.run")
    def test_generic_error(self, mock_run):
        """Test generic error handling."""
        mock_run.side_effect = OSError("Failed")

        result = _rebuild_initramfs("ubuntu")

        assert result is False


class TestCheckNvidiaPackagesInstalled:
    """Test check_nvidia_packages_installed function."""

    @patch("subprocess.run")
    def test_ubuntu_packages_found(self, mock_run):
        """Test finding Nvidia packages on Ubuntu."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ii  nvidia-driver-535       535.154.05-0ubuntu1\n"
            "ii  nvidia-dkms-535         535.154.05-0ubuntu1\n",
            stderr="",
        )

        result = check_nvidia_packages_installed("ubuntu")

        assert len(result) >= 1

    @patch("subprocess.run")
    def test_fedora_packages_found(self, mock_run):
        """Test finding Nvidia packages on Fedora."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="akmod-nvidia.x86_64        535.154.05-1.fc38\n"
            "xorg-x11-drv-nvidia.x86_64",
            stderr="",
        )

        result = check_nvidia_packages_installed("fedora")

        assert len(result) >= 1

    @patch("subprocess.run")
    def test_no_packages_found(self, mock_run):
        """Test when no Nvidia packages are found."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = check_nvidia_packages_installed("ubuntu")

        assert result == []

    @patch("subprocess.run")
    def test_command_failure(self, mock_run):
        """Test handling command failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

        result = check_nvidia_packages_installed("ubuntu")

        assert result == []

    @patch("subprocess.run")
    def test_unknown_distro(self, mock_run):
        """Test handling unknown distribution."""
        result = check_nvidia_packages_installed("unknown")

        assert result == []
        mock_run.assert_not_called()


class TestRevertResult:
    """Test RevertResult dataclass."""

    def test_result_creation(self):
        """Test creating RevertResult."""
        result = RevertResult(
            success=True,
            packages_removed=["nvidia-driver-535"],
            versionlock_removed=["akmod-nvidia-580.*"],
            apt_preferences_removed=["/etc/apt/preferences.d/nvidia"],
            errors=[],
            message="Success",
        )

        assert result.success is True
        assert len(result.packages_removed) == 1
        assert len(result.versionlock_removed) == 1
        assert len(result.apt_preferences_removed) == 1
        assert len(result.errors) == 0
