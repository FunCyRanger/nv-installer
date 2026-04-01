"""Tests for installer/driver.py module."""

import pytest
from unittest.mock import patch, MagicMock

from nvidia_inst.installer.driver import (
    DriverInstallError,
    NouveauLoadedError,
    SecureBootError,
    KernelIncompatibleError,
    InstallResult,
    check_nouveau,
    check_nvidia_open_installed,
    check_nvidia_open_available,
    check_nonfree_available,
    check_secure_boot,
    get_current_driver_type,
    disable_nouveau,
)


class TestExceptions:
    """Tests for exception classes."""

    def test_driver_install_error(self):
        """Test DriverInstallError can be raised."""
        with pytest.raises(DriverInstallError):
            raise DriverInstallError("Test error")

    def test_nouveau_loaded_error(self):
        """Test NouveauLoadedError can be raised."""
        with pytest.raises(NouveauLoadedError):
            raise NouveauLoadedError("Nouveau loaded")

    def test_secure_boot_error(self):
        """Test SecureBootError can be raised."""
        with pytest.raises(SecureBootError):
            raise SecureBootError("Secure boot enabled")

    def test_kernel_incompatible_error(self):
        """Test KernelIncompatibleError can be raised."""
        with pytest.raises(KernelIncompatibleError):
            raise KernelIncompatibleError("Kernel incompatible")


class TestInstallResult:
    """Tests for InstallResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = InstallResult(success=True, message="Test")
        assert result.success is True
        assert result.message == "Test"
        assert result.packages_installed == []

    def test_with_packages(self):
        """Test with packages installed."""
        result = InstallResult(
            success=True,
            message="Installed",
            packages_installed=["nvidia-driver-535"],
        )
        assert result.packages_installed == ["nvidia-driver-535"]


class TestCheckNouveau:
    """Tests for check_nouveau function."""

    @patch("subprocess.run")
    def test_nouveau_loaded(self, mock_run):
        """Test when nouveau is loaded."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="nouveau  1638400  0\nnvidia  123456  0\n",
        )
        result = check_nouveau()
        assert result is True

    @patch("subprocess.run")
    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_nouveau_not_loaded(self, mock_distro, mock_run):
        """Test when nouveau is not loaded."""
        mock_run.return_value = MagicMock(returncode=0, stdout="nvidia  123456  0\n")
        mock_distro.return_value = MagicMock(id="fedora")

        with patch(
            "nvidia_inst.installer.driver._check_nouveau_packages", return_value=False
        ):
            result = check_nouveau()
            assert result is False

    @patch("subprocess.run")
    def test_lsmod_fails(self, mock_run):
        """Test when lsmod fails."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "lsmod")

        with patch(
            "nvidia_inst.installer.driver._check_nouveau_packages", return_value=False
        ):
            result = check_nouveau()
            assert result is False


class TestCheckSecureBoot:
    """Tests for check_secure_boot function."""

    @patch("subprocess.run")
    def test_secure_boot_enabled(self, mock_run):
        """Test when secure boot is enabled."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="SecureBoot: enabled",
        )
        result = check_secure_boot()
        assert result is True

    @patch("subprocess.run")
    def test_secure_boot_disabled(self, mock_run):
        """Test when secure boot is disabled."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="SecureBoot: disabled",
        )
        result = check_secure_boot()
        assert result is False

    @patch("subprocess.run")
    def test_mokutil_not_found(self, mock_run):
        """Test when mokutil is not found."""
        mock_run.side_effect = FileNotFoundError()
        result = check_secure_boot()
        assert result is False


class TestGetCurrentDriverType:
    """Tests for get_current_driver_type function."""

    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.installer.driver.check_nvidia_open_installed")
    def test_proprietary_driver(self, mock_open, mock_working):
        """Test when proprietary driver is installed."""
        mock_working.return_value = MagicMock(is_working=True)
        mock_open.return_value = False
        result = get_current_driver_type()
        assert result == "proprietary"

    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.installer.driver.check_nvidia_open_installed")
    def test_nvidia_open_driver(self, mock_open, mock_working):
        """Test when nvidia-open driver is installed."""
        mock_working.return_value = MagicMock(is_working=True)
        mock_open.return_value = True
        result = get_current_driver_type()
        assert result == "nvidia_open"

    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.installer.driver.check_nouveau")
    def test_nouveau_driver(self, mock_nouveau, mock_working):
        """Test when nouveau driver is loaded."""
        mock_working.return_value = MagicMock(is_working=False)
        mock_nouveau.return_value = True
        result = get_current_driver_type()
        assert result == "nouveau"

    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.installer.driver.check_nouveau")
    def test_no_driver(self, mock_nouveau, mock_working):
        """Test when no driver is loaded."""
        mock_working.return_value = MagicMock(is_working=False)
        mock_nouveau.return_value = False
        result = get_current_driver_type()
        assert result == "none"


class TestCheckNvidiaOpenAvailable:
    """Tests for check_nvidia_open_available function."""

    @patch("nvidia_inst.distro.detector.detect_distro")
    @patch("subprocess.run")
    def test_ubuntu_nvidia_open_available(self, mock_run, mock_distro):
        """Test nvidia-open available on Ubuntu."""
        mock_distro.return_value = MagicMock(id="ubuntu")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="nvidia-driver-535-open:\n  Candidate: 535.154.05\n",
        )
        result = check_nvidia_open_available()
        assert result is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    @patch("subprocess.run")
    def test_fedora_nvidia_open_available(self, mock_run, mock_distro):
        """Test nvidia-open available on Fedora."""
        mock_distro.return_value = MagicMock(id="fedora")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="xorg-x11-drv-nvidia-open.x86_64\n",
        )
        result = check_nvidia_open_available()
        assert result is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_opensuse_nvidia_open_available(self, mock_distro):
        """Test nvidia-open available on openSUSE."""
        mock_distro.return_value = MagicMock(id="opensuse")
        result = check_nvidia_open_available()
        assert result is True


class TestCheckNonfreeAvailable:
    """Tests for check_nonfree_available function."""

    @patch("nvidia_inst.distro.detector.detect_distro")
    @patch("subprocess.run")
    def test_fedora_nonfree_available(self, mock_run, mock_distro):
        """Test nonfree available on Fedora."""
        mock_distro.return_value = MagicMock(id="fedora")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="rpmfusion-nonfree\n",
        )
        result = check_nonfree_available()
        assert result is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_opensuse_nonfree_available(self, mock_distro):
        """Test nonfree available on openSUSE."""
        mock_distro.return_value = MagicMock(id="opensuse")
        result = check_nonfree_available()
        assert result is True


class TestDisableNouveau:
    """Tests for disable_nouveau function."""

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    @patch("nvidia_inst.distro.detector.detect_distro")
    @patch("builtins.open", create=True)
    @patch("subprocess.run")
    def test_disable_nouveau_success(
        self, mock_run, mock_open, mock_distro, mock_is_root
    ):
        """Test successful nouveau disable."""
        mock_distro.return_value = MagicMock(id="fedora")
        mock_run.return_value = MagicMock(returncode=0)

        result = disable_nouveau()
        assert result is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    @patch("subprocess.run")
    def test_disable_nouveau_sudo(self, mock_run, mock_is_root):
        """Test nouveau disable with sudo."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch("nvidia_inst.distro.detector.detect_distro") as mock_distro:
            mock_distro.return_value = MagicMock(id="ubuntu")
            result = disable_nouveau()
            assert result is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_disable_nouveau_os_error(self, mock_open, mock_is_root):
        """Test nouveau disable with OS error."""
        result = disable_nouveau()
        assert result is False
