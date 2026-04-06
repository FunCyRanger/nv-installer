"""Tests for CUDA installation module (installer/cuda.py).

Covers:
- CUDAInstaller abstract base class
- UbuntuCUDAInstaller, FedoraCUDAInstaller, ArchCUDAInstaller
- detect_installed_cuda_version() - rpm, dpkg, pacman detection
- setup_cuda_environment() / remove_cuda_environment()
- pin_cuda_to_major_version() / pin_cuda_to_exact_version()
- check_cuda_driver_compatibility() - CUDA 10/11/12 vs driver versions
- get_cuda_packages_for_version() - all distros
- get_uninstall_cuda_packages() - all distros, with/without version
"""

import subprocess
from unittest.mock import MagicMock, patch

from nvidia_inst.installer.cuda import (
    ArchCUDAInstaller,
    FedoraCUDAInstaller,
    UbuntuCUDAInstaller,
    check_cuda_driver_compatibility,
    detect_installed_cuda_version,
    get_cuda_installer,
    get_cuda_packages_for_version,
    get_uninstall_cuda_packages,
    pin_cuda_to_exact_version,
    pin_cuda_to_major_version,
    remove_cuda_environment,
    setup_cuda_environment,
)


class TestUbuntuCUDAInstaller:
    """Tests for UbuntuCUDAInstaller."""

    def test_get_cuda_packages_no_version(self):
        """Test getting CUDA packages without version."""
        installer = UbuntuCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert "cuda-toolkit" in packages

    def test_get_cuda_packages_with_version(self):
        """Test getting CUDA packages with version."""
        installer = UbuntuCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert "cuda-toolkit" in packages

    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    def test_is_cuda_installed_true(self, mock_detect):
        """Test is_cuda_installed returns True."""
        mock_detect.return_value = "12.2"
        installer = UbuntuCUDAInstaller()
        assert installer.is_cuda_installed() is True

    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    def test_is_cuda_installed_false(self, mock_detect):
        """Test is_cuda_installed returns False."""
        mock_detect.return_value = None
        installer = UbuntuCUDAInstaller()
        assert installer.is_cuda_installed() is False

    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    def test_get_installed_cuda_version(self, mock_detect):
        """Test get_installed_cuda_version."""
        mock_detect.return_value = "12.2"
        installer = UbuntuCUDAInstaller()
        assert installer.get_installed_cuda_version() == "12.2"


class TestFedoraCUDAInstaller:
    """Tests for FedoraCUDAInstaller."""

    def test_get_cuda_packages_no_version(self):
        """Test getting CUDA packages without version."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert "cuda-toolkit" in packages

    def test_get_cuda_packages_with_version(self):
        """Test getting CUDA packages with version."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert any("cuda-toolkit-12" in pkg for pkg in packages)

    def test_get_cuda_packages_version_13(self):
        """Test getting CUDA 13 packages."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages("13.0")
        assert any("cuda-toolkit-13" in pkg for pkg in packages)

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    def test_is_cuda_installed_true(self, mock_detect):
        """Test is_cuda_installed returns True."""
        mock_detect.return_value = "12.6"
        installer = FedoraCUDAInstaller()
        assert installer.is_cuda_installed() is True

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    def test_is_cuda_installed_false(self, mock_detect):
        """Test is_cuda_installed returns False."""
        mock_detect.return_value = None
        installer = FedoraCUDAInstaller()
        assert installer.is_cuda_installed() is False

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    def test_get_installed_cuda_version(self, mock_detect):
        """Test get_installed_cuda_version."""
        mock_detect.return_value = "12.6"
        installer = FedoraCUDAInstaller()
        assert installer.get_installed_cuda_version() == "12.6"


class TestArchCUDAInstaller:
    """Tests for ArchCUDAInstaller."""

    def test_get_cuda_packages_no_version(self):
        """Test getting CUDA packages without version."""
        installer = ArchCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert "cuda" in packages

    def test_get_cuda_packages_with_version(self):
        """Test getting CUDA packages with version (Arch ignores version)."""
        installer = ArchCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert "cuda" in packages

    @patch("nvidia_inst.installer.cuda._detect_via_pacman")
    def test_is_cuda_installed_true(self, mock_detect):
        """Test is_cuda_installed returns True."""
        mock_detect.return_value = "12.6"
        installer = ArchCUDAInstaller()
        assert installer.is_cuda_installed() is True

    @patch("nvidia_inst.installer.cuda._detect_via_pacman")
    def test_is_cuda_installed_false(self, mock_detect):
        """Test is_cuda_installed returns False."""
        mock_detect.return_value = None
        installer = ArchCUDAInstaller()
        assert installer.is_cuda_installed() is False


class TestGetCudaInstaller:
    """Tests for get_cuda_installer() factory function."""

    def test_ubuntu_installer(self):
        """Test getting Ubuntu installer."""
        installer = get_cuda_installer("ubuntu")
        assert isinstance(installer, UbuntuCUDAInstaller)

    def test_debian_installer(self):
        """Test getting Debian installer."""
        installer = get_cuda_installer("debian")
        assert isinstance(installer, UbuntuCUDAInstaller)

    def test_fedora_installer(self):
        """Test getting Fedora installer."""
        installer = get_cuda_installer("fedora")
        assert isinstance(installer, FedoraCUDAInstaller)

    def test_rhel_installer(self):
        """Test getting RHEL installer."""
        installer = get_cuda_installer("rhel")
        assert isinstance(installer, FedoraCUDAInstaller)

    def test_arch_installer(self):
        """Test getting Arch installer."""
        installer = get_cuda_installer("arch")
        assert isinstance(installer, ArchCUDAInstaller)

    def test_manjaro_installer(self):
        """Test getting Manjaro installer."""
        installer = get_cuda_installer("manjaro")
        assert isinstance(installer, ArchCUDAInstaller)

    def test_unknown_distro_fallback(self):
        """Test unknown distro falls back to Ubuntu installer."""
        installer = get_cuda_installer("unknown")
        assert isinstance(installer, UbuntuCUDAInstaller)


class TestDetectInstalledCudaVersion:
    """Tests for detect_installed_cuda_version()."""

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    @patch("nvidia_inst.installer.cuda._detect_via_pacman")
    def test_detect_via_rpm_first(self, mock_pacman, mock_dpkg, mock_rpm):
        """Test RPM detection is tried first."""
        mock_rpm.return_value = "12.6"
        result = detect_installed_cuda_version()
        assert result == "12.6"
        mock_dpkg.assert_not_called()

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    @patch("nvidia_inst.installer.cuda._detect_via_pacman")
    def test_detect_via_dpkg_fallback(self, mock_pacman, mock_dpkg, mock_rpm):
        """Test DPKG detection when RPM fails."""
        mock_rpm.return_value = None
        mock_dpkg.return_value = "12.2"
        result = detect_installed_cuda_version()
        assert result == "12.2"
        mock_pacman.assert_not_called()

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    @patch("nvidia_inst.installer.cuda._detect_via_pacman")
    def test_detect_via_pacman_fallback(self, mock_pacman, mock_dpkg, mock_rpm):
        """Test pacman detection when RPM and DPKG fail."""
        mock_rpm.return_value = None
        mock_dpkg.return_value = None
        mock_pacman.return_value = "12.6"
        result = detect_installed_cuda_version()
        assert result == "12.6"

    @patch("nvidia_inst.installer.cuda._detect_via_rpm")
    @patch("nvidia_inst.installer.cuda._detect_via_dpkg")
    @patch("nvidia_inst.installer.cuda._detect_via_pacman")
    def test_detect_no_cuda(self, mock_pacman, mock_dpkg, mock_rpm):
        """Test when no CUDA is installed."""
        mock_rpm.return_value = None
        mock_dpkg.return_value = None
        mock_pacman.return_value = None
        result = detect_installed_cuda_version()
        assert result is None


class TestSetupCudaEnvironment:
    """Tests for setup_cuda_environment()."""

    @patch("subprocess.run")
    def test_setup_success(self, mock_run):
        """Test successful CUDA environment setup."""
        mock_run.return_value = MagicMock(returncode=0)
        success, message = setup_cuda_environment()
        assert success is True
        assert "cuda.sh" in message

    @patch("subprocess.run")
    def test_setup_failure(self, mock_run):
        """Test CUDA environment setup failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        success, message = setup_cuda_environment()
        assert success is False

    @patch("subprocess.run")
    def test_setup_timeout(self, mock_run):
        """Test CUDA environment setup timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)
        success, message = setup_cuda_environment()
        assert success is False
        assert "Timeout" in message


class TestRemoveCudaEnvironment:
    """Tests for remove_cuda_environment()."""

    @patch("subprocess.run")
    def test_remove_success(self, mock_run):
        """Test successful CUDA environment removal."""
        mock_run.return_value = MagicMock(returncode=0)
        success, message = remove_cuda_environment()
        assert success is True
        assert "Removed" in message

    @patch("subprocess.run")
    def test_remove_failure(self, mock_run):
        """Test CUDA environment removal failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        success, message = remove_cuda_environment()
        assert success is False


class TestPinCudaToMajorVersion:
    """Tests for pin_cuda_to_major_version()."""

    def test_pin_ubuntu_major(self):
        """Test pinning CUDA to major version on Ubuntu."""
        mock_pkg_mgr = MagicMock()
        mock_pkg_mgr.pin_version.return_value = True
        result = pin_cuda_to_major_version("ubuntu", "12", mock_pkg_mgr)
        assert result is True

    def test_pin_fedora_major(self):
        """Test pinning CUDA to major version on Fedora."""
        mock_pkg_mgr = MagicMock()
        mock_pkg_mgr.pin_version.return_value = True
        result = pin_cuda_to_major_version("fedora", "12", mock_pkg_mgr)
        assert result is True

    def test_pin_partial_failure(self):
        """Test pinning with partial failure."""
        mock_pkg_mgr = MagicMock()
        mock_pkg_mgr.pin_version.side_effect = [True, False]
        result = pin_cuda_to_major_version("ubuntu", "12", mock_pkg_mgr)
        assert result is False


class TestPinCudaToExactVersion:
    """Tests for pin_cuda_to_exact_version()."""

    def test_pin_ubuntu_exact(self):
        """Test pinning CUDA to exact version on Ubuntu."""
        mock_pkg_mgr = MagicMock()
        mock_pkg_mgr.pin_version.return_value = True
        result = pin_cuda_to_exact_version("ubuntu", "12.2", mock_pkg_mgr)
        assert result is True

    def test_pin_failure(self):
        """Test pinning exact version failure."""
        mock_pkg_mgr = MagicMock()
        mock_pkg_mgr.pin_version.return_value = False
        result = pin_cuda_to_exact_version("ubuntu", "12.2", mock_pkg_mgr)
        assert result is False


class TestCheckCudaDriverCompatibility:
    """Tests for check_cuda_driver_compatibility()."""

    def test_cuda_12_with_modern_driver(self):
        """Test CUDA 12 compatibility with modern driver."""
        result = check_cuda_driver_compatibility("12.0", "535.154.05")
        assert result == (True, "CUDA 12.0 is compatible with driver 535.154.05")

    def test_cuda_11_with_modern_driver(self):
        """Test CUDA 11 compatibility with modern driver."""
        result = check_cuda_driver_compatibility("11.8", "535.154.05")
        assert result == (True, "CUDA 11.8 is compatible with driver 535.154.05")

    def test_cuda_10_with_old_driver(self):
        """Test CUDA 10 compatibility with older driver."""
        result = check_cuda_driver_compatibility("10.2", "450.80.02")
        assert result == (True, "CUDA 10.2 is compatible with driver 450.80.02")

    def test_cuda_12_with_old_driver(self):
        """Test CUDA 12 with very old driver (incompatible)."""
        result = check_cuda_driver_compatibility("12.0", "470.57.02")
        assert result == (False, "CUDA 12.0 requires driver 525+, found 470.57.02")


class TestGetCudaPackagesForVersion:
    """Tests for get_cuda_packages_for_version()."""

    def test_ubuntu_packages(self):
        """Test CUDA packages for Ubuntu."""
        packages = get_cuda_packages_for_version("ubuntu", "12.2")
        assert isinstance(packages, list)
        assert len(packages) > 0

    def test_fedora_packages(self):
        """Test CUDA packages for Fedora."""
        packages = get_cuda_packages_for_version("fedora", "12.2")
        assert isinstance(packages, list)
        assert len(packages) > 0

    def test_arch_packages(self):
        """Test CUDA packages for Arch."""
        packages = get_cuda_packages_for_version("arch", "12.2")
        assert isinstance(packages, list)

    def test_opensuse_packages(self):
        """Test CUDA packages for openSUSE."""
        packages = get_cuda_packages_for_version("opensuse", "12.2")
        assert isinstance(packages, list)

    def test_unknown_distro(self):
        """Test CUDA packages for unknown distro."""
        packages = get_cuda_packages_for_version("unknown", "12.2")
        assert isinstance(packages, list)


class TestGetUninstallCudaPackages:
    """Tests for get_uninstall_cuda_packages()."""

    def test_ubuntu_uninstall(self):
        """Test CUDA uninstall packages for Ubuntu."""
        packages = get_uninstall_cuda_packages("ubuntu")
        assert isinstance(packages, list)

    def test_fedora_uninstall(self):
        """Test CUDA uninstall packages for Fedora."""
        packages = get_uninstall_cuda_packages("fedora")
        assert isinstance(packages, list)

    def test_arch_uninstall(self):
        """Test CUDA uninstall packages for Arch."""
        packages = get_uninstall_cuda_packages("arch")
        assert isinstance(packages, list)

    def test_opensuse_uninstall(self):
        """Test CUDA uninstall packages for openSUSE."""
        packages = get_uninstall_cuda_packages("opensuse")
        assert isinstance(packages, list)

    def test_unknown_distro_uninstall(self):
        """Test CUDA uninstall packages for unknown distro."""
        packages = get_uninstall_cuda_packages("unknown")
        assert packages == []

    def test_uninstall_with_version(self):
        """Test CUDA uninstall packages with specific version."""
        packages = get_uninstall_cuda_packages("ubuntu", "12.2")
        assert isinstance(packages, list)
