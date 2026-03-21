"""Tests for error handling scenarios."""

from unittest.mock import MagicMock, patch

from nvidia_inst.installer.driver import (
    DriverInstallError,
    NouveauLoadedError,
    SecureBootError,
    check_nouveau,
    check_secure_boot,
    disable_nouveau,
)


class TestNoGPUDetected:
    """Tests for no GPU detected scenarios."""

    def test_has_nvidia_gpu_returns_false(self):
        """Test has_nvidia_gpu returns False when no GPU."""
        with patch("nvidia_inst.gpu.detector.has_nvidia_gpu", return_value=False):
            from nvidia_inst.gpu.detector import has_nvidia_gpu

            assert has_nvidia_gpu() is False

    def test_detect_gpu_returns_none(self):
        """Test detect_gpu returns None when no GPU."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            from nvidia_inst.gpu.detector import detect_gpu

            result = detect_gpu()
            assert result is None

    def test_lspci_no_nvidia(self):
        """Test lspci returns empty when no Nvidia GPU."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="VGA compatible controller: Intel Corporation\nAudio device: Realtek",
            )
            from nvidia_inst.gpu.detector import _detect_gpu_lspci

            result = _detect_gpu_lspci()
            assert result is None


class TestNouveauLoaded:
    """Tests for Nouveau kernel module detection."""

    def test_nouveau_loaded_detected(self):
        """Test Nouveau loaded is detected."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="nouveau  1638400  0\n",
            )
            assert check_nouveau() is True

    def test_nouveau_not_loaded(self):
        """Test Nouveau not loaded."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Module                  Size  Used by\n",
            )
            assert check_nouveau() is False


class TestSecureBoot:
    """Tests for Secure Boot detection."""

    def test_secure_boot_enabled(self):
        """Test Secure Boot enabled is detected."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="SecureBoot enabled\n",
            )
            assert check_secure_boot() is True

    def test_secure_boot_disabled(self):
        """Test Secure Boot disabled."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="SecureBoot disabled\n",
            )
            assert check_secure_boot() is False

    def test_mokutil_not_found(self):
        """Test check_secure_boot handles mokutil not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("mokutil not found")
            assert check_secure_boot() is False


class TestDistroDetectionErrors:
    """Tests for distribution detection errors."""

    def test_distro_info_dataclass(self):
        """Test DistroInfo dataclass can be instantiated."""
        from nvidia_inst.distro.detector import DistroInfo

        distro = DistroInfo(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0-91-generic",
        )
        assert distro.id == "ubuntu"
        assert distro.version_id == "22.04"
        assert "Ubuntu" in str(distro)


class TestGPUDetectionErrors:
    """Tests for GPU detection errors."""

    def test_gpu_info_dataclass(self):
        """Test GPUInfo dataclass can be instantiated."""
        from nvidia_inst.gpu.detector import GPUInfo

        gpu = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            compute_capability=8.6,
            driver_version="535.154.05",
        )
        assert gpu.model == "NVIDIA GeForce RTX 3080"
        assert gpu.compute_capability == 8.6


class TestPackageManagerErrors:
    """Tests for package manager error handling."""

    def test_package_manager_instantiation(self):
        """Test package managers can be instantiated."""
        from nvidia_inst.distro.apt import AptManager
        from nvidia_inst.distro.dnf import DnfManager
        from nvidia_inst.distro.pacman import PacmanManager
        from nvidia_inst.distro.zypper import ZypperManager

        assert AptManager() is not None
        assert DnfManager() is not None
        assert PacmanManager() is not None
        assert ZypperManager() is not None


class TestNetworkErrors:
    """Tests for network-related errors."""

    def test_matrix_manager_instantiation(self):
        """Test MatrixManager can be instantiated."""
        from nvidia_inst.gpu.matrix.manager import MatrixManager

        manager = MatrixManager()
        assert manager is not None
        assert hasattr(manager, "check_for_updates")


class TestValidationErrors:
    """Tests for post-installation validation errors."""

    def test_driver_not_loaded_after_install(self):
        """Test validation when driver not loaded after install."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="No devices found")
            mock_pm = MagicMock()
            mock_pm.get_installed_version.return_value = None
            from nvidia_inst.installer.validation import post_install_validate

            result = post_install_validate("ubuntu", ["nvidia-driver-535"], mock_pm)
            assert result.success is False

    def test_nvidia_smi_fails_post_install(self):
        """Test validation when nvidia-smi fails post-install (needs reboot)."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            mock_pm = MagicMock()
            mock_pm.get_installed_version.return_value = "535.154.05"
            from nvidia_inst.installer.validation import post_install_validate

            result = post_install_validate("ubuntu", ["nvidia-driver-535"], mock_pm)
            assert result.success is True
            assert result.nvidia_smi_works is False
            assert any("reboot" in w.lower() for w in result.warnings)


class TestDisableNouveauErrors:
    """Tests for disable_nouveau error handling."""

    def test_disable_nouveau_no_root(self):
        """Test disable_nouveau fails without root."""
        with patch("os.geteuid", return_value=1000):
            assert disable_nouveau() is False

    def test_disable_nouveau_file_write_error(self):
        """Test disable_nouveau handles file write error."""
        with (
            patch("os.geteuid", return_value=0),
            patch("builtins.open", side_effect=OSError("Disk full")),
        ):
            assert disable_nouveau() is False

    def test_disable_nouveau_initramfs_fails(self):
        """Test disable_nouveau handles initramfs rebuild failure."""
        with (
            patch("os.geteuid", return_value=0),
            patch("builtins.open", create=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="Error")
            assert disable_nouveau() is False

    def test_disable_nouveau_unknown_error(self):
        """Test disable_nouveau handles unknown errors."""
        with (
            patch("os.geteuid", return_value=0),
            patch("builtins.open", create=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = Exception("Unknown error")
            assert disable_nouveau() is False


class TestDriverInstallErrors:
    """Tests for driver installation errors."""

    def test_driver_install_error(self):
        """Test DriverInstallError is raised correctly."""
        error = DriverInstallError("Installation failed")
        assert str(error) == "Installation failed"
        assert isinstance(error, Exception)

    def test_nouveau_loaded_error(self):
        """Test NouveauLoadedError is raised correctly."""
        error = NouveauLoadedError("Nouveau is loaded")
        assert str(error) == "Nouveau is loaded"
        assert isinstance(error, Exception)

    def test_secure_boot_error(self):
        """Test SecureBootError is raised correctly."""
        error = SecureBootError("Secure Boot enabled")
        assert str(error) == "Secure Boot enabled"
        assert isinstance(error, Exception)


class TestPrerequisitesErrors:
    """Tests for prerequisites checking errors."""

    def test_package_manager_not_found(self):
        """Test when package manager not found."""
        with patch("pathlib.Path.exists", return_value=False):
            from nvidia_inst.installer.prerequisites import PrerequisitesChecker

            checker = PrerequisitesChecker()
            pm_available, pm_name = checker._check_package_manager()
            assert pm_available is False
            assert pm_name == ""

    def test_repo_check_timeout(self):
        """Test repo check handles timeout."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)
            from nvidia_inst.installer.prerequisites import PrerequisitesChecker

            checker = PrerequisitesChecker()
            exists = checker._repo_exists("rpmfusion")
            assert exists is False
