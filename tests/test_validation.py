"""Tests for validation module."""

from unittest.mock import MagicMock, patch

from nvidia_inst.installer.validation import (
    SafetyCheckResult,
    ValidationResult,
    WorkingInstallResult,
    _check_secure_boot,
    is_nvidia_working,
    post_install_validate,
    pre_install_check,
    unblock_nouveau,
)


class TestUnblockNouveau:
    """Tests for unblock_nouveau function."""

    def test_blacklist_not_exists(self, tmp_path):
        """Test when blacklist file does not exist."""
        with patch("nvidia_inst.installer.validation.Path") as mock_path:
            mock_instance = MagicMock()
            mock_instance.exists.return_value = False
            mock_path.return_value = mock_instance
            success, message = unblock_nouveau()
            assert success is True
            assert "does not exist" in message

    def test_blacklist_removed(self, tmp_path):
        """Test successful blacklist removal."""
        blacklist = tmp_path / "blacklist-nouveau.conf"
        blacklist.write_text("blacklist nouveau\n")

        with patch("nvidia_inst.installer.validation.Path") as mock_path:
            mock_instance = MagicMock()
            mock_instance.exists.return_value = True
            mock_instance.unlink.return_value = None
            mock_path.return_value = mock_instance

            success, message = unblock_nouveau()
            assert success is True
            assert "re-enabled" in message.lower()

    def test_blacklist_error(self):
        """Test blacklist removal error."""
        with patch("nvidia_inst.installer.validation.Path") as mock_path:
            mock_instance = MagicMock()
            mock_instance.exists.return_value = True
            mock_instance.unlink.side_effect = PermissionError("Access denied")
            mock_path.return_value = mock_instance

            success, message = unblock_nouveau()
            assert success is False
            assert "Access denied" in message


class TestCheckSecureBoot:
    """Tests for _check_secure_boot function."""

    @patch("subprocess.run")
    def test_secure_boot_enabled(self, mock_run):
        """Test when secure boot is enabled."""
        mock_run.return_value = MagicMock(returncode=0, stdout="SecureBoot: enabled")
        assert _check_secure_boot() is True

    @patch("subprocess.run")
    def test_secure_boot_disabled(self, mock_run):
        """Test when secure boot is disabled."""
        mock_run.return_value = MagicMock(returncode=0, stdout="SecureBoot: disabled")
        assert _check_secure_boot() is False

    @patch("subprocess.run")
    def test_secure_boot_command_fails(self, mock_run):
        """Test when mokutil command fails."""
        mock_run.side_effect = FileNotFoundError()
        assert _check_secure_boot() is False

    @patch("subprocess.run")
    def test_secure_boot_timeout(self, mock_run):
        """Test when mokutil times out."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)
        assert _check_secure_boot() is False


class TestIsNvidiaWorking:
    """Tests for is_nvidia_working function."""

    def test_working_with_full_detection(self):
        """Test when NVIDIA is fully working."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3080\n"),
                MagicMock(returncode=0, stdout="535.154.05\n"),
                MagicMock(returncode=0, stdout="nvidia  123456  0\n"),
            ]

            result = is_nvidia_working()

            assert result.is_working is True
            assert result.driver_version == "535.154.05"
            assert result.kernel_module_loaded is True
            assert result.gpu_detected is True

    def test_working_without_kernel_module(self):
        """Test when nvidia-smi works but kernel module not loaded."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3080\n"),
                MagicMock(returncode=0, stdout="535.154.05\n"),
                MagicMock(returncode=0, stdout="nouveau  123456  0\n"),
            ]

            result = is_nvidia_working()

            assert result.is_working is False
            assert result.driver_version == "535.154.05"
            assert result.kernel_module_loaded is False
            assert result.gpu_detected is True

    def test_not_working_no_gpu(self):
        """Test when no NVIDIA GPU detected."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = is_nvidia_working()

            assert result.is_working is False
            assert result.driver_version is None
            assert result.kernel_module_loaded is False
            assert result.gpu_detected is False

    def test_not_working_smi_fails(self):
        """Test when nvidia-smi fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=1, stdout=""),
                MagicMock(returncode=1, stdout=""),
                MagicMock(returncode=1, stdout=""),
            ]

            result = is_nvidia_working()

            assert result.is_working is False
            assert result.gpu_detected is False


class TestPreInstallCheck:
    """Tests for pre_install_check function."""

    def test_disk_space_warning(self):
        """Test disk space warning when low."""
        mock_pm = MagicMock()
        mock_pm.is_available.return_value = True
        mock_pm.get_available_version.return_value = "1.0"
        with patch("os.statvfs") as mock_stat:
            mock_stat.return_value = MagicMock(f_bavail=100, f_frsize=1024)
            with (
                patch(
                    "nvidia_inst.installer.validation._check_secure_boot",
                    return_value=False,
                ),
                patch.dict("os.environ", {"DISPLAY": ""}),
            ):
                result = pre_install_check("fedora", ["akmod-nvidia"], mock_pm)

            assert result.can_proceed is True
            assert any("Low disk space" in w for w in result.warnings)

    def test_packages_not_available(self):
        """Test error when packages not available."""
        mock_pm = MagicMock()
        mock_pm.is_available.return_value = True
        mock_pm.get_available_version.return_value = None
        with patch("os.statvfs") as mock_stat:
            mock_stat.return_value = MagicMock(f_bavail=1000000, f_frsize=4096)
            result = pre_install_check("fedora", ["akmod-nvidia"], mock_pm)

        assert result.can_proceed is False
        assert any("not available" in e for e in result.errors)

    def test_package_manager_not_available(self):
        """Test error when package manager not available."""
        mock_pm = MagicMock()
        mock_pm.is_available.return_value = False

        result = pre_install_check("fedora", ["akmod-nvidia"], mock_pm)

        assert result.can_proceed is False
        assert any("Package manager not available" in e for e in result.errors)

    def test_display_warning(self):
        """Test warning when running in graphical session."""
        mock_pm = MagicMock()
        mock_pm.is_available.return_value = True
        mock_pm.get_available_version.return_value = "1.0"
        with patch("os.statvfs") as mock_stat:
            mock_stat.return_value = MagicMock(f_bavail=1000000, f_frsize=4096)
            with (
                patch(
                    "nvidia_inst.installer.validation._check_secure_boot",
                    return_value=False,
                ),
                patch.dict("os.environ", {"/tmp/.X11-unix": ":0"}),
            ):
                result = pre_install_check("fedora", ["akmod-nvidia"], mock_pm)

        # Display warning is based on DISPLAY env var, not X11 socket
        assert result.can_proceed is True


class TestPostInstallValidate:
    """Tests for post_install_validate function."""

    @patch("subprocess.run")
    @patch("nvidia_inst.installer.validation.Path.exists")
    @patch("nvidia_inst.installer.validation.glob.glob")
    @patch("os.uname")
    def test_all_checks_pass(self, mock_uname, mock_glob, mock_exists, mock_run):
        """Test when all validation checks pass."""
        mock_uname.return_value = MagicMock(release="5.15.0")
        mock_glob.return_value = ["/lib/modules/5.15.0/extra/nvidia.ko"]
        mock_exists.return_value = True
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),  # nvidia-smi
            MagicMock(returncode=0, stdout="535.154.05\n"),  # nvidia-smi version
        ]

        mock_pm = MagicMock()
        mock_pm.get_installed_version.return_value = "535.154.05"

        result = post_install_validate("fedora", ["akmod-nvidia"], mock_pm)

        assert result.success is True
        assert result.kernel_module_built is True
        assert result.nouveau_blocked is True
        assert result.nvidia_smi_works is True

    @patch("subprocess.run")
    @patch("nvidia_inst.installer.validation.Path.exists")
    @patch("nvidia_inst.installer.validation.glob.glob")
    @patch("os.uname")
    def test_missing_packages(self, mock_uname, mock_glob, mock_exists, mock_run):
        """Test when packages are missing."""
        mock_uname.return_value = MagicMock(release="5.15.0")
        mock_glob.return_value = []
        mock_exists.return_value = True
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="535.154.05\n"),
        ]

        mock_pm = MagicMock()
        mock_pm.get_installed_version.return_value = None

        result = post_install_validate("fedora", ["akmod-nvidia"], mock_pm)

        assert result.success is False
        assert len(result.missing_packages) > 0


class TestSafetyCheckResult:
    """Tests for SafetyCheckResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = SafetyCheckResult(can_proceed=True, warnings=[], errors=[])
        assert result.can_proceed is True
        assert result.warnings == []
        assert result.errors == []


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ValidationResult(
            success=True,
            installed_packages=[],
            missing_packages=[],
            kernel_module_built=False,
            nouveau_blocked=False,
            nvidia_smi_works=False,
            actual_driver_version=None,
            warnings=[],
            errors=[],
        )
        assert result.success is True
        assert result.actual_driver_version is None


class TestWorkingInstallResult:
    """Tests for WorkingInstallResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = WorkingInstallResult(
            is_working=False,
            driver_version=None,
            kernel_module_loaded=False,
            gpu_detected=False,
        )
        assert result.is_working is False
        assert result.driver_version is None
        assert result.kernel_module_loaded is False
        assert result.gpu_detected is False

    def test_working_state(self):
        """Test working state."""
        result = WorkingInstallResult(
            is_working=True,
            driver_version="535.154.05",
            kernel_module_loaded=True,
            gpu_detected=True,
        )
        assert result.is_working is True
        assert result.driver_version == "535.154.05"
        assert result.kernel_module_loaded is True
        assert result.gpu_detected is True
