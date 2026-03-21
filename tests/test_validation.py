"""Tests for validation module."""

from unittest.mock import MagicMock, patch

from nvidia_inst.installer.validation import (
    SafetyCheckResult,
    ValidationResult,
    WorkingInstallResult,
    is_nvidia_working,
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
