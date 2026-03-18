"""Tests for the installer module."""

from unittest.mock import MagicMock, patch

from nvidia_inst.installer.driver import (
    check_nouveau,
    check_secure_boot,
    disable_nouveau,
    get_compatible_driver_packages,
)


class TestNouveauCheck:
    """Test Nouveau kernel module detection."""

    @patch("subprocess.run")
    def test_nouveau_loaded(self, mock_run):
        """Test Nouveau is detected when loaded."""
        mock_run.return_value = MagicMock(
            stdout="nouveau  1638400  0\n",
            returncode=0,
        )
        assert check_nouveau() is True

    @patch("subprocess.run")
    def test_nouveau_not_loaded(self, mock_run):
        """Test Nouveau not detected when not loaded."""
        mock_run.return_value = MagicMock(
            stdout="Module                  Size  Used by\n",
            returncode=0,
        )
        assert check_nouveau() is False


class TestSecureBootCheck:
    """Test Secure Boot detection."""

    @patch("subprocess.run")
    def test_secure_boot_enabled(self, mock_run):
        """Test Secure Boot enabled detection."""
        mock_run.return_value = MagicMock(
            stdout="SecureBoot enabled\n",
            returncode=0,
        )
        assert check_secure_boot() is True

    @patch("subprocess.run")
    def test_secure_boot_disabled(self, mock_run):
        """Test Secure Boot disabled detection."""
        mock_run.return_value = MagicMock(
            stdout="SecureBoot disabled\n",
            returncode=0,
        )
        assert check_secure_boot() is False


class TestGetCompatiblePackages:
    """Test compatible package detection."""

    def test_ubuntu_packages(self):
        """Test Ubuntu driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="535.154.05",
            max_version=None,
            cuda_min="11.8",
            cuda_max="12.2",
            is_eol=False,
        )
        packages = get_compatible_driver_packages("ubuntu", driver_range)
        assert "nvidia-driver-535" in packages or "nvidia-driver-550" in packages

    def test_ubuntu_eol_packages(self):
        """Test Ubuntu EOL driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="470.256.02",
            max_version="470.256.02",
            cuda_min="9.0",
            cuda_max="9.0",
            is_eol=True,
        )
        packages = get_compatible_driver_packages("ubuntu", driver_range)
        assert "nvidia-driver-470" in packages

    def test_fedora_packages(self):
        """Test Fedora driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="535.154.05",
            max_version=None,
            cuda_min="11.8",
            cuda_max="12.2",
            is_eol=False,
        )
        packages = get_compatible_driver_packages("fedora", driver_range)
        assert "akmod-nvidia" in packages
        assert "xorg-x11-drv-nvidia" in packages

    def test_arch_packages(self):
        """Test Arch Linux driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="535.154.05",
            max_version=None,
            cuda_min="11.8",
            cuda_max="12.2",
            is_eol=False,
        )
        packages = get_compatible_driver_packages("arch", driver_range)
        assert "nvidia-open" in packages
        assert "nvidia-utils" in packages

    def test_arch_eol_packages(self):
        """Test Arch Linux EOL driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="470.256.02",
            max_version="470.256.02",
            cuda_min="9.0",
            cuda_max="9.0",
            is_eol=True,
        )
        packages = get_compatible_driver_packages("arch", driver_range)
        assert "nvidia-470xx-dkms" in packages
        assert "nvidia-470xx-utils" in packages

    def test_debian_packages(self):
        """Test Debian driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="535.154.05",
            max_version=None,
            cuda_min="11.8",
            cuda_max="12.2",
            is_eol=False,
        )
        packages = get_compatible_driver_packages("debian", driver_range)
        assert "nvidia-driver" in packages

    def test_opensuse_packages(self):
        """Test openSUSE driver packages."""
        from nvidia_inst.gpu.compatibility import DriverRange
        driver_range = DriverRange(
            min_version="535.154.05",
            max_version=None,
            cuda_min="11.8",
            cuda_max="12.2",
            is_eol=False,
        )
        packages = get_compatible_driver_packages("opensuse", driver_range)
        assert "x11-video-nvidiaG05" in packages
        assert "nvidia-computeG05" in packages


class TestDisableNouveau:
    """Test Nouveau disable function."""

    def test_disable_nouveau_no_root(self):
        """Test disable_nouveau fails without root privileges."""
        with patch("os.geteuid", return_value=1000):
            result = disable_nouveau()
            assert result is False

    @patch("os.geteuid")
    @patch("builtins.open", create=True)
    @patch("subprocess.run")
    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_disable_nouveau_ubuntu(self, mock_detect, mock_run, mock_open, mock_geteuid):
        """Test disable_nouveau on Ubuntu."""
        mock_geteuid.return_value = 0

        mock_file_instance = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_instance

        mock_detect.return_value = MagicMock(id="ubuntu")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = disable_nouveau()
        assert result is True

        mock_file_instance.write.assert_called()

    @patch("os.geteuid")
    @patch("builtins.open", create=True)
    @patch("subprocess.run")
    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_disable_nouveau_fedora(self, mock_detect, mock_run, mock_open, mock_geteuid):
        """Test disable_nouveau on Fedora uses dracut."""
        mock_geteuid.return_value = 0

        mock_file_instance = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_instance

        mock_detect.return_value = MagicMock(id="fedora")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = disable_nouveau()
        assert result is True

        call_args = mock_run.call_args[0][0]
        assert "dracut" in call_args

    @patch("os.geteuid")
    @patch("builtins.open", create=True)
    @patch("subprocess.run")
    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_disable_nouveau_arch(self, mock_detect, mock_run, mock_open, mock_geteuid):
        """Test disable_nouveau on Arch uses mkinitcpio."""
        mock_geteuid.return_value = 0

        mock_file_instance = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_instance

        mock_detect.return_value = MagicMock(id="arch")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = disable_nouveau()
        assert result is True

        call_args = mock_run.call_args[0][0]
        assert "mkinitcpio" in call_args

    @patch("os.geteuid")
    @patch("builtins.open", create=True)
    @patch("subprocess.run")
    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_disable_nouveau_initramfs_failure(self, mock_detect, mock_run, mock_open, mock_geteuid):
        """Test disable_nouveau handles initramfs rebuild failure."""
        mock_geteuid.return_value = 0

        mock_file_instance = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_instance

        mock_detect.return_value = MagicMock(id="ubuntu")

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        result = disable_nouveau()
        assert result is False

    @patch("os.geteuid")
    @patch("builtins.open", create=True)
    def test_disable_nouveau_file_write_error(self, mock_open, mock_geteuid):
        """Test disable_nouveau handles file write error."""
        mock_geteuid.return_value = 0
        mock_open.side_effect = OSError("Disk full")

        result = disable_nouveau()
        assert result is False

    @patch("os.geteuid")
    @patch("builtins.open", create=True)
    @patch("subprocess.run")
    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_disable_nouveau_unknown_distro(self, mock_detect, mock_run, mock_open, mock_geteuid):
        """Test disable_nouveau on unknown distro uses update-initramfs."""
        mock_geteuid.return_value = 0

        mock_file_instance = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_instance

        mock_detect.return_value = MagicMock(id="unknown")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = disable_nouveau()
        assert result is True

        call_args = mock_run.call_args[0][0]
        assert "update-initramfs" in call_args
