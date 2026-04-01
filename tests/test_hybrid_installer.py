"""Tests for installer/hybrid.py module."""

from unittest.mock import MagicMock, patch

from nvidia_inst.installer.hybrid import (
    configure_prime_env,
    get_hybrid_packages,
    get_power_profile,
    install_hybrid_packages,
    is_prime_env_configured,
    set_power_profile,
)


class TestGetHybridPackages:
    """Tests for get_hybrid_packages function."""

    def test_fedora_packages(self):
        """Test Fedora returns switcheroo-control."""
        packages = get_hybrid_packages("fedora")
        assert "switcheroo-control" in packages

    def test_ubuntu_packages(self):
        """Test Ubuntu returns empty list."""
        packages = get_hybrid_packages("ubuntu")
        assert packages == []

    def test_arch_packages(self):
        """Test Arch returns empty list."""
        packages = get_hybrid_packages("arch")
        assert packages == []

    def test_opensuse_packages(self):
        """Test openSUSE returns empty list."""
        packages = get_hybrid_packages("opensuse")
        assert packages == []

    def test_unknown_distro_packages(self):
        """Test unknown distro returns empty list."""
        packages = get_hybrid_packages("unknown")
        assert packages == []


class TestInstallHybridPackages:
    """Tests for install_hybrid_packages function."""

    def test_no_packages_needed(self):
        """Test when no packages are needed."""
        result = install_hybrid_packages([], "apt")
        assert result is True

    @patch("subprocess.run")
    def test_apt_install_success(self, mock_run):
        """Test successful apt install."""
        mock_run.return_value = MagicMock(returncode=0)
        result = install_hybrid_packages(["pkg1"], "apt")
        assert result is True

    @patch("subprocess.run")
    def test_apt_install_failure(self, mock_run):
        """Test failed apt install."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "apt")
        result = install_hybrid_packages(["pkg1"], "apt")
        assert result is False

    @patch("subprocess.run")
    def test_dnf_install_success(self, mock_run):
        """Test successful dnf install."""
        mock_run.return_value = MagicMock(returncode=0)
        result = install_hybrid_packages(["pkg1"], "dnf")
        assert result is True

    @patch("subprocess.run")
    def test_pacman_install_success(self, mock_run):
        """Test successful pacman install."""
        mock_run.return_value = MagicMock(returncode=0)
        result = install_hybrid_packages(["pkg1"], "pacman")
        assert result is True

    @patch("subprocess.run")
    def test_zypper_install_success(self, mock_run):
        """Test successful zypper install."""
        mock_run.return_value = MagicMock(returncode=0)
        result = install_hybrid_packages(["pkg1"], "zypper")
        assert result is True

    def test_unknown_package_manager(self):
        """Test unknown package manager returns False."""
        result = install_hybrid_packages(["pkg1"], "unknown")
        assert result is False


class TestConfigurePrimeEnv:
    """Tests for configure_prime_env function."""

    @patch("os.geteuid", return_value=1000)
    def test_non_root_fails(self, mock_geteuid):
        """Test non-root fails."""
        result = configure_prime_env()
        assert result is False

    @patch("os.geteuid", return_value=0)
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_root_success(self, mock_mkdir, mock_write, mock_geteuid):
        """Test root success."""
        result = configure_prime_env()
        assert result is True


class TestGetPowerProfile:
    """Tests for get_power_profile function."""

    @patch("subprocess.run")
    def test_prime_select(self, mock_run):
        """Test getting prime-select mode."""
        mock_run.return_value = MagicMock(stdout="nvidia\n", returncode=0)
        result = get_power_profile("nvidia-prime")
        assert result == "nvidia"

    @patch("subprocess.run")
    def test_switcherooctl(self, mock_run):
        """Test getting switcherooctl mode."""
        mock_run.return_value = MagicMock(stdout="integrated\n", returncode=0)
        result = get_power_profile("switcherooctl")
        assert result == "integrated"

    @patch("subprocess.run")
    def test_system76(self, mock_run):
        """Test getting system76-power mode."""
        mock_run.return_value = MagicMock(stdout="hybrid\n", returncode=0)
        result = get_power_profile("system76-power")
        assert result == "hybrid"

    def test_unknown_tool(self):
        """Test unknown tool returns None."""
        result = get_power_profile("unknown")
        assert result is None

    def test_none_tool(self):
        """Test None tool returns None."""
        result = get_power_profile(None)
        assert result is None


class TestSetPowerProfile:
    """Tests for set_power_profile function."""

    @patch("subprocess.run")
    def test_prime_select_intel(self, mock_run):
        """Test setting prime-select to intel."""
        mock_run.return_value = MagicMock(returncode=0)
        result = set_power_profile("intel", "nvidia-prime", "ubuntu")
        assert result is True

    @patch("subprocess.run")
    def test_prime_select_nvidia(self, mock_run):
        """Test setting prime-select to nvidia."""
        mock_run.return_value = MagicMock(returncode=0)
        result = set_power_profile("nvidia", "nvidia-prime", "ubuntu")
        assert result is True

    @patch("os.geteuid", return_value=0)
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.exists", return_value=False)
    def test_switcherooctl_intel(self, mock_exists, mock_write, mock_geteuid):
        """Test setting switcherooctl to intel."""
        result = set_power_profile("intel", "switcherooctl", "fedora")
        assert result is True

    @patch("subprocess.run")
    def test_system76_intel(self, mock_run):
        """Test setting system76-power to intel."""
        mock_run.return_value = MagicMock(returncode=0)
        result = set_power_profile("intel", "system76-power", "pop")
        assert result is True

    def test_none_tool(self):
        """Test None tool returns False."""
        result = set_power_profile("intel", None, "ubuntu")
        assert result is False


class TestIsPrimeEnvConfigured:
    """Tests for is_prime_env_configured function."""

    @patch("pathlib.Path.exists", return_value=False)
    def test_not_configured(self, mock_exists):
        """Test when not configured."""
        result = is_prime_env_configured()
        assert result is False

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text", return_value="__NV_PRIME_RENDER_OFFLOAD=1\n")
    def test_configured(self, mock_read, mock_exists):
        """Test when configured."""
        result = is_prime_env_configured()
        assert result is True

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text", return_value="# empty file\n")
    def test_file_exists_but_not_configured(self, mock_read, mock_exists):
        """Test when file exists but not configured."""
        result = is_prime_env_configured()
        assert result is False
