"""Tests for CLI installer orchestration functions.

Covers:
- get_packages_to_remove() for all tools
- remove_packages() - success, partial failure, timeout
- rebuild_initramfs() - all tools, root/non-root, failure
- install_driver_packages() - success, failure, timeout, empty
- install_cuda_packages() - success, failure, timeout, empty
- prompt_reboot() - yes/no response
"""

import subprocess
from unittest.mock import MagicMock, patch

from nvidia_inst.cli.installer import (
    InstallResult,
    get_packages_to_remove,
    install_cuda_packages,
    install_driver_packages,
    prompt_reboot,
    rebuild_initramfs,
    remove_packages,
)


class TestGetPackagesToRemove:
    """Test get_packages_to_remove() for all package manager tools."""

    def test_apt_packages(self):
        """Test apt package removal list."""
        packages = get_packages_to_remove("apt")
        assert "nvidia-driver-*" in packages
        assert "nvidia-dkms-*" in packages
        assert "xserver-xorg-video-nvidia" in packages

    def test_apt_get_packages(self):
        """Test apt-get package removal list."""
        packages = get_packages_to_remove("apt-get")
        assert "nvidia-driver-*" in packages
        assert packages == get_packages_to_remove("apt")

    def test_dnf_packages(self):
        """Test dnf package removal list."""
        packages = get_packages_to_remove("dnf")
        assert "akmod-nvidia" in packages
        assert "xorg-x11-drv-nvidia" in packages
        assert "nvidia-persistenced" in packages

    def test_dnf5_packages(self):
        """Test dnf5 package removal list."""
        packages = get_packages_to_remove("dnf5")
        assert "akmod-nvidia" in packages
        assert packages == get_packages_to_remove("dnf")

    def test_pacman_packages(self):
        """Test pacman package removal list."""
        packages = get_packages_to_remove("pacman")
        assert "nvidia" in packages
        assert "nvidia-open" in packages
        assert "nvidia-utils" in packages

    def test_zypper_packages(self):
        """Test zypper package removal list."""
        packages = get_packages_to_remove("zypper")
        assert "x11-video-nvidiaG05" in packages
        assert "nvidia-computeG05" in packages

    def test_unknown_tool(self):
        """Test unknown tool returns empty list."""
        packages = get_packages_to_remove("unknown")
        assert packages == []


class TestRemovePackages:
    """Test remove_packages() function."""

    @patch("subprocess.run")
    def test_remove_single_package(self, mock_run):
        """Test removing a single package."""
        mock_run.return_value = MagicMock(returncode=0)

        removed = remove_packages("apt", ["nvidia-driver-535"])

        assert removed == ["nvidia-driver-535"]
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_remove_multiple_packages(self, mock_run):
        """Test removing multiple packages."""
        mock_run.return_value = MagicMock(returncode=0)

        removed = remove_packages("apt", ["nvidia-driver-535", "nvidia-dkms-535"])

        assert len(removed) == 2
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_remove_partial_failure(self, mock_run):
        """Test partial package removal failure."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(returncode=0)
            return MagicMock(returncode=1)

        mock_run.side_effect = side_effect

        removed = remove_packages("apt", ["pkg1", "pkg2"])

        assert removed == ["pkg1"]

    @patch("subprocess.run")
    def test_remove_timeout(self, mock_run):
        """Test package removal timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 120)

        removed = remove_packages("apt", ["nvidia-driver-535"])

        assert removed == []

    @patch("subprocess.run")
    def test_remove_all_fail(self, mock_run):
        """Test all package removals fail."""
        mock_run.return_value = MagicMock(returncode=1)

        removed = remove_packages("apt", ["nvidia-driver-535"])

        assert removed == []

    @patch("subprocess.run")
    def test_remove_uses_correct_command(self, mock_run):
        """Test remove uses correct tool command."""
        mock_run.return_value = MagicMock(returncode=0)

        remove_packages("dnf", ["akmod-nvidia"])

        call_args = mock_run.call_args[0][0]
        assert "dnf" in call_args
        assert "remove" in call_args


class TestRebuildInitramfs:
    """Test rebuild_initramfs() function."""

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    @patch("nvidia_inst.cli.installer.get_initramfs_command")
    @patch("subprocess.run")
    def test_rebuild_success(self, mock_run, mock_cmd, mock_root):
        """Test successful initramfs rebuild."""
        mock_cmd.return_value = ["update-initramfs", "-u"]
        mock_run.return_value = MagicMock(returncode=0)

        result = rebuild_initramfs("apt")

        assert result is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    @patch("nvidia_inst.cli.installer.get_initramfs_command")
    @patch("subprocess.run")
    def test_rebuild_failure(self, mock_run, mock_cmd, mock_root):
        """Test failed initramfs rebuild."""
        mock_cmd.return_value = ["update-initramfs", "-u"]
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        result = rebuild_initramfs("apt")

        assert result is False

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    @patch("nvidia_inst.cli.installer.get_initramfs_command")
    @patch("subprocess.run")
    def test_rebuild_timeout(self, mock_run, mock_cmd, mock_root):
        """Test initramfs rebuild timeout."""
        mock_cmd.return_value = ["update-initramfs", "-u"]
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 180)

        result = rebuild_initramfs("apt")

        assert result is False

    @patch("nvidia_inst.cli.installer.is_root", return_value=False)
    @patch("nvidia_inst.cli.installer.get_initramfs_command")
    @patch("subprocess.run")
    def test_rebuild_with_sudo(self, mock_run, mock_cmd, mock_root):
        """Test initramfs rebuild uses sudo when not root."""
        mock_cmd.return_value = ["update-initramfs", "-u"]
        mock_run.return_value = MagicMock(returncode=0)

        rebuild_initramfs("apt")

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "sudo"


class TestInstallDriverPackages:
    """Test install_driver_packages() function."""

    @patch("subprocess.run")
    def test_install_success(self, mock_run):
        """Test successful driver package installation."""
        mock_run.return_value = MagicMock(returncode=0)

        result = install_driver_packages("apt", ["nvidia-driver-535"])

        assert result.success is True
        assert result.packages_installed == ["nvidia-driver-535"]
        assert "Installed:" in result.message

    @patch("subprocess.run")
    def test_install_failure(self, mock_run):
        """Test failed driver package installation."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        result = install_driver_packages("apt", ["nvidia-driver-535"])

        assert result.success is False
        assert "Installation failed" in result.message

    @patch("subprocess.run")
    def test_install_timeout(self, mock_run):
        """Test driver package installation timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)

        result = install_driver_packages("apt", ["nvidia-driver-535"])

        assert result.success is False
        assert "timed out" in result.message

    def test_install_empty_packages(self):
        """Test installation with empty package list."""
        result = install_driver_packages("apt", [])

        assert result.success is True
        assert "No packages" in result.message

    @patch("subprocess.run")
    def test_install_multiple_packages(self, mock_run):
        """Test installing multiple packages at once."""
        mock_run.return_value = MagicMock(returncode=0)

        result = install_driver_packages(
            "apt", ["nvidia-driver-535", "nvidia-dkms-535"]
        )

        assert result.success is True
        assert len(result.packages_installed) == 2

    @patch("subprocess.run")
    def test_install_uses_correct_command(self, mock_run):
        """Test installation uses correct tool command."""
        mock_run.return_value = MagicMock(returncode=0)

        install_driver_packages("dnf", ["akmod-nvidia"])

        call_args = mock_run.call_args[0][0]
        assert "dnf" in call_args
        assert "install" in call_args


class TestInstallCudaPackages:
    """Test install_cuda_packages() function."""

    @patch("subprocess.run")
    def test_cuda_install_success(self, mock_run):
        """Test successful CUDA package installation."""
        mock_run.return_value = MagicMock(returncode=0)

        result = install_cuda_packages("apt", ["cuda-toolkit-12-2"])

        assert result.success is True
        assert result.packages_installed == ["cuda-toolkit-12-2"]
        assert "Installed CUDA" in result.message

    @patch("subprocess.run")
    def test_cuda_install_failure(self, mock_run):
        """Test failed CUDA package installation."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        result = install_cuda_packages("apt", ["cuda-toolkit-12-2"])

        assert result.success is False
        assert "CUDA installation failed" in result.message

    @patch("subprocess.run")
    def test_cuda_install_timeout(self, mock_run):
        """Test CUDA package installation timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)

        result = install_cuda_packages("apt", ["cuda-toolkit-12-2"])

        assert result.success is False
        assert "timed out" in result.message

    def test_cuda_install_empty_packages(self):
        """Test CUDA installation with empty package list."""
        result = install_cuda_packages("apt", [])

        assert result.success is True
        assert "No CUDA packages" in result.message

    @patch("subprocess.run")
    def test_cuda_install_multiple_packages(self, mock_run):
        """Test installing multiple CUDA packages."""
        mock_run.return_value = MagicMock(returncode=0)

        result = install_cuda_packages("apt", ["cuda-toolkit-12-2", "cuda-drivers"])

        assert result.success is True
        assert len(result.packages_installed) == 2


class TestInstallResult:
    """Test InstallResult dataclass."""

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

    def test_empty_packages_explicit(self):
        """Test explicit empty packages list."""
        result = InstallResult(
            success=True,
            message="No packages",
            packages_installed=[],
        )
        assert result.packages_installed == []


class TestPromptReboot:
    """Test prompt_reboot() function."""

    @patch("subprocess.run")
    @patch("builtins.input", return_value="y")
    def test_prompt_reboot_yes(self, mock_input, mock_run):
        """Test reboot prompt with yes response."""
        prompt_reboot()

        mock_run.assert_called_once_with(["sudo", "reboot"])

    @patch("subprocess.run")
    @patch("builtins.input", return_value="yes")
    def test_prompt_reboot_yes_full(self, mock_input, mock_run):
        """Test reboot prompt with yes full response."""
        prompt_reboot()

        mock_run.assert_called_once_with(["sudo", "reboot"])

    @patch("subprocess.run")
    @patch("builtins.input", return_value="n")
    def test_prompt_reboot_no(self, mock_input, mock_run):
        """Test reboot prompt with no response."""
        prompt_reboot()

        mock_run.assert_not_called()

    @patch("subprocess.run")
    @patch("builtins.input", return_value="N")
    def test_prompt_reboot_no_uppercase(self, mock_input, mock_run):
        """Test reboot prompt with uppercase no response."""
        prompt_reboot()

        mock_run.assert_not_called()

    @patch("subprocess.run")
    @patch("builtins.input", return_value="y")
    def test_prompt_reboot_command_failure(self, mock_input, mock_run):
        """Test reboot prompt handles command failure."""
        mock_run.side_effect = Exception("Command failed")

        # Should not raise, just print error
        prompt_reboot()
