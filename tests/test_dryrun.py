"""Tests for cli/dryrun.py module."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from nvidia_inst.cli.dryrun import (
    get_initramfs_command,
    dry_run_generic,
    dry_run_change,
    dry_run_nvidia_open_install,
    dry_run_nouveau_install,
    dry_run_revert,
)
from nvidia_inst.distro.tools import PackageContext


class TestGetInitramfsCommand:
    """Tests for get_initramfs_command function."""

    def test_apt_initramfs(self):
        """Test apt initramfs command."""
        cmd = get_initramfs_command("apt")
        assert cmd == ["update-initramfs", "-u"]

    def test_dnf_initramfs(self):
        """Test dnf initramfs command."""
        cmd = get_initramfs_command("dnf")
        assert cmd == ["dracut", "-f", "--regenerate-all"]

    def test_pacman_initramfs(self):
        """Test pacman initramfs command."""
        cmd = get_initramfs_command("pacman")
        assert cmd == ["mkinitcpio", "-P"]

    def test_zypper_initramfs(self):
        """Test zypper initramfs command."""
        cmd = get_initramfs_command("zypper")
        assert cmd == ["dracut", "-f", "--regenerate-all"]

    def test_unknown_tool_initramfs(self):
        """Test unknown tool returns default initramfs command."""
        cmd = get_initramfs_command("unknown")
        assert cmd == ["update-initramfs", "-u"]


class TestDryRunGeneric:
    """Tests for dry_run_generic function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_generic_output(self, mock_stdout):
        """Test dry_run_generic outputs correctly."""
        dry_run_generic(
            title="Test Title",
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia-driver-535"],
            cuda_pkgs=["cuda-toolkit"],
            steps=["step 1", "step 2"],
        )

        output = mock_stdout.getvalue()
        assert "DRY-RUN MODE - Test Title" in output
        assert "Driver installed" in output
        assert "535.154.05" in output
        assert "nvidia-driver-535" in output
        assert "cuda-toolkit" in output
        assert "1. step 1" in output
        assert "2. step 2" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_generic_no_state(self, mock_stdout):
        """Test dry_run_generic without state message."""
        dry_run_generic(
            title="Test Title",
            state_message=None,
            current_version=None,
            packages=["nvidia-driver-535"],
            cuda_pkgs=[],
            steps=["step 1"],
        )

        output = mock_stdout.getvalue()
        assert "Current state" not in output


class TestDryRunChange:
    """Tests for dry_run_change function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_change_with_cuda(self, mock_stdout):
        """Test dry_run_change with CUDA packages."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )

        dry_run_change(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia-driver-535"],
            ctx=ctx,
            cuda_pkgs=["cuda-toolkit"],
        )

        output = mock_stdout.getvalue()
        assert "Driver Change" in output
        assert "apt-get remove -y --purge" in output
        assert "apt-get update" in output
        assert "apt-get install -y nvidia-driver-535" in output
        assert "apt-get install -y cuda-toolkit" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_change_without_cuda(self, mock_stdout):
        """Test dry_run_change without CUDA packages."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )

        dry_run_change(
            state_message="No driver",
            current_version=None,
            packages=["akmod-nvidia"],
            ctx=ctx,
            cuda_pkgs=None,
        )

        output = mock_stdout.getvalue()
        assert "Driver Change" in output
        assert "dnf install -y akmod-nvidia" in output
        assert "cuda-toolkit" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_change_pacman(self, mock_stdout):
        """Test dry_run_change with pacman."""
        ctx = PackageContext(
            tool="pacman", distro_id="arch", distro_family="arch", version_id=""
        )

        dry_run_change(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia"],
            ctx=ctx,
        )

        output = mock_stdout.getvalue()
        assert "pacman -Rns --noconfirm" in output
        assert "pacman -Sy" in output
        assert "pacman -S --noconfirm nvidia" in output


class TestDryRunNvidiaOpenInstall:
    """Tests for dry_run_nvidia_open_install function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_nvidia_open_with_cuda(self, mock_stdout):
        """Test dry_run_nvidia_open_install with CUDA."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )

        dry_run_nvidia_open_install(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia-open"],
            ctx=ctx,
            cuda_pkgs=["cuda-toolkit"],
        )

        output = mock_stdout.getvalue()
        assert "NVIDIA Open Installation" in output
        assert "nvidia-open" in output
        assert "cuda-toolkit" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_nvidia_open_without_cuda(self, mock_stdout):
        """Test dry_run_nvidia_open_install without CUDA."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )

        dry_run_nvidia_open_install(
            state_message=None,
            current_version=None,
            packages=["nvidia-open"],
            ctx=ctx,
            cuda_pkgs=None,
        )

        output = mock_stdout.getvalue()
        assert "NVIDIA Open Installation" in output
        assert "nvidia-open" in output


class TestDryRunNouveauInstall:
    """Tests for dry_run_nouveau_install function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_nouveau_install(self, mock_stdout):
        """Test dry_run_nouveau_install."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )

        dry_run_nouveau_install(
            packages=["xserver-xorg-video-nouveau"],
            ctx=ctx,
        )

        output = mock_stdout.getvalue()
        assert "Nouveau Installation" in output
        assert "xserver-xorg-video-nouveau" in output
        assert "apt-get remove -y --purge" in output


class TestDryRunRevert:
    """Tests for dry_run_revert function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_dry_run_revert(self, mock_stdout):
        """Test dry_run_revert."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )

        dry_run_revert(ctx=ctx)

        output = mock_stdout.getvalue()
        assert "Revert to Nouveau" in output
        assert "apt-get remove -y --purge" in output
        assert "reboot" in output
