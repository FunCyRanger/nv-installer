"""Tests for cli/simulate.py module."""

from io import StringIO
from unittest.mock import patch

from nvidia_inst.cli.simulate import (
    get_initramfs_command,
    simulate_change,
    simulate_generic,
    simulate_nouveau_install,
    simulate_nvidia_open_install,
    simulate_revert,
)


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


class TestSimulateGeneric:
    """Tests for simulate_generic function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_generic_output(self, mock_stdout):
        """Test simulate_generic outputs correctly."""
        simulate_generic(
            title="Test Title",
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia-driver-535"],
            cuda_pkgs=["cuda-toolkit"],
            steps=["step 1", "step 2"],
        )

        output = mock_stdout.getvalue()
        assert "SIMULATE MODE - Test Title" in output
        assert "Driver installed" in output
        assert "535.154.05" in output
        assert "nvidia-driver-535" in output
        assert "cuda-toolkit" in output
        assert "1. step 1" in output
        assert "2. step 2" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_generic_no_state(self, mock_stdout):
        """Test simulate_generic without state message."""
        simulate_generic(
            title="Test Title",
            state_message=None,
            current_version=None,
            packages=["nvidia-driver-535"],
            cuda_pkgs=[],
            steps=["step 1"],
        )

        output = mock_stdout.getvalue()
        assert "Current state" not in output


class TestSimulateChange:
    """Tests for simulate_change function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_change_ubuntu_with_cuda(self, mock_stdout):
        """Test simulate_change with Ubuntu and CUDA packages."""
        simulate_change(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia-driver-535"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
        )

        output = mock_stdout.getvalue()
        assert "Driver Change" in output
        assert "apt-get remove -y --purge" in output
        assert "apt-get update" in output
        assert "apt-get install -y nvidia-driver-535" in output
        assert "cuda-toolkit-12.2" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_change_fedora_without_cuda(self, mock_stdout):
        """Test simulate_change with Fedora without CUDA packages."""
        simulate_change(
            state_message="No driver",
            current_version=None,
            packages=["akmod-nvidia"],
            distro_id="fedora",
            with_cuda=False,
        )

        output = mock_stdout.getvalue()
        assert "Driver Change" in output
        assert "dnf install -y akmod-nvidia" in output
        assert "cuda-toolkit" not in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_change_arch(self, mock_stdout):
        """Test simulate_change with Arch."""
        simulate_change(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia"],
            distro_id="arch",
        )

        output = mock_stdout.getvalue()
        assert "pacman -Rns --noconfirm" in output
        assert "pacman -Sy" in output
        assert "pacman -S --noconfirm nvidia" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_change_opensuse(self, mock_stdout):
        """Test simulate_change with openSUSE."""
        simulate_change(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["x11-video-nvidiaG05"],
            distro_id="opensuse",
            with_cuda=False,
        )

        output = mock_stdout.getvalue()
        assert "zypper remove -y" in output
        assert "zypper refresh" in output
        assert "zypper install -y x11-video-nvidiaG05" in output


class TestSimulateNvidiaOpenInstall:
    """Tests for simulate_nvidia_open_install function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_nvidia_open_ubuntu_with_cuda(self, mock_stdout):
        """Test simulate_nvidia_open_install with Ubuntu and CUDA."""
        simulate_nvidia_open_install(
            state_message="Driver installed",
            current_version="535.154.05",
            packages=["nvidia-open"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
        )

        output = mock_stdout.getvalue()
        assert "NVIDIA Open Installation" in output
        assert "nvidia-open" in output
        assert "cuda-toolkit-12.2" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_nvidia_open_fedora_without_cuda(self, mock_stdout):
        """Test simulate_nvidia_open_install with Fedora without CUDA."""
        simulate_nvidia_open_install(
            state_message=None,
            current_version=None,
            packages=["nvidia-open"],
            distro_id="fedora",
            with_cuda=False,
        )

        output = mock_stdout.getvalue()
        assert "NVIDIA Open Installation" in output
        assert "nvidia-open" in output
        assert "cuda-toolkit" not in output


class TestSimulateNouveauInstall:
    """Tests for simulate_nouveau_install function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_nouveau_install_ubuntu(self, mock_stdout):
        """Test simulate_nouveau_install with Ubuntu."""
        simulate_nouveau_install(
            packages=["xserver-xorg-video-nouveau"],
            distro_id="ubuntu",
        )

        output = mock_stdout.getvalue()
        assert "Nouveau Installation" in output
        assert "xserver-xorg-video-nouveau" in output
        assert "apt-get remove -y --purge" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_nouveau_install_fedora(self, mock_stdout):
        """Test simulate_nouveau_install with Fedora."""
        simulate_nouveau_install(
            packages=["xorg-x11-drv-nouveau"],
            distro_id="fedora",
        )

        output = mock_stdout.getvalue()
        assert "Nouveau Installation" in output
        assert "xorg-x11-drv-nouveau" in output
        assert "dnf remove -y" in output


class TestSimulateRevert:
    """Tests for simulate_revert function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_revert_ubuntu(self, mock_stdout):
        """Test simulate_revert with Ubuntu."""
        simulate_revert(distro_id="ubuntu")

        output = mock_stdout.getvalue()
        assert "Revert to Nouveau" in output
        assert "apt-get remove -y --purge" in output
        assert "reboot" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_revert_fedora(self, mock_stdout):
        """Test simulate_revert with Fedora."""
        simulate_revert(distro_id="fedora")

        output = mock_stdout.getvalue()
        assert "Revert to Nouveau" in output
        assert "dnf remove -y" in output
        assert "reboot" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_revert_arch(self, mock_stdout):
        """Test simulate_revert with Arch."""
        simulate_revert(distro_id="arch")

        output = mock_stdout.getvalue()
        assert "Revert to Nouveau" in output
        assert "pacman -Rns --noconfirm" in output
        assert "reboot" in output
