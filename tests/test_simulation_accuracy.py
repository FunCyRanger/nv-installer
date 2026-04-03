"""E2E simulation accuracy tests.

These tests verify that --simulate output matches what would actually be
installed for different GPU generations, distros, CUDA states, and driver
states.
"""

import os
import sys
from io import StringIO

import pytest

from nvidia_inst.cli.simulate import (
    simulate_change,
    simulate_nouveau_install,
    simulate_nvidia_open_install,
    simulate_revert,
)
from nvidia_inst.distro.tools import (
    get_install_command,
    get_remove_command,
    get_update_command,
)
from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.gpu.detector import GPUInfo


def is_fedora_container() -> bool:
    """Check if running in a Fedora container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            return "fedora" in f.read().lower()
    return False


def is_ubuntu_container() -> bool:
    """Check if running in an Ubuntu container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            content = f.read().lower()
            return "ubuntu" in content
    return False


# ---------------------------------------------------------------------------
# Helper: capture simulate output
# ---------------------------------------------------------------------------


def _capture_simulate(fn, *args, **kwargs) -> str:
    """Run a simulate function and capture its stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        fn(*args, **kwargs)
        return sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# GPU generation simulation accuracy
# ---------------------------------------------------------------------------


class TestSimulateGpuGenerations:
    """Test simulate output for different GPU generations."""

    def _make_driver_range(
        self,
        min_ver: str = "520.56.06",
        max_ver: str | None = None,
        branch: str | None = "590",
        cuda_min: str = "11.0",
        cuda_max: str = "12.8",
        cuda_locked: bool = False,
        cuda_locked_major: str | None = None,
        is_eol: bool = False,
        is_limited: bool = False,
    ) -> DriverRange:
        return DriverRange(
            min_version=min_ver,
            max_version=max_ver,
            cuda_min=cuda_min,
            cuda_max=cuda_max,
            max_branch=branch,
            cuda_is_locked=cuda_locked,
            cuda_locked_major=cuda_locked_major,
            is_eol=is_eol,
            is_limited=is_limited,
        )

    def test_kepler_eol_simulation(self):
        """Test simulate output for Kepler (EOL) GPU."""
        dr = self._make_driver_range(
            min_ver="390.157.0",
            max_ver="470.256.02",
            branch="470",
            cuda_min="7.5",
            cuda_max="11.8",
            cuda_locked=True,
            cuda_locked_major="11",
            is_eol=True,
            is_limited=True,
        )
        gpu = GPUInfo(model="NVIDIA GeForce GTX 750", generation="kepler")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-470", "nvidia-dkms-470"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="11.8",
            driver_range=dr,
            gpu=gpu,
        )

        assert "SIMULATE MODE" in output
        assert "470" in output
        assert "Lock driver to 470" in output
        assert "Lock CUDA to 11" in output
        assert "nvidia-driver-470" in output
        assert "nvidia-dkms-470" in output

    def test_pascal_limited_simulation(self):
        """Test simulate output for Pascal (limited) GPU."""
        dr = self._make_driver_range(
            min_ver="450.191.0",
            max_ver="580.142",
            branch="580",
            cuda_min="8.0",
            cuda_max="12.x",
            cuda_locked=True,
            cuda_locked_major="12",
            is_eol=False,
            is_limited=True,
        )
        gpu = GPUInfo(model="NVIDIA GeForce GTX 1080", generation="pascal")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-580", "nvidia-dkms-580"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.0",
            driver_range=dr,
            gpu=gpu,
        )

        assert "SIMULATE MODE" in output
        assert "Lock driver to 580" in output
        assert "Lock CUDA to 12" in output

    def test_ampere_full_simulation(self):
        """Test simulate output for Ampere (full support) GPU."""
        dr = self._make_driver_range(
            min_ver="520.56.06",
            max_ver=None,
            branch="590",
            cuda_min="11.0",
            cuda_max="12.8",
            cuda_locked=False,
            is_eol=False,
            is_limited=False,
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590", "nvidia-dkms-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "SIMULATE MODE" in output
        assert "Lock driver to 590" in output
        # CUDA not locked for Ampere
        assert "Lock CUDA" not in output

    def test_ada_full_simulation(self):
        """Test simulate output for Ada (full support) GPU."""
        dr = self._make_driver_range(
            min_ver="525.60.13",
            max_ver=None,
            branch="590",
            cuda_min="11.8",
            cuda_max="12.8",
            cuda_locked=False,
            is_eol=False,
            is_limited=False,
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 4090", generation="ada")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590", "nvidia-dkms-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.8",
            driver_range=dr,
            gpu=gpu,
        )

        assert "SIMULATE MODE" in output
        assert "Lock driver to 590" in output

    def test_blackwell_simulation(self):
        """Test simulate output for Blackwell GPU."""
        dr = self._make_driver_range(
            min_ver="550.127.05",
            max_ver=None,
            branch="590",
            cuda_min="12.4",
            cuda_max="13.x",
            cuda_locked=False,
            is_eol=False,
            is_limited=False,
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 5090", generation="blackwell")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590", "nvidia-dkms-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.4",
            driver_range=dr,
            gpu=gpu,
        )

        assert "SIMULATE MODE" in output
        assert "Lock driver to 590" in output


# ---------------------------------------------------------------------------
# CUDA enabled/disabled simulation
# ---------------------------------------------------------------------------


class TestSimulateCudaStates:
    """Test simulate output with CUDA enabled/disabled."""

    def test_simulation_with_cuda_enabled(self):
        """Test simulate output includes CUDA packages when enabled."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "cuda-toolkit-12.2" in output
        assert "Install CUDA toolkit" in output

    def test_simulation_with_cuda_disabled(self):
        """Test simulate output excludes CUDA packages when disabled."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=False,
            driver_range=dr,
            gpu=gpu,
        )

        assert "cuda-toolkit" not in output
        assert "Install CUDA toolkit" not in output


# ---------------------------------------------------------------------------
# Distro-specific simulation accuracy
# ---------------------------------------------------------------------------


class TestSimulateDistroAccuracy:
    """Test simulate output for different distros."""

    def _make_driver_range(self) -> DriverRange:
        return DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )

    def test_ubuntu_simulation_uses_apt(self):
        """Test Ubuntu simulation uses apt-get commands."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590", "nvidia-dkms-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "apt-get" in output
        assert "update-initramfs" in output
        assert "apt-get install" in output

    def test_fedora_simulation_uses_dnf(self):
        """Test Fedora simulation uses dnf commands."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["akmod-nvidia", "xorg-x11-drv-nvidia"],
            distro_id="fedora",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "dnf" in output
        assert "dracut" in output
        assert "dnf install" in output

    def test_arch_simulation_uses_pacman(self):
        """Test Arch simulation uses pacman commands."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia", "nvidia-utils"],
            distro_id="arch",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "pacman" in output
        assert "mkinitcpio" in output

    def test_opensuse_simulation_uses_zypper(self):
        """Test openSUSE simulation uses zypper commands."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["x11-video-nvidiaG05", "nvidia-computeG05"],
            distro_id="opensuse",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "zypper" in output
        assert "dracut" in output


# ---------------------------------------------------------------------------
# Driver state simulation accuracy
# ---------------------------------------------------------------------------


class TestSimulateDriverStates:
    """Test simulate output for different driver states."""

    def _make_driver_range(self) -> DriverRange:
        return DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )

    def test_simulation_nothing_installed(self):
        """Test simulate when nothing is installed."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        # Should NOT have a "Remove old packages" step
        assert "Remove old packages" not in output
        # Should have install step
        assert "Install driver packages" in output

    def test_simulation_upgrade_needed(self):
        """Test simulate when upgrading from older driver."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="Driver 580.142 may not be optimal",
            current_version="580.142",
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        # Should have a "Remove old packages" step
        assert "Remove old packages" in output
        # Should have install step
        assert "Install driver packages" in output

    def test_simulation_optimal_state(self):
        """Test simulate when driver is already optimal."""
        dr = self._make_driver_range()
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="NVIDIA driver 590.48.01 is working optimally",
            current_version="590.48.01",
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        # Should still show remove + install (upgrade path)
        assert "Remove old packages" in output
        assert "Install driver packages" in output


# ---------------------------------------------------------------------------
# Version locking in simulation
# ---------------------------------------------------------------------------


class TestSimulateVersionLocking:
    """Test that simulate output includes version locking steps."""

    def test_simulation_includes_driver_lock(self):
        """Test simulate includes driver version lock step."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "Lock driver to 590" in output
        assert "Version Lock Details" in output

    def test_simulation_includes_cuda_lock_when_locked(self):
        """Test simulate includes CUDA lock when GPU has CUDA lock."""
        dr = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="8.0",
            cuda_max="12.x",
            max_branch="580",
            cuda_is_locked=True,
            cuda_locked_major="12",
            is_limited=True,
        )
        gpu = GPUInfo(model="NVIDIA GeForce GTX 1080", generation="pascal")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-580"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.0",
            driver_range=dr,
            gpu=gpu,
        )

        assert "Lock CUDA to 12" in output

    def test_simulation_no_cuda_lock_when_unlocked(self):
        """Test simulate does NOT include CUDA lock when GPU is unlocked."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
            cuda_is_locked=False,
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "Lock CUDA" not in output

    def test_simulation_includes_initramfs_rebuild(self):
        """Test simulate includes initramfs rebuild step."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "Rebuild initramfs" in output
        assert "update-initramfs" in output

    def test_simulation_includes_reboot(self):
        """Test simulate includes reboot step."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        assert "Reboot" in output
        assert "sudo reboot" in output


# ---------------------------------------------------------------------------
# simulate_nouveau_install accuracy
# ---------------------------------------------------------------------------


class TestSimulateNouveauAccuracy:
    """Test simulate_nouveau_install output accuracy."""

    def test_nouveau_simulation_ubuntu(self):
        """Test Nouveau simulation for Ubuntu."""
        output = _capture_simulate(
            simulate_nouveau_install,
            packages=["xserver-xorg-video-nouveau"],
            distro_id="ubuntu",
        )

        assert "Nouveau Installation" in output
        assert "xserver-xorg-video-nouveau" in output
        assert "apt-get" in output
        assert "update-initramfs" in output
        assert "reboot" in output

    def test_nouveau_simulation_fedora(self):
        """Test Nouveau simulation for Fedora."""
        output = _capture_simulate(
            simulate_nouveau_install,
            packages=["xorg-x11-drv-nouveau"],
            distro_id="fedora",
        )

        assert "Nouveau Installation" in output
        assert "xorg-x11-drv-nouveau" in output
        assert "dnf" in output
        assert "dracut" in output


# ---------------------------------------------------------------------------
# simulate_revert accuracy
# ---------------------------------------------------------------------------


class TestSimulateRevertAccuracy:
    """Test simulate_revert output accuracy."""

    def test_revert_simulation_ubuntu(self):
        """Test revert simulation for Ubuntu."""
        output = _capture_simulate(simulate_revert, distro_id="ubuntu")

        assert "Revert to Nouveau" in output
        assert "apt-get" in output
        assert "update-initramfs" in output
        assert "reboot" in output

    def test_revert_simulation_fedora(self):
        """Test revert simulation for Fedora."""
        output = _capture_simulate(simulate_revert, distro_id="fedora")

        assert "Revert to Nouveau" in output
        assert "dnf" in output
        assert "dracut" in output


# ---------------------------------------------------------------------------
# simulate_nvidia_open_install accuracy
# ---------------------------------------------------------------------------


class TestSimulateNvidiaOpenAccuracy:
    """Test simulate_nvidia_open_install output accuracy."""

    def test_open_install_ubuntu_with_cuda(self):
        """Test NVIDIA Open simulation for Ubuntu with CUDA."""
        output = _capture_simulate(
            simulate_nvidia_open_install,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590-open", "nvidia-dkms-590-open"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
        )

        assert "NVIDIA Open Installation" in output
        assert "nvidia-driver-590-open" in output
        assert "cuda-toolkit-12.2" in output

    def test_open_install_fedora_without_cuda(self):
        """Test NVIDIA Open simulation for Fedora without CUDA."""
        output = _capture_simulate(
            simulate_nvidia_open_install,
            state_message=None,
            current_version=None,
            packages=["xorg-x11-drv-nvidia-open"],
            distro_id="fedora",
            with_cuda=False,
        )

        assert "NVIDIA Open Installation" in output
        assert "xorg-x11-drv-nvidia-open" in output
        assert "cuda-toolkit" not in output


# ---------------------------------------------------------------------------
# Real distro package command accuracy
# ---------------------------------------------------------------------------


class TestRealDistroPackageCommands:
    """Test that simulate output uses correct real package manager commands."""

    @pytest.mark.skipif("not is_ubuntu_container()")
    def test_ubuntu_real_commands(self):
        """Test that Ubuntu simulate output matches real apt commands."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["nvidia-driver-590", "nvidia-dkms-590"],
            distro_id="ubuntu",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        # Verify commands match real apt tool
        install_cmd = get_install_command("apt")
        remove_cmd = get_remove_command("apt")
        update_cmd = get_update_command("apt")

        assert " ".join(install_cmd) in output
        assert " ".join(remove_cmd) in output
        assert " ".join(update_cmd) in output

    @pytest.mark.skipif("not is_fedora_container()")
    def test_fedora_real_commands(self):
        """Test that Fedora simulate output matches real dnf commands."""
        dr = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="12.8",
            max_branch="590",
        )
        gpu = GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")

        output = _capture_simulate(
            simulate_change,
            state_message="No NVIDIA driver installed",
            current_version=None,
            packages=["akmod-nvidia", "xorg-x11-drv-nvidia"],
            distro_id="fedora",
            with_cuda=True,
            cuda_version="12.2",
            driver_range=dr,
            gpu=gpu,
        )

        # Verify commands match real dnf tool
        install_cmd = get_install_command("dnf")
        update_cmd = get_update_command("dnf")

        assert " ".join(install_cmd) in output
        assert " ".join(update_cmd) in output
        # No "Remove old packages" step when current_version is None
        assert "Remove old packages" not in output
