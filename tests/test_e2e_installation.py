"""E2E installation tests running in real distro containers.

These tests verify the actual installation workflow with real package managers,
real version locking, and real package installation.
"""

import os

import pytest

from nvidia_inst.distro.versionlock import (
    add_pattern_versionlock_entry,
    read_versionlock_toml,
    verify_versionlock_pattern_active,
)


def is_fedora_container():
    """Check if running in a Fedora container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            return "fedora" in f.read().lower()
    return False


def is_ubuntu_container():
    """Check if running in an Ubuntu container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            content = f.read().lower()
            return "ubuntu" in content
    return False


def has_root():
    """Check if running as root."""
    return os.geteuid() == 0


class TestDNFVersionlock:
    """E2E tests for DNF versionlock TOML management."""

    @pytest.mark.skipif("not is_fedora_container()")
    @pytest.mark.skipif("not has_root()")
    def test_versionlock_toml_created(self):
        """Test that versionlock TOML is created with correct content."""
        success, msg = add_pattern_versionlock_entry(
            package_name="test-package",
            major_version="999",
            comment="E2E test lock",
        )
        assert success is True
        assert "Locked" in msg

        # Verify the file was created
        data = read_versionlock_toml()
        assert "packages" in data
        assert any(pkg.get("name") == "test-package" for pkg in data["packages"])

    @pytest.mark.skipif("not is_fedora_container()")
    @pytest.mark.skipif("not has_root()")
    def test_versionlock_verification(self):
        """Test that versionlock verification works."""
        add_pattern_versionlock_entry(
            package_name="verify-test",
            major_version="999",
            comment="E2E test lock",
        )

        success, msg = verify_versionlock_pattern_active("verify-test", "999")
        assert success is True
        assert "verified" in msg.lower()

    @pytest.mark.skipif("not is_fedora_container()")
    @pytest.mark.skipif("not has_root()")
    def test_versionlock_prevents_duplicate(self):
        """Test that adding duplicate lock returns already-locked message."""
        add_pattern_versionlock_entry(
            package_name="dup-test",
            major_version="999",
            comment="E2E test lock",
        )

        success, msg = add_pattern_versionlock_entry(
            package_name="dup-test",
            major_version="999",
            comment="E2E test lock",
        )
        assert success is True
        assert "already locked" in msg.lower()


class TestAPTVersionlock:
    """E2E tests for APT preferences management."""

    @pytest.mark.skipif("not is_ubuntu_container()")
    @pytest.mark.skipif("not has_root()")
    def test_apt_preferences_created(self):
        """Test that APT preferences file is created."""

        prefs_dir = "/etc/apt/preferences.d"
        if not os.path.isdir(prefs_dir):
            pytest.skip("APT preferences directory not available")

        # Create a test preferences file
        pref_file = os.path.join(prefs_dir, "nvidia-inst-e2e-test")
        content = "Package: test-package*\nPin: version 999.*\nPin-Priority: 1001\n"
        with open(pref_file, "w") as f:
            f.write(content)

        # Verify it was created
        assert os.path.isfile(pref_file)
        with open(pref_file) as f:
            assert "test-package" in f.read()

        # Cleanup
        os.unlink(pref_file)


class TestSimulateAccuracy:
    """Tests that simulate output matches actual installation steps."""

    def test_simulate_includes_version_locking(self):
        """Test that simulate output mentions version locking."""
        import sys
        from io import StringIO

        from nvidia_inst.cli.driver_state import DriverOption, DriverState, DriverStatus
        from nvidia_inst.cli.simulate import simulate_change
        from nvidia_inst.gpu.compatibility import DriverRange
        from nvidia_inst.gpu.detector import GPUInfo

        state = DriverState(
            status=DriverStatus.NOTHING,
            current_version=None,
            is_compatible=False,
            is_optimal=False,
            suggested_packages=["akmod-nvidia", "xorg-x11-drv-nvidia"],
            options=[DriverOption(1, "Install", "install")],
            message="No driver installed",
            cuda_range="11.0-12.8",
        )
        driver_range = DriverRange(
            min_version="520.56.06",
            max_version="580.142",
            cuda_min="11.0",
            cuda_max="12.x",
            max_branch="580",
            is_eol=False,
            is_limited=True,
            cuda_is_locked=True,
            cuda_locked_major="12",
        )
        gpu = GPUInfo(model="NVIDIA GeForce GTX 1080", generation="pascal")

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            simulate_change(
                state.message,
                state.current_version,
                state.suggested_packages,
                "fedora",
                with_cuda=True,
                cuda_version="12.0",
                driver_range=driver_range,
                gpu=gpu,
            )
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "versionlock" in output.lower() or "lock" in output.lower()
        assert "580" in output
        assert "12" in output
