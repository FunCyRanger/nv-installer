"""E2E tests for driver state table display on real distros."""

import os
from unittest.mock import patch

import pytest

from nvidia_inst.cli.driver_state import (
    DriverOption,
    DriverState,
    DriverStatus,
    _format_status_table,
    _get_current_locks,
)
from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.gpu.detector import GPUInfo


@pytest.fixture
def ampere_gpu():
    return GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")


@pytest.fixture
def full_range():
    return DriverRange(
        min_version="520.56.06",
        max_version="590.48.01",
        cuda_min="11.0",
        cuda_max="12.8",
        max_branch="590",
        is_eol=False,
        is_limited=False,
        cuda_is_locked=False,
    )


class TestLockDetectionRealDistro:
    """Test lock detection on real distribution containers."""

    def test_get_distro_id(self):
        """Verify we can detect the distro ID."""
        if os.path.isfile("/etc/os-release"):
            with open("/etc/os-release") as f:
                content = f.read()
            assert "ID=" in content

    def test_lock_detection_returns_list(self):
        """Test that lock detection returns a list on any distro."""
        distro_id = "unknown"
        if os.path.isfile("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("ID="):
                        distro_id = line.split("=", 1)[1].strip().strip('"')
                        break

        locks = _get_current_locks(distro_id)
        assert isinstance(locks, list)

    def test_table_format_on_real_distro(self, ampere_gpu, full_range):
        """Test that table formatting works on real distro."""
        state = DriverState(
            status=DriverStatus.NOTHING,
            current_version=None,
            is_compatible=False,
            is_optimal=False,
            suggested_packages=["nvidia-driver-590"],
            options=[
                DriverOption(1, "NVIDIA proprietary", "install"),
                DriverOption(2, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
            ],
            message="No NVIDIA driver installed",
            cuda_range="11.0-12.8",
        )

        with patch(
            "nvidia_inst.cli.driver_state.get_current_driver_type",
            return_value="none",
        ):
            with patch(
                "nvidia_inst.cli.driver_state._get_current_locks", return_value=[]
            ):
                with patch(
                    "nvidia_inst.cli.driver_state._get_nouveau_version",
                    return_value="-",
                ):
                    table = _format_status_table(state, full_range, ampere_gpu, "test")

        assert "Driver Status" in table
        assert "RTX 3080" in table
        assert "NVIDIA proprietary" in table
