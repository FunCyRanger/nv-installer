"""Tests for compatibility checking."""

import pytest
from nvidia_inst.gpu.compatibility import (
    get_driver_range,
    get_max_driver_version,
    is_driver_compatible,
    get_latest_driver,
    format_driver_version,
    DriverRange,
)
from nvidia_inst.gpu.detector import GPUInfo


class TestDriverRange:
    """Test driver range calculation."""

    def test_kepler_driver_range(self):
        """Test Kepler GPU driver range."""
        gpu = GPUInfo(model="GTX 680", generation="kepler")
        driver_range = get_driver_range(gpu)

        assert driver_range.is_eol is True
        assert driver_range.max_version == "470.256.02"
        assert "470" in driver_range.eol_message

    def test_maxwell_driver_range(self):
        """Test Maxwell GPU driver range."""
        gpu = GPUInfo(model="GTX 980", generation="maxwell")
        driver_range = get_driver_range(gpu)

        assert driver_range.is_eol is False
        assert driver_range.max_version == "580.142"
        assert driver_range.max_branch == "580"

    def test_pascal_driver_range(self):
        """Test Pascal GPU driver range."""
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        driver_range = get_driver_range(gpu)

        assert driver_range.is_eol is False
        assert driver_range.max_version == "580.142"
        assert driver_range.max_branch == "580"

    def test_turing_driver_range(self):
        """Test Turing GPU driver range."""
        gpu = GPUInfo(model="RTX 2080", generation="turing")
        driver_range = get_driver_range(gpu)

        assert driver_range.is_eol is False
        assert driver_range.max_version == "590.48.01"
        assert driver_range.max_branch == "590"

    def test_ampere_driver_range(self):
        """Test Ampere GPU driver range."""
        gpu = GPUInfo(model="RTX 3080", generation="ampere")
        driver_range = get_driver_range(gpu)

        assert driver_range.is_eol is False
        assert driver_range.max_version == "590.48.01"
        assert driver_range.min_version == "520.56.06"

    def test_ada_driver_range(self):
        """Test Ada GPU driver range."""
        gpu = GPUInfo(model="RTX 4090", generation="ada")
        driver_range = get_driver_range(gpu)

        assert driver_range.is_eol is False
        assert driver_range.max_version == "590.48.01"

    def test_unknown_generation_range(self):
        """Test unknown generation uses default range."""
        gpu = GPUInfo(model="Unknown GPU", generation="unknown")
        driver_range = get_driver_range(gpu)

        assert driver_range.min_version == "520.56.06"


class TestMaxDriverVersion:
    """Test max driver version for EOL GPUs."""

    def test_kepler_max_version(self):
        """Test Kepler max version."""
        assert get_max_driver_version("GTX 680") == "470.256.02"
        assert get_max_driver_version("GTX 780") == "470.256.02"

    def test_maxwell_max_version(self):
        """Test Maxwell max version."""
        assert get_max_driver_version("GTX 980") == "580.142"

    def test_pascal_max_version(self):
        """Test Pascal max version."""
        assert get_max_driver_version("GTX 1080") == "580.142"

    def test_volta_max_version(self):
        """Test Volta max version."""
        assert get_max_driver_version("V100") == "580.142"

    def test_modern_gpu_no_max(self):
        """Test modern GPUs have no max version."""
        assert get_max_driver_version("RTX 3080") is None
        assert get_max_driver_version("RTX 4090") is None


class TestDriverCompatibility:
    """Test driver compatibility checking."""

    def test_pascal_compatible_within_range(self):
        """Test Pascal driver within range."""
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        assert is_driver_compatible("525.147.05", gpu) is True

    def test_pascal_incompatible_too_new(self):
        """Test Pascal driver too new (590.xx not compatible with Pascal)."""
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        assert is_driver_compatible("590.48.01", gpu) is False

    def test_ampere_compatible_latest(self):
        """Test Ampere accepts latest drivers."""
        gpu = GPUInfo(model="RTX 3080", generation="ampere")
        assert is_driver_compatible("550.127.05", gpu) is True


class TestLatestDriver:
    """Test latest driver selection."""

    def test_kepler_latest(self):
        """Test Kepler latest driver."""
        assert get_latest_driver("kepler") == "470.256.02"

    def test_ampere_latest(self):
        """Test Ampere latest driver."""
        assert get_latest_driver("ampere") == "590.48.01"

    def test_ada_latest(self):
        """Test Ada latest driver."""
        assert get_latest_driver("ada") == "590.48.01"

    def test_maxwell_latest(self):
        """Test Maxwell latest driver."""
        assert get_latest_driver("maxwell") == "580.142"

    def test_pascal_latest(self):
        """Test Pascal latest driver."""
        assert get_latest_driver("pascal") == "580.142"


class TestVersionFormatting:
    """Test version string formatting."""

    def test_format_normal_version(self):
        """Test formatting normal version."""
        assert format_driver_version("535.154.05") == "535.154.05"

    def test_format_version_with_prefix(self):
        """Test formatting version with prefix."""
        assert format_driver_version("nvidia-535.154.05") == "535.154.05"

    def test_format_short_version(self):
        """Test formatting short version."""
        assert format_driver_version("535") == "535"
