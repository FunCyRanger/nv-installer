"""Tests for gpu/compatibility.py module."""

import pytest
from unittest.mock import patch, MagicMock

from nvidia_inst.gpu.compatibility import (
    DriverRange,
    get_driver_range,
    validate_cuda_version,
    validate_cuda_version_with_lock,
    get_recommended_cuda_version,
    get_cuda_major_version_lock,
    validate_driver_version,
    is_driver_compatible_with_branch,
    get_max_driver_version,
    is_driver_compatible,
    _compare_versions,
    _get_cuda_range,
    _get_branch_max_minor,
)
from nvidia_inst.gpu.detector import GPUInfo


class TestDriverRange:
    """Tests for DriverRange dataclass."""

    def test_default_values(self):
        """Test default values."""
        driver_range = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        assert driver_range.min_version == "450.191.0"
        assert driver_range.max_version == "580.142"
        assert driver_range.cuda_min == "7.5"
        assert driver_range.cuda_max == "12.8"
        assert driver_range.is_eol is False
        assert driver_range.is_limited is False
        assert driver_range.max_branch is None
        assert driver_range.eol_message is None
        assert driver_range.cuda_locked_major is None
        assert driver_range.cuda_is_locked is False

    def test_eol_gpu(self):
        """Test EOL GPU values."""
        driver_range = DriverRange(
            min_version="390.157.0",
            max_version="470.256.02",
            cuda_min="7.5",
            cuda_max="11.8",
            is_eol=True,
            is_limited=True,
            max_branch="470",
            eol_message="GPU is end-of-life",
            cuda_locked_major="11",
            cuda_is_locked=True,
        )
        assert driver_range.is_eol is True
        assert driver_range.cuda_locked_major == "11"
        assert driver_range.cuda_is_locked is True


class TestValidateCudaVersion:
    """Tests for validate_cuda_version function."""

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_cuda_version_compatible(self, mock_get_range):
        """Test compatible CUDA version."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        gpu = GPUInfo(model="RTX 3080", generation="ampere")
        valid, msg = validate_cuda_version("12.2", gpu)
        assert valid is True
        assert "compatible" in msg.lower()

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_cuda_version_below_min(self, mock_get_range):
        """Test CUDA version below minimum."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        gpu = GPUInfo(model="RTX 3080", generation="ampere")
        valid, msg = validate_cuda_version("6.0", gpu)
        assert valid is False
        assert "below minimum" in msg.lower()

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_cuda_version_above_max(self, mock_get_range):
        """Test CUDA version above maximum."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        gpu = GPUInfo(model="RTX 3080", generation="ampere")
        valid, msg = validate_cuda_version("13.0", gpu)
        assert valid is False
        assert "exceeds maximum" in msg.lower()

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_cuda_version_wildcard_max(self, mock_get_range):
        """Test CUDA version with wildcard maximum."""
        mock_get_range.return_value = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="13.x",
        )
        gpu = GPUInfo(model="RTX 4090", generation="ada")
        valid, msg = validate_cuda_version("12.8", gpu)
        assert valid is True


class TestValidateCudaVersionWithLock:
    """Tests for validate_cuda_version_with_lock function."""

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_locked_version_compatible(self, mock_get_range):
        """Test locked CUDA version compatible."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
            cuda_locked_major="12",
            cuda_is_locked=True,
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        valid, msg = validate_cuda_version_with_lock("12.2", gpu)
        assert valid is True

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_locked_version_wrong_major(self, mock_get_range):
        """Test locked CUDA version wrong major."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
            cuda_locked_major="12",
            cuda_is_locked=True,
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        valid, msg = validate_cuda_version_with_lock("13.0", gpu)
        assert valid is False
        assert "locked" in msg.lower()

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_not_locked(self, mock_get_range):
        """Test non-locked CUDA version."""
        mock_get_range.return_value = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="13.x",
            cuda_is_locked=False,
        )
        gpu = GPUInfo(model="RTX 4090", generation="ada")
        valid, msg = validate_cuda_version_with_lock("12.8", gpu)
        assert valid is True


class TestGetRecommendedCudaVersion:
    """Tests for get_recommended_cuda_version function."""

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_locked_gpu(self, mock_get_range):
        """Test locked GPU returns locked major + .0."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
            cuda_locked_major="12",
            cuda_is_locked=True,
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        version = get_recommended_cuda_version(gpu)
        assert version == "12.0"

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_unlocked_gpu(self, mock_get_range):
        """Test unlocked GPU returns min version."""
        mock_get_range.return_value = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="13.x",
            cuda_is_locked=False,
        )
        gpu = GPUInfo(model="RTX 4090", generation="ada")
        version = get_recommended_cuda_version(gpu)
        assert version == "11.0"


class TestGetCudaMajorVersionLock:
    """Tests for get_cuda_major_version_lock function."""

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_locked_gpu(self, mock_get_range):
        """Test locked GPU returns locked major."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
            cuda_locked_major="12",
            cuda_is_locked=True,
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        lock = get_cuda_major_version_lock(gpu)
        assert lock == "12"

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_not_locked_gpu(self, mock_get_range):
        """Test not locked GPU returns None."""
        mock_get_range.return_value = DriverRange(
            min_version="520.56.06",
            max_version=None,
            cuda_min="11.0",
            cuda_max="13.x",
            cuda_is_locked=False,
        )
        gpu = GPUInfo(model="RTX 4090", generation="ada")
        lock = get_cuda_major_version_lock(gpu)
        assert lock is None


class TestValidateDriverVersion:
    """Tests for validate_driver_version function."""

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_driver_compatible(self, mock_get_range):
        """Test compatible driver version."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        valid, msg = validate_driver_version("535.154.05", gpu)
        assert valid is True
        assert "compatible" in msg.lower()

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_driver_below_min(self, mock_get_range):
        """Test driver below minimum."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        valid, msg = validate_driver_version("400.0.0", gpu)
        assert valid is False
        assert "below minimum" in msg.lower()

    @patch("nvidia_inst.gpu.compatibility.get_driver_range")
    def test_driver_above_max(self, mock_get_range):
        """Test driver above maximum."""
        mock_get_range.return_value = DriverRange(
            min_version="450.191.0",
            max_version="580.142",
            cuda_min="7.5",
            cuda_max="12.8",
        )
        gpu = GPUInfo(model="GTX 1080", generation="pascal")
        valid, msg = validate_driver_version("590.0.0", gpu)
        assert valid is False
        assert "exceeds maximum" in msg.lower()


class TestIsDriverCompatibleWithBranch:
    """Tests for is_driver_compatible_with_branch function."""

    def test_within_branch(self):
        """Test driver within branch."""
        assert is_driver_compatible_with_branch("580.142", "580") is True

    def test_below_branch(self):
        """Test driver below branch."""
        assert is_driver_compatible_with_branch("570.0.0", "580") is True

    def test_above_branch(self):
        """Test driver above branch."""
        assert is_driver_compatible_with_branch("590.0.0", "580") is False

    def test_no_max_branch(self):
        """Test with no max branch."""
        assert is_driver_compatible_with_branch("590.0.0", "") is True

    def test_invalid_version(self):
        """Test invalid version."""
        assert is_driver_compatible_with_branch("invalid", "580") is False


class TestCompareVersions:
    """Tests for _compare_versions function."""

    def test_v1_greater(self):
        """Test v1 > v2."""
        assert _compare_versions("580.142", "535.154.05") is True

    def test_v1_equal(self):
        """Test v1 == v2."""
        assert _compare_versions("535.154.05", "535.154.05") is True

    def test_v1_less(self):
        """Test v1 < v2."""
        assert _compare_versions("535.154.05", "580.142") is False


class TestGetCudaRange:
    """Tests for _get_cuda_range function."""

    def test_kepler_range(self):
        """Test Kepler CUDA range."""
        cuda_min, cuda_max = _get_cuda_range("kepler")
        assert cuda_min == "7.5"
        assert cuda_max == "11.8"

    def test_maxwell_range(self):
        """Test Maxwell CUDA range."""
        cuda_min, cuda_max = _get_cuda_range("maxwell")
        assert cuda_min == "7.5"
        assert cuda_max == "12.8"

    def test_ampere_range(self):
        """Test Ampere CUDA range."""
        cuda_min, cuda_max = _get_cuda_range("ampere")
        assert cuda_min == "11.0"
        assert cuda_max == "12.8"

    def test_unknown_range(self):
        """Test unknown generation CUDA range."""
        cuda_min, cuda_max = _get_cuda_range("unknown")
        assert cuda_min == "11.0"
        assert cuda_max == "12.8"


class TestGetBranchMaxMinor:
    """Tests for _get_branch_max_minor function."""

    @patch("nvidia_inst.gpu.compatibility._get_matrix_manager")
    def test_known_branch(self, mock_manager):
        """Test known branch."""
        mock_manager.return_value.get_branch_info.return_value = None
        result = _get_branch_max_minor("580")
        assert result == "580.142"

    @patch("nvidia_inst.gpu.compatibility._get_matrix_manager")
    def test_unknown_branch(self, mock_manager):
        """Test unknown branch."""
        mock_manager.return_value.get_branch_info.return_value = None
        result = _get_branch_max_minor("999")
        assert result == "590.48.01"
