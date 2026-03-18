"""Tests for the matrix system."""



from nvidia_inst.gpu.matrix.data import (
    DRIVER_BRANCHES,
    GPU_GENERATIONS,
    ComputeCapability,
    CUDARange,
    DriverBranchInfo,
    GPUGenerationInfo,
    SupportStatus,
    get_branch_info,
    get_generation_info,
    get_max_branch_for_generation,
    is_generation_supported,
)


class TestSupportStatus:
    """Test SupportStatus enum."""

    def test_full_status(self):
        """Test FULL status value."""
        assert SupportStatus.FULL.value == "full"
        assert SupportStatus.FULL.name == "FULL"

    def test_limited_status(self):
        """Test LIMITED status value."""
        assert SupportStatus.LIMITED.value == "limited"
        assert SupportStatus.LIMITED.name == "LIMITED"

    def test_eol_status(self):
        """Test EOL status value."""
        assert SupportStatus.EOL.value == "eol"
        assert SupportStatus.EOL.name == "EOL"


class TestGPUGenerationInfo:
    """Test GPUGenerationInfo dataclass."""

    def test_is_eol_property(self):
        """Test is_eol property."""
        info = GPUGenerationInfo(
            name="kepler",
            display_name="Kepler",
            compute_cap=ComputeCapability(min=3.0, max=3.7),
            cuda=CUDARange(min_version="7.5"),
            branches=["470"],
            status=SupportStatus.EOL,
            min_driver="390.157.0",
        )
        assert info.is_eol is True
        assert info.is_limited is False

    def test_is_limited_property(self):
        """Test is_limited property."""
        info = GPUGenerationInfo(
            name="maxwell",
            display_name="Maxwell",
            compute_cap=ComputeCapability(min=5.0, max=5.2),
            cuda=CUDARange(min_version="7.5"),
            branches=["580"],
            status=SupportStatus.LIMITED,
            min_driver="450.191.0",
        )
        assert info.is_limited is True
        assert info.is_eol is False

    def test_full_support_property(self):
        """Test properties for FULL status."""
        info = GPUGenerationInfo(
            name="ampere",
            display_name="Ampere",
            compute_cap=ComputeCapability(min=8.0, max=8.6),
            cuda=CUDARange(min_version="11.0"),
            branches=["590"],
            status=SupportStatus.FULL,
            min_driver="520.56.06",
        )
        assert info.is_eol is False
        assert info.is_limited is False


class TestDriverBranchInfo:
    """Test DriverBranchInfo dataclass."""

    def test_is_eol_property_future(self):
        """Test is_eol returns False for future EOL date."""
        info = DriverBranchInfo(
            number="590",
            name="New Feature",
            latest_version="590.48.01",
            release_date="2025-01-15",
            eol_date="2030-01-01",
        )
        assert info.is_eol is False

    def test_is_eol_property_past(self):
        """Test is_eol returns True for past EOL date."""
        info = DriverBranchInfo(
            number="470",
            name="Legacy",
            latest_version="470.256.02",
            release_date="2022-03-22",
            eol_date="2020-01-01",
        )
        assert info.is_eol is True

    def test_is_eol_property_none(self):
        """Test is_eol returns False when no EOL date."""
        info = DriverBranchInfo(
            number="595",
            name="New Feature",
            latest_version="595.45.04",
            release_date="2026-01-15",
            eol_date=None,
        )
        assert info.is_eol is False


class TestGPUGenerations:
    """Test GPU_GENERATIONS dictionary."""

    def test_blackwell_in_generations(self):
        """Test Blackwell generation exists."""
        assert "blackwell" in GPU_GENERATIONS
        info = GPU_GENERATIONS["blackwell"]
        assert info.name == "blackwell"
        assert info.status == SupportStatus.FULL
        assert "590" in info.branches

    def test_ampere_in_generations(self):
        """Test Ampere generation exists."""
        assert "ampere" in GPU_GENERATIONS
        info = GPU_GENERATIONS["ampere"]
        assert info.name == "ampere"
        assert info.status == SupportStatus.FULL
        assert info.min_driver == "520.56.06"

    def test_turing_in_generations(self):
        """Test Turing generation exists."""
        assert "turing" in GPU_GENERATIONS
        info = GPU_GENERATIONS["turing"]
        assert info.name == "turing"
        assert info.status == SupportStatus.FULL

    def test_maxwell_in_generations(self):
        """Test Maxwell generation is LIMITED."""
        assert "maxwell" in GPU_GENERATIONS
        info = GPU_GENERATIONS["maxwell"]
        assert info.status == SupportStatus.LIMITED
        assert "580" in info.branches

    def test_kepler_in_generations(self):
        """Test Kepler generation is EOL."""
        assert "kepler" in GPU_GENERATIONS
        info = GPU_GENERATIONS["kepler"]
        assert info.status == SupportStatus.EOL
        assert info.max_driver == "470.256.02"


class TestDriverBranches:
    """Test DRIVER_BRANCHES dictionary."""

    def test_590_branch(self):
        """Test 590 branch exists."""
        assert "590" in DRIVER_BRANCHES
        branch = DRIVER_BRANCHES["590"]
        assert branch.number == "590"
        assert branch.latest_version == "590.48.01"

    def test_580_branch(self):
        """Test 580 branch exists."""
        assert "580" in DRIVER_BRANCHES
        branch = DRIVER_BRANCHES["580"]
        assert branch.number == "580"
        assert branch.latest_version == "580.142"

    def test_470_branch(self):
        """Test 470 branch exists."""
        assert "470" in DRIVER_BRANCHES
        branch = DRIVER_BRANCHES["470"]
        assert branch.number == "470"
        assert branch.is_eol is True


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_generation_info_ampere(self):
        """Test get_generation_info for Ampere."""
        info = get_generation_info("ampere")
        assert info is not None
        assert info.name == "ampere"

    def test_get_generation_info_case_insensitive(self):
        """Test get_generation_info is case insensitive."""
        info = get_generation_info("AMPERE")
        assert info is not None
        assert info.name == "ampere"

    def test_get_generation_info_not_found(self):
        """Test get_generation_info returns None for unknown."""
        info = get_generation_info("unknown")
        assert info is None

    def test_get_branch_info_590(self):
        """Test get_branch_info for 590."""
        info = get_branch_info("590")
        assert info is not None
        assert info.number == "590"

    def test_get_branch_info_not_found(self):
        """Test get_branch_info returns None for unknown."""
        info = get_branch_info("999")
        assert info is None

    def test_get_max_branch_for_generation(self):
        """Test get_max_branch_for_generation."""
        branch = get_max_branch_for_generation("ampere")
        assert branch == "590"

        branch = get_max_branch_for_generation("maxwell")
        assert branch == "580"

    def test_get_max_branch_for_unknown_generation(self):
        """Test get_max_branch_for_generation for unknown."""
        branch = get_max_branch_for_generation("unknown")
        assert branch is None

    def test_is_generation_supported(self):
        """Test is_generation_supported."""
        assert is_generation_supported("ampere") is True
        assert is_generation_supported("kepler") is True
        assert is_generation_supported("unknown") is False


class TestCUDARange:
    """Test CUDARange dataclass."""

    def test_cuda_range_creation(self):
        """Test CUDARange creation."""
        cuda = CUDARange(min_version="11.0", max_version="12.8", recommended="12.2")
        assert cuda.min_version == "11.0"
        assert cuda.max_version == "12.8"
        assert cuda.recommended == "12.2"

    def test_cuda_range_defaults(self):
        """Test CUDARange default values."""
        cuda = CUDARange(min_version="11.0")
        assert cuda.max_version is None
        assert cuda.recommended == "12.2"


class TestComputeCapability:
    """Test ComputeCapability dataclass."""

    def test_compute_capability_creation(self):
        """Test ComputeCapability creation."""
        cc = ComputeCapability(min=8.0, max=8.6)
        assert cc.min == 8.0
        assert cc.max == 8.6
