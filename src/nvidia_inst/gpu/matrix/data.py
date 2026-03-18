"""Data structures for Nvidia driver compatibility matrix."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SupportStatus(Enum):
    """Driver support status for GPU generations."""

    FULL = "full"
    LIMITED = "limited"
    EOL = "eol"


@dataclass(frozen=True)
class CUDARange:
    """CUDA toolkit compatibility range for a GPU generation."""

    min_version: str
    max_version: str | None = None
    recommended: str = "12.2"


@dataclass(frozen=True)
class ComputeCapability:
    """GPU compute capability range."""

    min: float
    max: float


@dataclass(frozen=True)
class GPUGenerationInfo:
    """Compatibility information for a GPU architecture generation."""

    name: str
    display_name: str
    compute_cap: ComputeCapability
    cuda: CUDARange
    branches: list[str]
    status: SupportStatus
    min_driver: str
    max_driver: str | None = None
    eol_message: str | None = None

    @property
    def is_eol(self) -> bool:
        return self.status == SupportStatus.EOL

    @property
    def is_limited(self) -> bool:
        return self.status == SupportStatus.LIMITED


@dataclass(frozen=True)
class DriverBranchInfo:
    """Information about a specific driver branch."""

    number: str
    name: str
    latest_version: str
    release_date: str
    eol_date: str | None = None
    gpu_generations: list[str] = field(default_factory=list)

    @property
    def is_eol(self) -> bool:
        if not self.eol_date:
            return False
        try:
            eol = datetime.strptime(self.eol_date, "%Y-%m-%d")
            return eol < datetime.now()
        except ValueError:
            return False


@dataclass
class MatrixMeta:
    """Metadata for the compatibility matrix."""

    version: str
    last_updated: str
    sources: list[str] = field(default_factory=list)
    generator: str = "nvidia-inst"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "sources": self.sources,
            "generator": self.generator,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatrixMeta":
        return cls(
            version=data.get("version", "0.0.0"),
            last_updated=data.get("last_updated", ""),
            sources=data.get("sources", []),
            generator=data.get("generator", "nvidia-inst"),
        )


GPU_GENERATIONS: dict[str, GPUGenerationInfo] = {
    "blackwell": GPUGenerationInfo(
        name="blackwell",
        display_name="GeForce RTX 50 Series",
        compute_cap=ComputeCapability(min=9.0, max=12.1),
        cuda=CUDARange(min_version="12.4", max_version="13.x", recommended="12.6"),
        branches=["590", "595"],
        status=SupportStatus.FULL,
        min_driver="550.127.05",
        eol_message=None,
    ),
    "ada": GPUGenerationInfo(
        name="ada",
        display_name="GeForce RTX 40 Series, L40, L10",
        compute_cap=ComputeCapability(min=8.9, max=8.9),
        cuda=CUDARange(min_version="11.8", max_version="12.8", recommended="12.2"),
        branches=["590"],
        status=SupportStatus.FULL,
        min_driver="525.60.13",
        eol_message=None,
    ),
    "ampere": GPUGenerationInfo(
        name="ampere",
        display_name="GeForce RTX 30 Series, A100, A30, A40",
        compute_cap=ComputeCapability(min=8.0, max=8.6),
        cuda=CUDARange(min_version="11.0", max_version="12.8", recommended="12.2"),
        branches=["590"],
        status=SupportStatus.FULL,
        min_driver="520.56.06",
        eol_message=None,
    ),
    "turing": GPUGenerationInfo(
        name="turing",
        display_name="GeForce RTX 20 Series, GTX 16 Series, T4, Quadro RTX",
        compute_cap=ComputeCapability(min=7.5, max=7.5),
        cuda=CUDARange(min_version="10.0", max_version="12.8", recommended="12.2"),
        branches=["590"],
        status=SupportStatus.FULL,
        min_driver="520.56.06",
        eol_message=None,
    ),
    "volta": GPUGenerationInfo(
        name="volta",
        display_name="Tesla V100, Titan V",
        compute_cap=ComputeCapability(min=7.0, max=7.0),
        cuda=CUDARange(min_version="9.0", max_version="12.8", recommended="11.8"),
        branches=["580"],
        status=SupportStatus.LIMITED,
        min_driver="515.65.01",
        max_driver="580.142",
        eol_message="Volta uses driver branch 580.x. Will receive branch updates through October 2028.",
    ),
    "pascal": GPUGenerationInfo(
        name="pascal",
        display_name="GeForce GTX 10 Series, P100, Quadro P-series",
        compute_cap=ComputeCapability(min=6.0, max=6.1),
        cuda=CUDARange(min_version="8.0", max_version="12.8", recommended="11.8"),
        branches=["580"],
        status=SupportStatus.LIMITED,
        min_driver="450.191.0",
        max_driver="580.142",
        eol_message="Pascal uses driver branch 580.x. Will receive branch updates through October 2028.",
    ),
    "maxwell": GPUGenerationInfo(
        name="maxwell",
        display_name="GeForce GTX 900 Series, GTX 750 Series, M-series, Quadro M",
        compute_cap=ComputeCapability(min=5.0, max=5.2),
        cuda=CUDARange(min_version="7.5", max_version="12.8", recommended="11.8"),
        branches=["580"],
        status=SupportStatus.LIMITED,
        min_driver="450.191.0",
        max_driver="580.142",
        eol_message="Maxwell uses driver branch 580.x. Will receive branch updates through October 2028.",
    ),
    "kepler": GPUGenerationInfo(
        name="kepler",
        display_name="GeForce GTX 600/700 Series, K-series",
        compute_cap=ComputeCapability(min=3.0, max=3.7),
        cuda=CUDARange(min_version="7.5", max_version="9.0", recommended="9.0"),
        branches=["470"],
        status=SupportStatus.EOL,
        min_driver="390.157.0",
        max_driver="470.256.02",
        eol_message="Kepler is end-of-life. Maximum supported driver: 470.256.02 (security updates only).",
    ),
}

DRIVER_BRANCHES: dict[str, DriverBranchInfo] = {
    "595": DriverBranchInfo(
        number="595",
        name="New Feature Branch",
        latest_version="595.45.04",
        release_date="2026-01-15",
        eol_date=None,
        gpu_generations=["blackwell"],
    ),
    "590": DriverBranchInfo(
        number="590",
        name="New Feature Branch",
        latest_version="590.48.01",
        release_date="2025-01-15",
        eol_date="2028-08-01",
        gpu_generations=["turing", "ampere", "ada", "blackwell"],
    ),
    "580": DriverBranchInfo(
        number="580",
        name="Production Branch",
        latest_version="580.142",
        release_date="2024-10-01",
        eol_date="2028-10-01",
        gpu_generations=["maxwell", "pascal", "volta"],
    ),
    "470": DriverBranchInfo(
        number="470",
        name="Legacy Branch",
        latest_version="470.256.02",
        release_date="2022-03-22",
        eol_date="2025-12-31",
        gpu_generations=["kepler"],
    ),
}


def get_generation_info(name: str) -> GPUGenerationInfo | None:
    """Get compatibility info for a GPU generation by name."""
    return GPU_GENERATIONS.get(name.lower())


def get_branch_info(number: str) -> DriverBranchInfo | None:
    """Get info for a driver branch by number."""
    return DRIVER_BRANCHES.get(number)


def get_max_branch_for_generation(generation: str) -> str | None:
    """Get the maximum driver branch for a GPU generation."""
    info = get_generation_info(generation)
    if info and info.branches:
        return info.branches[0]
    return None


def is_generation_supported(generation: str) -> bool:
    """Check if a GPU generation has any driver support."""
    info = get_generation_info(generation)
    return info is not None
