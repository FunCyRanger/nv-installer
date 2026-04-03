"""Nvidia driver compatibility checking.

This module provides GPU/driver compatibility information, sourced from
the compatibility matrix system (src/nvidia_inst/gpu/matrix/).
"""

import re
from dataclasses import dataclass

from nvidia_inst.gpu.detector import GPUInfo
from nvidia_inst.gpu.matrix import MatrixManager
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)

_matrix_manager: MatrixManager | None = None


def _get_matrix_manager() -> MatrixManager:
    """Get or create the matrix manager singleton."""
    global _matrix_manager
    if _matrix_manager is None:
        _matrix_manager = MatrixManager()
    return _matrix_manager


GPU_DRIVER_MAX_VERSIONS: dict[str, str] = {
    "kepler": "470.256.02",
    "maxwell": "580.142",
    "pascal": "580.142",
    "volta": "580.142",
    "turing": "590.48.01",
}

GPU_DRIVER_MIN_VERSIONS: dict[str, str] = {
    "kepler": "390.157.0",
    "maxwell": "450.191.0",
    "pascal": "450.191.0",
    "volta": "515.65.01",
    "turing": "520.56.06",
    "ampere": "520.56.06",
    "ada": "525.60.13",
    "blackwell": "550.127.05",
}


def _init_from_matrix() -> None:
    """Initialize version dicts from matrix if available."""
    try:
        manager = _get_matrix_manager()
        for gen in manager.get_all_generations().values():
            if gen.max_driver:
                GPU_DRIVER_MAX_VERSIONS[gen.name] = gen.max_driver
            GPU_DRIVER_MIN_VERSIONS[gen.name] = gen.min_driver
    except Exception as e:
        logger.debug(f"Failed to initialize from matrix: {e}")


@dataclass
class DriverRange:
    """Range of compatible driver versions."""

    min_version: str
    max_version: str | None
    cuda_min: str
    cuda_max: str | None
    is_eol: bool = False
    is_limited: bool = False
    max_branch: str | None = None
    eol_message: str | None = None
    cuda_locked_major: str | None = None  # Lock CUDA to major version (e.g., "12")
    cuda_is_locked: bool = False  # Whether CUDA version is locked


def get_driver_range(gpu: GPUInfo) -> DriverRange:
    """Get the compatible driver version range for a GPU.

    Args:
        gpu: GPU information.

    Returns:
        DriverRange with min/max versions and CUDA support.
    """
    generation = gpu.generation or "unknown"

    try:
        manager = _get_matrix_manager()
        gen_info = manager.get_generation_info(generation)

        if gen_info:
            max_version = gen_info.max_driver
            if max_version is None and gen_info.branches:
                branch_info = manager.get_branch_info(gen_info.branches[0])
                if branch_info:
                    max_version = branch_info.latest_version

            return DriverRange(
                min_version=gen_info.min_driver,
                max_version=max_version,
                cuda_min=gen_info.cuda.min_version,
                cuda_max=gen_info.cuda.max_version,
                is_eol=gen_info.is_eol,
                is_limited=gen_info.is_limited,
                max_branch=gen_info.branches[0] if gen_info.branches else None,
                eol_message=gen_info.eol_message,
                cuda_locked_major=gen_info.cuda.locked_major,
                cuda_is_locked=gen_info.cuda.is_locked,
            )
    except Exception as e:
        logger.debug(f"Failed to get driver range from matrix: {e}")

    return _get_driver_range_fallback(generation)


def validate_cuda_version(cuda_version: str, gpu: GPUInfo) -> tuple[bool, str]:
    """Validate CUDA version compatibility with GPU generation.

    Args:
        cuda_version: CUDA version string (e.g., "12.2", "13.x").
        gpu: GPU information.

    Returns:
        Tuple of (is_compatible, message).
    """
    driver_range = get_driver_range(gpu)
    min_cuda = driver_range.cuda_min
    max_cuda = driver_range.cuda_max

    # Parse version, handle "x" wildcard
    def parse_version(ver: str):
        if ver.endswith(".x"):
            # "13.x" -> (13, 0, True) where True indicates wildcard
            return (int(ver[:-2]), 0, True)
        else:
            parts = ver.split(".")
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0, False)

    cuda_parsed = parse_version(cuda_version)
    min_parsed = parse_version(min_cuda)
    max_parsed = parse_version(max_cuda) if max_cuda else None

    # Check lower bound
    if cuda_parsed < min_parsed:
        return (
            False,
            f"CUDA {cuda_version} is below minimum supported version {min_cuda} for {gpu.generation}",
        )

    # Check upper bound
    if max_parsed:
        # If max has wildcard, treat as up to but not including next major version
        if max_parsed[2]:  # max_cuda ends with .x
            if cuda_parsed[0] > max_parsed[0]:
                return (
                    False,
                    f"CUDA {cuda_version} exceeds maximum supported version {max_cuda} for {gpu.generation}",
                )
        else:
            if cuda_parsed > max_parsed:
                return (
                    False,
                    f"CUDA {cuda_version} exceeds maximum supported version {max_cuda} for {gpu.generation}",
                )

    return (True, f"CUDA {cuda_version} is compatible with {gpu.generation}")


def validate_cuda_version_with_lock(
    cuda_version: str, gpu: GPUInfo
) -> tuple[bool, str]:
    """Validate CUDA version respecting locked major version.

    For locked GPUs:
    - Limited (locked_major="12"): Must be 12.x
    - EOL (locked_major="11"): Must be 11.x

    Args:
        cuda_version: CUDA version to validate (e.g., "12.2")
        gpu: GPU information

    Returns:
        Tuple of (is_valid, message)
    """
    driver_range = get_driver_range(gpu)

    if not driver_range.cuda_is_locked:
        return validate_cuda_version(cuda_version, gpu)

    if driver_range.cuda_locked_major is None:
        # Should not happen, fall back to range validation
        return validate_cuda_version(cuda_version, gpu)

    # Limited/EOL: major version lock
    cuda_major = cuda_version.split(".")[0]
    if cuda_major != driver_range.cuda_locked_major:
        return (
            False,
            f"CUDA for {gpu.generation} GPUs is locked to major version {driver_range.cuda_locked_major}.x. "
            f"Requested: {cuda_version}",
        )

    # Also check it's within the valid range
    return validate_cuda_version(cuda_version, gpu)


def get_recommended_cuda_version(gpu: GPUInfo) -> str:
    """Get recommended CUDA version for GPU.

    For locked GPUs, returns the locked major + .0 (e.g., "12.0").
    For unlocked GPUs, returns the recommended version from matrix.

    Args:
        gpu: GPU information

    Returns:
        Recommended CUDA version string
    """
    driver_range = get_driver_range(gpu)

    if driver_range.cuda_is_locked and driver_range.cuda_locked_major:
        return f"{driver_range.cuda_locked_major}.0"

    # Fall back to minimum version if no recommended
    return driver_range.cuda_min


def get_cuda_major_version_lock(gpu: GPUInfo) -> str | None:
    """Get the CUDA major version lock for a GPU, if any.

    Args:
        gpu: GPU information

    Returns:
        Major version string (e.g., "12") or None if not locked
    """
    driver_range = get_driver_range(gpu)
    if driver_range.cuda_is_locked:
        return driver_range.cuda_locked_major
    return None


def validate_driver_version(driver_version: str, gpu: GPUInfo) -> tuple[bool, str]:
    """Validate driver version compatibility with GPU generation.

    Args:
        driver_version: Driver version string (e.g., "535.154.05").
        gpu: GPU information.

    Returns:
        Tuple of (is_compatible, message).
    """
    driver_range = get_driver_range(gpu)
    min_driver = driver_range.min_version
    max_driver = driver_range.max_version

    # Simple version comparison: split into integers and compare
    def version_to_tuple(v: str):
        return tuple(map(int, v.split(".")))

    driver_parsed = version_to_tuple(driver_version)
    min_parsed = version_to_tuple(min_driver)

    if driver_parsed < min_parsed:
        return (
            False,
            f"Driver {driver_version} is below minimum supported version {min_driver} for {gpu.generation}",
        )

    if max_driver:
        max_parsed = version_to_tuple(max_driver)
        if driver_parsed > max_parsed:
            return (
                False,
                f"Driver {driver_version} exceeds maximum supported version {max_driver} for {gpu.generation}",
            )

    return (True, f"Driver {driver_version} is compatible with {gpu.generation}")


def _get_driver_range_fallback(generation: str) -> DriverRange:
    """Fallback driver range calculation using hardcoded values."""
    if generation == "kepler":
        max_version = GPU_DRIVER_MAX_VERSIONS.get(generation, "470.256.02")
        min_version = GPU_DRIVER_MIN_VERSIONS.get(generation, "450.191.0")
        cuda_range = _get_cuda_range(generation)

        return DriverRange(
            min_version=min_version,
            max_version=max_version,
            cuda_min=cuda_range[0],
            cuda_max=cuda_range[1],
            is_eol=True,
            is_limited=True,
            max_branch="470",
            eol_message=f"GPU generation '{generation}' is end-of-life. "
            f"Maximum supported driver: {max_version}. CUDA locked to 11.x.",
            cuda_locked_major="11",
            cuda_is_locked=True,
        )

    if generation in ("maxwell", "pascal", "volta"):
        min_version = GPU_DRIVER_MIN_VERSIONS.get(generation, "450.191.0")
        cuda_range = _get_cuda_range(generation)
        branch = get_driver_branch(generation)

        max_major_minor = _get_branch_max_minor(branch)

        return DriverRange(
            min_version=min_version,
            max_version=max_major_minor,
            cuda_min=cuda_range[0],
            cuda_max=cuda_range[1],
            is_eol=False,
            is_limited=True,
            max_branch=branch,
            eol_message=f"GPU generation '{generation}' uses driver branch {branch}. "
            f"CUDA support frozen at 12.x. "
            f"Will receive branch updates through October 2028.",
            cuda_locked_major="12",
            cuda_is_locked=True,
        )

    cuda_range = _get_cuda_range(generation)
    branch = get_driver_branch(generation)
    max_version = _get_branch_max_minor(branch)

    return DriverRange(
        min_version="520.56.06",
        max_version=max_version,
        cuda_min=cuda_range[0],
        cuda_max=cuda_range[1],
        is_eol=False,
        is_limited=False,
        max_branch=branch,
        cuda_locked_major=None,
        cuda_is_locked=False,
    )


def _get_cuda_range(generation: str) -> tuple[str, str | None]:
    """Get CUDA version range for a GPU generation.

    Based on official NVIDIA CUDA Toolkit and Architecture Matrix:
    - Kepler: CUDA 7.5 - 11.8 (EOL, last CUDA 11.8)
    - Maxwell/Pascal/Volta: CUDA 7.5-9.0 - 12.x (Limited, frozen at 12.x)
    - Turing+: Latest CUDA supported
    """
    cuda_versions = {
        "kepler": ("7.5", "11.8"),
        "maxwell": ("7.5", "12.x"),
        "pascal": ("8.0", "12.x"),
        "volta": ("9.0", "12.x"),
        "turing": ("10.0", "13.x"),
        "ampere": ("11.0", "13.x"),
        "ada": ("11.8", "13.x"),
        "blackwell": ("12.4", "13.x"),
        "unknown": ("11.0", "13.x"),
    }

    return cuda_versions.get(generation, ("11.0", "12.8"))


def _get_branch_max_minor(branch: str) -> str:
    """Get the current maximum minor version for a driver branch.

    Args:
        branch: Driver branch (e.g., "470", "580", "590").

    Returns:
        Maximum minor version (e.g., "580.142").
    """
    try:
        manager = _get_matrix_manager()
        branch_info = manager.get_branch_info(branch)
        if branch_info:
            return branch_info.latest_version
    except Exception:
        pass

    branch_max_versions = {
        "470": "470.256.02",
        "580": "580.142",
        "590": "590.48.01",
    }
    return branch_max_versions.get(branch, "590.48.01")


def is_driver_compatible_with_branch(driver_version: str, max_branch: str) -> bool:
    """Check if a driver version is within the allowed branch.

    Args:
        driver_version: Driver version to check (e.g., "580.142").
        max_branch: Maximum allowed branch (e.g., "580").

    Returns:
        True if driver is within branch, False otherwise.
    """
    if not max_branch:
        return True

    match = re.match(r"(\d+)\.", driver_version)
    if not match:
        return False

    driver_branch = match.group(1)
    return int(driver_branch) <= int(max_branch)


def get_max_driver_version(gpu_name: str) -> str | None:
    """Return max driver version for EOL/Limited GPUs.

    Args:
        gpu_name: GPU model name.

    Returns:
        Max driver version string if GPU has version limit, None for latest.
    """
    from nvidia_inst.gpu.detector import _get_gpu_generation

    generation = _get_gpu_generation(gpu_name)

    if not generation:
        return None

    eol_generations = {"kepler": "470.256.02"}
    limited_generations = {
        "maxwell": "580.142",
        "pascal": "580.142",
        "volta": "580.142",
    }

    if generation in eol_generations:
        max_version = eol_generations[generation]
        logger.info(f"GPU {gpu_name} is EOL (generation: {generation})")
        logger.info(f"Max driver version: {max_version}")
        return max_version

    if generation in limited_generations:
        max_version = limited_generations[generation]
        logger.info(
            f"GPU {gpu_name} has limited driver support (generation: {generation})"
        )
        logger.info(f"Max driver version: {max_version}")
        return max_version

    return None


def is_driver_compatible(driver_version: str, gpu: GPUInfo) -> bool:
    """Check if a driver version is compatible with the GPU.

    Args:
        driver_version: Driver version to check.
        gpu: GPU information.

    Returns:
        True if compatible, False otherwise.
    """
    driver_range = get_driver_range(gpu)

    if not _compare_versions(driver_version, driver_range.min_version):
        return False

    if driver_range.max_version:
        return _compare_versions(driver_range.max_version, driver_version)

    return True


def _compare_versions(v1: str, v2: str) -> bool:
    """Compare two version strings.

    Args:
        v1: First version.
        v2: Second version.

    Returns:
        True if v1 >= v2.
    """

    def parse_version(v: str) -> tuple:
        parts = re.findall(r"\d+", v)
        return tuple(int(p) for p in parts[:3])

    return parse_version(v1) >= parse_version(v2)


def get_latest_driver(generation: str) -> str:
    """Get the latest driver version for a GPU generation.

    Args:
        gpu_generation: GPU generation.

    Returns:
        Latest driver version string.
    """
    try:
        manager = _get_matrix_manager()
        gen_info = manager.get_generation_info(generation)
        if gen_info and gen_info.branches:
            branch_info = manager.get_branch_info(gen_info.branches[0])
            if branch_info:
                return branch_info.latest_version
    except Exception:
        pass

    latest_drivers = {
        "kepler": "470.256.02",
        "maxwell": "580.142",
        "pascal": "580.142",
        "volta": "580.142",
        "turing": "590.48.01",
        "ampere": "590.48.01",
        "ada": "590.48.01",
        "blackwell": "590.48.01",
    }

    return latest_drivers.get(generation, "590.48.01")


def is_driver_eol(generation: str) -> bool:
    """Check if a GPU generation is end-of-life (no longer receives regular driver updates).

    Args:
        generation: GPU generation.

    Returns:
        True if EOL (no more regular driver updates), False otherwise.
    """
    try:
        manager = _get_matrix_manager()
        gen_info = manager.get_generation_info(generation)
        if gen_info:
            return gen_info.is_eol
    except Exception:
        pass

    eol_generations = {"kepler"}
    return generation in eol_generations


def get_driver_branch(generation: str) -> str:
    """Get the driver branch name for a GPU generation.

    Args:
        generation: GPU generation.

    Returns:
        Driver branch name (e.g., "470", "580", "590").
    """
    try:
        manager = _get_matrix_manager()
        gen_info = manager.get_generation_info(generation)
        if gen_info and gen_info.branches:
            return gen_info.branches[0]
    except Exception:
        pass

    branch_map = {
        "kepler": "470",
        "maxwell": "580",
        "pascal": "580",
        "volta": "580",
        "turing": "590",
        "ampere": "590",
        "ada": "590",
        "blackwell": "590",
    }
    return branch_map.get(generation, "590")


def get_preferred_branch(generation: str, prefer_production: bool = True) -> str:
    """Get preferred driver branch for a GPU generation.

    For Blackwell GPUs, this allows choosing between:
    - 590: Production/New Feature branch (stable)
    - 595: New Feature branch (latest features)

    Args:
        generation: GPU generation (e.g., "blackwell")
        prefer_production: If True, prefer stable branches over new feature branches

    Returns:
        Preferred branch identifier
    """
    try:
        manager = _get_matrix_manager()
        gen_info = manager.get_generation_info(generation)
        if gen_info and gen_info.branches:
            # Blackwell has multiple branches - select based on preference
            if generation == "blackwell" and len(gen_info.branches) > 1:
                if prefer_production:
                    # Prefer 590 (production/stable) over 595 (new feature)
                    if "590" in gen_info.branches:
                        return "590"
                else:
                    # Prefer latest feature branch
                    return max(gen_info.branches)
            return gen_info.branches[0]
    except Exception:
        pass

    # Fallback
    return get_driver_branch(generation)


def format_driver_version(version: str) -> str:
    """Format driver version for display.

    Args:
        version: Raw version string.

    Returns:
        Formatted version (e.g., "535.154.05").
    """
    version = version.removeprefix("nvidia-")
    version = version.removeprefix("NVIDIA-")
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return version
