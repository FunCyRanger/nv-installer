"""Compatibility checking for nvidia-inst CLI.

This module provides compatibility checking functions for
system, driver, and CUDA compatibility.
"""

from typing import Any

from nvidia_inst.distro.detector import (
    DistroDetectionError,
    DistroInfo,
    detect_distro,
)
from nvidia_inst.gpu.compatibility import DriverRange, get_driver_range
from nvidia_inst.gpu.detector import (
    GPUDetectionError,
    GPUInfo,
    detect_gpu,
    has_nvidia_gpu,
)
from nvidia_inst.gpu.hybrid import detect_hybrid, get_native_tool
from nvidia_inst.installer.prerequisites import PrerequisitesChecker
from nvidia_inst.installer.validation import is_nvidia_working
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def check_compatibility() -> int:
    """Check system compatibility.

    Returns:
        0 if compatible, 1 otherwise.
    """
    from nvidia_inst.gpu.compatibility import is_driver_compatible

    try:
        distro = detect_distro()
        logger.info(f"Detected distribution: {distro}")
    except DistroDetectionError as e:
        logger.error(f"Failed to detect distribution: {e}")
        return 1

    if not has_nvidia_gpu():
        logger.info("No Nvidia GPU detected")
        print("\nNo Nvidia GPU detected. Nothing to do.")
        return 0

    try:
        gpu = detect_gpu()
        if not gpu:
            logger.error("Failed to detect GPU")
            return 1
    except GPUDetectionError as e:
        logger.error(f"Failed to detect GPU: {e}")
        return 1

    logger.info(f"Detected GPU: {gpu}")

    driver_range = get_driver_range(gpu)

    check_prerequisites(distro.id, distro.version_id, driver_range)

    print_compatibility_info(distro, gpu, driver_range)

    working_check = is_nvidia_working()
    if working_check.is_working:
        print(
            f"\n[OK] NVIDIA driver is working (version {working_check.driver_version})"
        )
        if working_check.driver_version and not is_driver_compatible(
            working_check.driver_version, gpu
        ):
            print(
                f"[WARNING] Installed driver {working_check.driver_version} may not be optimal for {gpu.model}"
            )
            print(
                f"  Recommended: {driver_range.min_version} - {driver_range.max_version or 'latest'}"
            )
            print("  -> Re-run this script to install the correct driver")
    elif working_check.gpu_detected:
        print("\n[INFO] NVIDIA GPU detected but driver not loaded")
    else:
        print("\n[INFO] No NVIDIA GPU detected")

    if driver_range.is_eol:
        print(f"\nWARNING: {driver_range.eol_message}")

    hybrid_info = detect_hybrid(distro.id)
    if hybrid_info:
        print("\n" + "-" * 50)
        print("HYBRID GRAPHICS DETECTED")
        print("-" * 50)
        print(f"System Type: {hybrid_info.system_type.capitalize()}")
        print(
            f"iGPU: {hybrid_info.igpu_type.upper() if hybrid_info.igpu_type else 'N/A'}"
        )
        print(f"dGPU: {hybrid_info.dgpu_model}")
        native_tool, _, _ = get_native_tool(distro.id)
        print(f"Native Tool: {native_tool or 'Environment file'}")
        print("\nUse --power-profile to configure: intel, hybrid, or nvidia")

    return 0


def print_compatibility_info(
    distro: DistroInfo,
    gpu: GPUInfo,
    driver_range: DriverRange,
) -> None:
    """Print compatibility information.

    Args:
        distro: Distribution information.
        gpu: GPU information.
        driver_range: Compatible driver range.
    """
    print("\n" + "=" * 50)
    print(" System Compatibility Check")
    print("=" * 50)

    # Distribution line
    print(f"\nDistribution: {distro}")

    # GPU line with optional compute capability and VRAM
    gpu_info = gpu.model
    details = []
    if gpu.compute_capability:
        details.append(f"Compute {gpu.compute_capability}")
    if gpu.vram:
        details.append(f"VRAM {gpu.vram}")
    if details:
        gpu_info += f" ({', '.join(details)})"
    print(f"GPU: {gpu_info}")

    # Driver line
    max_ver = driver_range.max_version or "latest"
    driver_status = (
        "EOL"
        if driver_range.is_eol
        else "Limited"
        if driver_range.is_limited
        else "Full"
    )
    print(f"Driver Range: {driver_range.min_version} - {max_ver} ({driver_status})")

    # CUDA line
    cuda_max = driver_range.cuda_max or "latest"
    if driver_range.cuda_is_locked:
        print(
            f"CUDA Range: {driver_range.cuda_min} - {cuda_max} (locked to {driver_range.cuda_locked_major}.x)"
        )
    else:
        print(f"CUDA Range: {driver_range.cuda_min} - {cuda_max}")


def check_prerequisites(
    distro_id: str,
    distro_version: str = "",
    driver_range: Any | None = None,
    fix: bool = False,
) -> int:
    """Check system prerequisites for driver installation.

    Args:
        distro_id: Distribution ID.
        distro_version: Distribution version.
        driver_range: Compatible driver range for the GPU.
        fix: Whether to automatically fix missing repositories.

    Returns:
        0 if all prerequisites met, 1 otherwise.
    """
    print("\n" + "=" * 50)
    print(" Prerequisites Check")
    print("=" * 50)

    checker = PrerequisitesChecker()
    result = checker.check_all(distro_id, distro_version, driver_range)

    print(
        f"\n[{'Y' if result.package_manager_available else 'N'}] Package manager: {result.package_manager}"
    )

    if result.repos_configured:
        for repo in result.repos_configured:
            print(f"[Y] {repo}")

    if result.repos_missing:
        for repo in result.repos_missing:
            print(f"[N] {repo}: NOT CONFIGURED")

    print(
        f"\n[{'Y' if result.driver_packages_available else 'N'}] Driver packages: ",
        end="",
    )
    if result.driver_packages_available:
        print(", ".join(result.driver_packages))
    else:
        print("NOT AVAILABLE (repos may need enabling)")

    if result.version_check:
        print_version_check(result.version_check, driver_range, distro_id)

    print("\n" + "-" * 50)

    if result.fix_commands:
        print("\nTo fix missing repositories, run:")
        for cmd in result.fix_commands:
            print(f"  {cmd}")

        if fix:
            print("\n--- Attempting to fix repositories ---")
            success, message = checker.fix_repositories(result.fix_commands)
            print(message)
            if success:
                print("\nRe-checking prerequisites...")
                return check_prerequisites(
                    distro_id, distro_version, driver_range, fix=False
                )

    print("\n" + "-" * 50)

    if result.success:
        print("\nStatus: READY - All prerequisites met")
    else:
        print("\nStatus: NOT READY - Some prerequisites not met")
        print("Please fix the issues above before installing.")

    print()

    return 0 if result.success else 1


def print_version_check(
    version_check: Any,
    driver_range: Any,
    distro_id: str,
) -> None:
    """Print version check results.

    Args:
        version_check: Version check result.
        driver_range: Compatible driver range.
        distro_id: Distribution ID.
    """
    print("\nVersion Check:")
    if version_check.success:
        print("[Y] Version check passed")
        if version_check.repo_versions:
            print(f"  Available in repos: {', '.join(version_check.repo_versions[:3])}")
        if version_check.official_versions:
            print(
                f"  Available from NVIDIA: {', '.join(version_check.official_versions[:3])}"
            )
    else:
        print("[N] Version check failed")
        for error in version_check.errors:
            print(f"  Error: {error}")

    for warning in version_check.warnings:
        print(f"  Warning: {warning}")
