"""Command-line interface for nvidia-inst."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from nvidia_inst.distro.detector import (
    DistroDetectionError,
    DistroInfo,
    detect_distro,
)
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.gpu.compatibility import (
    DriverRange,
    get_driver_range,
    is_driver_compatible,
)
from nvidia_inst.gpu.detector import (
    GPUDetectionError,
    GPUInfo,
    detect_gpu,
    has_nvidia_gpu,
)
from nvidia_inst.gpu.hybrid import (
    detect_hybrid,
    get_native_tool,
)
from nvidia_inst.installer.driver import (
    check_nonfree_available,
    check_nvidia_open_available,
    check_secure_boot,
    get_compatible_driver_packages,
    get_current_driver_type,
    get_nouveau_packages,
    get_nvidia_open_packages,
)
from nvidia_inst.installer.hybrid import (
    get_hybrid_packages,
    set_power_profile,
)
from nvidia_inst.installer.prerequisites import PrerequisitesChecker
from nvidia_inst.installer.secureboot import (
    SecureBootError,
    SecureBootState,
    enroll_mok_key,
    generate_mok_key,
    get_mok_key_paths,
    get_secure_boot_state,
    is_mok_enrolled,
    setup_auto_signing,
    sign_nvidia_modules,
)
from nvidia_inst.installer.uninstaller import (
    check_nvidia_packages_installed,
    revert_to_nouveau,
)
from nvidia_inst.installer.validation import (
    is_nvidia_working,
)
from nvidia_inst.utils.logger import get_logger, setup_logging


class DriverStatus(Enum):
    """Driver installation status."""

    OPTIMAL = "optimal"
    WRONG_BRANCH = "wrong_branch"
    NVIDIA_OPEN_ACTIVE = "nvidia_open_active"
    NOUVEAU_ACTIVE = "nouveau_active"
    BROKEN_INSTALL = "broken_install"
    NOTHING = "nothing"


@dataclass
class DriverOption:
    """A menu option for driver management."""

    number: int
    description: str
    action: str
    recommended: bool = False


@dataclass
class DriverState:
    """Current state of NVIDIA driver installation."""

    status: DriverStatus
    current_version: str | None
    is_compatible: bool
    is_optimal: bool
    suggested_packages: list[str] | None
    options: list[DriverOption]
    message: str
    cuda_range: str | None = None


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-distribution Nvidia driver installer with CUDA support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI (Tkinter or Zenity)",
    )

    parser.add_argument(
        "--gui-type",
        choices=["tkinter", "zenity"],
        help="Force specific GUI type",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check compatibility only, don't install",
    )

    parser.add_argument(
        "--driver-version",
        help="Specific driver version to install",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Install driver without CUDA",
    )

    parser.add_argument(
        "--cuda-version",
        help="Specific CUDA version to install",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    parser.add_argument(
        "--dry-run",
        "--simulate",
        action="store_true",
        help="Show what would be installed without actually installing",
    )

    parser.add_argument(
        "--revert-to-nouveau",
        action="store_true",
        help="Switch from proprietary driver to Nouveau (open-source)",
    )

    parser.add_argument(
        "--power-profile",
        choices=["intel", "hybrid", "nvidia"],
        help="Set hybrid graphics power profile",
    )

    return parser.parse_args()


def check_compatibility() -> int:
    """Check system compatibility.

    Returns:
        0 if compatible, 1 otherwise.
    """
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
            print("  → Re-run this script to install the correct driver")
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
    """Print compatibility information."""
    print("\n" + "=" * 50)
    print(" System Compatibility Check")
    print("=" * 50)

    print("\nDistribution:")
    print(f"  {distro}")

    print("\nGPU:")
    print(f"  {gpu.model}")
    if gpu.compute_capability:
        print(f"  Compute Capability: {gpu.compute_capability}")
    if gpu.vram:
        print(f"  VRAM: {gpu.vram}")

    print("\nCompatible Driver:")
    if driver_range.max_version:
        print(f"  {driver_range.min_version} - {driver_range.max_version}")
    else:
        print(f"  {driver_range.min_version} or later")

    print("\nCUDA Support:")
    if driver_range.cuda_max:
        print(f"  {driver_range.cuda_min} - {driver_range.cuda_max}")
    else:
        print(f"  {driver_range.cuda_min} or later")

    print("\nStatus:")
    print(f"  {'Compatible' if not driver_range.is_eol else 'Limited (EOL GPU)'}")


def _get_cuda_range_str(driver_range: DriverRange) -> str | None:
    """Format CUDA range for display."""
    if not driver_range.cuda_min:
        return None
    if driver_range.cuda_max:
        return f"CUDA {driver_range.cuda_min}-{driver_range.cuda_max}"
    return f"CUDA {driver_range.cuda_min}+"


def detect_driver_state(
    gpu: GPUInfo,
    driver_range: DriverRange,
    distro_id: str,
) -> DriverState:
    """Detect current driver state and available options.

    Args:
        gpu: Detected GPU information.
        driver_range: Compatible driver range for the GPU.
        distro_id: Distribution ID.

    Returns:
        DriverState with current status and available options.
    """
    driver_type = get_current_driver_type()
    working = is_nvidia_working()
    cuda_range = _get_cuda_range_str(driver_range)
    nonfree_available = check_nonfree_available()
    nvidia_open_available = check_nvidia_open_available()

    if working.is_working:
        compatible = (
            is_driver_compatible(working.driver_version, gpu)
            if working.driver_version
            else False
        )
        suggested = get_compatible_driver_packages(distro_id, driver_range)

        if compatible:
            return DriverState(
                status=DriverStatus.OPTIMAL,
                current_version=working.driver_version,
                is_compatible=True,
                is_optimal=True,
                suggested_packages=suggested,
                options=_build_optimal_options(
                    driver_type, cuda_range, nvidia_open_available
                ),
                message=f"NVIDIA driver {working.driver_version} is working optimally",
                cuda_range=cuda_range,
            )
        else:
            return DriverState(
                status=DriverStatus.WRONG_BRANCH,
                current_version=working.driver_version,
                is_compatible=False,
                is_optimal=False,
                suggested_packages=suggested,
                options=_build_wrong_branch_options(
                    driver_range, cuda_range, nvidia_open_available, nonfree_available
                ),
                message=f"Driver {working.driver_version} may not be optimal for {gpu.model}",
                cuda_range=cuda_range,
            )

    elif driver_type == "nouveau":
        suggested = get_compatible_driver_packages(distro_id, driver_range)
        return DriverState(
            status=DriverStatus.NOUVEAU_ACTIVE,
            current_version=None,
            is_compatible=False,
            is_optimal=False,
            suggested_packages=suggested,
            options=_build_nouveau_options(
                cuda_range, nvidia_open_available, nonfree_available
            ),
            message="Nouveau (open-source) driver is active",
            cuda_range=cuda_range,
        )

    elif driver_type == "nvidia_open":
        suggested = get_compatible_driver_packages(distro_id, driver_range)
        return DriverState(
            status=DriverStatus.NVIDIA_OPEN_ACTIVE,
            current_version=working.driver_version if working.is_working else None,
            is_compatible=True,
            is_optimal=True,
            suggested_packages=suggested,
            options=_build_nvidia_open_options(cuda_range, nonfree_available),
            message="NVIDIA Open driver is active",
            cuda_range=cuda_range,
        )

    else:
        suggested = get_compatible_driver_packages(distro_id, driver_range)
        return DriverState(
            status=DriverStatus.NOTHING,
            current_version=None,
            is_compatible=False,
            is_optimal=False,
            suggested_packages=suggested,
            options=_build_nothing_options(
                cuda_range, nvidia_open_available, nonfree_available
            ),
            message="No NVIDIA driver installed"
            + (" (non-free repos not enabled)" if not nonfree_available else ""),
            cuda_range=cuda_range,
        )


def _build_optimal_options(
    driver_type: str,
    cuda_range: str | None,
    nvidia_open_available: bool,
) -> list[DriverOption]:
    """Build options for optimal driver state."""
    options = [
        DriverOption(1, "Upgrade to latest", "upgrade", recommended=True),
        DriverOption(2, "Keep current driver", "keep"),
    ]

    cuda_suffix = f" ({cuda_range})" if cuda_range else ""

    if nvidia_open_available and driver_type != "nvidia_open":
        options.append(
            DriverOption(
                len(options) + 1,
                f"Switch to NVIDIA Open{cuda_suffix}"
                if cuda_suffix
                else "Switch to NVIDIA Open (open-source kernel module)",
                "switch_nvidia_open",
            )
        )

    options.append(
        DriverOption(
            len(options) + 1,
            "Switch to Nouveau (open-source)"
            + (f" ({cuda_range})" if cuda_range else ""),
            "switch_nouveau",
        )
    )

    return options


def _build_wrong_branch_options(
    driver_range: DriverRange,
    cuda_range: str | None,
    nvidia_open_available: bool,
    nonfree_available: bool,
) -> list[DriverOption]:
    """Build options for wrong branch driver state."""
    cuda_suffix = f" ({cuda_range})" if cuda_range else ""

    options = []

    if nonfree_available:
        branch_desc = (
            f"Install correct branch ({driver_range.max_branch}){cuda_suffix}"
            if cuda_suffix
            else f"Install correct branch ({driver_range.max_branch})"
        )
        options.append(DriverOption(1, branch_desc, "install", recommended=True))
    else:
        options.append(
            DriverOption(
                1,
                f"Enable non-free repos + install correct branch ({driver_range.max_branch}){cuda_suffix}",
                "install",
                recommended=True,
            )
        )

    options.append(DriverOption(2, "Keep current driver", "keep"))

    if nvidia_open_available:
        open_desc = (
            f"Switch to NVIDIA Open{cuda_suffix}"
            if cuda_suffix
            else "Switch to NVIDIA Open"
        )
        options.append(DriverOption(3, open_desc, "switch_nvidia_open"))

    options.append(
        DriverOption(
            len(options) + 1,
            "Switch to Nouveau (open-source, no CUDA support)",
            "switch_nouveau",
        )
    )

    return options


def _build_nouveau_options(
    cuda_range: str | None,
    nvidia_open_available: bool,
    nonfree_available: bool,
) -> list[DriverOption]:
    """Build options for Nouveau active state."""
    cuda_suffix = f" ({cuda_range})" if cuda_range else ""
    options = []

    if nonfree_available:
        options.append(
            DriverOption(
                1,
                f"Switch to proprietary driver{cuda_suffix}"
                if cuda_suffix
                else "Switch to proprietary driver",
                "install",
                recommended=True,
            )
        )
    else:
        options.append(
            DriverOption(
                1,
                f"Enable non-free repos + switch to proprietary{cuda_suffix}"
                if cuda_suffix
                else "Enable non-free repos + switch to proprietary",
                "install",
                recommended=True,
            )
        )

    if nvidia_open_available:
        open_suffix = cuda_suffix if cuda_range else ""
        options.append(
            DriverOption(
                len(options) + 1,
                f"Switch to NVIDIA Open{open_suffix}"
                if cuda_range
                else "Switch to NVIDIA Open",
                "install_nvidia_open",
            )
        )

    options.append(DriverOption(len(options) + 1, "Keep Nouveau (open-source)", "keep"))

    return options


def _build_nvidia_open_options(
    cuda_range: str | None,
    nonfree_available: bool,
) -> list[DriverOption]:
    """Build options for NVIDIA Open active state."""
    cuda_suffix = f" ({cuda_range})" if cuda_range else ""
    options = [
        DriverOption(1, "Upgrade to latest", "upgrade", recommended=True),
        DriverOption(2, "Keep NVIDIA Open", "keep"),
    ]

    if nonfree_available:
        options.append(
            DriverOption(
                3,
                f"Switch to proprietary{cuda_suffix}"
                if cuda_suffix
                else "Switch to proprietary driver",
                "install",
            )
        )

    options.append(
        DriverOption(
            len(options) + 1,
            f"Switch to Nouveau (open-source){cuda_suffix}"
            if cuda_suffix
            else "Switch to Nouveau (open-source)",
            "switch_nouveau",
        )
    )

    return options


def _build_nothing_options(
    cuda_range: str | None,
    nvidia_open_available: bool,
    nonfree_available: bool,
) -> list[DriverOption]:
    """Build options for no driver installed state."""
    cuda_suffix = f" ({cuda_range})" if cuda_range else ""
    options = []

    if nonfree_available:
        options.append(
            DriverOption(
                1,
                f"Install proprietary driver{cuda_suffix}"
                if cuda_suffix
                else "Install proprietary driver",
                "install",
                recommended=True,
            )
        )
    else:
        options.append(
            DriverOption(
                1,
                f"Enable non-free repos + install proprietary{cuda_suffix}"
                if cuda_suffix
                else "Enable non-free repos + install proprietary driver",
                "install",
                recommended=True,
            )
        )

    if nvidia_open_available:
        open_suffix = cuda_suffix if cuda_range else ""
        if nonfree_available:
            options.append(
                DriverOption(
                    len(options) + 1,
                    f"Install NVIDIA Open{open_suffix}"
                    if cuda_range
                    else "Install NVIDIA Open",
                    "install_nvidia_open",
                )
            )
        else:
            options.append(
                DriverOption(
                    len(options) + 1,
                    f"Enable non-free repos + install NVIDIA Open{open_suffix}"
                    if cuda_range
                    else "Enable non-free repos + install NVIDIA Open",
                    "install_nvidia_open",
                )
            )

    options.append(
        DriverOption(
            len(options) + 1,
            "Install Nouveau (open-source, no CUDA support)",
            "install_nouveau",
        )
    )

    options.append(DriverOption(len(options) + 1, "Cancel", "cancel"))

    return options


def show_driver_options(state: DriverState) -> int:
    """Show driver options menu and return selected option.

    Args:
        state: Current driver state with available options.

    Returns:
        Selected DriverOption, or None to cancel.
    """
    print(f"\n{'=' * 50}")
    print(" Driver Status")
    print(f"{'=' * 50}")
    print(f"\n{state.message}")

    if state.current_version:
        print(f"  Installed: {state.current_version}")

    if not state.is_compatible and state.suggested_packages:
        print(f"  Recommended: {' '.join(state.suggested_packages)}")

    print("\nOptions:")
    for opt in state.options:
        rec = " [RECOMMENDED]" if opt.recommended else ""
        print(f"  [{opt.number}] {opt.description}{rec}")

    while True:
        try:
            choice = input("\nSelect option: ")
            choice_num = int(choice)
            if any(opt.number == choice_num for opt in state.options):
                return choice_num
            print("Invalid option. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return -1


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
        f"\n[{'✓' if result.package_manager_available else '✗'}] Package manager: {result.package_manager}"
    )

    if result.repos_configured:
        for repo in result.repos_configured:
            print(f"[✓] {repo}")

    if result.repos_missing:
        for repo in result.repos_missing:
            print(f"[✗] {repo}: NOT CONFIGURED")

    print(
        f"\n[{'✓' if result.driver_packages_available else '✗'}] Driver packages: ",
        end="",
    )
    if result.driver_packages_available:
        print(", ".join(result.driver_packages))
    else:
        print("NOT AVAILABLE (repos may need enabling)")

    if result.version_check:
        print_version_check(result.version_check, driver_range)

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


def handle_secure_boot(
    distro_id: str,
    skip_confirmation: bool = False,
) -> tuple[bool, SecureBootState]:
    """Handle Secure Boot setup for NVIDIA driver installation.

    Args:
        distro_id: Distribution ID.
        skip_confirmation: Skip user prompts.

    Returns:
        Tuple of (setup_completed, secure_boot_state).
    """
    state = get_secure_boot_state()

    if state == SecureBootState.DISABLED:
        print("\n[INFO] Secure Boot is disabled - no signing required.")
        return (True, state)

    if state == SecureBootState.UNKNOWN:
        print("\n[WARNING] Cannot detect Secure Boot state.")
        print("  mokutil may not be installed. Driver may fail to load.")
        return (True, state)

    print(f"\n[INFO] Secure Boot is {state.value.replace('_', ' ').title()}")

    if state == SecureBootState.ENABLED:
        print("\n  Secure Boot is enabled. NVIDIA drivers need to be signed.")
        print("  We can set up MOK (Machine Owner Key) signing.")

    if state == SecureBootState.SETUP_MODE:
        print(
            "\n  System is in Setup Mode - can enroll keys directly (no reboot needed)."
        )

    key_paths = get_mok_key_paths(distro_id)

    if is_mok_enrolled(key_paths.public_cert):
        print("\n[INFO] MOK key already enrolled - modules should load correctly.")
        return (True, state)

    print("\n  Would you like to set up MOK signing?")
    print("  This will:")
    print("    1. Generate a signing key pair")
    print("    2. Enroll the public key in Secure Boot")
    print("    3. Set up automatic re-signing on kernel updates")

    if state == SecureBootState.ENABLED:
        print("\n  NOTE: Reboot will be required after enrollment to complete setup.")

    if not skip_confirmation:
        response = input("\nSet up MOK signing? [Y/n]: ")
        if response.lower() in ("n", "no"):
            print("\n[WARNING] Driver may fail to load without signing.")
            print("  You can set up signing later by running this script again.")
            return (False, state)

    try:
        key_dir = key_paths.private_key.parent
        print("\n[INFO] Generating MOK key pair...")
        generate_mok_key(key_dir)

        print("[INFO] Enrolling MOK key...")
        result = enroll_mok_key(key_paths.public_cert)

        if result.requires_reboot:
            print("\n" + "=" * 50)
            print(" IMPORTANT: Reboot Required")
            print("=" * 50)
            if result.reboot_instructions:
                print(f"\n{result.reboot_instructions}")
            print("\nAfter completing MOK enrollment, run this script again")
            print("to continue with driver installation.\n")

        print("\n[INFO] Setting up automatic signing for future updates...")
        setup_result = setup_auto_signing(
            key_paths.private_key,
            key_paths.public_cert,
            distro_id,
        )

        if setup_result.success:
            print("[INFO] Automatic signing configured successfully.")
        else:
            print(f"[WARNING] Auto-signing setup: {setup_result.message}")

        print("\n[INFO] Signing currently installed NVIDIA modules...")
        signed, failed = sign_nvidia_modules(
            key_paths.private_key,
            key_paths.public_cert,
        )
        print(f"  Signed: {signed}, Failed: {failed}")

        return (True, state)

    except SecureBootError as e:
        logger.error(f"Secure Boot setup failed: {e}")
        print(f"\n[ERROR] Secure Boot setup failed: {e}")
        print("\n  You may need to:")
        print("    - Install mokutil: sudo apt install mokutil (or equivalent)")
        print("    - Install openssl: sudo apt install openssl (or equivalent)")
        return (False, state)

    except Exception as e:
        logger.error(f"Unexpected error during Secure Boot setup: {e}")
        print(f"\n[ERROR] Unexpected error: {e}")
        return (False, state)


def print_version_check(version_check: Any, driver_range: Any) -> None:
    """Print version check results."""
    print("\n" + "-" * 50)
    print(" Version Availability Check")
    print("-" * 50)

    print("\nRepository:")
    if version_check.repo_versions:
        print(f"  Available: {', '.join(version_check.repo_versions[:5])}")
        if len(version_check.repo_versions) > 5:
            print(f"           ... ({len(version_check.repo_versions)} total)")
        print(f"  Latest: {version_check.repo_versions[0]}")
    else:
        print("  Unable to determine available versions")

    print("\nOfficial (NVIDIA Archive):")
    if version_check.official_versions:
        print(f"  Available: {', '.join(version_check.official_versions[:5])}")
        if len(version_check.official_versions) > 5:
            print(f"           ... ({len(version_check.official_versions)} total)")
    else:
        print("  Unable to fetch official versions")

    if version_check.installed_driver_version:
        print(f"\nInstalled Driver: {version_check.installed_driver_version}")

    print("\nCompatibility:")
    if version_check.compatible_versions:
        print(f"  ✓ Compatible: {', '.join(version_check.compatible_versions[:3])}")
        if len(version_check.compatible_versions) > 3:
            print(f"              ... ({len(version_check.compatible_versions)} total)")
    if version_check.incompatible_versions:
        print(f"  ✗ Incompatible: {', '.join(version_check.incompatible_versions[:3])}")
        if driver_range and driver_range.max_branch:
            print(f"    Your GPU requires: {driver_range.max_branch}.xx drivers")

    if not version_check.compatible:
        print("\n[✗] BLOCKED - No compatible driver available in repos")
    elif version_check.warnings:
        for warning in version_check.warnings:
            print(f"\n[⚠] {warning}")
    else:
        print("\n[✓] PASSED - Compatible driver available")


def _get_packages_to_remove(distro_id: str) -> list[str]:
    """Get list of NVIDIA packages to remove based on distro."""
    if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        return [
            "nvidia-driver-*",
            "nvidia-dkms-*",
            "nvidia-kernel-common-*",
            "nvidia-kernel-source-*",
            "nvidia-settings",
            "nvidia-utils-*",
            "libnvidia-*",
            "xserver-xorg-video-nvidia",
        ]
    if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        return [
            "akmod-nvidia",
            "xorg-x11-drv-nvidia",
            "xorg-x11-drv-nvidia-cuda",
            "xorg-x11-drv-nvidia-drm",
            "xorg-x11-drv-nvidia-kmodsrc",
            "nvidia-persistenced",
            "nvidia-settings",
        ]
    if distro_id in ("arch", "manjaro"):
        return [
            "nvidia",
            "nvidia-open",
            "nvidia-580xx-dkms",
            "nvidia-470xx-dkms",
            "nvidia-utils",
            "nvidia-settings",
            "lib32-nvidia-utils",
            "lib32-nvidia-580xx-utils",
            "lib32-nvidia-470xx-utils",
        ]
    if distro_id in ("opensuse", "sles"):
        return [
            "x11-video-nvidiaG05",
            "x11-video-nvidiaG04",
            "nvidia-computeG05",
            "nvidia-computeG04",
        ]
    return []


def _remove_packages(distro_id: str, packages: list[str]) -> list[str]:
    """Remove packages using the distribution's package manager."""
    removed = []
    for pkg_pattern in packages:
        try:
            if distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
                cmd = ["apt-get", "remove", "--purge", "-y", pkg_pattern]
            elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
                cmd = ["dnf", "remove", "-y", pkg_pattern]
            elif distro_id in ("arch", "manjaro"):
                cmd = ["pacman", "-Rns", "--noconfirm", pkg_pattern]
            elif distro_id in ("opensuse", "sles"):
                cmd = ["zypper", "remove", "-y", pkg_pattern]
            else:
                continue

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"Removed: {pkg_pattern}")
                removed.append(pkg_pattern)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Failed to remove {pkg_pattern}: {e}")

    return removed


def _rebuild_initramfs(distro_id: str) -> bool:
    """Rebuild initramfs for the distribution."""
    from nvidia_inst.utils.permissions import is_root

    try:
        if distro_id in (
            "fedora",
            "rhel",
            "centos",
            "rocky",
            "alma",
            "opensuse",
            "sles",
        ):
            cmd = ["dracut", "-f", "--regenerate-all"]
        elif distro_id in ("arch", "manjaro"):
            cmd = ["mkinitcpio", "-P"]
        else:
            cmd = ["update-initramfs", "-u"]

        if not is_root():
            cmd = ["sudo"] + cmd

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Initramfs rebuild failed: {e}")
        return False


def _prompt_reboot() -> None:
    """Prompt user to reboot."""
    print("\nPlease reboot your system for changes to take effect.")
    response = input("Reboot now? [y/N]: ")
    if response.lower() in ("y", "yes"):
        try:
            subprocess.run(["sudo", "reboot"])
        except Exception:
            print("Reboot command failed. Please reboot manually.")


def _dry_run_change(
    state: DriverState,
    packages: list[str],
    distro: DistroInfo,
) -> None:
    """Show dry-run output for driver change."""
    print("\n" + "=" * 50)
    print(" DRY-RUN MODE")
    print("=" * 50)

    print(f"\nCurrent state: {state.message}")
    if state.current_version:
        print(f"  Installed: {state.current_version}")

    print(f"\nTarget packages: {' '.join(packages)}")

    print("\nSteps to execute manually:")
    if state.current_version or state.status == DriverStatus.NOUVEAU_ACTIVE:
        if distro.id in ("ubuntu", "debian"):
            print("  1. sudo apt remove --purge -y nvidia-driver-* nvidia-dkms-*")
        elif distro.id in ("fedora", "rhel", "centos"):
            print("  1. sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia*")
        elif distro.id in ("arch", "manjaro"):
            print("  1. sudo pacman -Rns --noconfirm nvidia nvidia-utils")
        elif distro.id in ("opensuse", "sles"):
            print("  1. sudo zypper remove -y x11-video-nvidiaG05")
    if distro.id in ("ubuntu", "debian"):
        print("  2. sudo apt update")
        print(f"  3. sudo apt install -y {' '.join(packages)}")
        print("  4. sudo update-initramfs -u")
    elif distro.id in ("fedora", "rhel", "centos"):
        print("  2. sudo dnf makecache")
        print(f"  3. sudo dnf install -y {' '.join(packages)}")
        print("  4. sudo dracut -f --regenerate-all")
    elif distro.id in ("arch", "manjaro"):
        print("  2. sudo pacman -Sy")
        print(f"  3. sudo pacman -S --noconfirm {' '.join(packages)}")
        print("  4. sudo mkinitcpio -P")
    print("  5. sudo reboot")

    print("\nOr run this script without --dry-run:")
    print("  sudo nvidia-inst")


def _dry_run_nvidia_open_install(
    distro: DistroInfo,
    packages: list[str],
) -> None:
    """Show dry-run output for NVIDIA Open installation."""
    print("\n" + "=" * 50)
    print(" DRY-RUN MODE - NVIDIA Open Installation")
    print("=" * 50)

    print(f"\nTarget packages: {' '.join(packages)}")

    print("\nSteps to execute manually:")
    driver_type = get_current_driver_type()
    if driver_type in ("proprietary", "nvidia_open"):
        if distro.id in ("ubuntu", "debian"):
            print("  1. sudo apt remove --purge -y nvidia-driver-* nvidia-dkms-*")
        elif distro.id in ("fedora", "rhel", "centos"):
            print("  1. sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia*")
        elif distro.id in ("arch", "manjaro"):
            print("  1. sudo pacman -Rns --noconfirm nvidia nvidia-utils")
        elif distro.id in ("opensuse", "sles"):
            print("  1. sudo zypper remove -y x11-video-nvidiaG05")
        print()

    if distro.id in ("ubuntu", "debian"):
        print(f"  2. sudo apt install -y {' '.join(packages)}")
        print("  3. sudo update-initramfs -u")
    elif distro.id in ("fedora", "rhel", "centos"):
        print(f"  2. sudo dnf install -y {' '.join(packages)}")
        print("  3. sudo dracut -f --regenerate-all")
    elif distro.id in ("arch", "manjaro"):
        print(f"  2. sudo pacman -S --noconfirm {' '.join(packages)}")
        print("  3. sudo mkinitcpio -P")
    elif distro.id in ("opensuse", "sles"):
        print(f"  2. sudo zypper install -y {' '.join(packages)}")
        print("  3. sudo dracut -f --regenerate-all")
    print("  4. sudo reboot")

    print("\nOr run this script without --dry-run:")
    print("  sudo nvidia-inst")


def _dry_run_nouveau_install(
    distro: DistroInfo,
    packages: list[str],
) -> None:
    """Show dry-run output for Nouveau installation."""
    print("\n" + "=" * 50)
    print(" DRY-RUN MODE - Nouveau Installation")
    print("=" * 50)

    print(f"\nTarget packages: {' '.join(packages)}")

    print("\nSteps to execute manually:")
    driver_type = get_current_driver_type()
    if driver_type in ("proprietary", "nvidia_open"):
        if distro.id in ("ubuntu", "debian"):
            print("  1. sudo apt remove --purge -y nvidia-driver-* nvidia-dkms-*")
        elif distro.id in ("fedora", "rhel", "centos"):
            print("  1. sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia*")
        elif distro.id in ("arch", "manjaro"):
            print("  1. sudo pacman -Rns --noconfirm nvidia nvidia-utils")
        elif distro.id in ("opensuse", "sles"):
            print("  1. sudo zypper remove -y x11-video-nvidiaG05 x11-video-nvidiaG06")
        print()

    if distro.id in ("ubuntu", "debian"):
        print(f"  2. sudo apt install -y {' '.join(packages)}")
        print("  3. sudo update-initramfs -u")
    elif distro.id in ("fedora", "rhel", "centos"):
        print(f"  2. sudo dnf install -y {' '.join(packages)}")
        print("  3. sudo dracut -f --regenerate-all")
    elif distro.id in ("arch", "manjaro"):
        print(f"  2. sudo pacman -S --noconfirm {' '.join(packages)}")
        print("  3. sudo mkinitcpio -P")
    elif distro.id in ("opensuse", "sles"):
        print(f"  2. sudo zypper install -y {' '.join(packages)}")
        print("  3. sudo dracut -f --regenerate-all")
    print("  4. sudo reboot")

    print("\nOr run this script without --dry-run:")
    print("  sudo nvidia-inst")


def _dry_run_revert(distro: DistroInfo) -> None:
    """Show dry-run output for reverting to Nouveau."""
    from nvidia_inst.installer.uninstaller import _get_packages_to_remove

    packages = _get_packages_to_remove(distro.id)

    print("\n" + "=" * 50)
    print(" DRY-RUN MODE: Revert to Nouveau")
    print("=" * 50)

    print("\nRemoving NVIDIA packages:")
    if distro.id in ("ubuntu", "debian"):
        print(f"  sudo apt-get remove --purge -y {' '.join(packages)}")
    elif distro.id in ("fedora", "rhel", "centos"):
        for pkg in packages:
            print(f"  sudo dnf remove -y -- {pkg}")
    elif distro.id in ("arch", "manjaro"):
        print(f"  sudo pacman -Rns --noconfirm {' '.join(packages)}")
    elif distro.id in ("opensuse", "sles"):
        for pkg in packages:
            print(f"  sudo zypper remove -y -- {pkg}")

    if distro.id in ("fedora", "rhel", "centos"):
        print("\nRemoving versionlock entries:")
        for pkg in packages:
            print(f"  sudo dnf versionlock delete -- {pkg}")

    if distro.id in ("ubuntu", "debian"):
        print("\nRemoving apt preferences:")
        print("  sudo rm /etc/apt/preferences.d/nvidia")

    print("\nRebuilding initramfs:")
    if distro.id in ("fedora", "rhel", "centos", "opensuse", "sles"):
        print("  sudo dracut -f --regenerate-all")
    elif distro.id in ("arch", "manjaro"):
        print("  sudo mkinitcpio -P")
    else:
        print("  sudo update-initramfs -u")

    print("\nReboot:")
    print("  sudo reboot")

    print("\n" + "=" * 50)
    print(" Or run without --dry-run to execute:")
    print("  sudo nvidia-inst")
    print("=" * 50)


def execute_driver_change(
    option: DriverOption,
    state: DriverState,
    distro: DistroInfo,
    gpu: GPUInfo,
    driver_range: DriverRange,
    dry_run: bool = False,
) -> int:
    """Execute the selected driver change.

    Args:
        option: Selected driver option.
        state: Current driver state.
        distro: Distribution information.
        gpu: GPU information.
        driver_range: Compatible driver range.
        dry_run: If True, show what would happen without executing.

    Returns:
        0 on success, 1 on failure.
    """
    if option.action == "keep":
        print("\nNo changes made.")
        return 0

    if option.action == "cancel":
        print("\nCancelled.")
        return 0

    if option.action == "revert_nouveau":
        if dry_run:
            _dry_run_revert(distro)
            return 0

        from nvidia_inst.utils.permissions import require_root

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to remove drivers.")
            return 1

        print("\n[WARNING] This will remove NVIDIA proprietary driver.")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

        print("\nRemoving proprietary driver...")
        result = revert_to_nouveau(distro.id)
        if result.success:
            print(f"\n✓ {result.message}")
            _prompt_reboot()
        else:
            print(f"\n✗ Revert failed: {', '.join(result.errors)}")
            return 1
        return 0

    if option.action in ("install", "upgrade"):
        packages = state.suggested_packages or get_compatible_driver_packages(
            distro.id, driver_range
        )

        if dry_run:
            _dry_run_change(state, packages, distro)
            return 0

        from nvidia_inst.utils.permissions import require_root

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        driver_type = get_current_driver_type()
        if driver_type == "nouveau":
            print("\nNouveau is active. Installing proprietary driver.")
            print("Note: Nouveau blacklist will be created automatically.")
        else:
            print("\nRemoving old driver packages...")
            packages_to_remove = _get_packages_to_remove(distro.id)
            removed = _remove_packages(distro.id, packages_to_remove)
            if removed:
                print(f"  Removed: {', '.join(removed)}")

        print(f"\nInstalling: {' '.join(packages)}")
        pkg_manager = get_package_manager()
        try:
            pkg_manager.install(packages)
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            print(f"Installation failed: {e}")
            return 1

        print("\nRebuilding initramfs...")
        if not _rebuild_initramfs(distro.id):
            print("[WARNING] Initramfs rebuild had issues. Reboot may fail.")

        if check_secure_boot():
            print("\nSecure Boot detected. Re-signing modules...")
            key_paths = get_mok_key_paths(distro.id)
            signed, failed = sign_nvidia_modules(
                key_paths.private_key,
                key_paths.public_cert,
            )
            if signed > 0:
                print(f"  Signed: {signed}, Failed: {failed}")
            else:
                print(
                    "[WARNING] Module signing failed. Driver may not load with Secure Boot."
                )

        print("\n✓ Driver installed successfully")
        _prompt_reboot()
        return 0

    if option.action in ("install_nvidia_open", "switch_nvidia_open"):
        packages = get_nvidia_open_packages(distro.id, driver_range)

        if dry_run:
            _dry_run_nvidia_open_install(distro, packages)
            return 0

        from nvidia_inst.utils.permissions import require_root

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        driver_type = get_current_driver_type()
        if driver_type in ("proprietary", "nvidia_open"):
            print("\nRemoving old driver packages...")
            packages_to_remove = _get_packages_to_remove(distro.id)
            removed = _remove_packages(distro.id, packages_to_remove)
            if removed:
                print(f"  Removed: {', '.join(removed)}")

        if driver_type == "nouveau":
            print("\nNouveau is active. Installing NVIDIA Open driver.")

        print(f"\nInstalling: {' '.join(packages)}")
        pkg_manager = get_package_manager()
        try:
            pkg_manager.install(packages)
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            print(f"Installation failed: {e}")
            return 1

        print("\nRebuilding initramfs...")
        if not _rebuild_initramfs(distro.id):
            print("[WARNING] Initramfs rebuild had issues. Reboot may fail.")

        print("\n✓ NVIDIA Open installed successfully")
        _prompt_reboot()
        return 0

    if option.action in ("install_nouveau", "switch_nouveau"):
        packages = get_nouveau_packages(distro.id)

        if dry_run:
            _dry_run_nouveau_install(distro, packages)
            return 0

        from nvidia_inst.utils.permissions import require_root

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        print("\nInstalling Nouveau driver...")
        pkg_manager = get_package_manager()
        try:
            pkg_manager.install(packages)
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            print(f"Installation failed: {e}")
            return 1

        print("\nRebuilding initramfs...")
        if not _rebuild_initramfs(distro.id):
            print("[WARNING] Initramfs rebuild had issues. Reboot may fail.")

        print("\n✓ Nouveau driver installed successfully")
        _prompt_reboot()
        return 0

    return 0


def install_driver_cli(
    driver_version: str | None = None,
    with_cuda: bool = True,
    cuda_version: str | None = None,
    skip_confirmation: bool = False,
    dry_run: bool = False,
) -> int:
    """Install driver from CLI.

    Args:
        driver_version: Specific driver version.
        with_cuda: Install CUDA.
        cuda_version: CUDA version.
        skip_confirmation: Skip confirmation prompt.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        distro = detect_distro()
    except DistroDetectionError as e:
        logger.error(f"Failed to detect distribution: {e}")
        return 1

    if not has_nvidia_gpu():
        print("No Nvidia GPU detected. Nothing to do.")
        return 0

    try:
        gpu = detect_gpu()
        if not gpu:
            logger.error("Failed to detect GPU")
            return 1
    except GPUDetectionError as e:
        logger.error(f"Failed to detect GPU: {e}")
        return 1

    driver_range = get_driver_range(gpu)

    if driver_range.is_eol and driver_version is None:
        driver_version = driver_range.max_version
        logger.info(f"Using EOL driver version: {driver_version}")

    print_compatibility_info(distro, gpu, driver_range)

    if driver_range.is_eol:
        print(f"\nWARNING: {driver_range.eol_message}")

    if dry_run:
        return _run_dry_run(
            distro, gpu, driver_range, driver_version, with_cuda, cuda_version
        )

    state = detect_driver_state(gpu, driver_range, distro.id)
    selected = show_driver_options(state)

    if selected == -1:
        return 0

    option = next(opt for opt in state.options if opt.number == selected)

    return execute_driver_change(
        option, state, distro, gpu, driver_range, dry_run=False
    )


def _run_dry_run(
    distro: DistroInfo,
    gpu: GPUInfo,
    driver_range: DriverRange,
    driver_version: str | None,
    with_cuda: bool,
    cuda_version: str | None,
) -> int:
    """Run in dry-run mode to show what would be installed."""
    from nvidia_inst.distro.factory import get_package_manager
    from nvidia_inst.installer.driver import get_compatible_driver_packages

    print("\n" + "=" * 50)
    print(" SIMULATION MODE - No changes will be made")
    print("=" * 50)

    print("\n--- System Information ---")
    print(f"Distribution: {distro}")
    print(f"GPU: {gpu.model}")
    if gpu.compute_capability:
        print(f"Compute Capability: {gpu.compute_capability}")
    print(f"Driver Range: {driver_range.min_version}", end="")
    if driver_range.max_version:
        print(f" - {driver_range.max_version}")
    else:
        print(" or later")

    print("\n--- Prerequisites Check ---")
    check_prerequisites(distro.id, distro.version_id, driver_range)

    print("\n--- Pre-install Checks ---")

    from nvidia_inst.installer.driver import check_nouveau

    nouveau_loaded = check_nouveau()
    if nouveau_loaded:
        print("[ ] Nouveau kernel module - NEEDS TO BE DISABLED")
    else:
        print("[x] Nouveau kernel module - Not loaded")

    sb_enabled = check_secure_boot()
    if sb_enabled:
        print("[ ] Secure Boot - NEEDS ATTENTION")
    else:
        print("[x] Secure Boot - Disabled")

    print("\n--- Installation Plan ---")

    pkg_manager = get_package_manager()
    packages = get_compatible_driver_packages(distro.id, driver_range)

    if driver_version:
        print(f"Requested driver version: {driver_version}")
    elif driver_range.max_branch:
        if driver_range.is_eol:
            print(f"Selected driver version (EOL): {driver_range.max_version}")
        else:
            print(
                f"Selected driver branch: {driver_range.max_branch}.xx (current max: {driver_range.max_version})"
            )
            print(
                f"  -> Will receive branch updates (e.g., {driver_range.max_branch}.143, {driver_range.max_branch}.144)"
            )
    else:
        print("Selected driver version: Latest (590.xx)")

    print(f"\nPackage manager: {pkg_manager.__class__.__name__}")

    print("\nDriver packages to install:")
    for pkg in packages:
        print(f"  - {pkg}")

    if with_cuda:
        print("\nCUDA packages to install:")
        cuda_installer = _get_cuda_installer(distro.id)
        cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
        for pkg in cuda_pkgs:
            print(f"  - {pkg}")

    print("\n--- Commands to Execute ---")

    print("# Step 1: Update package lists:")
    if distro.id in ("ubuntu", "debian"):
        print("  sudo apt update")
    elif distro.id in ("fedora", "rhel", "centos"):
        print("  sudo dnf makecache")
    elif distro.id in ("arch", "manjaro"):
        print("  sudo pacman -Sy")
    elif distro.id in ("opensuse"):
        print("  sudo zypper refresh")

    if driver_range.max_branch and driver_range.is_limited:
        print(
            "\n# Step 2: Lock driver to correct branch (prevents incompatible drivers):"
        )
        if distro.id in ("fedora", "rhel", "centos"):
            print(
                f"  sudo dnf versionlock add --raw 'akmod-nvidia-{driver_range.max_branch}.*'"
            )
        elif distro.id in ("ubuntu", "debian"):
            print("  # Add to /etc/apt/preferences.d/nvidia:")
            print(
                f"  echo 'Package: nvidia-driver-*\nPin: version {driver_range.max_branch}.*\nPin-Priority: 1001' | sudo tee /etc/apt/preferences.d/nvidia"
            )
        elif distro.id in ("arch", "manjaro"):
            print("  # Lock current branch package:")
            print(f"  sudo pacman -D --lock nvidia-{driver_range.max_branch}xx")
        elif distro.id in ("opensuse"):
            print("  sudo zypper addlock x11-video-nvidiaG05")

    print("\n# Step 3: Install driver packages:")
    if distro.id in ("ubuntu", "debian"):
        cmd = f"  sudo apt install -y {' '.join(packages)}"
        print(cmd)
    elif distro.id in ("fedora", "rhel", "centos"):
        cmd = f"  sudo dnf install -y {' '.join(packages)}"
        print(cmd)
    elif distro.id in ("arch", "manjaro"):
        cmd = f"  sudo pacman -S --noconfirm {' '.join(packages)}"
        print(cmd)
    elif distro.id in ("opensuse"):
        cmd = f"  sudo zypper install -y {' '.join(packages)}"
        print(cmd)

    print("\n# Step 4: Rebuild initramfs:")
    if distro.id in ("fedora", "rhel", "centos", "opensuse", "sles"):
        print("  sudo dracut -f --regenerate-all")
    elif distro.id in ("arch", "manjaro"):
        print("  sudo mkinitcpio -P")
    else:
        print("  sudo update-initramfs -u")

    print("\n# Step 5: Reboot:")
    print("  sudo reboot")

    print("\n" + "=" * 50)
    print(" Dry-run complete. Run without --dry-run to install.")
    print("=" * 50 + "\n")

    return 0


def _get_wrong_branch(max_branch: str) -> str:
    """Get the incompatible driver branch to block.

    Args:
        max_branch: Maximum allowed branch (e.g., "580").

    Returns:
        Wrong branch to block (e.g., "590" for Maxwell which only supports 580).
    """
    branch_blocklist = {
        "470": None,  # Kepler - nothing newer works
        "580": "590",  # Maxwell/Pascal/Volta - block 590+
        "590": None,  # Turing+ - no restrictions
    }
    return branch_blocklist.get(max_branch) or ""


def _get_cuda_installer(distro_id: str):
    """Get CUDA installer for distribution."""
    from nvidia_inst.installer.cuda import get_cuda_installer

    return get_cuda_installer(distro_id)


def update_matrix_on_startup() -> None:
    """Update matrix on startup (non-blocking)."""
    try:
        from nvidia_inst.gpu.matrix.manager import MatrixManager

        manager = MatrixManager()
        updated, message = manager.check_for_updates()
        if updated:
            logger.info(f"Matrix updated: {message}")
        else:
            logger.debug(f"Matrix status: {message}")
    except Exception as e:
        logger.debug(f"Matrix update check failed: {e}")


def revert_to_nouveau_cli() -> int:
    """Switch from proprietary Nvidia driver to Nouveau open-source driver."""
    try:
        distro = detect_distro()
    except DistroDetectionError as e:
        logger.error(f"Failed to detect distribution: {e}")
        print(f"Failed to detect distribution: {e}")
        return 1

    installed = check_nvidia_packages_installed(distro.id)

    if not installed:
        print("\nNo proprietary Nvidia packages found.")
        print("Nouveau is already the active driver or no GPU detected.")
        return 0

    print("\n" + "=" * 50)
    print(" Revert to Nouveau")
    print("=" * 50)

    print(f"\nDistribution: {distro}")
    print(f"\nNvidia packages installed ({len(installed)}):")
    for pkg in installed[:10]:
        print(f"  - {pkg}")
    if len(installed) > 10:
        print(f"  ... and {len(installed) - 10} more")

    print("\nThis will:")
    print("  1. Remove proprietary Nvidia driver packages")
    print("  2. Remove Nouveau blacklist")
    print("  3. Rebuild initramfs")
    print("  4. Enable Nouveau (open-source) driver")

    print("\n" + "-" * 50)
    response = input("\nAre you sure you want to switch to Nouveau? [y/N]: ")
    if response.lower() not in ("y", "yes"):
        print("Cancelled.")
        return 1

    print("\nReverting to Nouveau...")

    result = revert_to_nouveau(distro.id)

    print(f"\n{result.message}")

    if result.packages_removed:
        print("\nRemoved packages:")
        for pkg in result.packages_removed[:10]:
            print(f"  - {pkg}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")

    return 0 if result.success else 1


def set_power_profile_cli(profile: str) -> int:
    """Set hybrid graphics power profile.

    Args:
        profile: Power profile to set ('intel', 'hybrid', 'nvidia').

    Returns:
        0 on success, 1 on failure.
    """
    from nvidia_inst.utils.permissions import require_root

    if not require_root(interactive=True):
        print("\n[ERROR] Root privileges required to modify hybrid graphics settings.")
        return 1
    try:
        distro = detect_distro()
    except DistroDetectionError:
        print("[ERROR] Could not detect distribution")
        return 1

    hybrid_info = detect_hybrid(distro.id)

    if not hybrid_info:
        print("[ERROR] No hybrid graphics detected on this system")
        return 1

    native_tool, _, needs_install = get_native_tool(distro.id)

    print(f"\nSetting power profile to: {profile}")

    if profile == "hybrid":
        if native_tool == "switcherooctl":
            print("\n[ERROR] 'hybrid' mode is not supported with switcherooctl.")
            print("switcherooctl supports: intel, nvidia")
            print("For per-app GPU selection, use your desktop environment's")
            print("right-click menu: 'Launch using Dedicated GPU'")
            return 1

        if needs_install and hybrid_info.needs_install:
            packages = get_hybrid_packages(distro.id)
            if packages:
                pm = get_package_manager()
                print(f"\nInstalling hybrid support packages: {', '.join(packages)}")
                try:
                    pm.install(packages)
                except Exception as e:
                    print(f"[WARNING] Failed to install hybrid packages: {e}")

        if not set_power_profile(profile, native_tool, distro.id):
            print(f"[ERROR] Failed to set power profile to {profile}")
            return 1

        print("\n[OK] Power-saving hybrid mode configured.")
        print("The NVIDIA GPU will be used only when needed.")

    else:
        if not native_tool:
            print("[ERROR] Native tool not available for this profile")
            if profile in ("intel", "nvidia"):
                print(
                    "Try running without specifying a profile to see available options"
                )
            return 1

        if not set_power_profile(profile, native_tool, distro.id):
            print(f"[ERROR] Failed to set power profile to {profile}")
            return 1

        print(f"\n[OK] Power profile set to: {profile}")

    print("\nNote: You may need to log out and log back in for changes to take effect.")

    return 0


def main() -> int:
    """Main entry point."""
    from nvidia_inst.utils.permissions import require_root

    args = parse_args()

    setup_logging(debug=args.debug, dry_run=args.dry_run)

    if args.version:
        from nvidia_inst import __version__

        print(f"nvidia-inst {__version__}")
        return 0

    logger.info("Starting nvidia-inst")

    if args.power_profile:
        return set_power_profile_cli(args.power_profile)

    if args.revert_to_nouveau:
        if not args.dry_run and not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to modify drivers.")
            return 1
        return revert_to_nouveau_cli()

    if args.gui or args.gui_type:
        return launch_gui(args)

    update_matrix_on_startup()

    if args.check:
        return check_compatibility()

    if not args.dry_run and not require_root(interactive=True):
        print("\n[ERROR] Root privileges required to modify drivers.")
        return 1

    return install_driver_cli(
        driver_version=args.driver_version,
        with_cuda=not args.no_cuda,
        cuda_version=args.cuda_version,
        skip_confirmation=args.yes,
        dry_run=args.dry_run,
    )


def launch_gui(args: argparse.Namespace) -> int:
    """Launch GUI interface.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    gui_type = args.gui_type

    if not gui_type:
        gui_type = detect_gui_type()

    if gui_type == "tkinter":
        try:
            from nvidia_inst.gui.tkinter_gui import run_gui

            return run_gui(args)
        except ImportError as e:
            logger.error(f"Failed to import Tkinter: {e}")
            print("Tkinter not available. Trying Zenity...")
            gui_type = "zenity"

    if gui_type == "zenity":
        try:
            from nvidia_inst.gui.zenity_gui import run_gui

            return run_gui(args)
        except ImportError as e:
            logger.error(f"Failed to import Zenity: {e}")
            print("Neither Tkinter nor Zenity is available.")
            return 1

    return 1


def detect_gui_type() -> str | None:
    """Detect available GUI type.

    Returns:
        'tkinter', 'zenity', or None.
    """
    import shutil

    if shutil.which("zenity"):
        return "zenity"

    try:
        import importlib.util

        if importlib.util.find_spec("tkinter"):
            return "tkinter"
    except ImportError:
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
