"""Command-line interface for nvidia-inst."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nvidia_inst.distro.tools import PackageContext

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
    validate_cuda_version,
    validate_driver_version,
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
from nvidia_inst.installer.cuda import get_cuda_installer
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
        "--ignore-compatibility",
        action="store_true",
        help="Ignore CUDA/driver compatibility warnings",
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
    driver_info = ""
    if driver_range.max_version:
        driver_info = f"{driver_range.min_version} - {driver_range.max_version}"
    else:
        driver_info = f"{driver_range.min_version} or later"
    print(f"Driver: {driver_info}")

    # CUDA line with lock info
    if driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            print(f"CUDA: {driver_range.cuda_locked_major}.x (locked to major version)")
        else:
            print(f"CUDA: {driver_range.cuda_max or driver_range.cuda_min} (locked)")
    elif driver_range.cuda_min:
        if driver_range.cuda_max:
            print(f"CUDA: {driver_range.cuda_min} - {driver_range.cuda_max}")
        else:
            print(f"CUDA: {driver_range.cuda_min} or later")

    # Status line
    if driver_range.is_eol:
        status = "EOL (security updates only)"
    elif driver_range.is_limited:
        status = "Limited support"
    else:
        status = "Full support"
    print(f"Status: {status}")


def _get_cuda_range_str(
    driver_range: DriverRange, gpu_generation: str | None = None
) -> str | None:
    """Format CUDA range for display.

    Args:
        driver_range: Driver range with CUDA info.
        gpu_generation: GPU generation name (e.g., "maxwell") for lock display.

    Returns:
        Formatted CUDA string or None.
    """
    if not driver_range.cuda_min:
        return None

    # Show lock info if CUDA is locked
    if driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            gen_str = f" for {gpu_generation}" if gpu_generation else ""
            return f"CUDA {driver_range.cuda_locked_major}.x (locked{gen_str})"
        elif driver_range.cuda_max:
            gen_str = f" for {gpu_generation}" if gpu_generation else ""
            return f"CUDA {driver_range.cuda_max} (locked{gen_str})"

    # Show range for unlocked
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
    cuda_range = _get_cuda_range_str(driver_range, gpu.generation)
    nonfree_available = check_nonfree_available()
    nvidia_open_available = check_nvidia_open_available()
    is_eol = driver_range.is_eol

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
                    driver_type, cuda_range, nvidia_open_available, is_eol
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
                    driver_range,
                    cuda_range,
                    nvidia_open_available,
                    nonfree_available,
                    is_eol,
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
                cuda_range, nvidia_open_available, nonfree_available, is_eol
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
            options=_build_nvidia_open_options(cuda_range, nonfree_available, is_eol),
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
                cuda_range, nvidia_open_available, nonfree_available, is_eol
            ),
            message="No NVIDIA driver installed"
            + (" (non-free repos not enabled)" if not nonfree_available else ""),
            cuda_range=cuda_range,
        )


def _build_optimal_options(
    driver_type: str,
    cuda_range: str | None,
    nvidia_open_available: bool,
    is_eol: bool = False,
) -> list[DriverOption]:
    """Build options for optimal driver state."""
    eol_suffix = " [EOL]" if is_eol else ""

    options = [
        DriverOption(1, "NVIDIA proprietary" + eol_suffix, "upgrade"),
        DriverOption(2, "Keep current driver", "keep"),
    ]

    if nvidia_open_available and driver_type != "nvidia_open":
        options.append(
            DriverOption(
                3,
                "NVIDIA Open" + eol_suffix,
                "switch_nvidia_open",
            )
        )

    options.append(
        DriverOption(
            len(options) + 1,
            "Nouveau (open-source, no CUDA)",
            "switch_nouveau",
        )
    )

    return options


def _build_wrong_branch_options(
    driver_range: DriverRange,
    cuda_range: str | None,
    nvidia_open_available: bool,
    nonfree_available: bool,
    is_eol: bool = False,
) -> list[DriverOption]:
    """Build options for wrong branch driver state."""
    eol_suffix = " [EOL]" if is_eol else ""
    options = []

    if nonfree_available:
        options.append(DriverOption(1, "NVIDIA proprietary" + eol_suffix, "install"))
    else:
        options.append(
            DriverOption(
                1,
                "Enable non-free + install NVIDIA proprietary" + eol_suffix,
                "install",
            )
        )

    options.append(DriverOption(2, "Keep current driver", "keep"))

    if nvidia_open_available:
        options.append(
            DriverOption(3, "NVIDIA Open" + eol_suffix, "switch_nvidia_open")
        )

    options.append(
        DriverOption(
            len(options) + 1,
            "Nouveau (open-source, no CUDA)",
            "switch_nouveau",
        )
    )

    return options


def _build_nouveau_options(
    cuda_range: str | None,
    nvidia_open_available: bool,
    nonfree_available: bool,
    is_eol: bool = False,
) -> list[DriverOption]:
    """Build options for Nouveau active state."""
    eol_suffix = " [EOL]" if is_eol else ""
    options = []

    if nonfree_available:
        options.append(
            DriverOption(
                1,
                "NVIDIA proprietary" + eol_suffix,
                "install",
            )
        )
    else:
        options.append(
            DriverOption(
                1,
                "Enable non-free + install NVIDIA proprietary" + eol_suffix,
                "install",
            )
        )

    if nvidia_open_available:
        options.append(
            DriverOption(
                2,
                "NVIDIA Open" + eol_suffix,
                "install_nvidia_open",
            )
        )

    options.append(DriverOption(len(options) + 1, "Keep Nouveau (no CUDA)", "keep"))

    return options


def _build_nvidia_open_options(
    cuda_range: str | None,
    nonfree_available: bool,
    is_eol: bool = False,
) -> list[DriverOption]:
    """Build options for NVIDIA Open active state."""
    eol_suffix = " [EOL]" if is_eol else ""
    options = [
        DriverOption(1, "Upgrade to latest" + eol_suffix, "upgrade"),
        DriverOption(2, "Keep NVIDIA Open", "keep"),
    ]

    if nonfree_available:
        options.append(
            DriverOption(
                3,
                "NVIDIA proprietary" + eol_suffix,
                "install",
            )
        )

    options.append(
        DriverOption(
            len(options) + 1,
            "Nouveau (open-source, no CUDA)",
            "switch_nouveau",
        )
    )

    return options


def _build_nothing_options(
    cuda_range: str | None,
    nvidia_open_available: bool,
    nonfree_available: bool,
    is_eol: bool = False,
) -> list[DriverOption]:
    """Build options for no driver installed state."""
    eol_suffix = " [EOL]" if is_eol else ""
    options = []

    if nonfree_available:
        options.append(
            DriverOption(
                1,
                "NVIDIA proprietary" + eol_suffix,
                "install",
            )
        )
    else:
        options.append(
            DriverOption(
                1,
                "Enable non-free + install NVIDIA proprietary" + eol_suffix,
                "install",
            )
        )

    if nvidia_open_available:
        if nonfree_available:
            options.append(
                DriverOption(
                    2,
                    "NVIDIA Open" + eol_suffix,
                    "install_nvidia_open",
                )
            )
        else:
            options.append(
                DriverOption(
                    2,
                    "Enable non-free + install NVIDIA Open" + eol_suffix,
                    "install_nvidia_open",
                )
            )

    options.append(
        DriverOption(
            len(options) + 1,
            "Nouveau (open-source, no CUDA)",
            "install_nouveau",
        )
    )

    options.append(DriverOption(len(options) + 1, "Cancel", "cancel"))

    return options


def show_driver_options(state: DriverState, distro_id: str) -> int:
    """Show driver options menu and return selected option.

    Args:
        state: Current driver state with available options.
        distro_id: Distribution ID for CUDA detection.

    Returns:
        Selected DriverOption, or None to cancel.
    """
    print(f"\n{'=' * 50}")
    print(" Driver Status")
    print(f"{'=' * 50}")
    print(f"\n{state.message}")

    if state.current_version:
        cuda_version = None
        try:
            cuda_installer = get_cuda_installer(distro_id)
            cuda_version = cuda_installer.get_installed_cuda_version()
        except Exception:
            pass
        if cuda_version:
            print(f"  Installed: {state.current_version} (CUDA {cuda_version})")
        else:
            print(f"  Installed: {state.current_version}")

    if not state.is_compatible and state.suggested_packages:
        print(f"  Recommended: {' '.join(state.suggested_packages)}")

    print("\nOptions:")
    for opt in state.options:
        print(f"  [{opt.number}] {opt.description}")

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


def print_version_check(version_check: Any, driver_range: Any, distro_id: str) -> None:
    """Print version check results.

    Args:
        version_check: Version check result.
        driver_range: Driver compatibility range.
        distro_id: Distribution ID for CUDA detection.
    """
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
        cuda_version = None
        try:
            cuda_installer = get_cuda_installer(distro_id)
            cuda_version = cuda_installer.get_installed_cuda_version()
        except Exception:
            pass

        cuda_info = ""
        if driver_range.cuda_min:
            if driver_range.cuda_max:
                cuda_info = (
                    f"supports {driver_range.cuda_min} - {driver_range.cuda_max}"
                )
            else:
                cuda_info = f"supports {driver_range.cuda_min} or later"

        if cuda_version and cuda_info:
            print(
                f"\nInstalled Driver: {version_check.installed_driver_version} (CUDA {cuda_version} installed, {cuda_info})"
            )
        elif cuda_version:
            print(
                f"\nInstalled Driver: {version_check.installed_driver_version} (CUDA {cuda_version} installed)"
            )
        elif cuda_info:
            print(
                f"\nInstalled Driver: {version_check.installed_driver_version} ({cuda_info})"
            )
        else:
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


def _install_cuda_packages(distro: DistroInfo, cuda_version: str | None = None) -> bool:
    """Install CUDA toolkit packages, ensuring repository is present.

    Args:
        distro: Distribution information.
        cuda_version: Specific CUDA version.

    Returns:
        True if successful, False otherwise.
    """
    from nvidia_inst.distro.factory import get_package_manager
    from nvidia_inst.installer.cuda import get_cuda_installer
    from nvidia_inst.installer.prerequisites import PrerequisitesChecker

    # Ensure CUDA repository is present
    checker = PrerequisitesChecker()
    fix_commands = checker.get_cuda_repo_fix_commands(distro.id, distro.version_id)
    if fix_commands:
        logger.info("Missing CUDA repository, attempting to add...")
        success, message = checker.fix_repositories(fix_commands)
        if not success:
            logger.error(f"Failed to add CUDA repository: {message}")
            print(f"[ERROR] Failed to add CUDA repository: {message}")
            return False
        print("[INFO] CUDA repository added successfully")

    # Get CUDA packages
    cuda_installer = get_cuda_installer(distro.id)
    cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
    if not cuda_pkgs:
        logger.info("No CUDA packages to install")
        return True

    print(f"\nInstalling CUDA packages: {' '.join(cuda_pkgs)}")
    pkg_manager = get_package_manager()
    try:
        pkg_manager.install(cuda_pkgs)
        print("[INFO] CUDA packages installed successfully")
        return True
    except Exception as e:
        logger.error(f"CUDA installation failed: {e}")
        print(f"[ERROR] CUDA installation failed: {e}")
        return False


def _verify_cuda_installation() -> tuple[bool, str]:
    """Verify CUDA installation by checking nvcc --version.

    Returns:
        Tuple of (success, version_string or error message)
    """
    import subprocess

    from nvidia_inst.utils.system import find_nvcc

    nvcc_path = find_nvcc()
    if not nvcc_path:
        return False, "nvcc not found in PATH or known locations"

    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.splitlines():
                if "release" in line:
                    import re

                    match = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        return True, f"nvcc {match.group(1)}"
            return True, "nvcc installed (version unknown)"
        else:
            return False, f"nvcc failed: {result.stderr[:100]}"
    except subprocess.TimeoutExpired:
        return False, "nvcc timeout"
    except Exception as e:
        return False, f"nvcc error: {e}"


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
    with_cuda: bool = True,
    cuda_version: str | None = None,
) -> None:
    """Show dry-run output for driver change."""
    from nvidia_inst.installer.cuda import get_cuda_installer
    from nvidia_inst.installer.prerequisites import PrerequisitesChecker

    print("\n" + "=" * 50)
    print(" DRY-RUN MODE")
    print("=" * 50)

    print(f"\nCurrent state: {state.message}")
    if state.current_version:
        print(f"  Installed: {state.current_version}")

    print(f"\nTarget driver packages: {' '.join(packages)}")

    # CUDA packages
    cuda_pkgs = []
    if with_cuda:
        cuda_installer = get_cuda_installer(distro.id)
        cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
        if cuda_pkgs:
            print(f"Target CUDA packages: {' '.join(cuda_pkgs)}")
        else:
            print("No CUDA packages to install")

    # Check CUDA repository
    checker = PrerequisitesChecker()
    fix_commands = checker.get_cuda_repo_fix_commands(distro.id, distro.version_id)
    if fix_commands:
        print("\n[!] CUDA repository missing. Commands to add:")
        for cmd in fix_commands:
            print(f"    {cmd}")

    print("\nSteps to execute manually:")
    step = 1
    if state.current_version or state.status == DriverStatus.NOUVEAU_ACTIVE:
        if distro.id in ("ubuntu", "debian"):
            print(f"  {step}. sudo apt remove --purge -y nvidia-driver-* nvidia-dkms-*")
        elif distro.id in ("fedora", "rhel", "centos"):
            print(f"  {step}. sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia*")
        elif distro.id in ("arch", "manjaro"):
            print(f"  {step}. sudo pacman -Rns --noconfirm nvidia nvidia-utils")
        elif distro.id in ("opensuse", "sles"):
            print(f"  {step}. sudo zypper remove -y x11-video-nvidiaG05")
        step += 1
    if distro.id in ("ubuntu", "debian"):
        print(f"  {step}. sudo apt update")
        step += 1
        print(f"  {step}. sudo apt install -y {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo apt install -y {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo update-initramfs -u")
        step += 1
    elif distro.id in ("fedora", "rhel", "centos"):
        print(f"  {step}. sudo dnf makecache")
        step += 1
        print(f"  {step}. sudo dnf install -y {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo dnf install -y {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo dracut -f --regenerate-all")
        step += 1
    elif distro.id in ("arch", "manjaro"):
        print(f"  {step}. sudo pacman -Sy")
        step += 1
        print(f"  {step}. sudo pacman -S --noconfirm {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo pacman -S --noconfirm {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo mkinitcpio -P")
        step += 1
    print(f"  {step}. sudo reboot")

    print("\nOr run this script without --dry-run:")
    print("  sudo nvidia-inst")


def _dry_run_change_tool_based(
    state: DriverState,
    packages: list[str],
    distro: DistroInfo,
    ctx: "PackageContext",
    with_cuda: bool = True,
    cuda_version: str | None = None,
) -> None:
    """Show dry-run output using tool-based detection.

    This function provides flexible dry-run output that works with any
    distro using supported package management tools.
    """
    from nvidia_inst.distro.tools import (
        get_install_command,
        get_remove_command,
        get_update_command,
    )
    from nvidia_inst.installer.cuda import get_cuda_packages_tool_based

    print("\n" + "=" * 50)
    print(" DRY-RUN MODE (Tool-Based)")
    print("=" * 50)

    print(f"\nCurrent state: {state.message}")
    if state.current_version:
        print(f"  Installed: {state.current_version}")

    print(f"\nPackage tool: {ctx.tool}")
    print(f"Package family: {ctx.distro_family}")
    print(f"Target packages: {' '.join(packages)}")

    # CUDA packages
    cuda_pkgs = []
    if with_cuda:
        cuda_pkgs = get_cuda_packages_tool_based(ctx, cuda_version)
        if cuda_pkgs:
            print(f"Target CUDA packages: {' '.join(cuda_pkgs)}")

    print("\nSteps to execute manually:")
    step = 1

    # Remove existing packages if needed
    if state.current_version or state.status == DriverStatus.NOUVEAU_ACTIVE:
        remove_cmd = get_remove_command(ctx.tool)
        print(f"  {step}. sudo {' '.join(remove_cmd)} nvidia-driver* nvidia-dkms*")
        step += 1

    # Update package lists
    update_cmd = get_update_command(ctx.tool)
    print(f"  {step}. sudo {' '.join(update_cmd)}")
    step += 1

    # Install driver packages
    install_cmd = get_install_command(ctx.tool)
    print(f"  {step}. sudo {' '.join(install_cmd)} {' '.join(packages)}")
    step += 1

    # Install CUDA packages
    if cuda_pkgs:
        print(f"  {step}. sudo {' '.join(install_cmd)} {' '.join(cuda_pkgs)}")
        step += 1

    # Rebuild initramfs
    initramfs_cmds = {
        "apt": "update-initramfs -u",
        "apt-get": "update-initramfs -u",
        "dnf": "dracut -f --regenerate-all",
        "dnf5": "dracut -f --regenerate-all",
        "yum": "dracut -f --regenerate-all",
        "pacman": "mkinitcpio -P",
        "pamac": "mkinitcpio -P",
        "paru": "mkinitcpio -P",
        "yay": "mkinitcpio -P",
        "zypper": "dracut -f --regenerate-all",
    }
    initramfs = initramfs_cmds.get(ctx.tool, "update-initramfs -u")
    print(f"  {step}. sudo {initramfs}")
    step += 1

    print(f"  {step}. sudo reboot")

    print("\nOr run this script without --dry-run:")
    print("  sudo nvidia-inst")


def _dry_run_nvidia_open_install(
    distro: DistroInfo,
    packages: list[str],
    with_cuda: bool = True,
    cuda_version: str | None = None,
) -> None:
    """Show dry-run output for NVIDIA Open installation."""
    from nvidia_inst.installer.cuda import get_cuda_installer
    from nvidia_inst.installer.prerequisites import PrerequisitesChecker

    print("\n" + "=" * 50)
    print(" DRY-RUN MODE - NVIDIA Open Installation")
    print("=" * 50)

    print(f"\nTarget driver packages: {' '.join(packages)}")

    # CUDA packages
    cuda_pkgs = []
    if with_cuda:
        cuda_installer = get_cuda_installer(distro.id)
        cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
        if cuda_pkgs:
            print(f"Target CUDA packages: {' '.join(cuda_pkgs)}")
        else:
            print("No CUDA packages to install")

    # Check CUDA repository
    checker = PrerequisitesChecker()
    fix_commands = checker.get_cuda_repo_fix_commands(distro.id, distro.version_id)
    if fix_commands:
        print("\n[!] CUDA repository missing. Commands to add:")
        for cmd in fix_commands:
            print(f"    {cmd}")

    print("\nSteps to execute manually:")
    step = 1
    driver_type = get_current_driver_type()
    if driver_type in ("proprietary", "nvidia_open"):
        if distro.id in ("ubuntu", "debian"):
            print(f"  {step}. sudo apt remove --purge -y nvidia-driver-* nvidia-dkms-*")
        elif distro.id in ("fedora", "rhel", "centos"):
            print(f"  {step}. sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia*")
        elif distro.id in ("arch", "manjaro"):
            print(f"  {step}. sudo pacman -Rns --noconfirm nvidia nvidia-utils")
        elif distro.id in ("opensuse", "sles"):
            print(f"  {step}. sudo zypper remove -y x11-video-nvidiaG05")
        step += 1
        print()

    if distro.id in ("ubuntu", "debian"):
        print(f"  {step}. sudo apt install -y {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo apt install -y {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo update-initramfs -u")
        step += 1
    elif distro.id in ("fedora", "rhel", "centos"):
        print(f"  {step}. sudo dnf install -y {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo dnf install -y {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo dracut -f --regenerate-all")
        step += 1
    elif distro.id in ("arch", "manjaro"):
        print(f"  {step}. sudo pacman -S --noconfirm {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo pacman -S --noconfirm {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo mkinitcpio -P")
        step += 1
    elif distro.id in ("opensuse", "sles"):
        print(f"  {step}. sudo zypper install -y {' '.join(packages)}")
        step += 1
        if cuda_pkgs:
            print(f"  {step}. sudo zypper install -y {' '.join(cuda_pkgs)}")
            step += 1
        print(f"  {step}. sudo dracut -f --regenerate-all")
        step += 1
    print(f"  {step}. sudo reboot")

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
    with_cuda: bool = True,
    cuda_version: str | None = None,
    skip_plan: bool = False,
) -> int:
    """Execute the selected driver change.

    Args:
        option: Selected driver option.
        state: Current driver state.
        distro: Distribution information.
        gpu: GPU information.
        driver_range: Compatible driver version range.
        dry_run: If True, show what would happen without executing.
        with_cuda: Install CUDA toolkit.
        cuda_version: Specific CUDA version.
        skip_plan: If True, skip showing the plan (already shown).

    Returns:
        0 on success, 1 on failure.
    """
    packages: list[str] = []
    if option.action == "keep":
        print("\nNo changes made.")
        return 0

    if option.action == "cancel":
        print("\nCancelled.")
        return 0

    if option.action == "revert_nouveau":
        if dry_run:
            _dry_run_change(
                state, packages, distro, with_cuda=with_cuda, cuda_version=cuda_version
            )
            return 0

        from nvidia_inst.utils.permissions import require_root

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to revert to Nouveau.")
            return 1

        print("\nReverting to Nouveau driver...")
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
            _dry_run_change(
                state,
                packages,
                distro,
                with_cuda=with_cuda,
                cuda_version=cuda_version,
            )
            return 0

        from nvidia_inst.utils.permissions import require_root

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        # Detect if we can lock before install (Pacman cannot)
        pkg_manager = get_package_manager()
        can_lock_before = pkg_manager.__class__.__name__ != "PacmanManager"

        step = 1

        # Auto-select CUDA if locked and not specified
        if with_cuda and cuda_version is None and driver_range.cuda_is_locked:
            if driver_range.cuda_locked_major:
                cuda_version = f"{driver_range.cuda_locked_major}.0"
                print(
                    f"\n[INFO] CUDA locked to {driver_range.cuda_locked_major}.x for {gpu.generation}"
                )
                print(f"[INFO] Auto-selecting CUDA {cuda_version}")
            elif driver_range.cuda_max:
                cuda_version = driver_range.cuda_max

        # Collect packages to install
        all_packages = list(packages)
        if with_cuda:
            cuda_installer = _get_cuda_installer(distro.id)
            cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
            all_packages.extend(cuda_pkgs)

        # Determine if we need locks
        needs_driver_lock = driver_range.is_limited or driver_range.is_eol
        needs_cuda_lock = driver_range.cuda_is_locked

        # ===== STEP 1: Remove old packages =====
        driver_type = get_current_driver_type()
        if driver_type == "nouveau":
            print(f"\n[Step {step}] Nouveau is active - installing proprietary driver")
        else:
            print(f"\n[Step {step}] Removing old driver packages...")
            packages_to_remove = _get_packages_to_remove(distro.id)
            removed = _remove_packages(distro.id, packages_to_remove)
            if removed:
                print(f"           Removed: {', '.join(removed)}")
        step += 1

        # ===== STEP 2-3: Apply locks BEFORE install (if supported) =====
        if can_lock_before and needs_driver_lock and driver_range.max_branch:
            print(
                f"\n[Step {step}] Locking driver to branch {driver_range.max_branch}.*"
            )
            lock_cmd = _get_driver_lock_command(distro.id, driver_range.max_branch)
            logger.info(f"Executing: {lock_cmd}")
            import subprocess

            lock_result = subprocess.run(
                lock_cmd, shell=True, capture_output=True, text=True
            )
            if lock_result.returncode == 0:
                print("           Driver lock applied")
            else:
                logger.warning(f"Driver lock failed: {lock_result.stderr}")
                print("           [WARNING] Driver lock failed")
            step += 1

        if can_lock_before and needs_cuda_lock and driver_range.cuda_locked_major:
            print(
                f"\n[Step {step}] Locking CUDA to major version {driver_range.cuda_locked_major}-*"
            )
            cuda_lock_cmd = _get_cuda_lock_command(
                distro.id, driver_range.cuda_locked_major
            )
            logger.info(f"Executing: {cuda_lock_cmd}")
            import subprocess

            lock_result = subprocess.run(
                cuda_lock_cmd, shell=True, capture_output=True, text=True
            )
            if lock_result.returncode == 0:
                print("           CUDA lock applied")
            else:
                logger.warning(f"CUDA lock failed: {lock_result.stderr}")
                print("           [WARNING] CUDA lock failed")
            step += 1

        # ===== STEP 2/4: Install ALL packages (driver + CUDA together) =====
        print(f"\n[Step {step}] Installing packages...")
        print(
            f"           {' '.join(all_packages[:5])}{'...' if len(all_packages) > 5 else ''}"
        )
        try:
            pkg_manager.install(all_packages)
            print("           Packages installed successfully")
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            print(f"           Installation failed: {e}")
            return 1
        step += 1

        # ===== STEP 3/5: Apply locks AFTER install (Pacman only) =====
        if not can_lock_before and needs_driver_lock and driver_range.max_branch:
            print(
                f"\n[Step {step}] Locking driver to branch {driver_range.max_branch}.*"
            )
            lock_cmd = _get_driver_lock_command(distro.id, driver_range.max_branch)
            logger.info(f"Executing: {lock_cmd}")
            import subprocess

            lock_result = subprocess.run(
                lock_cmd, shell=True, capture_output=True, text=True
            )
            if lock_result.returncode == 0:
                print("           Driver lock applied")
            else:
                logger.warning(f"Driver lock failed: {lock_result.stderr}")
                print("           [WARNING] Driver lock failed")
            step += 1

        if not can_lock_before and needs_cuda_lock and driver_range.cuda_locked_major:
            print(
                f"\n[Step {step}] Locking CUDA to major version {driver_range.cuda_locked_major}-*"
            )
            cuda_lock_cmd = _get_cuda_lock_command(
                distro.id, driver_range.cuda_locked_major
            )
            logger.info(f"Executing: {cuda_lock_cmd}")
            import subprocess

            lock_result = subprocess.run(
                cuda_lock_cmd, shell=True, capture_output=True, text=True
            )
            if lock_result.returncode == 0:
                print("           CUDA lock applied")
            else:
                logger.warning(f"CUDA lock failed: {lock_result.stderr}")
                print("           [WARNING] CUDA lock failed")
            step += 1

        # ===== Common post-install steps =====
        print(f"\n[Step {step}] Rebuilding initramfs...")
        if not _rebuild_initramfs(distro.id):
            print("           [WARNING] Initramfs rebuild had issues")
        else:
            print("           Initramfs rebuilt successfully")
        step += 1

        if check_secure_boot():
            print(f"\n[Step {step}] Secure Boot detected - re-signing modules...")
            key_paths = get_mok_key_paths(distro.id)
            signed, failed = sign_nvidia_modules(
                key_paths.private_key,
                key_paths.public_cert,
            )
            if signed > 0:
                print(f"           Signed: {signed} modules")
            else:
                print("           [WARNING] Module signing failed")
            step += 1

        # Verify CUDA actually works after installation
        if with_cuda:
            print(f"\n[Step {step}] Verifying CUDA installation...")
            nvcc_ok, nvcc_version = _verify_cuda_installation()
            if nvcc_ok:
                print(f"           CUDA verified: {nvcc_version}")
            else:
                print(
                    "           [WARNING] CUDA verification failed - nvcc not working"
                )
            step += 1

        print("\n" + "=" * 50)
        print("  Installation completed successfully!")
        print("=" * 50)
        _prompt_reboot()
        return 0

    if option.action in ("install_nvidia_open", "switch_nvidia_open"):
        packages = get_nvidia_open_packages(distro.id, driver_range)

        if dry_run:
            _dry_run_nvidia_open_install(
                distro, packages, with_cuda=with_cuda, cuda_version=cuda_version
            )
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

        # Install CUDA toolkit if requested
        if with_cuda:
            cuda_success = _install_cuda_packages(distro, cuda_version)
            if not cuda_success:
                print(
                    "[WARNING] CUDA toolkit installation failed, but driver is installed."
                )

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
    ignore_compatibility: bool = False,
) -> int:
    """Install driver from CLI.

    Args:
        driver_version: Specific driver version.
        with_cuda: Install CUDA.
        cuda_version: CUDA version.
        skip_confirmation: Skip confirmation prompt.
        dry_run: Dry run mode.
        ignore_compatibility: Ignore CUDA/driver compatibility warnings.

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

    # Validate specified driver and CUDA versions
    if not ignore_compatibility:
        if driver_version:
            compatible, message = validate_driver_version(driver_version, gpu)
            if not compatible:
                print(f"\nWARNING: {message}")
                print("         Use --ignore-compatibility to suppress this warning.")
        if with_cuda and cuda_version:
            compatible, message = validate_cuda_version(cuda_version, gpu)
            if not compatible:
                print(f"\nWARNING: {message}")
                print("         Use --ignore-compatibility to suppress this warning.")
    else:
        if driver_version:
            _, message = validate_driver_version(driver_version, gpu)
            logger.debug(f"Driver compatibility: {message}")
        if with_cuda and cuda_version:
            _, message = validate_cuda_version(cuda_version, gpu)
            logger.debug(f"CUDA compatibility: {message}")

    if driver_range.is_eol and driver_version is None:
        driver_version = driver_range.max_version
        logger.info(f"Using EOL driver version: {driver_version}")

    if dry_run:
        return _run_dry_run(
            distro, gpu, driver_range, driver_version, with_cuda, cuda_version
        )

    # Step 1: Show system analysis first (analyze the situation - no root needed)
    data = _print_system_analysis(distro, gpu, driver_range, with_cuda, cuda_version)

    # Step 2: Show options after analysis
    state = detect_driver_state(gpu, driver_range, distro.id)
    selected = show_driver_options(state, distro.id)

    if selected == -1:
        return 0

    option = next(opt for opt in state.options if opt.number == selected)

    # Actions that don't require confirmation (no system changes)
    no_confirm_actions = ("keep", "cancel")

    if option.action in no_confirm_actions:
        # Execute directly without plan or confirmation
        return execute_driver_change(
            option,
            state,
            distro,
            gpu,
            driver_range,
            dry_run=False,
            with_cuda=with_cuda,
            cuda_version=cuda_version,
        )

    # Step 3: Show action plan for selected option (only for actions that make changes)
    _print_action_plan(
        option.action,
        distro,
        driver_range,
        driver_version,
        with_cuda,
        data.get("cuda_version", cuda_version),
        data,
    )

    # Step 4: Ask for confirmation (unless skip_confirmation)
    if not skip_confirmation:
        print("\n" + "=" * 64)
        confirm = input("  Proceed with installation? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Installation cancelled.")
            return 0
        print("=" * 64)

    # Step 5: Request root only when installation actually starts
    from nvidia_inst.utils.permissions import require_root

    if not require_root(interactive=True):
        print("\n[ERROR] Root privileges required to modify drivers.")
        return 0

    return execute_driver_change(
        option,
        state,
        distro,
        gpu,
        driver_range,
        dry_run=False,
        with_cuda=with_cuda,
        cuda_version=cuda_version,
        skip_plan=True,  # Plan already shown above
    )


def _print_system_analysis(
    distro: DistroInfo,
    gpu: GPUInfo,
    driver_range: DriverRange,
    with_cuda: bool,
    cuda_version: str | None,
) -> dict:
    """Print system analysis sections and return computed data.

    Returns dict with: installed_cuda, nouveau_loaded, sb_enabled, packages,
    cuda_pkgs, cuda_warnings, cuda_version
    """
    from nvidia_inst.distro.factory import get_package_manager
    from nvidia_inst.installer.cuda import detect_installed_cuda_version
    from nvidia_inst.installer.driver import (
        check_nouveau,
        get_compatible_driver_packages,
    )

    # Detect system state
    installed_cuda = detect_installed_cuda_version()
    nouveau_loaded = check_nouveau()
    sb_enabled = check_secure_boot()
    pkg_manager = get_package_manager()
    packages = get_compatible_driver_packages(distro.id, driver_range)

    # Auto-select CUDA if locked and not specified
    if with_cuda and cuda_version is None and driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            cuda_version = f"{driver_range.cuda_locked_major}.0"
        elif driver_range.cuda_max:
            cuda_version = driver_range.cuda_max

    # Get CUDA packages
    cuda_pkgs = []
    if with_cuda and cuda_version:
        cuda_installer = _get_cuda_installer(distro.id)
        cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)

    # Check for incompatible installed CUDA
    cuda_warnings = []
    if (
        installed_cuda
        and driver_range.cuda_is_locked
        and driver_range.cuda_locked_major
    ):
        installed_major = installed_cuda.split(".")[0]
        if installed_major != driver_range.cuda_locked_major:
            cuda_warnings.append(
                f"Existing CUDA {installed_cuda} is INCOMPATIBLE (locked to {driver_range.cuda_locked_major}.x)"
            )

    # Print output
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  NVIDIA Driver Installation - System Analysis                 ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # DETECTED SYSTEM
    print("\n┌─── DETECTED SYSTEM ───────────────────────────────────────────┐")
    _print_row("Distribution", str(distro), 20)
    _print_row("Kernel", _get_kernel_version(), 20)
    _print_row("GPU", gpu.model, 20)
    if gpu.compute_capability:
        _print_row("Compute Cap.", str(gpu.compute_capability), 20)
    tool = _detect_tool_name()
    _print_row(
        "Package Manager", f"{pkg_manager.__class__.__name__} ({tool})", 20, last=True
    )

    # INSTALLED SOFTWARE
    print("\n┌─── INSTALLED SOFTWARE ────────────────────────────────────────┐")
    _print_row("Driver", _get_installed_driver_version() or "None", 20)
    _print_row("CUDA", installed_cuda or "None", 20, last=True)

    # COMPATIBILITY
    print("\n┌─── COMPATIBILITY ─────────────────────────────────────────────┐")
    if driver_range.max_branch:
        if driver_range.is_eol:
            driver_info = f"{driver_range.max_version} (EOL)"
        else:
            driver_info = f"{driver_range.min_version} - {driver_range.max_version} (branch {driver_range.max_branch})"
    else:
        driver_info = (
            f"{driver_range.min_version} - {driver_range.max_version or 'latest'}"
        )
    _print_row("Driver Range", driver_info, 20)

    cuda_info = f"{driver_range.cuda_min}"
    if driver_range.cuda_max:
        cuda_info += f" - {driver_range.cuda_max}"
    if driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            cuda_info += f" (locked to {driver_range.cuda_locked_major}.x)"
        else:
            cuda_info += " (locked)"
    _print_row("CUDA Range", cuda_info, 20)

    if driver_range.is_eol:
        status = "End-of-life (security updates only)"
    elif driver_range.is_limited:
        status = "Limited support"
    else:
        status = "Full support"
    _print_row("GPU Status", status, 20, last=True)

    # CHECKS
    print("\n┌─── PRE-INSTALL CHECKS ───────────────────────────────────────┐")
    _print_row(
        "Nouveau", "[ ] Disabled - REQUIRED" if nouveau_loaded else "[x] Not loaded", 20
    )
    _print_row(
        "Secure Boot",
        "[ ] Enabled - needs attention" if sb_enabled else "[x] Disabled",
        20,
    )
    _print_row(
        "CUDA Repo",
        _check_cuda_repo_status(distro.id, distro.version_id),
        20,
        last=True,
    )

    # WARNINGS
    if cuda_warnings or nouveau_loaded or sb_enabled:
        print("\n┌─── WARNINGS ──────────────────────────────────────────────────┐")
        if nouveau_loaded:
            print("│  ⚠  Nouveau kernel module must be disabled before install    │")
        if sb_enabled:
            print("│  ⚠  Secure Boot is enabled - may require MOK key enrollment  │")
        for warning in cuda_warnings:
            print(f"│  ⚠  {warning:<58}│")
        print("└────────────────────────────────────────────────────────────────┘")

    return {
        "installed_cuda": installed_cuda,
        "nouveau_loaded": nouveau_loaded,
        "sb_enabled": sb_enabled,
        "pkg_manager": pkg_manager,
        "packages": packages,
        "cuda_pkgs": cuda_pkgs,
        "cuda_warnings": cuda_warnings,
        "cuda_version": cuda_version,
    }


def _print_action_plan(
    action: str,
    distro: DistroInfo,
    driver_range: DriverRange,
    driver_version: str | None,
    with_cuda: bool,
    cuda_version: str | None,
    data: dict,
) -> None:
    """Print the action plan section."""
    nouveau_loaded = data["nouveau_loaded"]
    installed_cuda = data["installed_cuda"]
    cuda_warnings = data["cuda_warnings"]
    packages = data["packages"]
    cuda_pkgs = data["cuda_pkgs"]

    # PLANNED CHANGES
    print("\n┌─── PLANNED CHANGES ───────────────────────────────────────────┐")

    if action == "revert_nouveau":
        _print_row("Action", "Revert to Nouveau", 20)
        _print_row("Remove", "NVIDIA driver + CUDA", 20, last=True)
    elif action in ("install_nvidia_open", "switch_nvidia_open"):
        _print_row("Action", "Install NVIDIA Open driver", 20)
        _print_row("Driver", "NVIDIA Open (open-source)", 20)
        if with_cuda:
            _print_row("CUDA", f"Install {cuda_version or 'latest'}", 20)
        else:
            _print_row("CUDA", "Skip", 20)
        _print_row("Remove", "Old proprietary driver", 20, last=True)
    elif action in ("install_nouveau", "switch_nouveau"):
        _print_row("Action", "Install Nouveau driver", 20)
        _print_row("Remove", "NVIDIA proprietary driver", 20, last=True)
    else:  # install/upgrade
        if nouveau_loaded:
            _print_row("Action", "Disable Nouveau + Install", 20)
        elif driver_range.is_eol or _has_installed_driver():
            _print_row("Action", "Update/Replace driver", 20)
        else:
            _print_row("Action", "Fresh install", 20)

        if driver_version:
            driver_action = f"Install {driver_version}"
        elif driver_range.is_eol:
            driver_action = f"Install {driver_range.max_version} (EOL)"
        elif driver_range.max_branch:
            driver_action = f"Install {driver_range.max_branch}.xx (branch)"
        else:
            driver_action = "Install latest"
        _print_row("Driver", driver_action, 20)

        if with_cuda:
            if cuda_warnings:
                cuda_action = f"Install {cuda_version} (replaces {installed_cuda})"
            elif driver_range.cuda_is_locked:
                cuda_action = f"Install {cuda_version} (locked)"
            else:
                cuda_action = f"Install {cuda_version or 'latest'}"
        else:
            cuda_action = "Skip"
        _print_row("CUDA", cuda_action, 20)

        locks = []
        if driver_range.max_branch and driver_range.is_limited:
            locks.append(f"driver to {driver_range.max_branch}.*")
        if driver_range.cuda_is_locked and driver_range.cuda_locked_major:
            locks.append(f"CUDA to {driver_range.cuda_locked_major}.*")
        if locks:
            _print_row("Locks", ", ".join(locks), 20)

        remove = []
        if _has_installed_driver():
            remove.append("old driver")
        if cuda_warnings:
            remove.append(f"CUDA {installed_cuda}")
        if remove:
            _print_row("Remove", ", ".join(remove), 20, last=True)
        else:
            print("└────────────────────────────────────────────────────────────────┘")

    # PACKAGES
    if packages or cuda_pkgs:
        print("\n┌─── PACKAGES ──────────────────────────────────────────────────┐")
        if packages:
            _print_row(
                "Driver",
                " ".join(packages[:3]) + (" ..." if len(packages) > 3 else ""),
                20,
            )
        if cuda_pkgs:
            _print_row("CUDA", " ".join(cuda_pkgs), 20, last=True)
        else:
            _print_row("CUDA", "(none)", 20, last=True)

    # COMMANDS preview
    print("\n┌─── COMMANDS ──────────────────────────────────────────────────┐")
    if action == "revert_nouveau":
        print("│  1. sudo dnf remove -y akmod-nvidia xorg-x11-drv-nvidia*      │")
        print("│  2. sudo dnf install -y xorg-x11-drv-nouveau                  │")
        print("│  3. sudo dracut -f --regenerate-all                            │")
        print("│  4. sudo reboot                                               │")
    elif action in ("install_nvidia_open", "switch_nvidia_open"):
        print(f"│  1. {_get_update_command(distro.id):<55}│")
        print(
            f"│  2. {_get_install_command(distro.id, cuda_pkgs[:3] if cuda_pkgs else ['nvidia-open']):<55}│"
        )
        print(f"│  3. {_get_initramfs_command(distro.id):<55}│")
        print("│  4. sudo reboot                                               │")
    elif action in ("install_nouveau", "switch_nouveau"):
        print(f"│  1. {_get_update_command(distro.id):<55}│")
        print(f"│  2. {_get_install_command(distro.id, ['xorg-x11-drv-nouveau']):<55}│")
        print("│  3. sudo reboot                                               │")
    else:
        step = 1
        print(f"│  {step}. {_get_update_command(distro.id):<55}│")
        step += 1
        if nouveau_loaded:
            print(f"│  {step}. {_get_nouveau_remove_command(distro.id):<55}│")
            step += 1
        # Lock driver BEFORE install (if limited/EOL)
        if driver_range.is_limited and driver_range.max_branch:
            print(
                f"│  {step}. {_get_driver_lock_command(distro.id, driver_range.max_branch):<55}│"
            )
            step += 1
        # Lock CUDA BEFORE install (if locked)
        if driver_range.cuda_is_locked and driver_range.cuda_locked_major:
            print(
                f"│  {step}. {_get_cuda_lock_command(distro.id, driver_range.cuda_locked_major):<55}│"
            )
            step += 1
        # Install driver + CUDA together
        all_pkgs = packages + (cuda_pkgs if cuda_pkgs else [])
        print(f"│  {step}. {_get_install_command(distro.id, all_pkgs):<55}│")
        step += 1
        print(f"│  {step}. {_get_initramfs_command(distro.id):<55}│")
        step += 1
        print(
            f"│  {step}. sudo reboot                                                   │"
        )
    print("└────────────────────────────────────────────────────────────────┘")


def _run_dry_run(
    distro: DistroInfo,
    gpu: GPUInfo,
    driver_range: DriverRange,
    driver_version: str | None,
    with_cuda: bool,
    cuda_version: str | None,
) -> int:
    """Run in dry-run mode with compact table format."""
    from nvidia_inst.distro.factory import get_package_manager
    from nvidia_inst.installer.cuda import detect_installed_cuda_version
    from nvidia_inst.installer.driver import (
        check_nouveau,
        get_compatible_driver_packages,
    )

    # Detect system state
    installed_cuda = detect_installed_cuda_version()
    nouveau_loaded = check_nouveau()
    sb_enabled = check_secure_boot()
    pkg_manager = get_package_manager()
    packages = get_compatible_driver_packages(distro.id, driver_range)

    # Auto-select CUDA if locked and not specified
    if with_cuda and cuda_version is None and driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            cuda_version = f"{driver_range.cuda_locked_major}.0"
        elif driver_range.cuda_max:
            cuda_version = driver_range.cuda_max

    # Get CUDA packages
    cuda_pkgs = []
    if with_cuda:
        cuda_installer = _get_cuda_installer(distro.id)
        cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)

    # Check for incompatible installed CUDA
    cuda_warnings = []
    if (
        installed_cuda
        and driver_range.cuda_is_locked
        and driver_range.cuda_locked_major
    ):
        installed_major = installed_cuda.split(".")[0]
        if installed_major != driver_range.cuda_locked_major:
            cuda_warnings.append(
                f"Existing CUDA {installed_cuda} is INCOMPATIBLE (locked to {driver_range.cuda_locked_major}.x)"
            )

    # Print output
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  NVIDIA Driver Installation - Dry Run Mode                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # DETECTED SYSTEM
    print("\n┌─── DETECTED SYSTEM ───────────────────────────────────────────┐")

    _print_row("Distribution", str(distro), 20)
    _print_row("Kernel", _get_kernel_version(), 20)
    _print_row("GPU", gpu.model, 20)
    if gpu.compute_capability:
        _print_row("Compute Cap.", str(gpu.compute_capability), 20)

    # Package manager info
    tool = _detect_tool_name()
    _print_row(
        "Package Manager", f"{pkg_manager.__class__.__name__} ({tool})", 20, last=True
    )

    # INSTALLED SOFTWARE
    print("\n┌─── INSTALLED SOFTWARE ────────────────────────────────────────┐")
    _print_row("Driver", _get_installed_driver_version() or "None", 20)
    _print_row("CUDA", installed_cuda or "None", 20, last=True)

    # COMPATIBILITY
    print("\n┌─── COMPATIBILITY ─────────────────────────────────────────────┐")

    # Driver range
    if driver_range.max_branch:
        if driver_range.is_eol:
            driver_info = f"{driver_range.max_version} (EOL)"
        else:
            driver_info = f"{driver_range.min_version} - {driver_range.max_version} (branch {driver_range.max_branch})"
    else:
        driver_info = (
            f"{driver_range.min_version} - {driver_range.max_version or 'latest'}"
        )
    _print_row("Driver Range", driver_info, 20)

    # CUDA range
    cuda_info = f"{driver_range.cuda_min}"
    if driver_range.cuda_max:
        cuda_info += f" - {driver_range.cuda_max}"
    if driver_range.cuda_is_locked:
        if driver_range.cuda_locked_major:
            cuda_info += f" (locked to {driver_range.cuda_locked_major}.x)"
        else:
            cuda_info += " (locked)"
    _print_row("CUDA Range", cuda_info, 20)

    # GPU status
    if driver_range.is_eol:
        status = "End-of-life (security updates only)"
    elif driver_range.is_limited:
        status = "Limited support"
    else:
        status = "Full support"
    _print_row("GPU Status", status, 20, last=True)

    # CHECKS
    print("\n┌─── PRE-INSTALL CHECKS ───────────────────────────────────────┐")
    _print_row(
        "Nouveau", "[ ] Disabled - REQUIRED" if nouveau_loaded else "[x] Not loaded", 20
    )
    _print_row(
        "Secure Boot",
        "[ ] Enabled - needs attention" if sb_enabled else "[x] Disabled",
        20,
    )
    _print_row(
        "CUDA Repo",
        _check_cuda_repo_status(distro.id, distro.version_id),
        20,
        last=True,
    )

    # WARNINGS
    if cuda_warnings or nouveau_loaded or sb_enabled:
        print("\n┌─── WARNINGS ──────────────────────────────────────────────────┐")
        if nouveau_loaded:
            print("│  ⚠  Nouveau kernel module must be disabled before install    │")
        if sb_enabled:
            print("│  ⚠  Secure Boot is enabled - may require MOK key enrollment  │")
        for warning in cuda_warnings:
            print(f"│  ⚠  {warning:<58}│")
        print("└────────────────────────────────────────────────────────────────┘")

    # PLANNED CHANGES
    print("\n┌─── PLANNED CHANGES ───────────────────────────────────────────┐")

    # Action summary
    if nouveau_loaded:
        _print_row("Action", "Disable Nouveau + Install", 20)
    elif driver_range.is_eol or _has_installed_driver():
        _print_row("Action", "Update/Replace driver", 20)
    else:
        _print_row("Action", "Fresh install", 20)

    # Driver
    if driver_version:
        driver_action = f"Install {driver_version}"
    elif driver_range.is_eol:
        driver_action = f"Install {driver_range.max_version} (EOL)"
    elif driver_range.max_branch:
        driver_action = f"Install {driver_range.max_branch}.xx (branch)"
    else:
        driver_action = "Install latest"
    _print_row("Driver", driver_action, 20)

    # CUDA
    if with_cuda:
        if cuda_warnings:
            cuda_action = f"Install {cuda_version} (replaces {installed_cuda})"
        elif driver_range.cuda_is_locked:
            cuda_action = f"Install {cuda_version} (locked)"
        else:
            cuda_action = f"Install {cuda_version or 'latest'}"
    else:
        cuda_action = "Skip"
    _print_row("CUDA", cuda_action, 20)

    # Locks
    locks = []
    if driver_range.max_branch and driver_range.is_limited:
        locks.append(f"driver to {driver_range.max_branch}.*")
    if driver_range.cuda_is_locked and driver_range.cuda_locked_major:
        locks.append(f"CUDA to {driver_range.cuda_locked_major}.*")
    if locks:
        _print_row("Locks", ", ".join(locks), 20)

    # Remove
    remove = []
    if _has_installed_driver():
        remove.append("old driver")
    if cuda_warnings:
        remove.append(f"CUDA {installed_cuda}")
    if remove:
        _print_row("Remove", ", ".join(remove), 20, last=True)
    else:
        print("└────────────────────────────────────────────────────────────────┘")

    # PACKAGES
    print("\n┌─── PACKAGES ──────────────────────────────────────────────────┐")
    _print_row(
        "Driver", " ".join(packages[:3]) + (" ..." if len(packages) > 3 else ""), 20
    )
    if cuda_pkgs:
        _print_row("CUDA", " ".join(cuda_pkgs), 20, last=True)
    else:
        _print_row("CUDA", "(none)", 20, last=True)

    # COMMANDS
    print("\n┌─── COMMANDS ──────────────────────────────────────────────────┐")
    step = 1

    # Step 1: Update
    update_cmd = _get_update_command(distro.id)
    print(f"│  {step}. {update_cmd:<56} │")
    step += 1

    # Step 1b: Remove if needed
    if nouveau_loaded:
        remove_cmd = _get_nouveau_remove_command(distro.id)
        print(f"│  {step}. {remove_cmd:<56} │")
        step += 1

    # Step 2: Lock driver BEFORE install (if limited/EOL)
    if driver_range.is_limited and driver_range.max_branch:
        lock_cmd = _get_driver_lock_command(distro.id, driver_range.max_branch)
        print(f"│  {step}. {lock_cmd:<56} │")
        step += 1

    # Step 3: Lock CUDA BEFORE install (if locked)
    if driver_range.cuda_is_locked and driver_range.cuda_locked_major:
        cuda_lock_cmd = _get_cuda_lock_command(
            distro.id, driver_range.cuda_locked_major
        )
        print(f"│  {step}. {cuda_lock_cmd:<56} │")
        step += 1

    # Step 4: Install driver + CUDA together
    all_pkgs = packages + (cuda_pkgs if cuda_pkgs else [])
    install_cmd = _get_install_command(distro.id, all_pkgs)
    print(f"│  {step}. {install_cmd:<56} │")
    step += 1

    # Step 5: Initramfs
    initramfs_cmd = _get_initramfs_command(distro.id)
    print(f"│  {step}. {initramfs_cmd:<56} │")
    step += 1

    # Step 6: Reboot
    print(f"│  {step}. sudo reboot                                                   │")
    print("└────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 64)
    print("  Dry-run complete. Run without --dry-run to install.")
    print("=" * 64 + "\n")

    return 0


def _print_row(
    label: str, value: str, label_width: int = 20, last: bool = False
) -> None:
    """Print a single row in the compact table format."""
    print(f"│  {label:<{label_width}} │ {value:<40} │")
    if last:
        print("└────────────────────────────────────────────────────────────────┘")


def _get_kernel_version() -> str:
    """Get current kernel version."""
    import subprocess

    try:
        result = subprocess.run(["uname", "-r"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _detect_tool_name() -> str:
    """Detect package tool name."""
    from nvidia_inst.distro.tools import detect_package_tool

    return detect_package_tool() or "unknown"


def _get_installed_driver_version() -> str | None:
    """Get currently installed driver version."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _has_installed_driver() -> bool:
    """Check if NVIDIA driver is installed."""
    return _get_installed_driver_version() is not None


def _check_cuda_repo_status(distro_id: str, version_id: str) -> str:
    """Check CUDA repository status."""
    try:
        from nvidia_inst.installer.prerequisites import PrerequisitesChecker

        checker = PrerequisitesChecker()
        fix_commands = checker.get_cuda_repo_fix_commands(distro_id, version_id)
        if fix_commands:
            return "[ ] Not configured"
        return "[x] Configured"
    except Exception:
        return "[?] Unknown"


def _get_update_command(distro_id: str) -> str:
    """Get package update command."""
    commands = {
        "ubuntu": "sudo apt update",
        "debian": "sudo apt update",
        "fedora": "sudo dnf makecache",
        "rhel": "sudo dnf makecache",
        "centos": "sudo dnf makecache",
        "arch": "sudo pacman -Sy",
        "manjaro": "sudo pacman -Sy",
        "opensuse": "sudo zypper refresh",
    }
    return commands.get(distro_id, "sudo apt update")


def _get_nouveau_remove_command(distro_id: str) -> str:
    """Get command to remove nouveau."""
    commands = {
        "ubuntu": "sudo apt remove -y xserver-xorg-video-nouveau",
        "debian": "sudo apt remove -y xserver-xorg-video-nouveau",
        "fedora": "sudo dnf remove -y xorg-x11-drv-nouveau",
        "arch": "sudo pacman -Rns --noconfirm xf86-video-nouveau",
        "opensuse": "sudo zypper remove -y xf86-video-nouveau",
    }
    return commands.get(distro_id, "sudo apt remove -y xserver-xorg-video-nouveau")


def _detect_dnf_version() -> str:
    """Detect if system uses dnf4 or dnf5.

    Returns:
        'dnf4' or 'dnf5'
    """
    import subprocess

    try:
        result = subprocess.run(
            ["dnf", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "dnf5" in result.stdout.lower() or "dnf5" in result.stderr.lower():
            return "dnf5"
        return "dnf4"
    except Exception:
        return "dnf4"  # Default to dnf4 for safety


def _get_driver_lock_command(distro_id: str, branch: str) -> str:
    """Get command to lock driver to branch."""
    if distro_id in ("ubuntu", "debian"):
        return f"echo 'Pin: version {branch}.*' | sudo tee -a /etc/apt/preferences.d/nvidia"
    elif distro_id == "fedora":
        # Detect dnf4 vs dnf5 - only dnf4 uses --raw flag
        dnf_version = _detect_dnf_version()
        if dnf_version == "dnf5":
            return f"sudo dnf versionlock add 'akmod-nvidia-{branch}.*'"
        return f"sudo dnf versionlock add --raw 'akmod-nvidia-{branch}.*'"
    elif distro_id == "arch":
        return f"sudo pacman -D --lock nvidia-{branch}xx"
    elif distro_id == "opensuse":
        return "sudo zypper addlock x11-video-nvidiaG05"
    return f"# Lock to branch {branch}"


def _get_cuda_lock_command(distro_id: str, major: str) -> str:
    """Get command to lock CUDA to major version.

    Uses hyphen pattern for Fedora (cuda-toolkit-{major}-*) because
    packages are named cuda-toolkit-13-2, not cuda-toolkit-13.2
    """
    if distro_id in ("ubuntu", "debian"):
        return (
            f"echo 'Pin: version {major}.*' | sudo tee -a /etc/apt/preferences.d/cuda"
        )
    elif distro_id == "fedora":
        # Use hyphen pattern: cuda-toolkit-{major}-*
        # Packages are named cuda-toolkit-13-2 (hyphen) not cuda-toolkit-13.2 (dot)
        dnf_version = _detect_dnf_version()
        if dnf_version == "dnf5":
            return f"sudo dnf versionlock add 'cuda-toolkit-{major}-*'"
        return f"sudo dnf versionlock add --raw 'cuda-toolkit-{major}-*'"
    elif distro_id == "arch":
        return f"sudo pacman -D --lock cuda-{major}*"
    elif distro_id == "opensuse":
        return f"sudo zypper addlock 'cuda-toolkit-{major}-*'"
    return f"# Lock CUDA to {major}.*"


def _get_install_command(distro_id: str, packages: list[str]) -> str:
    """Get install command for packages."""
    pkg_str = " ".join(packages[:3])
    if len(packages) > 3:
        pkg_str += " ..."
    commands = {
        "ubuntu": f"sudo apt install -y {pkg_str}",
        "debian": f"sudo apt install -y {pkg_str}",
        "fedora": f"sudo dnf install -y {pkg_str}",
        "arch": f"sudo pacman -S --noconfirm {pkg_str}",
        "opensuse": f"sudo zypper install -y {pkg_str}",
    }
    return commands.get(distro_id, f"sudo apt install -y {pkg_str}")


def _get_initramfs_command(distro_id: str) -> str:
    """Get initramfs rebuild command."""
    commands = {
        "ubuntu": "sudo update-initramfs -u",
        "debian": "sudo update-initramfs -u",
        "fedora": "sudo dracut -f --regenerate-all",
        "rhel": "sudo dracut -f --regenerate-all",
        "centos": "sudo dracut -f --regenerate-all",
        "arch": "sudo mkinitcpio -P",
        "manjaro": "sudo mkinitcpio -P",
        "opensuse": "sudo dracut -f --regenerate-all",
    }
    return commands.get(distro_id, "sudo update-initramfs -u")


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

    # Request root only after user confirms
    from nvidia_inst.utils.permissions import require_root

    if not require_root(interactive=True):
        print("\n[ERROR] Root privileges required to modify drivers.")
        return 0

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
        # Root will be requested inside revert_to_nouveau_cli after confirmation
        return revert_to_nouveau_cli()

    if args.gui or args.gui_type:
        return launch_gui(args)

    update_matrix_on_startup()

    if args.check:
        return check_compatibility()

    # Note: Root request moved to install_driver_cli after user confirms plan
    return install_driver_cli(
        driver_version=args.driver_version,
        with_cuda=not args.no_cuda,
        cuda_version=args.cuda_version,
        skip_confirmation=args.yes,
        dry_run=args.dry_run,
        ignore_compatibility=args.ignore_compatibility,
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
