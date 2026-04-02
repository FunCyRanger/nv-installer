"""Command-line interface for nvidia-inst.

This module provides the main CLI entry point and delegates to
specialized modules for specific functionality.
"""

import subprocess
from pathlib import Path
from typing import Any

# Import from specialized modules
from nvidia_inst.cli.compatibility import (
    check_compatibility,
)
from nvidia_inst.cli.driver_state import (
    DriverOption,
    DriverState,
    detect_driver_state,
    show_driver_options,
)
from nvidia_inst.cli.installer import (
    get_packages_to_remove,
    prompt_reboot,
    rebuild_initramfs,
    remove_packages,
)
from nvidia_inst.cli.parser import parse_args
from nvidia_inst.cli.simulate import (
    simulate_change,
    simulate_nouveau_install,
)
from nvidia_inst.distro.detector import (
    DistroDetectionError,
    DistroInfo,
    detect_distro,
)
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.gpu.compatibility import (
    DriverRange,
    get_driver_range,
)
from nvidia_inst.gpu.detector import (
    GPUDetectionError,
    GPUInfo,
    detect_gpu,
    has_nvidia_gpu,
)
from nvidia_inst.gpu.hybrid import (
    get_native_tool,
)
from nvidia_inst.gui import launch_gui
from nvidia_inst.installer.cuda import get_cuda_installer
from nvidia_inst.installer.driver import (
    get_compatible_driver_packages,
    get_nvidia_open_packages,
)
from nvidia_inst.installer.hybrid import (
    set_power_profile,
)
from nvidia_inst.installer.prerequisites import PrerequisitesChecker
from nvidia_inst.installer.secureboot import (
    SecureBootState,
    enroll_mok_key,
    generate_mok_key,
    get_mok_key_paths,
    get_secure_boot_state,
    is_mok_enrolled,
    setup_auto_signing,
)
from nvidia_inst.installer.uninstaller import (
    check_nvidia_packages_installed,
    revert_to_nouveau,
)
from nvidia_inst.utils.logger import get_logger, setup_logging
from nvidia_inst.utils.permissions import require_root

logger = get_logger(__name__)

# Version info
__version__ = "2.0.0"


def _get_cuda_range_str(driver_range: DriverRange, generation: str) -> str:
    """Get CUDA range display string."""
    if driver_range.cuda_is_locked:
        return f"{driver_range.cuda_locked_major}.x (locked for {generation})"
    return f"{driver_range.cuda_min}-{driver_range.cuda_max}"


def _verify_cuda_installation() -> tuple[bool, str]:
    """Verify CUDA installation."""
    from nvidia_inst.installer.cuda import detect_installed_cuda_version

    cuda_version = detect_installed_cuda_version()
    if cuda_version:
        return True, f"CUDA {cuda_version} installed"
    return False, "CUDA not detected"


def _get_kernel_version() -> str:
    """Get current kernel version."""
    try:
        result = subprocess.run(
            ["uname", "-r"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_installed_driver_version() -> str | None:
    """Get installed driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _has_installed_driver() -> bool:
    """Check if any driver is installed."""
    return _get_installed_driver_version() is not None


def _check_cuda_repo_status(distro_id: str, version_id: str) -> str:
    """Check CUDA repository status."""
    try:
        checker = PrerequisitesChecker()
        fix_commands = checker.get_cuda_repo_fix_commands(distro_id, version_id)
        if fix_commands:
            return "MISSING"
        return "CONFIGURED"
    except Exception:
        return "UNKNOWN"


def _add_cuda_repo(distro: DistroInfo, driver_range: DriverRange) -> bool:
    """Add CUDA repository."""
    try:
        checker = PrerequisitesChecker()
        cuda_major = (
            driver_range.cuda_locked_major if driver_range.cuda_is_locked else None
        )
        fix_commands = checker.get_cuda_repo_fix_commands(
            distro.id, distro.version_id, cuda_major
        )
        if fix_commands:
            success, message = checker.fix_repositories(fix_commands)
            if success:
                print("[INFO] CUDA repository added successfully")
                return True
            else:
                print(f"[ERROR] Failed to add CUDA repository: {message}")
                return False
        return True
    except Exception as e:
        print(f"[ERROR] Failed to add CUDA repository: {e}")
        return False


def _cleanup_incorrect_versionlocks(
    distro_id: str, package_base: str, branch: str
) -> bool:
    """Clean up incorrect versionlock entries."""
    try:
        from nvidia_inst.distro.versionlock import (
            cleanup_incorrect_versionlocks as cleanup,
        )

        return cleanup(distro_id, package_base, branch)
    except Exception:
        return True


def _get_install_command(distro_id: str, packages: list[str]) -> str:
    """Get install command for display."""
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


def _get_initramfs_cmd(distro_id: str) -> str:
    """Get initramfs rebuild command for display."""
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
    """Get the incompatible driver branch to block."""
    branch_blocklist = {
        "470": None,  # Kepler - nothing newer works
        "580": "590",  # Maxwell/Pascal/Volta - block 590+
        "590": None,  # Turing+ - no restrictions
    }
    return branch_blocklist.get(max_branch) or ""


def _get_cuda_installer(distro_id: str):
    """Get CUDA installer for distribution."""
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
        if not skip_confirmation:
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                return (False, state)
        return (True, state)

    # Secure Boot is enabled
    print("\n[INFO] Secure Boot is enabled.")
    print("  The installer can set up automatic module signing.")

    mok_key, mok_cert = get_mok_key_paths()

    if is_mok_enrolled():
        print(f"\n[OK] MOK key already enrolled: {mok_key}")
        print("  Modules will be signed automatically during installation.")
        return (True, state)

    print(f"\n[INFO] MOK key not found: {mok_key}")
    print("\nThe installer will:")
    print("  1. Generate a Machine Owner Key (MOK)")
    print("  2. Set up DKMS to auto-sign modules")
    print("  3. Prompt to enroll the key on next reboot")

    if not skip_confirmation:
        response = input("\nProceed with MOK setup? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print(
                "\n[INFO] MOK setup skipped. Driver may fail to load with Secure Boot enabled."
            )
            return (False, state)

    print("\n--- Generating MOK key ---")
    try:
        generate_mok_key()
        print(f"[OK] MOK key generated: {mok_key}")
    except Exception as e:
        print(f"[ERROR] Failed to generate MOK key: {e}")
        return (False, state)

    print("\n--- Setting up auto-signing ---")
    try:
        setup_auto_signing()
        print("[OK] DKMS auto-signing configured")
    except Exception as e:
        print(f"[ERROR] Failed to setup auto-signing: {e}")
        return (False, state)

    print("\n--- Enrolling MOK key ---")
    try:
        enroll_mok_key()
        print("[OK] MOK key enrolled for next reboot")
        print("\n[IMPORTANT] On next reboot, the MOK Manager will appear.")
        print("  Select 'Enroll MOK' and confirm the enrollment.")
        print("  The system will then continue booting normally.")
    except Exception as e:
        print(f"[ERROR] Failed to enroll MOK key: {e}")
        return (False, state)

    return (True, state)


def execute_driver_change(
    option: DriverOption,
    state: DriverState,
    distro: DistroInfo,
    gpu: GPUInfo,
    driver_range: DriverRange,
    simulate: bool = False,
    with_cuda: bool = True,
    cuda_version: str | None = None,
    skip_plan: bool = False,
) -> int:
    """Execute the selected driver change."""
    packages: list[str] = []
    if option.action == "keep":
        print("\nNo changes made.")
        return 0

    if option.action == "cancel":
        print("\nCancelled.")
        return 0

    if option.action == "revert_nouveau":
        if simulate:
            simulate_nouveau_install(packages, distro.id)
            return 0

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to revert to Nouveau.")
            return 1

        print("\nReverting to Nouveau driver...")
        result = revert_to_nouveau(distro.id)
        if result.success:
            print(f"\n✓ {result.message}")
            prompt_reboot()
        else:
            print(f"\n✗ Revert failed: {', '.join(result.errors)}")
            return 1
        return 0

    if option.action == "switch_nvidia_open":
        packages = get_nvidia_open_packages(distro.id, driver_range)

        if simulate:
            simulate_change(
                state.message,
                state.current_version,
                packages,
                distro.id,
                with_cuda=with_cuda,
                cuda_version=cuda_version,
            )
            return 0

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        # Auto-select CUDA if locked and not specified
        if (
            with_cuda
            and cuda_version is None
            and driver_range.cuda_is_locked
            and driver_range.cuda_locked_major
        ):
            cuda_version = f"{driver_range.cuda_locked_major}.0"
            print(
                f"\n[INFO] CUDA locked to {driver_range.cuda_locked_major}.x for {gpu.generation}"
            )

        # Remove existing packages
        if state.current_version:
            print("\n--- Removing old driver packages ---")
            packages_to_remove = get_packages_to_remove(distro.id)
            removed = remove_packages(distro.id, packages_to_remove)
            if removed:
                print(f"Removed: {', '.join(removed)}")

        # Install NVIDIA Open packages
        print("\n--- Installing NVIDIA Open driver packages ---")
        try:
            install_cmd = get_package_manager()
            install_cmd.install(packages)
            print(f"[OK] Installed: {' '.join(packages)}")
        except Exception as e:
            print(f"[ERROR] Installation failed: {e}")
            return 1

        # Install CUDA if requested
        if with_cuda:
            print("\n--- Installing CUDA ---")
            cuda_installer = get_cuda_installer(distro.id)
            cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
            if cuda_pkgs:
                try:
                    install_cmd = get_package_manager()
                    install_cmd.install(cuda_pkgs)
                    print(f"[OK] Installed CUDA: {' '.join(cuda_pkgs)}")
                except Exception as e:
                    print(f"[ERROR] CUDA installation failed: {e}")

        # Rebuild initramfs
        print("\n--- Rebuilding initramfs ---")
        if rebuild_initramfs(distro.id):
            print("[OK] Initramfs rebuilt successfully")
        else:
            print("[WARNING] Initramfs rebuild may have failed")

        prompt_reboot()
        return 0

    if option.action == "install_nvidia_open":
        packages = get_nvidia_open_packages(distro.id, driver_range)

        if simulate:
            simulate_change(
                state.message,
                state.current_version,
                packages,
                distro.id,
                with_cuda=with_cuda,
                cuda_version=cuda_version,
            )
            return 0

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        # Auto-select CUDA if locked and not specified
        if (
            with_cuda
            and cuda_version is None
            and driver_range.cuda_is_locked
            and driver_range.cuda_locked_major
        ):
            cuda_version = f"{driver_range.cuda_locked_major}.0"
            print(
                f"\n[INFO] CUDA locked to {driver_range.cuda_locked_major}.x for {gpu.generation}"
            )

        # Remove existing packages (including Nouveau)
        print("\n--- Removing old driver packages ---")
        packages_to_remove = get_packages_to_remove(distro.id)
        removed = remove_packages(distro.id, packages_to_remove)
        if removed:
            print(f"Removed: {', '.join(removed)}")

        # Install NVIDIA Open packages
        print("\n--- Installing NVIDIA Open driver packages ---")
        try:
            install_cmd = get_package_manager()
            install_cmd.install(packages)
            print(f"[OK] Installed: {' '.join(packages)}")
        except Exception as e:
            print(f"[ERROR] Installation failed: {e}")
            return 1

        # Install CUDA if requested
        if with_cuda:
            print("\n--- Installing CUDA ---")
            cuda_installer = get_cuda_installer(distro.id)
            cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
            if cuda_pkgs:
                try:
                    install_cmd = get_package_manager()
                    install_cmd.install(cuda_pkgs)
                    print(f"[OK] Installed CUDA: {' '.join(cuda_pkgs)}")
                except Exception as e:
                    print(f"[ERROR] CUDA installation failed: {e}")

        # Rebuild initramfs
        print("\n--- Rebuilding initramfs ---")
        if rebuild_initramfs(distro.id):
            print("[OK] Initramfs rebuilt successfully")
        else:
            print("[WARNING] Initramfs rebuild may have failed")

        prompt_reboot()
        return 0

    if option.action in ("install", "upgrade"):
        packages = state.suggested_packages or get_compatible_driver_packages(
            distro.id, driver_range
        )

        if simulate:
            simulate_change(
                state.message,
                state.current_version,
                packages,
                distro.id,
                with_cuda=with_cuda,
                cuda_version=cuda_version,
            )
            return 0

        if not require_root(interactive=True):
            print("\n[ERROR] Root privileges required to install drivers.")
            return 1

        # Auto-select CUDA if locked and not specified
        if (
            with_cuda
            and cuda_version is None
            and driver_range.cuda_is_locked
            and driver_range.cuda_locked_major
        ):
            cuda_version = f"{driver_range.cuda_locked_major}.0"
            print(
                f"\n[INFO] CUDA locked to {driver_range.cuda_locked_major}.x for {gpu.generation}"
            )

        # Remove existing packages
        if state.current_version:
            print("\n--- Removing old driver packages ---")
            packages_to_remove = get_packages_to_remove(distro.id)
            removed = remove_packages(distro.id, packages_to_remove)
            if removed:
                print(f"Removed: {', '.join(removed)}")

        # Install new packages
        print("\n--- Installing driver packages ---")
        try:
            install_cmd = get_package_manager()
            install_cmd.install(packages)
            print(f"[OK] Installed: {' '.join(packages)}")
        except Exception as e:
            print(f"[ERROR] Installation failed: {e}")
            return 1

        # Install CUDA if requested
        if with_cuda:
            print("\n--- Installing CUDA ---")
            cuda_installer = get_cuda_installer(distro.id)
            cuda_pkgs = cuda_installer.get_cuda_packages(cuda_version)
            if cuda_pkgs:
                try:
                    install_cmd = get_package_manager()
                    install_cmd.install(cuda_pkgs)
                    print(f"[OK] Installed CUDA: {' '.join(cuda_pkgs)}")
                except Exception as e:
                    print(f"[ERROR] CUDA installation failed: {e}")

        # Rebuild initramfs
        print("\n--- Rebuilding initramfs ---")
        if rebuild_initramfs(distro.id):
            print("[OK] Initramfs rebuilt successfully")
        else:
            print("[WARNING] Initramfs rebuild may have failed")

        prompt_reboot()
        return 0

    print(f"\n[ERROR] Unknown action: {option.action}")
    return 1


def install_driver_cli() -> int:
    """Install driver via CLI."""
    args = parse_args()

    if args.version:
        print(f"nvidia-inst version {__version__}")
        return 0

    if args.debug:
        setup_logging("DEBUG")
    else:
        setup_logging()

    # Check for no GPU
    if not has_nvidia_gpu():
        print("\nNo NVIDIA GPU detected. Nothing to do.")
        return 0

    # Detect GPU
    try:
        gpu = detect_gpu()
        if not gpu:
            print("\n[ERROR] Failed to detect GPU")
            return 1
    except GPUDetectionError as e:
        print(f"\n[ERROR] Failed to detect GPU: {e}")
        return 1

    # Detect distro
    try:
        distro = detect_distro()
    except DistroDetectionError as e:
        print(f"\n[ERROR] Failed to detect distribution: {e}")
        return 1

    # Get driver range
    driver_range = get_driver_range(gpu)

    # Detect current state
    state = detect_driver_state(gpu, driver_range, distro.id)

    # Show options
    choice = show_driver_options(state, distro.id)
    if choice == -1:
        return 0

    # Find selected option
    selected = None
    for opt in state.options:
        if opt.number == choice:
            selected = opt
            break

    if not selected:
        print("\n[ERROR] Invalid selection")
        return 1

    # Execute the change
    return execute_driver_change(
        selected,
        state,
        distro,
        gpu,
        driver_range,
        simulate=args.simulate,
        with_cuda=not args.no_cuda,
        cuda_version=args.cuda_version,
    )


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
    print("  4. Require reboot")

    response = input("\nProceed? [y/N]: ")
    if response.lower() not in ("y", "yes"):
        print("Cancelled.")
        return 0

    if not require_root(interactive=True):
        print("\n[ERROR] Root privileges required.")
        return 1

    print("\nRemoving Nvidia packages...")
    result = revert_to_nouveau(distro.id)

    if result.success:
        print(f"\n✓ {result.message}")
        if result.packages_removed:
            print(f"  Removed: {', '.join(result.packages_removed)}")
        prompt_reboot()
        return 0
    else:
        print(f"\n✗ Revert failed: {', '.join(result.errors)}")
        return 1


def rollback_cli() -> int:
    """Rollback to previous driver state."""
    from nvidia_inst.installer.rollback import RollbackManager

    print("\n" + "=" * 50)
    print(" Rollback to Previous State")
    print("=" * 50)

    rollback_mgr = RollbackManager()
    snapshots = rollback_mgr.list_snapshots()

    if not snapshots:
        print("\nNo snapshots available for rollback.")
        print("Snapshots are created automatically before each installation.")
        return 1

    print(f"\nAvailable snapshots ({len(snapshots)}):")
    for i, snapshot in enumerate(snapshots[:5], 1):
        print(
            f"  [{i}] {snapshot['timestamp']} - Driver: {snapshot.get('driver_version', 'N/A')}, CUDA: {snapshot.get('cuda_version', 'N/A')}"
        )

    if len(snapshots) > 5:
        print(f"  ... and {len(snapshots) - 5} more")

    try:
        choice = int(input("\nSelect snapshot to rollback to [1]: ") or "1")
        if choice < 1 or choice > len(snapshots):
            print("Invalid selection.")
            return 1
    except ValueError:
        print("Invalid input.")
        return 1

    selected = snapshots[choice - 1]

    print(f"\nRolling back to snapshot from {selected['timestamp']}...")
    print("  Driver version: " + selected.get("driver_version", "N/A"))
    print("  CUDA version: " + selected.get("cuda_version", "N/A"))

    response = input("\nProceed with rollback? [y/N]: ")
    if response.lower() not in ("y", "yes"):
        print("Cancelled.")
        return 0

    if not require_root(interactive=True):
        print("\n[ERROR] Root privileges required.")
        return 1

    state_file = Path(selected["file"])
    state = rollback_mgr._load_snapshot(state_file)

    if state and rollback_mgr.rollback(state):
        print("\n✓ Rollback completed successfully")
        prompt_reboot()
        return 0
    else:
        print("\n✗ Rollback failed")
        return 1


def list_snapshots_cli() -> int:
    """List available system snapshots."""
    from nvidia_inst.installer.rollback import RollbackManager

    rollback_mgr = RollbackManager()
    snapshots = rollback_mgr.list_snapshots()

    if not snapshots:
        print("\nNo snapshots available.")
        return 0

    print("\n" + "=" * 50)
    print(" Available Snapshots")
    print("=" * 50)

    for i, snapshot in enumerate(snapshots, 1):
        print(f"\n[{i}] {snapshot['timestamp']}")
        print(f"    Distro: {snapshot['distro_id']}")
        print(f"    Driver: {snapshot.get('driver_version', 'N/A')}")
        print(f"    CUDA: {snapshot.get('cuda_version', 'N/A')}")

    return 0


def create_cache_cli(args: Any) -> int:
    """Create offline package cache."""
    from nvidia_inst.installer.offline import OfflineInstaller

    print("\n" + "=" * 50)
    print(" Creating Offline Package Cache")
    print("=" * 50)

    offline_installer = OfflineInstaller(cache_dir=args.cache_dir)

    try:
        distro = detect_distro()
        driver_range = get_driver_range(detect_gpu())

        # Get packages to cache
        packages = get_compatible_driver_packages(distro.id, driver_range)

        # Add CUDA packages if not disabled
        if not args.no_cuda:
            from nvidia_inst.installer.cuda import get_cuda_installer

            cuda_installer = get_cuda_installer(distro.id)
            cuda_pkgs = cuda_installer.get_cuda_packages(args.cuda_version)
            packages.extend(cuda_pkgs)

        print(f"\nDistribution: {distro}")
        print(f"Packages to cache: {', '.join(packages)}")
        print(f"Cache directory: {args.cache_dir}")

        if not args.yes:
            response = input("\nCreate offline cache? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                print("Cancelled.")
                return 0

        print("\nDownloading packages...")
        if offline_installer.create_cache(packages, distro.id):
            info = offline_installer.get_cache_info()
            print("\n✓ Cache created successfully")
            print(f"  Packages: {info['package_count']}")
            print(f"  Size: {info['total_size_mb']} MB")
            print(f"  Location: {args.cache_dir}")
            return 0
        else:
            print("\n✗ Failed to create cache")
            return 1

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


def verify_cache_cli(args: Any) -> int:
    """Verify offline package cache integrity."""
    from nvidia_inst.installer.offline import OfflineInstaller

    print("\n" + "=" * 50)
    print(" Verifying Offline Cache")
    print("=" * 50)

    offline_installer = OfflineInstaller(cache_dir=args.cache_dir)

    info = offline_installer.get_cache_info()
    if not info.get("exists"):
        print(f"\nNo cache found at {args.cache_dir}")
        return 1

    print(f"\nCache created: {info['created_at']}")
    print(f"Package count: {info['package_count']}")
    print(f"Total size: {info['total_size_mb']} MB")

    print("\nVerifying integrity...")
    if offline_installer.verify_cache_integrity():
        print("\n✓ Cache integrity verified")
        return 0
    else:
        print("\n✗ Cache integrity check failed")
        return 1


def set_power_profile_cli(profile: str) -> int:
    """Set hybrid graphics power profile."""
    try:
        distro = detect_distro()
    except DistroDetectionError as e:
        logger.error(f"Failed to detect distribution: {e}")
        print(f"Failed to detect distribution: {e}")
        return 1

    native_tool, _, _ = get_native_tool(distro.id)

    if not native_tool:
        print("\nNo native hybrid graphics tool found.")
        print("Using environment file configuration instead.")
        native_tool = "environment"

    print(f"\nSetting power profile to: {profile}")
    print(f"Using tool: {native_tool}")

    if not require_root(interactive=True):
        print("\n[ERROR] Root privileges required.")
        return 1

    success = set_power_profile(profile, native_tool, distro.id)

    if success:
        print(f"\n✓ Power profile set to: {profile}")
        print("  Changes will take effect after reboot.")
        return 0
    else:
        print("\n✗ Failed to set power profile")
        return 1


def main() -> int:
    """Main entry point for nvidia-inst CLI."""
    args = parse_args()

    if args.version:
        print(f"nvidia-inst version {__version__}")
        return 0

    if args.debug:
        setup_logging("DEBUG")
    else:
        setup_logging()

    update_matrix_on_startup()

    # GUI mode
    if args.gui:
        return launch_gui(args)

    # Check mode
    if args.check:
        return check_compatibility()

    # Revert to Nouveau
    if args.revert_to_nouveau:
        return revert_to_nouveau_cli()

    # Power profile
    if args.power_profile:
        return set_power_profile_cli(args.power_profile)

    # Rollback options
    if args.rollback:
        return rollback_cli()

    if args.list_snapshots:
        return list_snapshots_cli()

    # Offline installation options
    if args.create_cache:
        return create_cache_cli(args)

    if args.verify_cache:
        return verify_cache_cli(args)

    if args.offline:
        return install_driver_cli(offline=True, cache_dir=args.cache_dir)

    # Default: install driver
    return install_driver_cli()
