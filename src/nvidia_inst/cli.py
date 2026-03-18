"""Command-line interface for nvidia-inst."""

import argparse
import sys
from typing import Any

from nvidia_inst.distro.detector import (
    DistroDetectionError,
    DistroInfo,
    detect_distro,
)
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.gpu.compatibility import DriverRange, get_driver_range
from nvidia_inst.gpu.detector import (
    GPUDetectionError,
    GPUInfo,
    detect_gpu,
    has_nvidia_gpu,
)
from nvidia_inst.gpu.matrix.data import GPUGenerationInfo
from nvidia_inst.installer.driver import (
    DriverInstallError,
    check_nouveau,
    check_secure_boot,
    disable_nouveau,
    get_compatible_driver_packages,
)
from nvidia_inst.installer.prerequisites import PrerequisitesChecker
from nvidia_inst.installer.uninstaller import (
    check_nvidia_packages_installed,
    revert_to_nouveau,
)
from nvidia_inst.installer.validation import (
    post_install_validate,
    pre_install_check,
    unblock_nouveau,
)
from nvidia_inst.utils.logger import get_logger, setup_logging

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
        "--fix",
        action="store_true",
        help="Automatically fix missing repositories",
    )

    parser.add_argument(
        "--revert-to-nouveau",
        action="store_true",
        help="Switch from proprietary driver to Nouveau (open-source)",
    )

    parser.add_argument(
        "--update-matrix",
        action="store_true",
        help="Force update of compatibility matrix",
    )

    parser.add_argument(
        "--matrix-info",
        action="store_true",
        help="Show compatibility matrix information",
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

    if driver_range.is_eol:
        print(f"\nWARNING: {driver_range.eol_message}")

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

    print(f"\n[{'✓' if result.package_manager_available else '✗'}] Package manager: {result.package_manager}")

    if result.repos_configured:
        for repo in result.repos_configured:
            print(f"[✓] {repo}")

    if result.repos_missing:
        for repo in result.repos_missing:
            print(f"[✗] {repo}: NOT CONFIGURED")

    print(f"\n[{'✓' if result.driver_packages_available else '✗'}] Driver packages: ", end="")
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
                return check_prerequisites(distro_id, distro_version, driver_range, fix=False)

    print("\n" + "-" * 50)

    if result.success:
        print("\nStatus: READY - All prerequisites met")
    else:
        print("\nStatus: NOT READY - Some prerequisites not met")
        print("Please fix the issues above before installing.")

    print()

    return 0 if result.success else 1


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
        return _run_dry_run(distro, gpu, driver_range, driver_version, with_cuda, cuda_version)

    if not skip_confirmation:
        response = input("\nProceed with installation? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print("Installation cancelled.")
            return 1

    nouveau_disabled = False
    if check_nouveau():
        logger.warning("Nouveau kernel module is loaded")
        response = input("Nouveau is loaded. Disable it? [y/N]: ")
        if response.lower() in ("y", "yes"):
            disable_nouveau()
            nouveau_disabled = True
            print("Nouveau has been disabled.")

    if check_secure_boot():
        logger.warning("Secure Boot is enabled")
        print("Secure Boot is enabled. You may need to sign the driver.")
        print("Consider disabling Secure Boot in BIOS/UEFI settings.")

    try:
        pkg_manager = get_package_manager()
        packages = get_compatible_driver_packages(
            distro.id,
            driver_range,
        )

        # Pre-installation safety check
        print("\nRunning safety checks...")
        safety = pre_install_check(distro.id, packages)

        if not safety.can_proceed:
            print("\n[ERROR] Cannot proceed with installation:")
            for err in safety.errors:
                print(f"  - {err}")
            return 1

        if safety.warnings:
            print("\n[WARNING] Potential issues detected:")
            for warn in safety.warnings:
                print(f"  - {warn}")

            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() not in ("y", "yes"):
                print("Installation cancelled.")
                return 1

        # Install packages
        print("\nInstalling driver packages...")
        for pkg in packages:
            print(f"  - {pkg}")
        print("(This may take a few minutes...)\n")

        pkg_manager.install(packages)

        # Post-installation validation
        print("Validating installation...")
        validation = post_install_validate(distro.id, packages)

        # ALWAYS show success message
        logger.info("Driver installation completed")
        print("\nDriver installed successfully!")

        # Show installed packages
        print("\nInstalled packages:")
        for pkg in packages:
            if pkg in validation.installed_packages:
                print(f"  ✓ {pkg}")
            else:
                print(f"  ✗ {pkg} (not found)")

        # Show validation issues (if any)
        if validation.warnings or validation.errors:
            print("\nValidation notes:")
            for err in validation.errors:
                print(f"  ✗ {err}")
            for warn in validation.warnings:
                print(f"  ⚠ {warn}")

        # If validation failed and Nouveau was disabled, try to re-enable it
        if not validation.success and nouveau_disabled:
            print("\n[WARNING] Validation issues detected.")
            print("Attempting to re-enable Nouveau for bootable system...")
            success, message = unblock_nouveau()
            if success:
                print(f"  ✓ {message}")
                print("\nIMPORTANT: Driver may not work properly.")
                print("If system fails to boot to graphical mode:")
                print("  - Boot to recovery mode")
                print("  - Run: sudo rm /etc/modprobe.d/blacklist-nouveau.conf")
                print("  - Run: sudo dracut -f && reboot")
                print("\nAfter fixing, you can manually block Nouveau once driver works.")
                nouveau_disabled = False
            else:
                print(f"  ✗ Could not re-enable Nouveau: {message}")
                print("  Manual fix: sudo rm /etc/modprobe.d/blacklist-nouveau.conf")

        # ALWAYS prompt for reboot
        if nouveau_disabled:
            print("\nIMPORTANT: Nouveau was disabled. You MUST reboot now.")
            print("Please reboot your system for changes to take effect.")
        else:
            print("\nPlease reboot your system for changes to take effect.")

    except DriverInstallError as e:
        logger.error(f"Installation failed: {e}")
        print(f"Installation failed: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return 1

    return 0


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

    from nvidia_inst.installer.driver import check_nouveau, check_secure_boot
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
            print(f"Selected driver branch: {driver_range.max_branch}.xx (current max: {driver_range.max_version})")
            print(f"  -> Will receive branch updates (e.g., {driver_range.max_branch}.143, {driver_range.max_branch}.144)")
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

    # Step 1: Update package lists first
    print("# Step 1: Update package lists:")
    if distro.id in ("ubuntu", "debian"):
        print("  sudo apt update")
    elif distro.id in ("fedora", "rhel", "centos"):
        print("  sudo dnf makecache")
    elif distro.id in ("arch", "manjaro"):
        print("  sudo pacman -Sy")
    elif distro.id in ("opensuse"):
        print("  sudo zypper refresh")

    # Step 2: Block wrong branch if GPU is limited (before install)
    if driver_range.max_branch and driver_range.is_limited:
        wrong_branch = _get_wrong_branch(driver_range.max_branch)
        if wrong_branch:
            print("\n# Step 2: BLOCK wrong driver branch (IMPORTANT!):")
            if distro.id in ("fedora", "rhel", "centos"):
                print(f"  # Block {wrong_branch}.xx drivers - these are incompatible with your GPU!")
                print(f"  sudo dnf versionlock add '*{wrong_branch}.*' || true")
            elif distro.id in ("ubuntu", "debian"):
                print(f"  # Block {wrong_branch}.xx drivers in /etc/apt/preferences.d/")
                print("  # This prevents installing incompatible drivers!")

    # Step 3: Install driver packages (now done before reboot)
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

    # Step 4: Lock driver to branch
    if driver_range.max_branch and driver_range.is_limited:
        if driver_range.is_eol:
            print("\n# Step 4: Lock driver to exact version (EOL GPU):")
            if distro.id in ("ubuntu", "debian"):
                print(f"  # Pin nvidia-driver to {driver_range.max_version}")
            elif distro.id in ("fedora", "rhel", "centos"):
                print(f"  sudo dnf versionlock add 'akmod-nvidia-{driver_range.max_branch}.*'")
        else:
            print(f"\n# Step 4: Lock driver to branch {driver_range.max_branch}.xx (optional):")
            if distro.id in ("ubuntu", "debian"):
                print(f"  # Pin to branch {driver_range.max_branch}.* in /etc/apt/preferences.d/")
            elif distro.id in ("fedora", "rhel", "centos"):
                print(f"  sudo dnf versionlock add 'akmod-nvidia-{driver_range.max_branch}.*'")

    # Step 5: Reboot (now after driver is installed)
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


def show_matrix_info() -> int:
    """Show compatibility matrix information."""
    try:
        from nvidia_inst.gpu.matrix.manager import MatrixManager
        manager = MatrixManager()

        print("\n" + "=" * 60)
        print(" Compatibility Matrix Information")
        print("=" * 60)

        print(f"\nVersion: {manager.get_version()}")
        print(f"Last Updated: {manager.get_last_update_time() or 'unknown'}")
        print(f"Data Source: {'Online' if manager.is_online_data else 'Fallback'}")

        branches = manager.get_all_branches()
        if branches:
            print("\nDriver Branches:")
            for branch in sorted(branches.keys()):
                branch_info = branches[branch]
                eol = f" (EOL: {branch_info.eol_date})" if branch_info.eol_date else ""
                print(f"  {branch}: {branch_info.name} - {branch_info.latest_version}{eol}")

        generations = manager.get_all_generations()
        if generations:
            print("\nGPU Generations:")
            for name in sorted(generations.keys()):
                gen_info: GPUGenerationInfo = generations[name]
                status_icon = {"full": "[+]", "limited": "[~]", "eol": "[-]"}
                icon = status_icon.get(gen_info.status.value, "[?]")
                print(f"  {icon} {gen_info.display_name}")

        print("\n" + "=" * 60 + "\n")
        return 0

    except Exception as e:
        logger.error(f"Failed to load matrix info: {e}")
        print(f"Error: {e}")
        return 1


def update_matrix_cli() -> int:
    """Update compatibility matrix from online sources."""
    try:
        from nvidia_inst.gpu.matrix.manager import MatrixManager
        manager = MatrixManager(force_update=True)
        updated, message = manager.check_for_updates()

        if updated:
            print("\nMatrix updated successfully!")
            print(f"Version: {manager.get_version()}")
        else:
            print(f"\nMatrix update: {message}")

        return 0

    except Exception as e:
        logger.error(f"Failed to update matrix: {e}")
        print(f"Error: {e}")
        return 1


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


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(debug=args.debug, dry_run=args.dry_run)

    if args.version:
        from nvidia_inst import __version__
        print(f"nvidia-inst {__version__}")
        return 0

    logger.info("Starting nvidia-inst")

    if args.matrix_info:
        return show_matrix_info()

    if args.update_matrix:
        return update_matrix_cli()

    if args.revert_to_nouveau:
        return revert_to_nouveau_cli()

    if args.gui or args.gui_type:
        return launch_gui(args)

    update_matrix_on_startup()

    if args.check:
        return check_compatibility()

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
