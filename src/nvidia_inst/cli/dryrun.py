"""Dry-run output generation for nvidia-inst.

This module provides dry-run output functionality using the tool-based
approach from tools.py, allowing support for any distro that uses
supported package managers.
"""

from nvidia_inst.distro.tools import (
    PackageContext,
    get_install_command,
    get_remove_command,
    get_update_command,
)
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def get_initramfs_command(tool: str) -> list[str]:
    """Get initramfs rebuild command for a tool.

    Args:
        tool: Package manager tool name

    Returns:
        List of command arguments for rebuilding initramfs
    """
    commands: dict[str, list[str]] = {
        "apt": ["update-initramfs", "-u"],
        "apt-get": ["update-initramfs", "-u"],
        "dnf": ["dracut", "-f", "--regenerate-all"],
        "dnf5": ["dracut", "-f", "--regenerate-all"],
        "yum": ["dracut", "-f", "--regenerate-all"],
        "pacman": ["mkinitcpio", "-P"],
        "pamac": ["mkinitcpio", "-P"],
        "paru": ["mkinitcpio", "-P"],
        "yay": ["mkinitcpio", "-P"],
        "zypper": ["dracut", "-f", "--regenerate-all"],
    }
    return commands.get(tool, ["update-initramfs", "-u"])


def dry_run_generic(
    title: str,
    state_message: str | None,
    current_version: str | None,
    packages: list[str],
    cuda_pkgs: list[str],
    steps: list[str],
) -> None:
    """Generic dry-run output.

    Args:
        title: Title for the dry-run output
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        cuda_pkgs: Target CUDA packages
        steps: List of steps to execute
    """
    print("\n" + "=" * 50)
    print(f" DRY-RUN MODE - {title}")
    print("=" * 50)

    if state_message:
        print(f"\nCurrent state: {state_message}")
        if current_version:
            print(f"  Installed: {current_version}")

    print(f"\nTarget packages: {' '.join(packages)}")
    if cuda_pkgs:
        print(f"Target CUDA packages: {' '.join(cuda_pkgs)}")

    print("\nSteps to execute manually:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    print("\nOr run this script without --dry-run:")
    print("  sudo nvidia-inst")


def dry_run_change(
    state_message: str,
    current_version: str | None,
    packages: list[str],
    ctx: PackageContext,
    cuda_pkgs: list[str] | None = None,
) -> None:
    """Show dry-run output for driver change.

    Args:
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        ctx: Package context with tool information
        cuda_pkgs: Target CUDA packages (optional)
    """
    steps = []

    # Remove existing packages if needed
    if current_version:
        remove_cmd = get_remove_command(ctx.tool)
        steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Update package lists
    update_cmd = get_update_command(ctx.tool)
    steps.append(f"sudo {' '.join(update_cmd)}")

    # Install driver packages
    install_cmd = get_install_command(ctx.tool)
    steps.append(f"sudo {' '.join(install_cmd)} {' '.join(packages)}")

    # Install CUDA packages
    if cuda_pkgs:
        steps.append(f"sudo {' '.join(install_cmd)} {' '.join(cuda_pkgs)}")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(ctx.tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    dry_run_generic(
        title="Driver Change",
        state_message=state_message,
        current_version=current_version,
        packages=packages,
        cuda_pkgs=cuda_pkgs or [],
        steps=steps,
    )


def dry_run_nvidia_open_install(
    state_message: str | None,
    current_version: str | None,
    packages: list[str],
    ctx: PackageContext,
    cuda_pkgs: list[str] | None = None,
) -> None:
    """Show dry-run output for NVIDIA Open installation.

    Args:
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        ctx: Package context with tool information
        cuda_pkgs: Target CUDA packages (optional)
    """
    steps = []

    # Remove existing packages if needed
    if current_version:
        remove_cmd = get_remove_command(ctx.tool)
        steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Install driver packages
    install_cmd = get_install_command(ctx.tool)
    steps.append(f"sudo {' '.join(install_cmd)} {' '.join(packages)}")

    # Install CUDA packages
    if cuda_pkgs:
        steps.append(f"sudo {' '.join(install_cmd)} {' '.join(cuda_pkgs)}")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(ctx.tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    dry_run_generic(
        title="NVIDIA Open Installation",
        state_message=state_message,
        current_version=current_version,
        packages=packages,
        cuda_pkgs=cuda_pkgs or [],
        steps=steps,
    )


def dry_run_nouveau_install(
    packages: list[str],
    ctx: PackageContext,
) -> None:
    """Show dry-run output for Nouveau installation.

    Args:
        packages: Target packages
        ctx: Package context with tool information
    """
    steps = []

    # Remove existing NVIDIA packages
    remove_cmd = get_remove_command(ctx.tool)
    steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Install nouveau packages
    install_cmd = get_install_command(ctx.tool)
    steps.append(f"sudo {' '.join(install_cmd)} {' '.join(packages)}")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(ctx.tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    dry_run_generic(
        title="Nouveau Installation",
        state_message=None,
        current_version=None,
        packages=packages,
        cuda_pkgs=[],
        steps=steps,
    )


def dry_run_revert(
    ctx: PackageContext,
) -> None:
    """Show dry-run output for revert to nouveau.

    Args:
        ctx: Package context with tool information
    """
    steps = []

    # Remove NVIDIA packages
    remove_cmd = get_remove_command(ctx.tool)
    steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(ctx.tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    dry_run_generic(
        title="Revert to Nouveau",
        state_message=None,
        current_version=None,
        packages=[],
        cuda_pkgs=[],
        steps=steps,
    )
