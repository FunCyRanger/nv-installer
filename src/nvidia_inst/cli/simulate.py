"""Simulate output generation for nvidia-inst CLI.

This module provides simulate output functionality using the tool-based
approach from tools.py, allowing support for any distro that uses
supported package managers.
"""

from nvidia_inst.distro.tools import (
    get_install_command,
    get_remove_command,
    get_update_command,
)
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)

# Distro ID to tool mapping
DISTRO_TO_TOOL: dict[str, str] = {
    "ubuntu": "apt",
    "debian": "apt",
    "linuxmint": "apt",
    "pop": "apt",
    "fedora": "dnf",
    "rhel": "dnf",
    "centos": "dnf",
    "rocky": "dnf",
    "alma": "dnf",
    "arch": "pacman",
    "manjaro": "pacman",
    "endeavouros": "pacman",
    "opensuse": "zypper",
    "sles": "zypper",
}


def _get_tool_for_distro(distro_id: str) -> str:
    """Get the package manager tool for a distro.

    Args:
        distro_id: Distribution ID

    Returns:
        Tool name (apt, dnf, pacman, zypper)
    """
    return DISTRO_TO_TOOL.get(distro_id, "apt")


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


def simulate_generic(
    title: str,
    state_message: str | None,
    current_version: str | None,
    packages: list[str],
    cuda_pkgs: list[str],
    steps: list[str],
) -> None:
    """Generic simulate output.

    Args:
        title: Title for the simulate output
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        cuda_pkgs: Target CUDA packages
        steps: List of steps to execute
    """
    print("\n" + "=" * 50)
    print(f" SIMULATE MODE - {title}")
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

    print("\nOr run this script without --simulate:")
    print("  sudo nvidia-inst")


def simulate_change(
    state_message: str,
    current_version: str | None,
    packages: list[str],
    distro_id: str,
    with_cuda: bool = True,
    cuda_version: str | None = None,
) -> None:
    """Show simulate output for driver change.

    Args:
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora')
        with_cuda: Whether CUDA will be installed
        cuda_version: Specific CUDA version (optional)
    """
    tool = _get_tool_for_distro(distro_id)
    steps = []

    # Remove existing packages if needed
    if current_version:
        remove_cmd = get_remove_command(tool)
        steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Update package lists
    update_cmd = get_update_command(tool)
    steps.append(f"sudo {' '.join(update_cmd)}")

    # Install driver packages
    install_cmd = get_install_command(tool)
    steps.append(f"sudo {' '.join(install_cmd)} {' '.join(packages)}")

    # Install CUDA packages if requested
    cuda_pkgs: list[str] = []
    if with_cuda:
        cuda_display = cuda_version or "default"
        cuda_pkgs = [f"cuda-toolkit-{cuda_display}"]
        steps.append(f"sudo {' '.join(install_cmd)} {' '.join(cuda_pkgs)}")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    simulate_generic(
        title="Driver Change",
        state_message=state_message,
        current_version=current_version,
        packages=packages,
        cuda_pkgs=cuda_pkgs,
        steps=steps,
    )


def simulate_nvidia_open_install(
    state_message: str | None,
    current_version: str | None,
    packages: list[str],
    distro_id: str,
    with_cuda: bool = True,
    cuda_version: str | None = None,
) -> None:
    """Show simulate output for NVIDIA Open installation.

    Args:
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        distro_id: Distribution ID
        with_cuda: Whether CUDA will be installed
        cuda_version: Specific CUDA version (optional)
    """
    tool = _get_tool_for_distro(distro_id)
    steps = []

    # Remove existing packages if needed
    if current_version:
        remove_cmd = get_remove_command(tool)
        steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Install driver packages
    install_cmd = get_install_command(tool)
    steps.append(f"sudo {' '.join(install_cmd)} {' '.join(packages)}")

    # Install CUDA packages if requested
    cuda_pkgs: list[str] = []
    if with_cuda:
        cuda_display = cuda_version or "default"
        cuda_pkgs = [f"cuda-toolkit-{cuda_display}"]
        steps.append(f"sudo {' '.join(install_cmd)} {' '.join(cuda_pkgs)}")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    simulate_generic(
        title="NVIDIA Open Installation",
        state_message=state_message,
        current_version=current_version,
        packages=packages,
        cuda_pkgs=cuda_pkgs,
        steps=steps,
    )


def simulate_nouveau_install(
    packages: list[str],
    distro_id: str,
) -> None:
    """Show simulate output for Nouveau installation.

    Args:
        packages: Target packages
        distro_id: Distribution ID
    """
    tool = _get_tool_for_distro(distro_id)
    steps = []

    # Remove existing NVIDIA packages
    remove_cmd = get_remove_command(tool)
    steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Install nouveau packages
    install_cmd = get_install_command(tool)
    steps.append(f"sudo {' '.join(install_cmd)} {' '.join(packages)}")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    simulate_generic(
        title="Nouveau Installation",
        state_message=None,
        current_version=None,
        packages=packages,
        cuda_pkgs=[],
        steps=steps,
    )


def simulate_revert(
    distro_id: str,
) -> None:
    """Show simulate output for revert to nouveau.

    Args:
        distro_id: Distribution ID
    """
    tool = _get_tool_for_distro(distro_id)
    steps = []

    # Remove NVIDIA packages
    remove_cmd = get_remove_command(tool)
    steps.append(f"sudo {' '.join(remove_cmd)} nvidia*")

    # Rebuild initramfs
    initramfs_cmd = get_initramfs_command(tool)
    steps.append(f"sudo {' '.join(initramfs_cmd)}")

    steps.append("sudo reboot")

    simulate_generic(
        title="Revert to Nouveau",
        state_message=None,
        current_version=None,
        packages=[],
        cuda_pkgs=[],
        steps=steps,
    )
