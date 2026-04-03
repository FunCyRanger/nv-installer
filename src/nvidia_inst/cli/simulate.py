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


def _format_lock_details(distro_id: str, driver_range, packages: list[str]) -> str:
    """Format version lock details for display.

    Args:
        distro_id: Distribution ID
        driver_range: Driver range with lock info
        packages: Driver packages to lock

    Returns:
        Formatted lock details string
    """
    lines = []
    branch = driver_range.max_branch
    cuda_locked = driver_range.cuda_locked_major

    if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        lines.append(
            "The following entries will be written to /etc/dnf/versionlock.toml:"
        )
        lines.append("")
        for pkg in packages:
            lines.append("  [[packages]]")
            lines.append(f'  name = "{pkg}"')
            lines.append("  [[packages.conditions]]")
            lines.append('  key = "evr"')
            lines.append('  comparator = ">="')
            lines.append(f'  value = "{branch}"')
            lines.append("  [[packages.conditions]]")
            lines.append('  key = "evr"')
            lines.append('  comparator = "<"')
            lines.append(f'  value = "{int(branch) + 1}"')
            lines.append("")
        if cuda_locked:
            lines.append("  [[packages]]")
            lines.append('  name = "cuda-toolkit"')
            lines.append("  [[packages.conditions]]")
            lines.append('  key = "evr"')
            lines.append('  comparator = ">="')
            lines.append(f'  value = "{cuda_locked}"')
            lines.append("  [[packages.conditions]]")
            lines.append('  key = "evr"')
            lines.append('  comparator = "<"')
            lines.append(f'  value = "{int(cuda_locked) + 1}"')
            lines.append("")

    elif distro_id in ("ubuntu", "debian", "linuxmint", "pop"):
        lines.append("The following APT preferences files will be created:")
        lines.append("")
        for pkg in packages:
            pref_name = pkg.replace("*", "").replace("-", "_")
            lines.append(f"  File: /etc/apt/preferences.d/nvidia-inst-{pref_name}")
            lines.append(f"    Package: {pkg}")
            lines.append(f"    Pin: version {branch}.*")
            lines.append("    Pin-Priority: 1001")
            lines.append("")

    elif distro_id in ("opensuse", "sles"):
        lines.append("The following zypper lock commands will be executed:")
        lines.append("")
        for pkg in packages:
            lines.append(f"  sudo zypper addlock {pkg}")
            lines.append(f'  sudo zypper addlock "{pkg} >= {int(branch) + 1}"')
        lines.append("")

    elif distro_id in ("arch", "manjaro"):
        lines.append("No version locking needed — Arch uses branch-specific packages")
        lines.append(
            f"(e.g., nvidia-{branch}xx-dkms) that stay on the correct branch automatically."
        )

    return "\n".join(lines)


def simulate_change(
    state_message: str,
    current_version: str | None,
    packages: list[str],
    distro_id: str,
    with_cuda: bool = True,
    cuda_version: str | None = None,
    driver_range=None,
    gpu=None,
) -> None:
    """Show simulate output for driver change.

    Args:
        state_message: Current driver state message
        current_version: Currently installed driver version
        packages: Target driver packages
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora')
        with_cuda: Whether CUDA will be installed
        cuda_version: Specific CUDA version (optional)
        driver_range: Compatible driver range for lock details
        gpu: GPU info for display
    """
    tool = _get_tool_for_distro(distro_id)

    print("\n" + "=" * 50)
    print(" SIMULATE MODE - Driver Change")
    print("=" * 50)

    if state_message:
        print(f"\nCurrent state: {state_message}")
        if current_version:
            print(f"  Installed: {current_version}")

    print(f"\nTarget driver packages: {' '.join(packages)}")

    # Show CUDA packages
    cuda_pkgs: list[str] = []
    if with_cuda:
        cuda_display = cuda_version or "default"
        cuda_pkgs = [f"cuda-toolkit-{cuda_display}"]
        print(f"Target CUDA packages:   {' '.join(cuda_pkgs)}")

    # Build step table
    print("\n--- Planned Steps ---")
    print("")
    print(" # | Action                    | Details")
    print(
        "---+---------------------------+--------------------------------------------------"
    )

    step_num = 1

    # Step 1: Remove old packages
    if current_version:
        remove_cmd = get_remove_command(tool)
        pkg_str = " ".join(packages[:2])
        print(
            f" {step_num} | Remove old packages       | sudo {' '.join(remove_cmd)} {pkg_str}"
        )
        step_num += 1

    # Step 2: Lock driver packages (BEFORE install)
    if driver_range and driver_range.max_branch:
        lock_cmd = _get_lock_command(tool, packages, driver_range.max_branch)
        print(
            f" {step_num} | Lock driver to {driver_range.max_branch}.x{' ' * (14 - len(driver_range.max_branch))}| {lock_cmd}"
        )
        step_num += 1

    # Step 3: Lock CUDA packages
    if (
        with_cuda
        and driver_range
        and driver_range.cuda_is_locked
        and driver_range.cuda_locked_major
    ):
        cuda_lock_cmd = _get_cuda_lock_command(tool, driver_range.cuda_locked_major)
        print(
            f" {step_num} | Lock CUDA to {driver_range.cuda_locked_major}.x{' ' * (16 - len(driver_range.cuda_locked_major))}| {cuda_lock_cmd}"
        )
        step_num += 1

    # Step 4: Update package cache
    update_cmd = get_update_command(tool)
    print(f" {step_num} | Update package cache      | sudo {' '.join(update_cmd)}")
    step_num += 1

    # Step 5: Install driver packages
    install_cmd = get_install_command(tool)
    print(
        f" {step_num} | Install driver packages   | sudo {' '.join(install_cmd)} {' '.join(packages)}"
    )
    step_num += 1

    # Step 6: Install CUDA
    if with_cuda and cuda_pkgs:
        print(
            f" {step_num} | Install CUDA toolkit      | sudo {' '.join(install_cmd)} {' '.join(cuda_pkgs)}"
        )
        step_num += 1

    # Step 7: Rebuild initramfs
    initramfs_cmd = get_initramfs_command(tool)
    print(f" {step_num} | Rebuild initramfs         | sudo {' '.join(initramfs_cmd)}")
    step_num += 1

    # Step 8: Reboot
    print(f" {step_num} | Reboot                    | sudo reboot")

    # Show lock details if available
    if driver_range and driver_range.max_branch:
        print("")
        print("--- Version Lock Details ---")
        print("")
        lock_details = _format_lock_details(distro_id, driver_range, packages)
        print(lock_details)

    print("\n--- Manual Commands ---")
    print("\nTo execute these steps manually, run each command from the table above.")
    print("Or run this script without --simulate:")
    print("  sudo nvidia-inst")


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


def _get_lock_command(tool: str, packages: list[str], branch: str) -> str:
    """Get version lock command for display.

    Args:
        tool: Package manager tool
        packages: Packages to lock
        branch: Branch version

    Returns:
        Formatted lock command string
    """
    if tool in ("dnf", "dnf5", "yum"):
        return "(writes to /etc/dnf/versionlock.toml)"
    elif tool == "apt":
        return "(creates /etc/apt/preferences.d/nvidia-inst-*)"
    elif tool == "zypper":
        pkgs = ", ".join(packages)
        return f"sudo zypper addlock {pkgs}"
    else:
        return "(no locking needed)"


def _get_cuda_lock_command(tool: str, cuda_major: str) -> str:
    """Get CUDA lock command for display.

    Args:
        tool: Package manager tool
        cuda_major: CUDA major version

    Returns:
        Formatted lock command string
    """
    if tool in ("dnf", "dnf5", "yum"):
        return "(writes to /etc/dnf/versionlock.toml)"
    elif tool == "apt":
        return "(creates /etc/apt/preferences.d/nvidia-inst-cuda)"
    elif tool == "zypper":
        return "sudo zypper addlock cuda"
    else:
        return "(no locking needed)"


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
