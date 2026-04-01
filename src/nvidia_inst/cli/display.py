"""Display functions for nvidia-inst CLI.

This module provides display and output functions for the CLI interface.
"""

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def print_row(label: str, value: str, width: int = 35) -> None:
    """Print a formatted row with label and value.

    Args:
        label: Label text
        value: Value text
        width: Total width for the row
    """
    print(f"  {label:<{width}} {value}")


def print_section_header(title: str, char: str = "=") -> None:
    """Print a section header.

    Args:
        title: Section title
        char: Character to use for the line
    """
    print(f"\n{char * 50}")
    print(f" {title}")
    print(f"{char * 50}")


def print_step(step_num: int, description: str) -> None:
    """Print a numbered step.

    Args:
        step_num: Step number
        description: Step description
    """
    print(f"  {step_num}. {description}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message
    """
    print(f"\n[!] {message}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Error message
    """
    print(f"\n[ERROR] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message
    """
    print(f"\n[INFO] {message}")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message
    """
    print(f"\n[OK] {message}")


def format_package_list(packages: list[str], max_display: int = 3) -> str:
    """Format a package list for display.

    Args:
        packages: List of package names
        max_display: Maximum number of packages to display

    Returns:
        Formatted string
    """
    if len(packages) <= max_display:
        return " ".join(packages)
    return " ".join(packages[:max_display]) + " ..."


def print_driver_status(
    current_version: str | None,
    is_working: bool,
    driver_type: str | None = None,
) -> None:
    """Print driver status information.

    Args:
        current_version: Currently installed driver version
        is_working: Whether driver is working
        driver_type: Type of driver (proprietary, nvidia_open, nouveau, none)
    """
    print("\nDriver Status:")
    if current_version:
        status = "Working" if is_working else "Not working"
        print(f"  Version: {current_version} ({status})")
        if driver_type:
            print(f"  Type: {driver_type}")
    else:
        print("  No NVIDIA driver installed")


def print_gpu_info(
    model: str,
    generation: str | None = None,
    compute_capability: float | None = None,
) -> None:
    """Print GPU information.

    Args:
        model: GPU model name
        generation: GPU generation (e.g., 'ampere', 'turing')
        compute_capability: Compute capability version
    """
    print("\nGPU Information:")
    print(f"  Model: {model}")
    if generation:
        print(f"  Generation: {generation}")
    if compute_capability:
        print(f"  Compute Capability: {compute_capability:.1f}")


def print_distro_info(
    distro_id: str,
    version_id: str | None = None,
    kernel: str | None = None,
) -> None:
    """Print distribution information.

    Args:
        distro_id: Distribution ID
        version_id: Distribution version
        kernel: Kernel version
    """
    print("\nDistribution:")
    print(f"  ID: {distro_id}")
    if version_id:
        print(f"  Version: {version_id}")
    if kernel:
        print(f"  Kernel: {kernel}")
