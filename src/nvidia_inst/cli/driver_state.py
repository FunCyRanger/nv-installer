"""Driver state detection and option building for nvidia-inst CLI.

This module provides driver state detection and option building
functionality for the CLI interface.
"""

from dataclasses import dataclass, field
from enum import Enum

from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.gpu.detector import GPUInfo
from nvidia_inst.installer.driver import (
    check_nvidia_open_available,
    check_nonfree_available,
    get_current_driver_type,
)
from nvidia_inst.installer.validation import is_nvidia_working
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


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


def _get_cuda_range_str(driver_range: DriverRange, generation: str) -> str:
    """Get CUDA range display string.

    Args:
        driver_range: Driver range with CUDA info.
        generation: GPU generation name.

    Returns:
        Formatted CUDA range string.
    """
    if driver_range.cuda_is_locked:
        return f"{driver_range.cuda_locked_major}.x (locked for {generation})"
    return f"{driver_range.cuda_min}-{driver_range.cuda_max}"


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
    from nvidia_inst.gpu.compatibility import is_driver_compatible
    from nvidia_inst.installer.driver import get_compatible_driver_packages

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
            from nvidia_inst.installer.cuda import get_cuda_installer

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
