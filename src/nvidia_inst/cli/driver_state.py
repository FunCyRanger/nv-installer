"""Driver state detection and option building for nvidia-inst CLI.

This module provides driver state detection and option building
functionality for the CLI interface.
"""

import os
from dataclasses import dataclass
from enum import Enum

from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.gpu.detector import GPUInfo
from nvidia_inst.installer.driver import (
    check_nonfree_available,
    check_nvidia_open_available,
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
            "revert_nouveau",
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
            "revert_nouveau",
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
            "revert_nouveau",
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
            "revert_nouveau",
        )
    )

    options.append(DriverOption(len(options) + 1, "Cancel", "cancel"))

    return options


def _get_nouveau_version() -> str:
    """Detect Mesa version for Nouveau driver."""
    import subprocess

    try:
        result = subprocess.run(
            ["glxinfo", "-B"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "OpenGL version" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    version = parts[1].strip().split()[0]
                    return version
    except Exception:
        pass
    return "-"


def _format_versionlock_conditions(conditions: list[dict]) -> str:
    """Format versionlock conditions into human-readable string.

    Args:
        conditions: List of condition dicts with key/comparator/value.

    Returns:
        Human-readable version string (e.g., "580.x", "12.x").
    """
    lower = None
    upper = None
    for cond in conditions:
        if cond.get("key") == "evr":
            if cond.get("comparator") == ">=":
                lower = cond.get("value")
            elif cond.get("comparator") == "<":
                upper = cond.get("value")

    if lower and upper:
        return f"{lower}.x"
    elif lower:
        return f">= {lower}"
    elif upper:
        return f"< {upper}"
    return "*"


def _get_current_locks(distro_id: str) -> list[str]:
    """Read current package manager version locks.

    Returns:
        List of lock descriptions (e.g., ["akmod-nvidia (580.x)", "cuda-toolkit (12.x)"]).
    """
    locks = []

    if distro_id in ("ubuntu", "debian", "linuxmint"):
        prefs_dir = "/etc/apt/preferences.d"
        if os.path.isdir(prefs_dir):
            for fname in os.listdir(prefs_dir):
                if fname.startswith("nvidia-inst-"):
                    try:
                        with open(os.path.join(prefs_dir, fname)) as f:
                            content = f.read()
                            pkg = None
                            pin = None
                            for line in content.splitlines():
                                if line.startswith("Package:"):
                                    pkg = line.split(":", 1)[1].strip()
                                elif line.startswith("Pin:"):
                                    pin = line.split(":", 1)[1].strip()
                            if pkg:
                                display = f"{pkg} ({pin})" if pin else pkg
                                locks.append(display)
                    except OSError:
                        pass

    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
        import tomllib

        vlock_path = "/etc/dnf/versionlock.toml"
        if os.path.isfile(vlock_path):
            try:
                with open(vlock_path, "rb") as f:
                    data = tomllib.load(f)
                for pkg in data.get("packages", []):
                    name = pkg.get("name", "")
                    if "nvidia" in name.lower() or "cuda" in name.lower():
                        conditions = pkg.get("conditions", [])
                        version_str = _format_versionlock_conditions(conditions)
                        locks.append(f"{name} ({version_str})")
            except Exception:
                pass

    elif distro_id in (
        "opensuse",
        "opensuse-leap",
        "opensuse-tumbleweed",
        "sles",
    ):
        import subprocess

        try:
            result = subprocess.run(
                ["zypper", "locks"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "nvidia" in line.lower() or "cuda" in line.lower():
                    parts = line.split("|")
                    if len(parts) >= 2:
                        locks.append(parts[1].strip())
        except Exception:
            pass

    return locks


def _get_option_locks(driver_range: DriverRange, action: str) -> str:
    """Compute what would be locked for a given option.

    Args:
        driver_range: GPU's driver range with constraint info.
        action: The option action.

    Returns:
        Lock description string or "-" if no locks.
    """
    if action in ("keep", "cancel", "revert_nouveau"):
        return "-"

    if not driver_range.is_limited and not driver_range.cuda_is_locked:
        return "-"

    branch = driver_range.max_branch
    cuda_locked = driver_range.cuda_is_locked

    if not branch and not cuda_locked:
        return "-"

    lock_parts = []
    if branch:
        if "open" in action:
            lock_parts.append(f"nvidia-driver-{branch}-open*")
        else:
            lock_parts.append(f"nvidia-driver-{branch}*")
        lock_parts.append("cuda-*")
    elif cuda_locked and driver_range.cuda_locked_major:
        lock_parts.append(f"cuda-toolkit-{driver_range.cuda_locked_major}*")

    return ", ".join(lock_parts) if lock_parts else "-"


def _get_constraints(driver_range: DriverRange) -> list[str]:
    """Get constraint labels from driver range."""
    constraints = []
    if driver_range.is_limited or (
        driver_range.max_branch and driver_range.cuda_is_locked
    ):
        constraints.append("Branch")
    if driver_range.cuda_is_locked:
        constraints.append("CUDA")
    if driver_range.is_eol:
        constraints.append("EOL")
    return constraints


def _get_warning_line(driver_range: DriverRange, gpu: GPUInfo) -> str | None:
    """Get warning message for limited/EOL GPUs."""
    if driver_range.eol_message:
        return driver_range.eol_message

    if driver_range.is_limited and driver_range.max_branch:
        cuda_str = ""
        if driver_range.cuda_is_locked and driver_range.cuda_locked_major:
            cuda_str = f" CUDA frozen at {driver_range.cuda_locked_major}.x."
        return (
            f"{gpu.generation.capitalize()} GPUs are limited to branch "
            f"{driver_range.max_branch}.{cuda_str}"
        )

    return None


def _format_status_table(
    state: DriverState,
    driver_range: DriverRange,
    gpu: GPUInfo,
    distro_id: str,
) -> str:
    """Format the driver status comparison table.

    Args:
        state: Current driver state.
        driver_range: Compatible driver range for the GPU.
        gpu: Detected GPU information.
        distro_id: Distribution ID.

    Returns:
        Formatted table string.
    """
    lines = []

    lines.append(f"\n{'=' * 50}")
    lines.append(" Driver Status")
    lines.append(f"{'=' * 50}")
    lines.append(f"\nGPU: {gpu.model} ({gpu.generation.capitalize()})")
    lines.append(f"Distribution: {distro_id}")

    current_cuda = "-"
    if state.current_version:
        try:
            from nvidia_inst.installer.cuda import get_cuda_installer

            cuda_installer = get_cuda_installer(distro_id)
            ver = cuda_installer.get_installed_cuda_version()
            current_cuda = ver if ver else "-"
        except Exception:
            pass

    driver_type = get_current_driver_type()
    if driver_type == "proprietary":
        current_label = "Proprietary (active)"
    elif driver_type == "nvidia_open":
        current_label = "Open-source (active)"
    elif driver_type == "nouveau":
        current_label = "Nouveau (active)"
    else:
        current_label = "Nouveau (active)"

    current_locks = _get_current_locks(distro_id)
    current_lock_str = ", ".join(current_locks) if current_locks else "-"

    option_version = driver_range.max_version or driver_range.min_version
    option_branch = driver_range.max_branch or "-"
    option_cuda = (
        f"{driver_range.cuda_locked_major}.x"
        if driver_range.cuda_is_locked
        else (driver_range.cuda_max or driver_range.cuda_min)
    )

    lines.append("")
    lines.append(" # | Driver               | Version   | Branch | CUDA | Locked")
    lines.append(
        "---+----------------------+-----------+--------+------+---------------------------"
    )

    current_version = state.current_version or "-"
    current_branch = "-"
    if state.current_version:
        current_branch = state.current_version.split(".")[0]

    lines.append(
        f" * | {current_label:<20} | {current_version:<9} | {current_branch:<6} | {current_cuda:<4} | {current_lock_str}"
    )

    for opt in state.options:
        if opt.action in ("keep", "cancel"):
            continue

        if opt.action in ("install", "upgrade"):
            label = "NVIDIA proprietary"
            version = option_version
            branch = option_branch
            cuda = option_cuda
            locked = _get_option_locks(driver_range, opt.action)
        elif opt.action in ("switch_nvidia_open", "install_nvidia_open"):
            label = "NVIDIA open-source"
            version = option_version
            branch = option_branch
            cuda = option_cuda
            locked = _get_option_locks(driver_range, opt.action)
        elif opt.action == "revert_nouveau":
            label = "Nouveau"
            version = "-"
            branch = "-"
            cuda = "None"
            locked = "-"
        else:
            continue

        lines.append(
            f" {opt.number} | {label:<20} | {version:<9} | {branch:<6} | {cuda:<4} | {locked}"
        )

    warning = _get_warning_line(driver_range, gpu)
    if warning:
        lines.append("")
        lines.append(f"[!] {warning}")

    return "\n".join(lines)


def show_driver_options(
    state: DriverState,
    driver_range: DriverRange,
    gpu: GPUInfo,
    distro_id: str,
) -> int:
    """Show driver options menu and return selected option.

    Args:
        state: Current driver state with available options.
        driver_range: Compatible driver range for the GPU.
        gpu: Detected GPU information.
        distro_id: Distribution ID for CUDA detection.

    Returns:
        Selected option number, or -1 to cancel.
    """
    table = _format_status_table(state, driver_range, gpu, distro_id)
    print(table)

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
