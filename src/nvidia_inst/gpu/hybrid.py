"""Hybrid graphics detection and management for NVIDIA Optimus systems."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)

TOOL_SERVICE_PACKAGES = {
    "switcherooctl": "switcheroo-control",
    "nvidia-prime": None,
    "system76-power": None,
    "cachyos-settings": None,
}


@dataclass
class HybridInfo:
    """Information about hybrid graphics configuration."""

    is_hybrid: bool
    igpu_type: Literal["intel", "amd", None]
    igpu_model: str | None
    dgpu_model: str | None
    dgpu_count: int
    system_type: Literal["laptop", "desktop", "unknown"]
    native_tool: str | None
    needs_install: bool
    available_modes: list[str]
    env_file_path: str = "/etc/environment.d/90-nvidia-hybrid.conf"


@dataclass
class GPUDevice:
    """Represents a GPU device."""

    index: int
    name: str
    gpu_type: Literal["discrete", "integrated"]
    vendor: str


def detect_all_gpus() -> list[GPUDevice]:
    """Detect all NVIDIA GPUs using nvidia-smi.

    Returns:
        List of GPUDevice objects for all detected NVIDIA GPUs.
    """
    gpus = []

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().splitlines():
            if line.startswith("GPU "):
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    index = int(parts[0].replace("GPU ", "").strip())
                    name = parts[1].strip()
                    gpus.append(
                        GPUDevice(
                            index=index,
                            name=name,
                            gpu_type="discrete",
                            vendor="NVIDIA",
                        )
                    )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("nvidia-smi not available or failed")

    return gpus


def detect_integrated_gpu() -> tuple[Literal["intel", "amd", None], str | None]:
    """Detect integrated GPU (Intel or AMD) from lspci.

    Returns:
        Tuple of (igpu_type, igpu_model) or (None, None) if not found.
    """
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.splitlines():
            line_lower = line.lower()
            if "vga compatible controller" in line_lower:
                if "intel" in line_lower or "uhd" in line_lower or "iris" in line_lower:
                    return ("intel", line)
                elif "amd" in line_lower or "radeon" in line_lower:
                    return ("amd", line)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("lspci not available or failed")

    return (None, None)


def detect_system_type() -> Literal["laptop", "desktop", "unknown"]:
    """Detect if system is a laptop or desktop.

    Returns:
        'laptop' if battery present, 'desktop' otherwise.
    """
    battery_paths = [
        Path("/sys/class/power_supply/BAT0"),
        Path("/sys/class/power_supply/BAT1"),
        Path("/sys/class/power_supply/BAT"),
    ]

    for path in battery_paths:
        if path.exists():
            return "laptop"

    if Path("/sys/class/power_supply").exists():
        for path in Path("/sys/class/power_supply").iterdir():
            if path.name.startswith("BAT"):
                return "laptop"

    return "desktop"


def is_hybrid_system() -> bool:
    """Check if system has hybrid graphics (iGPU + dGPU).

    Returns:
        True if hybrid system detected, False otherwise.
    """
    nvidia_gpus = detect_all_gpus()
    igpu_type, _ = detect_integrated_gpu()

    return len(nvidia_gpus) > 0 and igpu_type is not None


def get_native_tool(distro_id: str) -> tuple[str | None, str | None, bool]:
    """Detect which native hybrid graphics tool is available.

    Args:
        distro_id: Distribution ID from detect_distro().

    Returns:
        Tuple of (tool_name, check_method, needs_install) where:
        - tool_name: The native tool name, or None if not available
        - check_method: Method to check the tool
        - needs_install: True if service package needs to be installed
    """
    tools = {
        "ubuntu": ("nvidia-prime", "prime-select"),
        "pop": ("nvidia-prime", "prime-select"),
        "linuxmint": ("nvidia-prime", "prime-select"),
        "fedora": ("switcherooctl", "switcherooctl"),
        "rhel": ("switcherooctl", "switcherooctl"),
        "centos": ("switcherooctl", "switcherooctl"),
        "rocky": ("switcherooctl", "switcherooctl"),
        "alma": ("switcherooctl", "switcherooctl"),
        "opensuse": ("switcherooctl", "switcherooctl"),
        "sles": ("switcherooctl", "switcherooctl"),
        "debian": ("primes", "glxinfo"),
        "arch": (None, None),
        "manjaro": (None, None),
        "cachyos": ("cachyos-settings", "cachyos-settings"),
        "endeavouros": (None, None),
    }

    tool_info = tools.get(distro_id, (None, None))
    tool_name, check_method = tool_info

    if tool_name is None:
        return (None, None, False)

    needs_install = False
    if distro_id in ("ubuntu", "pop", "linuxmint"):
        if _command_exists("prime-select"):
            return ("nvidia-prime", check_method, False)
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma", "opensuse", "sles"):
        if _command_exists("switcherooctl"):
            service_package = TOOL_SERVICE_PACKAGES.get("switcherooctl")
            if service_package and not is_service_installed(service_package):
                needs_install = True
            return ("switcherooctl", check_method, needs_install)
    elif distro_id == "debian":
        if _command_exists("glxinfo"):
            return ("primes", check_method, False)
    elif distro_id == "cachyos":
        if _command_exists("cachyos-settings"):
            return ("cachyos-settings", check_method, False)
        if _command_exists("switcherooctl"):
            service_package = TOOL_SERVICE_PACKAGES.get("switcherooctl")
            if service_package and not is_service_installed(service_package):
                needs_install = True
            return ("switcherooctl", check_method, needs_install)

    return (None, None, False)


def _command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH.

    Args:
        cmd: Command name to check.

    Returns:
        True if command exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_service_installed(package: str) -> bool:
    """Check if a service/package is installed.

    Args:
        package: Package name to check.

    Returns:
        True if package is installed, False otherwise.
    """
    if Path("/usr/bin/rpm").exists():
        try:
            result = subprocess.run(
                ["rpm", "-q", package],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            pass

    if Path("/usr/bin/dpkg-query").exists():
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Status}", package],
                capture_output=True,
                text=True,
                check=False,
            )
            return "install ok installed" in result.stdout
        except Exception:
            pass

    if Path("/usr/bin/pacman").exists():
        try:
            result = subprocess.run(
                ["pacman", "-Q", package],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            pass

    if Path("/usr/bin/rpm").exists():
        try:
            result = subprocess.run(
                ["rpm", "-q", package],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            pass

    return False


def detect_hybrid(distro_id: str) -> HybridInfo | None:
    """Detect hybrid graphics configuration.

    Args:
        distro_id: Distribution ID from detect_distro().

    Returns:
        HybridInfo if hybrid system detected, None otherwise.
    """
    if not is_hybrid_system():
        return None

    gpus = detect_all_gpus()
    igpu_type, igpu_model = detect_integrated_gpu()
    system_type = detect_system_type()
    native_tool, _, needs_install = get_native_tool(distro_id)

    dgpu_model = gpus[0].name if gpus else None

    available_modes = _get_available_modes(native_tool, distro_id)

    return HybridInfo(
        is_hybrid=True,
        igpu_type=igpu_type,
        igpu_model=igpu_model,
        dgpu_model=dgpu_model,
        dgpu_count=len(gpus),
        system_type=system_type,
        native_tool=native_tool,
        needs_install=needs_install,
        available_modes=available_modes,
    )


def _get_available_modes(native_tool: str | None, distro_id: str) -> list[str]:
    """Get available power modes for the hybrid system.

    Args:
        native_tool: Name of native tool available.
        distro_id: Distribution ID.

    Returns:
        List of available power modes.
    """
    modes_map = {
        "nvidia-prime": ["intel", "on-demand", "nvidia"],
        "switcherooctl": ["intel", "nvidia"],
        "system76-power": ["integrated", "hybrid", "nvidia", "compute"],
    }
    if native_tool:
        return modes_map.get(native_tool, ["hybrid", "intel", "nvidia"])
    return ["hybrid", "intel", "nvidia"]
