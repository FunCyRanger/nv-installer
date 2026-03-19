import glob
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SafetyCheckResult:
    can_proceed: bool
    warnings: list[str]
    errors: list[str]


@dataclass
class ValidationResult:
    success: bool
    installed_packages: list[str]
    missing_packages: list[str]
    kernel_module_built: bool
    nouveau_blocked: bool
    nvidia_smi_works: bool
    actual_driver_version: str | None
    warnings: list[str]
    errors: list[str]


def unblock_nouveau() -> tuple[bool, str]:
    """Remove Nouveau blacklist to ensure bootable system.

    Returns:
        Tuple of (success, message).
    """
    blacklist_file = Path("/etc/modprobe.d/blacklist-nouveau.conf")

    if not blacklist_file.exists():
        return True, "Nouveau blacklist does not exist"

    try:
        blacklist_file.unlink()
        return True, "Nouveau re-enabled"
    except Exception as e:
        return False, str(e)


def pre_install_check(distro_id: str, packages: list[str]) -> SafetyCheckResult:
    """Run pre-installation safety checks."""
    result = SafetyCheckResult(can_proceed=True, warnings=[], errors=[])

    # 1. Check disk space (500MB free in /var)
    try:
        stat = os.statvfs("/var")
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        if free_mb < 500:
            result.warnings.append(
                f"Low disk space: {free_mb:.0f}MB free (500MB recommended)"
            )
    except Exception:
        pass

    # 2. Check package availability
    available = _check_packages_available(distro_id, packages)
    if not available:
        result.errors.append("Required packages not available in repos")
        result.can_proceed = False

    # 3. Check kernel build deps
    if not _check_kernel_devel():
        result.warnings.append(
            "Kernel development packages missing - akmod may fail to build"
        )

    # 4. Check Secure Boot
    if _check_secure_boot():
        result.warnings.append("Secure Boot is enabled - driver may need signing")

    # 5. Check running environment
    if os.environ.get("DISPLAY"):
        result.warnings.append(
            "Running in graphical session - recommend running from tty or SSH"
        )

    return result


def post_install_validate(
    distro_id: str, expected_packages: list[str]
) -> ValidationResult:
    """Run post-installation validation."""
    result = ValidationResult(
        success=True,
        installed_packages=[],
        missing_packages=[],
        kernel_module_built=False,
        nouveau_blocked=False,
        nvidia_smi_works=False,
        actual_driver_version=None,
        warnings=[],
        errors=[],
    )

    # 1. Check packages installed
    try:
        for pkg in expected_packages:
            check = subprocess.run(
                ["rpm", "-q", pkg],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if check.returncode == 0:
                result.installed_packages.append(pkg)
            else:
                result.missing_packages.append(pkg)

        if result.missing_packages:
            result.warnings.append(
                f"Packages not installed: {', '.join(result.missing_packages)}"
            )
    except Exception as e:
        result.errors.append(f"Could not verify packages: {e}")

    # 2. Check akmod built kernel module
    try:
        kernel_version = os.uname().release
        module_pattern = f"/lib/modules/{kernel_version}/extra/nvidia*.ko*"
        modules = glob.glob(module_pattern)
        result.kernel_module_built = len(modules) > 0

        if not result.kernel_module_built:
            result.warnings.append(
                "Kernel module not built yet (may need reboot or akmod failed)"
            )
    except Exception:
        result.warnings.append("Could not verify kernel module")

    # 3. Check nouveau blocked
    result.nouveau_blocked = Path("/etc/modprobe.d/blacklist-nouveau.conf").exists()
    if not result.nouveau_blocked:
        result.warnings.append(
            "Nouveau is not blocked - may conflict with nvidia driver"
        )

    # 4. Test nvidia-smi
    try:
        check = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result.nvidia_smi_works = check.returncode == 0

        if result.nvidia_smi_works:
            try:
                ver_check = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                result.actual_driver_version = ver_check.stdout.strip()
            except Exception:
                pass
        else:
            result.warnings.append(
                "nvidia-smi not available (driver may work but needs reboot)"
            )
    except FileNotFoundError:
        result.nvidia_smi_works = False
        result.warnings.append("nvidia-smi not found (will be available after reboot)")
    except Exception:
        result.warnings.append("Could not verify nvidia-smi")

    result.success = len(result.errors) == 0 and len(result.missing_packages) == 0
    return result


def _check_packages_available(distro_id: str, packages: list[str]) -> bool:
    """Check if packages are available in repos."""
    try:
        if distro_id in ("fedora", "rhel", "centos", "rocky", "alma"):
            for pkg in packages:
                check = subprocess.run(
                    ["dnf", "repoquery", pkg],
                    capture_output=True,
                    timeout=30,
                )
                if check.returncode != 0:
                    return False
        return True
    except Exception:
        return False


def _check_kernel_devel() -> bool:
    """Check if kernel development packages are installed."""
    try:
        result = subprocess.run(
            ["rpm", "-q", "kernel-devel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "kernel-devel" in result.stdout
    except Exception:
        return False


def _check_secure_boot() -> bool:
    """Check if Secure Boot is enabled."""
    try:
        result = subprocess.run(
            ["mokutil", "--sb-state"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "enabled" in result.stdout.lower()
    except Exception:
        return False


@dataclass
class WorkingInstallResult:
    """Result of checking if NVIDIA is working."""

    is_working: bool
    driver_version: str | None
    kernel_module_loaded: bool
    gpu_detected: bool


def is_nvidia_working() -> WorkingInstallResult:
    """Check if NVIDIA driver is currently working.

    Returns:
        WorkingInstallResult with working status and details.
    """
    result = WorkingInstallResult(
        is_working=False,
        driver_version=None,
        kernel_module_loaded=False,
        gpu_detected=False,
    )

    try:
        smi_check = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi_check.returncode == 0 and smi_check.stdout.strip():
            result.gpu_detected = True
    except Exception:
        return result

    try:
        ver_check = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if ver_check.returncode == 0:
            result.driver_version = ver_check.stdout.strip()
    except Exception:
        pass

    try:
        lsmod_check = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result.kernel_module_loaded = "nvidia" in lsmod_check.stdout
    except Exception:
        pass

    result.is_working = (
        result.gpu_detected
        and result.driver_version is not None
        and result.kernel_module_loaded
    )

    return result
