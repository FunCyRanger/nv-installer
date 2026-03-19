"""Secure Boot support for NVIDIA driver installation.

Handles MOK (Machine Owner Key) generation, enrollment, and module signing
for systems with Secure Boot enabled.
"""

import glob
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


class SecureBootState(Enum):
    """Secure Boot state enumeration."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    SETUP_MODE = "setup_mode"
    UNKNOWN = "unknown"


class SecureBootError(Exception):
    """Raised when Secure Boot operations fail."""

    pass


class MokutilNotFoundError(SecureBootError):
    """Raised when mokutil is not installed."""

    pass


@dataclass
class MokKeyPaths:
    """Paths for MOK keys and certificates."""

    private_key: Path
    public_cert: Path


@dataclass
class SecureBootResult:
    """Result of a Secure Boot operation."""

    success: bool
    message: str
    requires_reboot: bool = False
    reboot_instructions: str | None = None


def get_secure_boot_state() -> SecureBootState:
    """Check Secure Boot state.

    Returns:
        SecureBootState: The current Secure Boot state.
    """
    try:
        result = subprocess.run(
            ["mokutil", "--sb-state"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.lower()

        if "enabled" in output:
            return SecureBootState.ENABLED
        elif "setup mode" in output:
            return SecureBootState.SETUP_MODE
        else:
            return SecureBootState.DISABLED

    except FileNotFoundError:
        logger.warning("mokutil not found - Secure Boot management unavailable")
        return SecureBootState.UNKNOWN
    except subprocess.TimeoutExpired:
        logger.warning("mokutil timed out")
        return SecureBootState.UNKNOWN
    except Exception as e:
        logger.warning(f"Failed to check Secure Boot state: {e}")
        return SecureBootState.UNKNOWN


def check_mokutil_available() -> bool:
    """Check if mokutil is available.

    Returns:
        True if mokutil is installed, False otherwise.
    """
    try:
        subprocess.run(
            ["mokutil", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def get_mok_key_paths(distro_id: str) -> MokKeyPaths:
    """Get default MOK key paths for a distribution.

    Args:
        distro_id: Distribution ID (e.g., 'ubuntu', 'fedora').

    Returns:
        MokKeyPaths with default key locations.
    """
    if distro_id in ("ubuntu", "linuxmint", "pop"):
        return MokKeyPaths(
            private_key=Path("/var/lib/shim-signed/mok/MOK.priv"),
            public_cert=Path("/var/lib/shim-signed/mok/MOK.der"),
        )
    elif distro_id in ("fedora", "rhel", "centos", "rocky", "alma", "opensuse", "sles"):
        return MokKeyPaths(
            private_key=Path("/etc/pki/akmods/private/private_key.priv"),
            public_cert=Path("/etc/pki/akmods/certs/public_key.der"),
        )
    elif distro_id in ("arch", "manjaro", "endeavouros"):
        return MokKeyPaths(
            private_key=Path("/etc/secureboot/keys/MOK.priv"),
            public_cert=Path("/etc/secureboot/keys/MOK.der"),
        )
    elif distro_id == "debian":
        return MokKeyPaths(
            private_key=Path("/var/lib/dkms/mok.key"),
            public_cert=Path("/var/lib/dkms/mok.pub"),
        )
    else:
        return MokKeyPaths(
            private_key=Path("/etc/secureboot/mok/MOK.priv"),
            public_cert=Path("/etc/secureboot/mok/MOK.der"),
        )


def is_mok_enrolled(cert_path: Path) -> bool:
    """Check if a MOK certificate is already enrolled.

    Args:
        cert_path: Path to the public certificate.

    Returns:
        True if the key is enrolled, False otherwise.
    """
    if not cert_path.exists():
        return False

    try:
        result = subprocess.run(
            ["mokutil", "--test-key", str(cert_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "already enrolled" in result.stdout.lower()

    except Exception as e:
        logger.warning(f"Failed to check MOK enrollment: {e}")
        return False


def generate_mok_key(
    key_dir: Path,
    key_name: str = "MOK",
    key_bits: int = 2048,
    validity_days: int = 36500,
) -> MokKeyPaths:
    """Generate a new MOK key pair.

    Args:
        key_dir: Directory to store keys.
        key_name: Base name for key files.
        key_bits: RSA key size (default 2048).
        validity_days: Key validity in days.

    Returns:
        MokKeyPaths with paths to generated keys.

    Raises:
        SecureBootError: If key generation fails.
    """
    key_dir.mkdir(parents=True, exist_ok=True)

    private_key = key_dir / f"{key_name}.priv"
    public_cert = key_dir / f"{key_name}.der"

    subject = f"/CN=nvidia-inst-{key_name}/"

    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-x509",
                "-newkey",
                f"rsa:{key_bits}",
                "-keyout",
                str(private_key),
                "-outform",
                "DER",
                "-out",
                str(public_cert),
                "-nodes",
                "-days",
                str(validity_days),
                "-subj",
                subject,
            ],
            check=True,
            timeout=30,
        )

        private_key.chmod(0o600)
        public_cert.chmod(0o644)

        logger.info(f"Generated MOK key pair in {key_dir}")
        return MokKeyPaths(private_key=private_key, public_cert=public_cert)

    except subprocess.CalledProcessError as e:
        raise SecureBootError(f"Failed to generate MOK key: {e}") from e
    except FileNotFoundError as e:
        raise SecureBootError("openssl not found - please install openssl") from e


def enroll_mok_key(
    cert_path: Path,
    password: str | None = None,
) -> SecureBootResult:
    """Enroll a MOK certificate.

    Args:
        cert_path: Path to the public certificate (DER format).
        password: Optional password for MOK Manager (prompts if not provided).

    Returns:
        SecureBootResult with enrollment status and instructions.
    """
    if not cert_path.exists():
        return SecureBootResult(
            success=False,
            message=f"Certificate not found: {cert_path}",
        )

    state = get_secure_boot_state()

    if state == SecureBootState.DISABLED:
        return SecureBootResult(
            success=True,
            message="Secure Boot is disabled - no MOK enrollment needed",
        )

    if state == SecureBootState.UNKNOWN:
        return SecureBootResult(
            success=False,
            message="Cannot determine Secure Boot state - mokutil may not be available",
        )

    try:
        cmd = ["mokutil", "--import", str(cert_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0 and "already enrolled" not in result.stdout.lower():
            return SecureBootResult(
                success=False,
                message=f"MOK enrollment failed: {result.stderr or result.stdout}",
            )

        if state == SecureBootState.SETUP_MODE:
            return SecureBootResult(
                success=True,
                message="MOK key enrolled successfully (Setup Mode - no reboot required)",
                requires_reboot=False,
            )

        return SecureBootResult(
            success=True,
            message="MOK key enrollment scheduled for next reboot",
            requires_reboot=True,
            reboot_instructions=(
                "On next boot, the MOK Manager will appear:\n"
                "  1. Select 'Enroll MOK'\n"
                "  2. Select 'Continue'\n"
                "  3. Select 'Yes' to confirm\n"
                "  4. Enter the password you set\n"
                "  5. Select 'Reboot'\n"
                "After reboot, your MOK will be enrolled and modules can be signed."
            ),
        )

    except subprocess.TimeoutExpired:
        return SecureBootResult(
            success=False,
            message="MOK enrollment timed out",
        )
    except FileNotFoundError:
        return SecureBootResult(
            success=False,
            message="mokutil not found - please install mokutil package",
        )
    except Exception as e:
        return SecureBootResult(
            success=False,
            message=f"MOK enrollment failed: {e}",
        )


def get_sign_file_path(kernel_version: str | None = None) -> Path | None:
    """Find the kernel sign-file script.

    Args:
        kernel_version: Kernel version (uses running kernel if None).

    Returns:
        Path to sign-file script, or None if not found.
    """
    if kernel_version is None:
        kernel_version = os.uname().release

    search_paths = [
        f"/usr/src/linux-headers-{kernel_version}/scripts/sign-file",
        f"/lib/modules/{kernel_version}/build/scripts/sign-file",
        f"/usr/src/kernels/{kernel_version}/scripts/sign-file",
    ]

    for path in search_paths:
        p = Path(path)
        if p.exists():
            return p

    return None


def find_nvidia_modules(kernel_version: str | None = None) -> list[Path]:
    """Find NVIDIA kernel modules.

    Args:
        kernel_version: Kernel version (uses running kernel if None).

    Returns:
        List of paths to NVIDIA .ko modules.
    """
    if kernel_version is None:
        kernel_version = os.uname().release

    search_patterns = [
        f"/lib/modules/{kernel_version}/updates/dkms/nvidia*.ko*",
        f"/lib/modules/{kernel_version}/extra/nvidia*.ko*",
        f"/lib/modules/{kernel_version}/kernel/drivers/video/nvidia*.ko*",
    ]

    modules: list[Path] = []
    for pattern in search_patterns:
        modules.extend(Path(p) for p in glob.glob(pattern))

    return sorted(set(modules))


def sign_module(
    module_path: Path,
    private_key: Path,
    public_cert: Path,
    kernel_version: str | None = None,
) -> bool:
    """Sign a single kernel module.

    Args:
        module_path: Path to the .ko module file.
        private_key: Path to private key.
        public_cert: Path to public certificate.
        kernel_version: Kernel version (uses running kernel if None).

    Returns:
        True if signing succeeded, False otherwise.
    """
    sign_file = get_sign_file_path(kernel_version)
    if not sign_file:
        logger.error("sign-file not found in kernel headers")
        return False

    if not module_path.exists():
        logger.error(f"Module not found: {module_path}")
        return False

    if not private_key.exists():
        logger.error(f"Private key not found: {private_key}")
        return False

    if not public_cert.exists():
        logger.error(f"Public certificate not found: {public_cert}")
        return False

    try:
        temp_module = None
        actual_path = module_path

        if str(module_path).endswith(".zst"):
            import tempfile

            temp_fd, temp_module = tempfile.mkstemp(suffix=".ko")
            os.close(temp_fd)

            subprocess.run(
                ["zstd", "-d", str(module_path), "-o", temp_module],
                check=True,
            )
            actual_path = Path(temp_module)

        result = subprocess.run(
            [
                str(sign_file),
                "sha256",
                str(private_key),
                str(public_cert),
                str(actual_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if temp_module:
            subprocess.run(
                ["zstd", "-f", str(actual_path), "-o", str(module_path)],
                check=True,
            )
            Path(temp_module).unlink()

        if result.returncode != 0:
            logger.error(f"Failed to sign {module_path}: {result.stderr}")
            return False

        logger.info(f"Signed module: {module_path.name}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sign module {module_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error signing module {module_path}: {e}")
        return False


def sign_nvidia_modules(
    private_key: Path,
    public_cert: Path,
    kernel_version: str | None = None,
) -> tuple[int, int]:
    """Sign all NVIDIA kernel modules.

    Args:
        private_key: Path to private key.
        public_cert: Path to public certificate.
        kernel_version: Kernel version (uses running kernel if None).

    Returns:
        Tuple of (signed_count, failed_count).
    """
    modules = find_nvidia_modules(kernel_version)

    if not modules:
        logger.warning("No NVIDIA modules found to sign")
        return (0, 0)

    signed = 0
    failed = 0

    for module in modules:
        if sign_module(module, private_key, public_cert, kernel_version):
            signed += 1
        else:
            failed += 1

    return (signed, failed)


def setup_dkms_hook(
    signing_script: Path,
    distro_id: str,
) -> bool:
    """Set up DKMS post-build hook for automatic signing.

    Args:
        signing_script: Path to the signing script.
        distro_id: Distribution ID.

    Returns:
        True if hook was created successfully.
    """
    if distro_id not in ("ubuntu", "linuxmint", "pop", "debian"):
        logger.debug(f"DKMS hook not needed for {distro_id}")
        return True

    hook_dir = Path("/etc/dkms/post-build.d")
    hook_path = hook_dir / "zz-nvidia-sign"

    try:
        hook_dir.mkdir(parents=True, exist_ok=True)

        hook_content = f"""#!/bin/bash
# Auto-generated by nvidia-inst for Secure Boot signing
# Signs NVIDIA modules after DKMS builds

KERNEL_VERSION="${{1:-$(_uname -r)}}"
"{signing_script}" "$KERNEL_VERSION"
"""
        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)

        logger.info(f"Created DKMS hook: {hook_path}")
        return True

    except PermissionError:
        logger.warning(f"Cannot create DKMS hook (need root): {hook_path}")
        return False
    except Exception as e:
        logger.warning(f"Failed to create DKMS hook: {e}")
        return False


def setup_pacman_hook(
    signing_script: Path,
) -> bool:
    """Set up pacman hook for automatic signing on Arch Linux.

    Args:
        signing_script: Path to the signing script.

    Returns:
        True if hook was created successfully.
    """
    hooks_dir = Path("/etc/pacman.d/hooks")
    hook_path = hooks_dir / "nvidia-sign.hook"

    try:
        hooks_dir.mkdir(parents=True, exist_ok=True)

        hook_content = f"""[Trigger]
Type = Package
Operation = Install
Operation = Upgrade
Target = nvidia*

[Action]
Description = Signing NVIDIA modules for Secure Boot...
When = PostTransaction
Exec = {signing_script}
Depends = openssl
"""
        hook_path.write_text(hook_content)

        logger.info(f"Created pacman hook: {hook_path}")
        return True

    except PermissionError:
        logger.warning(f"Cannot create pacman hook (need root): {hook_path}")
        return False
    except Exception as e:
        logger.warning(f"Failed to create pacman hook: {e}")
        return False


def install_signing_script(
    script_path: Path,
    private_key: Path,
    public_cert: Path,
) -> bool:
    """Install the NVIDIA module signing script.

    Args:
        script_path: Where to install the script.
        public_key: Path to public certificate.

    Returns:
        True if script was installed successfully.
    """
    script_content = f"""#!/usr/bin/env bash
# Auto-generated by nvidia-inst for Secure Boot signing
# Signs NVIDIA kernel modules after kernel/driver updates

set -euo pipefail

KVER="${{1:-$(uname -r)}}"
MOK_PRIV="{private_key}"
MOK_CERT="{public_cert}"

SIGN_FILE=$(find /usr/src/linux-headers-"$KVER" /lib/modules/"$KVER"/build -name sign-file 2>/dev/null | head -1)

if [ -z "$SIGN_FILE" ]; then
    echo "Error: sign-file not found for kernel $KVER" >&2
    exit 1
fi

if [ ! -f "$MOK_PRIV" ] || [ ! -f "$MOK_CERT" ]; then
    echo "Error: MOK keys not found" >&2
    exit 1
fi

for dir in "/lib/modules/$KVER/updates/dkms" "/lib/modules/$KVER/extra"; do
    [ -d "$dir" ] || continue

    for mod in "$dir"/nvidia*.ko*; do
        [ -f "$mod" ] || continue

        if [[ "$mod" == *.zst ]]; then
            temp_dir=$(mktemp -d)
            temp_mod="$temp_dir/$(basename "$mod" .zst)"
            zstd -d "$mod" -o "$temp_mod"
            "$SIGN_FILE" sha256 "$MOK_PRIV" "$MOK_CERT" "$temp_mod"
            zstd -f "$temp_mod" -o "$mod"
            rm -rf "$temp_dir"
        else
            "$SIGN_FILE" sha256 "$MOK_PRIV" "$MOK_CERT" "$mod"
        fi

        echo "Signed: $mod"
    done
done
"""

    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        logger.info(f"Installed signing script: {script_path}")
        return True

    except PermissionError:
        logger.warning(f"Cannot install signing script (need root): {script_path}")
        return False
    except Exception as e:
        logger.warning(f"Failed to install signing script: {e}")
        return False


def setup_auto_signing(
    private_key: Path,
    public_cert: Path,
    distro_id: str,
    script_dir: Path = Path("/usr/local/bin"),
) -> SecureBootResult:
    """Set up automatic signing for future kernel/driver updates.

    Args:
        private_key: Path to private key.
        public_cert: Path to public certificate.
        distro_id: Distribution ID.
        script_dir: Directory for signing script.

    Returns:
        SecureBootResult with setup status.
    """
    signing_script = script_dir / "sign-nvidia-modules"

    if not install_signing_script(signing_script, private_key, public_cert):
        return SecureBootResult(
            success=False,
            message="Failed to install signing script",
        )

    hook_success = True
    if distro_id in ("ubuntu", "linuxmint", "pop", "debian"):
        hook_success = setup_dkms_hook(signing_script, distro_id)
    elif distro_id in ("arch", "manjaro", "endeavouros"):
        hook_success = setup_pacman_hook(signing_script)

    if not hook_success:
        return SecureBootResult(
            success=True,
            message="Signing script installed, but auto-signing hooks require manual setup",
        )

    return SecureBootResult(
        success=True,
        message="Automatic signing set up successfully",
    )


def disable_secure_boot_validation() -> bool:
    """Disable Secure Boot validation via mokutil.

    This is an alternative to MOK enrollment - disables the shim validation.

    Returns:
        True if disabled successfully, False otherwise.
    """
    try:
        result = subprocess.run(
            ["mokutil", "--disable-validation"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to disable Secure Boot validation: {e}")
        return False
