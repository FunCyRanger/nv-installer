"""Permission and privilege management utilities."""

import os
import subprocess

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)

_sudo_cached: bool | None = None


def is_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0


def have_sudo() -> bool:
    """Check if sudo is cached/available without prompting."""
    global _sudo_cached
    if _sudo_cached is not None:
        return _sudo_cached

    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            timeout=5,
        )
        _sudo_cached = result.returncode == 0
        return _sudo_cached
    except Exception:
        _sudo_cached = False
        return False


def require_root(interactive: bool = True) -> bool:
    """Ensure we have root privileges.

    Caches the result so multiple operations don't prompt multiple times.
    Sudo credentials are cached for ~5 minutes by default.

    Args:
        interactive: If True, prompt user for sudo. If False, return False.

    Returns:
        True if we have root, False otherwise.
    """
    global _sudo_cached

    if is_root():
        return True

    if _sudo_cached:
        return True

    if not interactive:
        return False

    print("\n[INFO] Root privileges required for this operation.")
    try:
        result = subprocess.run(
            ["sudo", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            _sudo_cached = True
            logger.info("Acquired root via sudo")
            return True
        print("[ERROR] Failed to acquire root privileges")
        return False
    except Exception as e:
        logger.error(f"Sudo failed: {e}")
        print("[ERROR] Could not acquire root privileges")
        return False


def check_root_required(operation: str = "this operation") -> int:
    """Check for root and exit with helpful message if missing.

    Args:
        operation: Description of the operation requiring root.

    Returns:
        0 if root available, exits with 1 otherwise.
    """
    if require_root(interactive=False):
        return 0

    print(f"\n[ERROR] Root privileges required for {operation}.")
    print("\nPlease run with: sudo nvidia-inst")
    print("\nOr for dry-run (no changes): nvidia-inst --dry-run")
    return 1
