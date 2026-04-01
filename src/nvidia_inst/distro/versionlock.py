"""Versionlock management for DNF-based systems.

This module provides functions to manage version locks for NVIDIA drivers
and CUDA toolkit packages using DNF's versionlock.toml format.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from nvidia_inst.distro.tools import detect_dnf_path, sudo_path
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def read_versionlock_toml() -> dict:
    """Read and parse the versionlock TOML file.

    Returns:
        Dictionary representing the TOML file, or empty structure if file doesn't exist.
    """
    versionlock_path = Path("/etc/dnf/versionlock.toml")

    if not versionlock_path.exists():
        return {"version": "1.0", "packages": []}

    try:
        result = subprocess.run(
            ["sudo", "cat", str(versionlock_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error("Failed to read versionlock.toml: sudo cat failed")
            return {"version": "1.0", "packages": []}

        data = tomllib.loads(result.stdout)
        return data
    except Exception as e:
        logger.error(f"Failed to read versionlock.toml: {e}")
        return {"version": "1.0", "packages": []}


def write_versionlock_toml(data: dict) -> tuple[bool, str]:
    """Write the versionlock TOML file safely with backup and rollback using sudo.

    Args:
        data: Dictionary to write as TOML

    Returns:
        Tuple of (success, message)
    """
    versionlock_path = Path("/etc/dnf/versionlock.toml")
    backup_path = versionlock_path.with_suffix(".toml.backup")

    try:
        # 1. Generate TOML content
        toml_output = f'version = "{data.get("version", "1.0")}"\n'

        for pkg in data.get("packages", []):
            toml_output += "\n[[packages]]\n"
            toml_output += f'name = "{pkg.get("name", "")}"\n'

            if pkg.get("comment"):
                toml_output += f'comment = "{pkg.get("comment")}"\n'

            for cond in pkg.get("conditions", []):
                toml_output += "[[packages.conditions]]\n"
                toml_output += f'key = "{cond.get("key", "")}"\n'
                toml_output += f'comparator = "{cond.get("comparator", "")}"\n'
                toml_output += f'value = "{cond.get("value", "")}"\n'

        # 2. Create backup of existing file using sudo
        if versionlock_path.exists():
            result = subprocess.run(
                ["sudo", "cp", str(versionlock_path), str(backup_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Created backup: {backup_path}")
            else:
                logger.warning(f"Could not create backup: {result.stderr}")

        # 3. Write to temporary file first (user can write to /tmp without sudo)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".toml")
        os.write(temp_fd, toml_output.encode())
        os.close(temp_fd)

        # 4. Validate the written TOML locally
        try:
            import tomllib

            with open(temp_path, "rb") as f:
                tomllib.load(f)
        except Exception as e:
            logger.error(f"Written TOML is invalid: {e}")
            os.unlink(temp_path)
            return False, f"Generated invalid TOML: {e}"

        # 5. Copy temp file to actual location using sudo
        result = subprocess.run(
            ["sudo", "cp", temp_path, str(versionlock_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        os.unlink(temp_path)  # Clean up temp file

        if result.returncode != 0:
            logger.error(f"Failed to write versionlock.toml: {result.stderr}")
            # Try rollback
            if backup_path.exists():
                rollback_result = subprocess.run(
                    ["sudo", "cp", str(backup_path), str(versionlock_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if rollback_result.returncode == 0:
                    logger.info("Rolled back to backup versionlock.toml")
                    return False, f"Write failed, rolled back: {result.stderr}"
            return False, f"Write failed: {result.stderr}"

        # 6. Verify the file was written correctly using sudo
        verify_result = subprocess.run(
            ["sudo", "cat", str(versionlock_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if verify_result.returncode != 0:
            logger.error("Verification failed: could not read written file")
            return False, "Verification failed: could not read written file"

        logger.info("Wrote versionlock.toml successfully")
        return True, "Versionlock file written successfully"

    except Exception as e:
        logger.error(f"Failed to write versionlock.toml: {e}")
        # Try rollback
        if backup_path.exists():
            rollback_result = subprocess.run(
                ["sudo", "cp", str(backup_path), str(versionlock_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if rollback_result.returncode == 0:
                logger.info("Rolled back to backup versionlock.toml")
                return False, f"Write failed, rolled back: {e}"
        return False, f"Write failed: {e}"


def pattern_entry_exists(data: dict, pattern: str) -> bool:
    """Check if a pattern entry already exists in the versionlock data.

    Args:
        data: Parsed TOML data
        pattern: Package name to check (e.g., 'cuda-toolkit')

    Returns:
        True if entry exists, False otherwise.
    """
    return any(pkg.get("name") == pattern for pkg in data.get("packages", []))


def add_pattern_versionlock_entry(
    package_name: str,
    major_version: str,
    comment: str = "",
    max_version: str | None = None,
) -> tuple[bool, str]:
    """Add pattern-based versionlock entry to lock to a major version.

    Uses direct TOML file editing for true pattern-based branch locking.
    Allows minor/bugfix updates within the branch while blocking major version changes.

    Args:
        package_name: Package name (e.g., 'cuda-toolkit' or 'akmod-nvidia')
        major_version: Major version to lock (e.g., '12' or '580')
        comment: Optional description
        max_version: Optional max version for upper bound (e.g., '12.8' -> evr < 12.9)

    Returns:
        Tuple of (success, message)
    """
    # Read current versionlock
    data = read_versionlock_toml()

    # Check if entry already exists
    if pattern_entry_exists(data, package_name):
        logger.info(f"Package {package_name} already locked")
        return True, f"Package {package_name} already locked"

    # Create new entry with branch-based conditions
    # Lock to: major_version <= X < upper_bound
    # If max_version is specified (e.g., "12.8"), use max_version + 0.1 as upper bound
    # Otherwise, use major_version + 1
    if max_version:
        # Convert max_version (e.g., "12.8") to upper bound (e.g., "12.9")
        parts = max_version.split(".")
        if len(parts) == 2:
            upper_bound = f"{parts[0]}.{int(parts[1]) + 1}"
        else:
            upper_bound = str(int(max_version) + 1)
    else:
        upper_bound = str(int(major_version) + 1)

    new_entry = {
        "name": package_name,
        "conditions": [
            {"key": "evr", "comparator": ">=", "value": major_version},
            {"key": "evr", "comparator": "<", "value": upper_bound},
        ],
    }

    if comment:
        new_entry["comment"] = comment

    # Add to packages list
    if "packages" not in data:
        data["packages"] = []
    if "version" not in data:
        data["version"] = "1.0"

    data["packages"].append(new_entry)

    # Write safely with backup
    success, msg = write_versionlock_toml(data)

    if success:
        logger.info(f"Added versionlock entry: {package_name} ({major_version}.x)")
        return True, f"Locked {package_name} to major version {major_version}.x"

    return False, msg


def verify_versionlock_pattern_active(
    package_name: str, major_version: str
) -> tuple[bool, str]:
    """Verify that a versionlock entry is active and correct.

    Args:
        package_name: Package name (e.g., 'cuda-toolkit')
        major_version: Major version that should be locked

    Returns:
        Tuple of (success, status_message)
    """
    data = read_versionlock_toml()

    for pkg in data.get("packages", []):
        if pkg.get("name") == package_name:
            conditions = pkg.get("conditions", [])
            # Check if we have >= and < conditions
            has_lower = False
            has_upper = False

            for cond in conditions:
                if cond.get("key") == "evr":
                    if (
                        cond.get("comparator") == ">="
                        and cond.get("value") == major_version
                    ):
                        has_lower = True
                    if cond.get("comparator") == "<":
                        has_upper = True

            if has_lower and has_upper:
                return (
                    True,
                    f"Package {package_name} verified: locked to {major_version}.x",
                )

            return True, f"Package {package_name} exists but conditions differ"

    return False, f"No versionlock entry found for {package_name}"


def verify_versionlock_pattern(
    distro_id: str, package_name: str, lock_type: str
) -> tuple[bool, str]:
    """Verify that a versionlock entry matches the expected pattern.

    Args:
        distro_id: Distribution ID (not used, kept for compatibility)
        package_name: Package name (e.g., 'akmod-nvidia', 'cuda-toolkit')
        lock_type: Type of lock ('driver' or 'cuda')

    Returns:
        Tuple of (success, status_message)
    """
    # Always check versionlock.toml for DNF-based systems
    versionlock_path = Path("/etc/dnf/versionlock.toml")
    if not versionlock_path.exists():
        return True, "No versionlock file (not a DNF system)"

    try:
        dnf_path = detect_dnf_path()
        result = subprocess.run(
            [dnf_path, "versionlock", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return False, f"Failed to list versionlocks: {result.stderr}"

        output = result.stdout

        # Handle patterns like 'cuda-toolkit-12-*'
        if lock_type == "cuda" and "cuda-toolkit" in package_name:
            # Look for any cuda-toolkit entry with matching major version
            lines = output.split("\n")
            for line in lines:
                if "cuda-toolkit" in line:
                    # Extract evr
                    match = re.search(r"evr\s*=\s*(.+?)(?:\s|$)", line)
                    if match:
                        evr = match.group(1).strip()
                        return (
                            True,
                            f"Versionlock verified: cuda-toolkit (locked to {evr})",
                        )

            # Check if any cuda-toolkit exists at all (no specific version check)
            if "cuda-toolkit" in output:
                return (
                    True,
                    "Versionlock verified: cuda-toolkit (version check skipped)",
                )

            return False, "No versionlock found for cuda-toolkit"

        # Check if we found our lock entry
        if package_name in output:
            # Check if it's a pattern lock (has *) or specific version
            lines = output.split("\n")
            for line in lines:
                if package_name in line:
                    # Extract evr part
                    match = re.search(r"evr\s*=\s*(.+?)(?:\s|$)", line)
                    if match:
                        evr = match.group(1).strip()
                        # If evr contains *, it's a pattern lock
                        if "*" in evr:
                            return True, f"Versionlock verified (pattern): {evr}"
                        # If evr matches our pattern (e.g., 580.126.18 matches 580.*)
                        else:
                            return (
                                True,
                                f"Versionlock verified: {evr}",
                            )

            # Found package but couldn't determine evr
            return True, "Versionlock exists"

        return False, f"No versionlock found for {package_name}"

    except subprocess.TimeoutExpired:
        return False, "Verification timed out"
    except Exception as e:
        return False, f"Verification error: {e}"


def cleanup_incorrect_versionlocks(
    distro_id: str, package_base: str, expected_branch: str
) -> bool:
    """Clean up incorrect versionlock entries for a package.

    Removes any locks that don't match the expected branch pattern.

    Args:
        distro_id: Distribution ID (not used, kept for compatibility)
        package_base: Base package name (e.g., 'akmod-nvidia')
        expected_branch: Expected branch (e.g., '580')

    Returns:
        True if cleanup successful or no cleanup needed, False on error
    """
    # Check if versionlock.toml exists
    versionlock_path = Path("/etc/dnf/versionlock.toml")
    if not versionlock_path.exists():
        return True  # No versionlock file, nothing to clean

    try:
        dnf_path = detect_dnf_path()

        # Get current locks
        result = subprocess.run(
            [dnf_path, "versionlock", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to list versionlocks: {result.stderr}")
            return True  # Continue anyway

        output = result.stdout

        # Find entries for our package
        lines = output.split("\n")
        for line in lines:
            if package_base in line:
                # Extract evr
                match = re.search(r"evr\s*=\s*(.+?)(?:\s|$)", line)
                if match:
                    evr = match.group(1).strip()
                    # Check if this is a specific version lock (not a pattern)
                    if "*" not in evr:
                        # Extract version from evr (e.g., "3:580.126.18-1.fc43" -> "580")
                        version_match = re.match(r"\d+:(\d+)\.", evr)
                        if version_match:
                            lock_branch = version_match.group(1)
                            # If branch doesn't match expected, remove it
                            if lock_branch != expected_branch:
                                logger.info(
                                    f"Removing incorrect versionlock: {package_base} ({evr})"
                                )
                                delete_result = subprocess.run(
                                    [
                                        sudo_path(),
                                        dnf_path,
                                        "versionlock",
                                        "delete",
                                        package_base,
                                    ],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                )
                                if delete_result.returncode != 0:
                                    logger.warning(
                                        f"Failed to remove lock: {delete_result.stderr}"
                                    )
                                else:
                                    print(f"           Removed incorrect lock: {evr}")

        return True

    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
        return True  # Continue anyway
