"""Rollback capability for installation recovery.

This module provides functionality to create snapshots of system state
and rollback to previous states on installation failure.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nvidia_inst.distro.detector import detect_distro
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemState:
    """Snapshot of system state before installation."""

    timestamp: str = ""
    distro_id: str = ""
    installed_packages: list[str] = field(default_factory=list)
    loaded_modules: list[str] = field(default_factory=list)
    driver_version: str | None = None
    cuda_version: str | None = None
    blacklist_files: list[Any] = field(default_factory=list)
    versionlock_entries: list[str] = field(default_factory=list)
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RollbackManager:
    """Manages installation rollback."""

    def __init__(self, state_dir: str = "/var/lib/nvidia-inst/state"):
        """Initialize rollback manager.

        Args:
            state_dir: Directory for state snapshots
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self) -> SystemState:
        """Create snapshot of current system state.

        Returns:
            SystemState snapshot
        """
        try:
            distro = detect_distro()

            state = SystemState(
                distro_id=distro.id,
                installed_packages=self._get_installed_packages(),
                loaded_modules=self._get_loaded_modules(),
                driver_version=self._get_current_driver_version(),
                cuda_version=self._get_current_cuda_version(),
                blacklist_files=self._get_blacklist_files(),
                versionlock_entries=self._get_versionlock_entries(),
            )

            # Calculate checksum
            state.checksum = self._calculate_checksum(state)

            # Save snapshot
            self._save_snapshot(state)

            logger.info(f"Created system snapshot at {state.timestamp}")
            return state

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    def rollback(self, state: SystemState | None = None) -> bool:
        """Rollback to previous system state.

        Args:
            state: State to rollback to (uses latest if None)

        Returns:
            True if rollback successful
        """
        try:
            if state is None:
                state = self._load_latest_snapshot()

            if state is None:
                logger.error("No snapshot available for rollback")
                return False

            logger.info(f"Rolling back to state from {state.timestamp}")

            # Step 1: Remove newly installed packages
            current_packages = self._get_installed_packages()
            new_packages = set(current_packages) - set(state.installed_packages)

            if new_packages:
                logger.info(
                    f"Removing newly installed packages: {', '.join(new_packages)}"
                )
                self._remove_packages(list(new_packages))

            # Step 2: Restore previous packages
            missing_packages = set(state.installed_packages) - set(current_packages)

            if missing_packages:
                logger.info(
                    f"Restoring missing packages: {', '.join(missing_packages)}"
                )
                self._install_packages(list(missing_packages))

            # Step 3: Restore blacklist files
            self._restore_blacklist_files(state.blacklist_files)

            # Step 4: Restore versionlock entries
            self._restore_versionlock_entries(state.versionlock_entries)

            # Step 5: Rebuild initramfs
            logger.info("Rebuilding initramfs...")
            self._rebuild_initramfs()

            logger.info("Rollback completed successfully")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List available snapshots.

        Returns:
            List of snapshot metadata
        """
        snapshots = []

        for state_file in sorted(self.state_dir.glob("state_*.json"), reverse=True):
            try:
                state = self._load_snapshot(state_file)
                if state:
                    snapshots.append(
                        {
                            "timestamp": state.timestamp,
                            "distro_id": state.distro_id,
                            "driver_version": state.driver_version,
                            "cuda_version": state.cuda_version,
                            "file": str(state_file),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load snapshot {state_file}: {e}")

        return snapshots

    def cleanup_old_snapshots(self, keep_count: int = 5) -> int:
        """Remove old snapshots, keeping only the most recent.

        Args:
            keep_count: Number of snapshots to keep

        Returns:
            Number of snapshots removed
        """
        snapshots = sorted(self.state_dir.glob("state_*.json"), reverse=True)

        if len(snapshots) <= keep_count:
            return 0

        removed = 0
        for snapshot_file in snapshots[keep_count:]:
            try:
                snapshot_file.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove snapshot {snapshot_file}: {e}")

        return removed

    def _get_installed_packages(self) -> list[str]:
        """Get list of installed NVIDIA packages."""
        packages = []

        try:
            distro = detect_distro()

            if distro.id in ("ubuntu", "debian", "linuxmint", "pop"):
                # APT-based systems
                result = subprocess.run(
                    ["dpkg-query", "-W", "-f=${Package}\n"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if "nvidia" in line.lower() or "cuda" in line.lower():
                            packages.append(line.strip())

            elif distro.id in ("fedora", "rhel", "centos", "rocky", "alma"):
                # DNF-based systems
                result = subprocess.run(
                    ["rpm", "-qa"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if "nvidia" in line.lower() or "cuda" in line.lower():
                            packages.append(line.strip())

            elif distro.id in ("arch", "manjaro"):
                # Pacman-based systems
                result = subprocess.run(
                    ["pacman", "-Q"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if "nvidia" in line.lower() or "cuda" in line.lower():
                            pkg_name = line.split()[0]
                            packages.append(pkg_name)

            elif distro.id in ("opensuse", "sles"):
                # Zypper-based systems
                result = subprocess.run(
                    ["rpm", "-qa"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if "nvidia" in line.lower() or "cuda" in line.lower():
                            packages.append(line.strip())

        except Exception as e:
            logger.warning(f"Failed to get installed packages: {e}")

        return packages

    def _get_loaded_modules(self) -> list[str]:
        """Get list of loaded NVIDIA kernel modules."""
        modules = []

        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("nvidia") or line.startswith("nouveau"):
                        module_name = line.split()[0]
                        modules.append(module_name)

        except Exception as e:
            logger.warning(f"Failed to get loaded modules: {e}")

        return modules

    def _get_current_driver_version(self) -> str | None:
        """Get current NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

        except Exception:
            pass

        return None

    def _get_current_cuda_version(self) -> str | None:
        """Get current CUDA version."""
        try:
            # Try nvcc first
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line:
                        parts = line.split("release")
                        if len(parts) > 1:
                            version = parts[1].strip().split(",")[0].strip()
                            return version

        except Exception:
            pass

        # Try checking /usr/local/cuda
        cuda_path = Path("/usr/local/cuda/version.txt")
        if cuda_path.exists():
            try:
                content = cuda_path.read_text()
                if "CUDA Version" in content:
                    version = content.split("CUDA Version")[1].strip()
                    return version
            except Exception:
                pass

        return None

    def _get_blacklist_files(self) -> list[Any]:
        """Get list of blacklist configuration files."""
        files: list[Any] = []

        modprobe_dir = Path("/etc/modprobe.d")
        if modprobe_dir.exists():
            for file in modprobe_dir.glob("*nouveau*"):
                try:
                    files.append(
                        {
                            "path": str(file),
                            "content": file.read_text(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read {file}: {e}")

        return files

    def _get_versionlock_entries(self) -> list[str]:
        """Get versionlock entries."""
        entries = []

        try:
            # DNF versionlock
            result = subprocess.run(
                ["dnf", "versionlock", "list"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "nvidia" in line.lower() or "cuda" in line.lower():
                        entries.append(line.strip())

        except Exception:
            pass

        return entries

    def _calculate_checksum(self, state: SystemState) -> str:
        """Calculate checksum for state validation."""
        import hashlib

        data = json.dumps(asdict(state), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _save_snapshot(self, state: SystemState) -> None:
        """Save snapshot to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = self.state_dir / f"state_{timestamp}.json"

        data = asdict(state)

        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved snapshot to {state_file}")

    def _load_snapshot(self, state_file: Path) -> SystemState | None:
        """Load snapshot from file."""
        try:
            with open(state_file) as f:
                data = json.load(f)

            return SystemState(**data)

        except Exception as e:
            logger.warning(f"Failed to load snapshot {state_file}: {e}")
            return None

    def _load_latest_snapshot(self) -> SystemState | None:
        """Load the most recent snapshot."""
        snapshots = sorted(self.state_dir.glob("state_*.json"), reverse=True)

        if not snapshots:
            return None

        return self._load_snapshot(snapshots[0])

    def _remove_packages(self, packages: list[str]) -> None:
        """Remove packages."""
        try:
            pkg_manager = get_package_manager()
            pkg_manager.remove(packages)
        except Exception as e:
            logger.error(f"Failed to remove packages: {e}")
            raise

    def _install_packages(self, packages: list[str]) -> None:
        """Install packages."""
        try:
            pkg_manager = get_package_manager()
            pkg_manager.install(packages)
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            raise

    def _restore_blacklist_files(self, blacklist_files: list[Any]) -> None:
        """Restore blacklist configuration files."""
        modprobe_dir = Path("/etc/modprobe.d")
        modprobe_dir.mkdir(parents=True, exist_ok=True)

        for entry in blacklist_files:
            try:
                if isinstance(entry, dict):
                    file_path = Path(entry["path"])
                    file_path.write_text(entry["content"])
                    logger.debug(f"Restored {file_path}")
            except Exception as e:
                logger.warning(f"Failed to restore blacklist file: {e}")

    def _restore_versionlock_entries(self, entries: list[str]) -> None:
        """Restore versionlock entries."""
        try:
            # Clear existing versionlock
            subprocess.run(
                ["dnf", "versionlock", "clear"],
                capture_output=True,
            )

            # Restore entries
            for entry in entries:
                subprocess.run(
                    ["dnf", "versionlock", "add", entry],
                    capture_output=True,
                )

        except Exception as e:
            logger.warning(f"Failed to restore versionlock: {e}")

    def _rebuild_initramfs(self) -> None:
        """Rebuild initramfs."""
        try:
            distro = detect_distro()

            if distro.id in ("ubuntu", "debian", "linuxmint", "pop"):
                subprocess.run(
                    ["update-initramfs", "-u"],
                    capture_output=True,
                    check=True,
                )
            elif distro.id in ("fedora", "rhel", "centos", "rocky", "alma"):
                subprocess.run(
                    ["dracut", "-f"],
                    capture_output=True,
                    check=True,
                )
            elif distro.id in ("arch", "manjaro"):
                subprocess.run(
                    ["mkinitcpio", "-P"],
                    capture_output=True,
                    check=True,
                )
            elif distro.id in ("opensuse", "sles"):
                subprocess.run(
                    ["dracut", "-f"],
                    capture_output=True,
                    check=True,
                )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to rebuild initramfs: {e}")
            raise
        except Exception as e:
            logger.warning(f"Initramfs rebuild may have failed: {e}")
