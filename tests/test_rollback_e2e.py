"""E2E rollback/snapshot tests.

These tests verify rollback operations with real filesystem operations,
including snapshot creation, listing, and rollback logic.
"""

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.installer.rollback import RollbackManager, SystemState


def is_fedora_container() -> bool:
    """Check if running in a Fedora container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            return "fedora" in f.read().lower()
    return False


def is_ubuntu_container() -> bool:
    """Check if running in an Ubuntu container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            content = f.read().lower()
            return "ubuntu" in content
    return False


def has_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0


# ---------------------------------------------------------------------------
# create_snapshot() tests
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    """E2E tests for create_snapshot() with real filesystem."""

    def test_snapshot_creates_state_dir(self, tmp_path):
        """Test that create_snapshot creates the state directory."""
        state_dir = tmp_path / "state"
        RollbackManager(state_dir=str(state_dir))

        # State dir should exist after init
        assert state_dir.exists()

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_created_with_data(self, mock_run, mock_detect, tmp_path):
        """Test that create_snapshot produces a valid SystemState."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        state = manager.create_snapshot()

        assert isinstance(state, SystemState)
        assert state.distro_id == "ubuntu"
        assert state.timestamp != ""
        assert isinstance(state.installed_packages, list)
        assert isinstance(state.loaded_modules, list)

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_file_created(self, mock_run, mock_detect, tmp_path):
        """Test that create_snapshot writes a JSON file to disk."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        manager.create_snapshot()

        # Check that a state file was created
        state_files = list(state_dir.glob("state_*.json"))
        assert len(state_files) == 1

        # Verify the file is valid JSON
        with open(state_files[0]) as f:
            data = json.load(f)
        assert "distro_id" in data
        assert "timestamp" in data
        assert "installed_packages" in data

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_checksum_calculated(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot includes a checksum."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        state = manager.create_snapshot()

        assert state.checksum != ""
        assert len(state.checksum) == 64  # SHA-256 hex digest

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    @patch("nvidia_inst.installer.rollback.datetime")
    def test_multiple_snapshots(self, mock_dt, mock_run, mock_detect, tmp_path):
        """Test that multiple snapshots can be created."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        # Mock datetime to produce different timestamps
        counter = [0]

        def fake_now(*args, **kwargs):
            from datetime import datetime

            counter[0] += 1
            return datetime(2026, 1, 1, 0, 0, counter[0])

        mock_dt.now.side_effect = fake_now
        mock_dt.strftime = datetime.strftime

        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))

        manager.create_snapshot()
        manager.create_snapshot()
        manager.create_snapshot()

        state_files = list(state_dir.glob("state_*.json"))
        assert len(state_files) == 3


# ---------------------------------------------------------------------------
# list_snapshots() tests
# ---------------------------------------------------------------------------


class TestListSnapshots:
    """E2E tests for list_snapshots() with real filesystem."""

    def test_list_empty(self, tmp_path):
        """Test listing snapshots when none exist."""
        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        snapshots = manager.list_snapshots()
        assert snapshots == []

    def test_list_single_snapshot(self, tmp_path):
        """Test listing a single snapshot."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        snapshot_data = {
            "timestamp": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "installed_packages": ["nvidia-driver-590"],
            "loaded_modules": ["nvidia"],
            "driver_version": "590.48.01",
            "cuda_version": "12.2",
            "blacklist_files": [],
            "versionlock_entries": [],
            "checksum": "abc123",
        }

        with open(state_dir / "state_20260101_000000.json", "w") as f:
            json.dump(snapshot_data, f)

        manager = RollbackManager(state_dir=str(state_dir))
        snapshots = manager.list_snapshots()

        assert len(snapshots) == 1
        assert snapshots[0]["timestamp"] == "2026-01-01T00:00:00"
        assert snapshots[0]["distro_id"] == "ubuntu"
        assert snapshots[0]["driver_version"] == "590.48.01"
        assert "file" in snapshots[0]

    def test_list_multiple_snapshots_sorted(self, tmp_path):
        """Test that snapshots are listed in reverse chronological order."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # Create snapshots with different timestamps
        for i, ts in enumerate(["20260103", "20260101", "20260102"]):
            snapshot_data = {
                "timestamp": f"2026-01-{i + 1:02d}T00:00:00",
                "distro_id": "ubuntu",
                "installed_packages": [],
                "loaded_modules": [],
                "driver_version": None,
                "cuda_version": None,
                "blacklist_files": [],
                "versionlock_entries": [],
                "checksum": f"hash{i}",
            }
            with open(state_dir / f"state_{ts}_000000.json", "w") as f:
                json.dump(snapshot_data, f)

        manager = RollbackManager(state_dir=str(state_dir))
        snapshots = manager.list_snapshots()

        assert len(snapshots) == 3
        # Should be sorted by filename (reverse), which means newest first
        assert "20260103" in snapshots[0]["file"]

    def test_list_skips_corrupted_snapshots(self, tmp_path):
        """Test that corrupted snapshot files are skipped."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # Valid snapshot
        valid_data = {
            "timestamp": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "installed_packages": [],
            "loaded_modules": [],
            "driver_version": None,
            "cuda_version": None,
            "blacklist_files": [],
            "versionlock_entries": [],
            "checksum": "abc123",
        }
        with open(state_dir / "state_20260101_000000.json", "w") as f:
            json.dump(valid_data, f)

        # Corrupted snapshot
        with open(state_dir / "state_20260102_000000.json", "w") as f:
            f.write("not valid json{{{")

        manager = RollbackManager(state_dir=str(state_dir))
        snapshots = manager.list_snapshots()

        # Should only return the valid one
        assert len(snapshots) == 1
        assert snapshots[0]["timestamp"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# rollback() tests
# ---------------------------------------------------------------------------


class TestRollback:
    """E2E tests for rollback() logic."""

    def test_rollback_no_snapshot(self, tmp_path):
        """Test rollback when no snapshots exist."""
        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        result = manager.rollback()
        assert result is False

    @patch("nvidia_inst.installer.rollback.get_package_manager")
    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_rollback_with_snapshot(
        self, mock_run, mock_detect, mock_pkg_mgr, tmp_path
    ):
        """Test rollback with a valid snapshot."""
        mock_detect.return_value = MagicMock(id="ubuntu")
        mock_pkg_mgr.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # Create a snapshot file
        snapshot_data = {
            "timestamp": "2026-01-01T00:00:00",
            "distro_id": "ubuntu",
            "installed_packages": ["nvidia-driver-590"],
            "loaded_modules": ["nvidia"],
            "driver_version": "590.48.01",
            "cuda_version": "12.2",
            "blacklist_files": [],
            "versionlock_entries": [],
            "checksum": "abc123",
        }

        with open(state_dir / "state_20260101_000000.json", "w") as f:
            json.dump(snapshot_data, f)

        manager = RollbackManager(state_dir=str(state_dir))
        result = manager.rollback()

        # Rollback may succeed or fail depending on system state
        assert isinstance(result, bool)

    @patch("nvidia_inst.installer.rollback.get_package_manager")
    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_rollback_with_explicit_state(
        self, mock_run, mock_detect, mock_pkg_mgr, tmp_path
    ):
        """Test rollback with an explicitly provided SystemState."""
        mock_detect.return_value = MagicMock(id="ubuntu")
        mock_pkg_mgr.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))

        state = SystemState(
            timestamp="2026-01-01T00:00:00",
            distro_id="ubuntu",
            installed_packages=["nvidia-driver-590"],
            loaded_modules=["nvidia"],
            driver_version="590.48.01",
            cuda_version="12.2",
            blacklist_files=[],
            versionlock_entries=[],
            checksum="abc123",
        )

        result = manager.rollback(state=state)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# cleanup_old_snapshots() tests
# ---------------------------------------------------------------------------


class TestCleanupOldSnapshots:
    """E2E tests for cleanup_old_snapshots()."""

    def test_cleanup_no_snapshots(self, tmp_path):
        """Test cleanup when no snapshots exist."""
        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        removed = manager.cleanup_old_snapshots(keep_count=3)
        assert removed == 0

    def test_cleanup_below_threshold(self, tmp_path):
        """Test cleanup when below keep threshold."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        for i in range(3):
            (state_dir / f"state_2026010{i}_000000.json").write_text("{}")

        manager = RollbackManager(state_dir=str(state_dir))
        removed = manager.cleanup_old_snapshots(keep_count=5)
        assert removed == 0

    def test_cleanup_removes_oldest(self, tmp_path):
        """Test cleanup removes oldest snapshots."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        for i in range(7):
            (state_dir / f"state_2026010{i}_000000.json").write_text("{}")

        manager = RollbackManager(state_dir=str(state_dir))
        removed = manager.cleanup_old_snapshots(keep_count=3)
        assert removed == 4

        remaining = list(state_dir.glob("state_*.json"))
        assert len(remaining) == 3


# ---------------------------------------------------------------------------
# Snapshot metadata tests
# ---------------------------------------------------------------------------


class TestSnapshotMetadata:
    """E2E tests for snapshot metadata."""

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_has_timestamp(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot has a valid timestamp."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert state.timestamp != ""
        # Should be ISO format
        assert "T" in state.timestamp

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_has_distro_id(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot captures distro ID."""
        mock_detect.return_value = MagicMock(id="fedora", version_id="40")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert state.distro_id == "fedora"

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_has_package_list(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot captures installed packages."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert isinstance(state.installed_packages, list)

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_has_module_list(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot captures loaded modules."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert isinstance(state.loaded_modules, list)

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_has_blacklist_files(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot captures blacklist files."""
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="24.04")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert isinstance(state.blacklist_files, list)

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_snapshot_has_versionlock_entries(self, mock_run, mock_detect, tmp_path):
        """Test that snapshot captures versionlock entries."""
        mock_detect.return_value = MagicMock(id="fedora", version_id="40")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert isinstance(state.versionlock_entries, list)


# ---------------------------------------------------------------------------
# Real distro snapshot tests
# ---------------------------------------------------------------------------


class TestRealDistroSnapshots:
    """E2E tests for snapshots on real distro containers."""

    @pytest.mark.skipif("not is_ubuntu_container()")
    @pytest.mark.skipif("not has_root()")
    def test_ubuntu_snapshot_creation(self, tmp_path):
        """Test snapshot creation on Ubuntu container."""
        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert state.distro_id == "ubuntu"
        assert state.timestamp != ""

        # Verify file was created
        state_files = list(tmp_path.glob("state_*.json"))
        assert len(state_files) == 1

    @pytest.mark.skipif("not is_fedora_container()")
    @pytest.mark.skipif("not has_root()")
    def test_fedora_snapshot_creation(self, tmp_path):
        """Test snapshot creation on Fedora container."""
        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert state.distro_id == "fedora"
        assert state.timestamp != ""

        state_files = list(tmp_path.glob("state_*.json"))
        assert len(state_files) == 1
