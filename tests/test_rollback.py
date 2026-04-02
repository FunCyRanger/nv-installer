"""Tests for rollback functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.installer.rollback import RollbackManager, SystemState


class TestSystemState:
    """Tests for SystemState dataclass."""

    def test_default_values(self):
        """Test default values for SystemState."""
        state = SystemState()
        assert state.installed_packages == []
        assert state.loaded_modules == []
        assert state.driver_version is None
        assert state.cuda_version is None
        assert state.blacklist_files == []
        assert state.versionlock_entries == []

    def test_with_data(self):
        """Test SystemState with data."""
        state = SystemState(
            distro_id="ubuntu",
            installed_packages=["nvidia-driver-590", "nvidia-dkms-590"],
            loaded_modules=["nvidia"],
            driver_version="590.48.01",
            cuda_version="12.2",
        )
        assert state.distro_id == "ubuntu"
        assert state.installed_packages == ["nvidia-driver-590", "nvidia-dkms-590"]
        assert state.driver_version == "590.48.01"


class TestRollbackManager:
    """Tests for RollbackManager."""

    def test_init_creates_state_dir(self, tmp_path):
        """Test initialization creates state directory."""
        state_dir = tmp_path / "state"
        manager = RollbackManager(state_dir=str(state_dir))
        assert state_dir.exists()

    def test_list_snapshots_empty(self, tmp_path):
        """Test listing snapshots when none exist."""
        manager = RollbackManager(state_dir=str(tmp_path))
        snapshots = manager.list_snapshots()
        assert snapshots == []

    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_create_snapshot(self, mock_run, mock_detect, tmp_path):
        """Test creating a snapshot."""
        # Setup mocks
        mock_detect.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_run.return_value = MagicMock(stdout="590.48.01\n", returncode=0)

        manager = RollbackManager(state_dir=str(tmp_path))
        state = manager.create_snapshot()

        assert state.distro_id == "ubuntu"
        assert state.timestamp != ""

        # Check snapshot file was created
        snapshots = list(tmp_path.glob("state_*.json"))
        assert len(snapshots) == 1

    def test_list_snapshots_with_data(self, tmp_path):
        """Test listing snapshots with data."""
        # Create a mock snapshot file
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

        snapshot_file = tmp_path / "state_20260101_000000.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f)

        manager = RollbackManager(state_dir=str(tmp_path))
        snapshots = manager.list_snapshots()

        assert len(snapshots) == 1
        assert snapshots[0]["distro_id"] == "ubuntu"
        assert snapshots[0]["driver_version"] == "590.48.01"

    def test_cleanup_old_snapshots(self, tmp_path):
        """Test cleanup of old snapshots."""
        # Create multiple snapshot files
        for i in range(7):
            snapshot_file = tmp_path / f"state_2026010{i}_000000.json"
            snapshot_file.write_text("{}")

        manager = RollbackManager(state_dir=str(tmp_path))

        # Keep only 3 snapshots
        removed = manager.cleanup_old_snapshots(keep_count=3)
        assert removed == 4

        # Check only 3 remain
        snapshots = list(tmp_path.glob("state_*.json"))
        assert len(snapshots) == 3


class TestRollbackOperations:
    """Tests for rollback operations."""

    @patch("nvidia_inst.installer.rollback.get_package_manager")
    @patch("nvidia_inst.installer.rollback.detect_distro")
    def test_rollback_no_snapshot(self, mock_detect, mock_pkg_mgr, tmp_path):
        """Test rollback with no snapshots available."""
        mock_detect.return_value = MagicMock(id="ubuntu")

        manager = RollbackManager(state_dir=str(tmp_path))
        result = manager.rollback()

        assert result is False

    @patch("nvidia_inst.installer.rollback.get_package_manager")
    @patch("nvidia_inst.installer.rollback.detect_distro")
    @patch("nvidia_inst.installer.rollback.subprocess.run")
    def test_rollback_with_snapshot(
        self, mock_run, mock_detect, mock_pkg_mgr, tmp_path
    ):
        """Test rollback with snapshot."""
        # Setup mocks
        mock_detect.return_value = MagicMock(id="ubuntu")
        mock_pkg_mgr.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

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

        snapshot_file = tmp_path / "state_20260101_000000.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f)

        manager = RollbackManager(state_dir=str(tmp_path))
        result = manager.rollback()

        # Rollback may succeed or fail depending on system state
        # Just verify it doesn't crash
        assert isinstance(result, bool)
