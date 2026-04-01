"""Tests for distro/versionlock.py module."""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from nvidia_inst.distro.versionlock import (
    read_versionlock_toml,
    write_versionlock_toml,
    pattern_entry_exists,
    add_pattern_versionlock_entry,
    verify_versionlock_pattern_active,
)


class TestReadVersionlockToml:
    """Tests for read_versionlock_toml function."""

    @patch("nvidia_inst.distro.versionlock.Path.exists", return_value=False)
    def test_read_versionlock_not_exists(self, mock_exists):
        """Test reading when versionlock file doesn't exist."""
        result = read_versionlock_toml()
        assert result == {"version": "1.0", "packages": []}

    @patch("subprocess.run")
    @patch("nvidia_inst.distro.versionlock.Path.exists", return_value=True)
    def test_read_versionlock_success(self, mock_exists, mock_run):
        """Test successful read of versionlock file."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='version = "1.0"\n\n[[packages]]\nname = "akmod-nvidia"',
        )

        result = read_versionlock_toml()
        assert result["version"] == "1.0"
        assert len(result["packages"]) == 1
        assert result["packages"][0]["name"] == "akmod-nvidia"

    @patch("subprocess.run")
    @patch("nvidia_inst.distro.versionlock.Path.exists", return_value=True)
    def test_read_versionlock_failure(self, mock_exists, mock_run):
        """Test failed read of versionlock file."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Permission denied")

        result = read_versionlock_toml()
        assert result == {"version": "1.0", "packages": []}


class TestPatternEntryExists:
    """Tests for pattern_entry_exists function."""

    def test_pattern_exists(self):
        """Test when pattern exists."""
        data = {
            "packages": [
                {"name": "akmod-nvidia", "conditions": []},
                {"name": "cuda-toolkit", "conditions": []},
            ]
        }
        assert pattern_entry_exists(data, "akmod-nvidia") is True

    def test_pattern_not_exists(self):
        """Test when pattern doesn't exist."""
        data = {
            "packages": [
                {"name": "cuda-toolkit", "conditions": []},
            ]
        }
        assert pattern_entry_exists(data, "akmod-nvidia") is False

    def test_empty_packages(self):
        """Test with empty packages list."""
        data = {"packages": []}
        assert pattern_entry_exists(data, "akmod-nvidia") is False


class TestAddPatternVersionlockEntry:
    """Tests for add_pattern_versionlock_entry function."""

    @patch("nvidia_inst.distro.versionlock.read_versionlock_toml")
    @patch("nvidia_inst.distro.versionlock.write_versionlock_toml")
    def test_add_entry_success(self, mock_write, mock_read):
        """Test successful entry addition."""
        mock_read.return_value = {"version": "1.0", "packages": []}
        mock_write.return_value = (True, "Success")

        success, msg = add_pattern_versionlock_entry(
            package_name="akmod-nvidia",
            major_version="580",
            comment="Test lock",
        )

        assert success is True
        assert "Locked" in msg
        mock_write.assert_called_once()

    @patch("nvidia_inst.distro.versionlock.read_versionlock_toml")
    def test_add_entry_already_exists(self, mock_read):
        """Test adding entry that already exists."""
        mock_read.return_value = {
            "version": "1.0",
            "packages": [{"name": "akmod-nvidia"}],
        }

        success, msg = add_pattern_versionlock_entry(
            package_name="akmod-nvidia",
            major_version="580",
        )

        assert success is True
        assert "already locked" in msg

    @patch("nvidia_inst.distro.versionlock.read_versionlock_toml")
    @patch("nvidia_inst.distro.versionlock.write_versionlock_toml")
    def test_add_entry_with_max_version(self, mock_write, mock_read):
        """Test adding entry with max_version."""
        mock_read.return_value = {"version": "1.0", "packages": []}
        mock_write.return_value = (True, "Success")

        success, msg = add_pattern_versionlock_entry(
            package_name="cuda-toolkit",
            major_version="12",
            max_version="12.8",
        )

        assert success is True
        # Check that the write was called with correct upper bound
        call_args = mock_write.call_args[0][0]
        package = call_args["packages"][0]
        assert package["conditions"][1]["value"] == "12.9"


class TestVerifyVersionlockPatternActive:
    """Tests for verify_versionlock_pattern_active function."""

    @patch("nvidia_inst.distro.versionlock.read_versionlock_toml")
    def test_verify_success(self, mock_read):
        """Test successful verification."""
        mock_read.return_value = {
            "version": "1.0",
            "packages": [
                {
                    "name": "akmod-nvidia",
                    "conditions": [
                        {"key": "evr", "comparator": ">=", "value": "580"},
                        {"key": "evr", "comparator": "<", "value": "581"},
                    ],
                }
            ],
        }

        success, msg = verify_versionlock_pattern_active("akmod-nvidia", "580")

        assert success is True
        assert "verified" in msg.lower()

    @patch("nvidia_inst.distro.versionlock.read_versionlock_toml")
    def test_verify_not_found(self, mock_read):
        """Test verification when entry not found."""
        mock_read.return_value = {"version": "1.0", "packages": []}

        success, msg = verify_versionlock_pattern_active("akmod-nvidia", "580")

        assert success is False
        assert "found" in msg.lower()

    @patch("nvidia_inst.distro.versionlock.read_versionlock_toml")
    def test_verify_different_conditions(self, mock_read):
        """Test verification with different conditions."""
        mock_read.return_value = {
            "version": "1.0",
            "packages": [
                {
                    "name": "akmod-nvidia",
                    "conditions": [
                        {"key": "evr", "comparator": ">=", "value": "570"},
                    ],
                }
            ],
        }

        success, msg = verify_versionlock_pattern_active("akmod-nvidia", "580")

        assert success is True
        assert "conditions differ" in msg.lower()
