"""Tests for driver state table formatting."""

from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.cli.driver_state import (
    DriverOption,
    DriverState,
    DriverStatus,
    _format_status_table,
    _format_versionlock_conditions,
    _get_constraints,
    _get_current_locks,
    _get_nouveau_version,
    _get_option_locks,
    _get_warning_line,
)
from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.gpu.detector import GPUInfo


@pytest.fixture
def ampere_gpu():
    return GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")


@pytest.fixture
def pascal_gpu():
    return GPUInfo(model="NVIDIA GeForce GTX 1080", generation="pascal")


@pytest.fixture
def kepler_gpu():
    return GPUInfo(model="NVIDIA GeForce GTX 680", generation="kepler")


@pytest.fixture
def full_range():
    return DriverRange(
        min_version="520.56.06",
        max_version="590.48.01",
        cuda_min="11.0",
        cuda_max="12.8",
        max_branch="590",
        is_eol=False,
        is_limited=False,
        cuda_is_locked=False,
    )


@pytest.fixture
def limited_range():
    return DriverRange(
        min_version="450.191.0",
        max_version="580.142",
        cuda_min="8.0",
        cuda_max="12.x",
        max_branch="580",
        is_eol=False,
        is_limited=True,
        cuda_is_locked=True,
        cuda_locked_major="12",
    )


@pytest.fixture
def eol_range():
    return DriverRange(
        min_version="390.157.0",
        max_version="470.256.02",
        cuda_min="7.5",
        cuda_max="11.x",
        max_branch="470",
        is_eol=True,
        is_limited=True,
        cuda_is_locked=True,
        cuda_locked_major="11",
        eol_message="Kepler is end-of-life. Maximum supported driver: 470.256.02.",
    )


class TestGetNouveauVersion:
    def test_mesa_detected(self):
        """Test Nouveau version detection when Mesa is available."""
        mock_result = MagicMock()
        mock_result.stdout = (
            "OpenGL version string: 4.6 (Compatibility Profile) Mesa 24.0.0\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            version = _get_nouveau_version()
            assert version == "4.6"

    def test_mesa_not_available(self):
        """Test fallback when Mesa is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            version = _get_nouveau_version()
            assert version == "-"


class TestGetCurrentLocks:
    def test_apt_no_locks(self):
        """Test APT with no lock files."""
        with patch("os.path.isdir", return_value=False):
            locks = _get_current_locks("ubuntu")
            assert locks == []

    def test_apt_with_locks(self):
        """Test APT reads nvidia-inst preferences with version info."""
        mock_content = (
            "Package: nvidia-driver-580*\nPin: version 580.*\nPin-Priority: 1001\n"
        )
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = mock_content

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["nvidia-inst-nvidia-driver-580_"]):
                with patch("builtins.open", return_value=mock_file):
                    locks = _get_current_locks("ubuntu")
                    assert "nvidia-driver-580*" in locks[0]
                    assert "580.*" in locks[0]

    def test_dnf_no_locks(self):
        """Test DNF with no versionlock file."""
        with patch("os.path.isfile", return_value=False):
            locks = _get_current_locks("fedora")
            assert locks == []

    def test_dnf_locks_with_conditions(self):
        """Test DNF lock detection shows version ranges."""
        mock_toml = (
            b'version = "1.0"\n'
            b"\n"
            b"[[packages]]\n"
            b'name = "akmod-nvidia"\n'
            b"[[packages.conditions]]\n"
            b'key = "evr"\n'
            b'comparator = ">="\n'
            b'value = "580"\n'
            b"[[packages.conditions]]\n"
            b'key = "evr"\n'
            b'comparator = "<"\n'
            b'value = "581"\n'
            b"\n"
            b"[[packages]]\n"
            b'name = "cuda-toolkit"\n'
            b"[[packages.conditions]]\n"
            b'key = "evr"\n'
            b'comparator = ">="\n'
            b'value = "12"\n'
            b"[[packages.conditions]]\n"
            b'key = "evr"\n'
            b'comparator = "<"\n'
            b'value = "13"\n'
        )
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = mock_toml

        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", return_value=mock_file):
                locks = _get_current_locks("fedora")
                assert "akmod-nvidia (580.x)" in locks
                assert "cuda-toolkit (12.x)" in locks

    def test_dnf_locks_no_conditions(self):
        """Test DNF lock detection with no conditions falls back to *."""
        mock_toml = b'version = "1.0"\n\n[[packages]]\nname = "akmod-nvidia"\n'
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = mock_toml

        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", return_value=mock_file):
                locks = _get_current_locks("fedora")
                assert "akmod-nvidia (*)" in locks

    def test_arch_no_locks(self):
        """Test Arch returns empty (no locking)."""
        locks = _get_current_locks("arch")
        assert locks == []


class TestFormatVersionlockConditions:
    def test_lower_and_upper(self):
        """Test formatting with both lower and upper bounds."""
        conditions = [
            {"key": "evr", "comparator": ">=", "value": "580"},
            {"key": "evr", "comparator": "<", "value": "581"},
        ]
        result = _format_versionlock_conditions(conditions)
        assert result == "580.x"

    def test_lower_only(self):
        """Test formatting with only lower bound."""
        conditions = [{"key": "evr", "comparator": ">=", "value": "12"}]
        result = _format_versionlock_conditions(conditions)
        assert result == ">= 12"

    def test_upper_only(self):
        """Test formatting with only upper bound."""
        conditions = [{"key": "evr", "comparator": "<", "value": "13"}]
        result = _format_versionlock_conditions(conditions)
        assert result == "< 13"

    def test_empty_conditions(self):
        """Test formatting with no conditions."""
        result = _format_versionlock_conditions([])
        assert result == "*"


class TestGetOptionLocks:
    def test_limited_gpu_install(self, limited_range):
        """Test lock generation for limited GPU install."""
        locks = _get_option_locks(limited_range, "install")
        assert "nvidia-driver-580*" in locks
        assert "cuda-*" in locks

    def test_limited_gpu_open(self, limited_range):
        """Test lock generation for NVIDIA Open on limited GPU."""
        locks = _get_option_locks(limited_range, "switch_nvidia_open")
        assert "nvidia-driver-580-open*" in locks
        assert "cuda-*" in locks

    def test_full_gpu_no_locks(self, full_range):
        """Test no locks for full support GPU."""
        locks = _get_option_locks(full_range, "install")
        assert locks == "-"

    def test_keep_action(self, limited_range):
        """Test keep action returns no locks."""
        locks = _get_option_locks(limited_range, "keep")
        assert locks == "-"

    def test_revert_nouveau(self, limited_range):
        """Test revert_nouveau returns no locks."""
        locks = _get_option_locks(limited_range, "revert_nouveau")
        assert locks == "-"


class TestGetConstraints:
    def test_full_support(self, full_range):
        """Test no constraints for full support GPU."""
        constraints = _get_constraints(full_range)
        assert constraints == []

    def test_limited_support(self, limited_range):
        """Test Branch+CUDA constraints for limited GPU."""
        constraints = _get_constraints(limited_range)
        assert "Branch" in constraints
        assert "CUDA" in constraints

    def test_eol(self, eol_range):
        """Test Branch+CUDA+EOL for EOL GPU."""
        constraints = _get_constraints(eol_range)
        assert "Branch" in constraints
        assert "CUDA" in constraints
        assert "EOL" in constraints


class TestGetWarningLine:
    def test_eol_message(self, eol_range, kepler_gpu):
        """Test EOL warning uses eol_message."""
        warning = _get_warning_line(eol_range, kepler_gpu)
        assert "end-of-life" in warning

    def test_limited_warning(self, limited_range, pascal_gpu):
        """Test limited GPU warning."""
        warning = _get_warning_line(limited_range, pascal_gpu)
        assert "580" in warning
        assert "12.x" in warning

    def test_full_no_warning(self, full_range, ampere_gpu):
        """Test no warning for full support GPU."""
        warning = _get_warning_line(full_range, ampere_gpu)
        assert warning is None


class TestFormatStatusTable:
    def _make_state(self, status, current_version, options, message):
        return DriverState(
            status=status,
            current_version=current_version,
            is_compatible=status == DriverStatus.OPTIMAL,
            is_optimal=status == DriverStatus.OPTIMAL,
            suggested_packages=["nvidia-driver-590"],
            options=options,
            message=message,
            cuda_range="11.0-12.8",
        )

    def test_full_support_table(self, ampere_gpu, full_range):
        """Test table format for full support GPU."""
        state = self._make_state(
            DriverStatus.OPTIMAL,
            "590.48.01",
            [
                DriverOption(1, "NVIDIA proprietary", "upgrade"),
                DriverOption(2, "Keep current driver", "keep"),
                DriverOption(3, "NVIDIA Open", "switch_nvidia_open"),
                DriverOption(4, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
            ],
            "NVIDIA driver 590.48.01 is working optimally",
        )

        with patch(
            "nvidia_inst.cli.driver_state.get_current_driver_type",
            return_value="proprietary",
        ):
            with patch(
                "nvidia_inst.cli.driver_state._get_current_locks", return_value=[]
            ):
                with patch(
                    "nvidia_inst.cli.driver_state._get_nouveau_version",
                    return_value="24.0.0",
                ):
                    table = _format_status_table(
                        state, full_range, ampere_gpu, "ubuntu"
                    )

        assert "Driver Status" in table
        assert "RTX 3080" in table
        assert "590.48.01" in table
        assert "Proprietary (active)" in table
        assert "NVIDIA proprietary" in table
        assert "NVIDIA open-source" in table
        assert "Nouveau" in table
        assert "[!]" not in table

    def test_limited_support_table(self, pascal_gpu, limited_range):
        """Test table format for limited support GPU."""
        state = self._make_state(
            DriverStatus.WRONG_BRANCH,
            "590.48.01",
            [
                DriverOption(1, "NVIDIA proprietary", "install"),
                DriverOption(2, "Keep current driver", "keep"),
                DriverOption(3, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
            ],
            "Driver 590.48.01 may not be optimal for GTX 1080",
        )

        with patch(
            "nvidia_inst.cli.driver_state.get_current_driver_type",
            return_value="proprietary",
        ):
            with patch(
                "nvidia_inst.cli.driver_state._get_current_locks", return_value=[]
            ):
                with patch(
                    "nvidia_inst.cli.driver_state._get_nouveau_version",
                    return_value="24.0.0",
                ):
                    table = _format_status_table(
                        state, limited_range, pascal_gpu, "ubuntu"
                    )

        assert "GTX 1080" in table
        assert "580" in table
        assert "12.x" in table
        assert "[!]" in table

    def test_no_driver_table(self, ampere_gpu, full_range):
        """Test table format when no driver is installed."""
        state = self._make_state(
            DriverStatus.NOTHING,
            None,
            [
                DriverOption(1, "NVIDIA proprietary", "install"),
                DriverOption(2, "NVIDIA Open", "install_nvidia_open"),
                DriverOption(3, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
                DriverOption(4, "Cancel", "cancel"),
            ],
            "No NVIDIA driver installed",
        )

        with patch(
            "nvidia_inst.cli.driver_state.get_current_driver_type",
            return_value="none",
        ):
            with patch(
                "nvidia_inst.cli.driver_state._get_current_locks", return_value=[]
            ):
                with patch(
                    "nvidia_inst.cli.driver_state._get_nouveau_version",
                    return_value="-",
                ):
                    table = _format_status_table(
                        state, full_range, ampere_gpu, "ubuntu"
                    )

        assert "Nouveau (active)" in table
        assert "*" in table

    def test_table_with_existing_locks(self, ampere_gpu, full_range):
        """Test table shows existing package manager locks."""
        state = self._make_state(
            DriverStatus.OPTIMAL,
            "590.48.01",
            [
                DriverOption(1, "NVIDIA proprietary", "upgrade"),
                DriverOption(2, "Keep current driver", "keep"),
            ],
            "NVIDIA driver 590.48.01 is working optimally",
        )

        with patch(
            "nvidia_inst.cli.driver_state.get_current_driver_type",
            return_value="proprietary",
        ):
            with patch(
                "nvidia_inst.cli.driver_state._get_current_locks",
                return_value=["nvidia-driver-590*", "cuda-*"],
            ):
                with patch(
                    "nvidia_inst.cli.driver_state._get_nouveau_version",
                    return_value="24.0.0",
                ):
                    table = _format_status_table(
                        state, full_range, ampere_gpu, "ubuntu"
                    )

        assert "nvidia-driver-590*" in table
        assert "cuda-*" in table

    def test_table_header_columns(self, ampere_gpu, full_range):
        """Test that table has correct column headers."""
        state = self._make_state(
            DriverStatus.NOTHING,
            None,
            [
                DriverOption(1, "NVIDIA proprietary", "install"),
                DriverOption(2, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
            ],
            "No NVIDIA driver installed",
        )

        with patch(
            "nvidia_inst.cli.driver_state.get_current_driver_type",
            return_value="none",
        ):
            with patch(
                "nvidia_inst.cli.driver_state._get_current_locks", return_value=[]
            ):
                with patch(
                    "nvidia_inst.cli.driver_state._get_nouveau_version",
                    return_value="-",
                ):
                    table = _format_status_table(
                        state, full_range, ampere_gpu, "ubuntu"
                    )

        assert "Driver" in table
        assert "Version" in table
        assert "Branch" in table
        assert "CUDA" in table
        assert "Locked" in table
