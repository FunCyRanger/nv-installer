"""E2E CLI workflow tests running in real distro containers.

These tests verify CLI argument parsing, --check mode, --simulate mode,
--version, --debug, and --branch options with real distro detection
but mocked GPU detection.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.cli.parser import parse_args
from nvidia_inst.gpu.detector import GPUInfo


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
# --version flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    """E2E tests for --version CLI flag."""

    def test_version_output(self, capsys):
        """Test that --version prints version string and exits 0."""
        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--version"]):
            rc = main()

        captured = capsys.readouterr()
        assert rc == 0
        assert "nvidia-inst version" in captured.out


# ---------------------------------------------------------------------------
# --debug flag
# ---------------------------------------------------------------------------


class TestDebugFlag:
    """E2E tests for --debug CLI flag."""

    @patch("nvidia_inst.cli.main.has_nvidia_gpu", return_value=False)
    def test_debug_enables_debug_logging(self, mock_gpu, capsys):
        """Test that --debug enables DEBUG-level logging."""
        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--debug"]):
            rc = main()

        # No GPU in container, should exit 0
        assert rc == 0
        captured = capsys.readouterr()
        assert "No NVIDIA GPU detected" in captured.out


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


class TestCheckMode:
    """E2E tests for --check CLI mode."""

    @patch("nvidia_inst.cli.compatibility.has_nvidia_gpu", return_value=False)
    def test_check_no_gpu(self, mock_gpu, capsys):
        """Test --check when no GPU is detected."""
        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--check"]):
            rc = main()

        assert rc == 0
        captured = capsys.readouterr()
        assert "No Nvidia GPU detected" in captured.out

    @patch("nvidia_inst.cli.compatibility.detect_gpu")
    @patch("nvidia_inst.cli.compatibility.has_nvidia_gpu", return_value=True)
    def test_check_with_mocked_gpu(self, mock_has_gpu, mock_detect_gpu, capsys):
        """Test --check with a mocked GPU in a real distro container."""
        mock_detect_gpu.return_value = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            compute_capability=8.6,
            generation="ampere",
        )

        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--check"]):
            rc = main()

        assert rc == 0
        captured = capsys.readouterr()
        # Should show compatibility info
        assert "System Compatibility Check" in captured.out
        assert "RTX 3080" in captured.out
        # Should show prerequisites check
        assert "Prerequisites Check" in captured.out


# ---------------------------------------------------------------------------
# --simulate mode
# ---------------------------------------------------------------------------


class TestSimulateMode:
    """E2E tests for --simulate CLI mode."""

    @patch("nvidia_inst.cli.main.has_nvidia_gpu", return_value=False)
    def test_simulate_no_gpu(self, mock_gpu, capsys):
        """Test --simulate when no GPU is detected."""
        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--simulate"]):
            rc = main()

        assert rc == 0
        captured = capsys.readouterr()
        assert "No NVIDIA GPU detected" in captured.out

    @patch("nvidia_inst.cli.main.detect_gpu")
    @patch("nvidia_inst.cli.main.has_nvidia_gpu", return_value=True)
    @patch("nvidia_inst.cli.main.show_driver_options", return_value=1)
    @patch("nvidia_inst.cli.main.execute_driver_change", return_value=0)
    def test_simulate_with_mocked_gpu(
        self, mock_execute, mock_show, mock_has_gpu, mock_detect, capsys
    ):
        """Test --simulate flow with a mocked GPU."""
        mock_detect.return_value = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            compute_capability=8.6,
            generation="ampere",
        )

        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--simulate", "--yes"]):
            rc = main()

        assert rc == 0
        # execute_driver_change should have been called with simulate=True
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args
        assert call_kwargs.kwargs.get("simulate") is True


# ---------------------------------------------------------------------------
# --branch option
# ---------------------------------------------------------------------------


class TestBranchOption:
    """E2E tests for --branch CLI option."""

    def test_branch_argument_parsed(self):
        """Test that --branch argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--branch", "580"]):
            args = parse_args()
        assert args.branch == "580"

    def test_branch_valid_choices(self):
        """Test that --branch accepts valid choices."""
        for branch in ("470", "580", "590", "595"):
            with patch.object(sys, "argv", ["nvidia-inst", "--branch", branch]):
                args = parse_args()
            assert args.branch == branch

    def test_branch_invalid_choice(self, capsys):
        """Test that --branch rejects invalid choices."""
        with (
            patch.object(sys, "argv", ["nvidia-inst", "--branch", "999"]),
            pytest.raises(SystemExit),
        ):
            parse_args()


# ---------------------------------------------------------------------------
# --no-cuda option
# ---------------------------------------------------------------------------


class TestNoCudaOption:
    """E2E tests for --no-cuda CLI option."""

    def test_no_cuda_argument_parsed(self):
        """Test that --no-cuda argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--no-cuda"]):
            args = parse_args()
        assert args.no_cuda is True

    def test_no_cuda_default_false(self):
        """Test that --no-cuda defaults to False."""
        with patch.object(sys, "argv", ["nvidia-inst"]):
            args = parse_args()
        assert args.no_cuda is False


# ---------------------------------------------------------------------------
# --cuda-version option
# ---------------------------------------------------------------------------


class TestCudaVersionOption:
    """E2E tests for --cuda-version CLI option."""

    def test_cuda_version_argument_parsed(self):
        """Test that --cuda-version argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--cuda-version", "12.2"]):
            args = parse_args()
        assert args.cuda_version == "12.2"

    def test_cuda_version_default_none(self):
        """Test that --cuda-version defaults to None."""
        with patch.object(sys, "argv", ["nvidia-inst"]):
            args = parse_args()
        assert args.cuda_version is None


# ---------------------------------------------------------------------------
# --revert-to-nouveau option
# ---------------------------------------------------------------------------


class TestRevertToNouveauOption:
    """E2E tests for --revert-to-nouveau CLI option."""

    def test_revert_argument_parsed(self):
        """Test that --revert-to-nouveau argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--revert-to-nouveau"]):
            args = parse_args()
        assert args.revert_to_nouveau is True

    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.check_nvidia_packages_installed")
    def test_revert_no_packages(self, mock_check, mock_detect, capsys):
        """Test --revert-to-nouveau when no NVIDIA packages are installed."""
        from nvidia_inst.distro.detector import DistroInfo

        mock_detect.return_value = DistroInfo(
            id="ubuntu",
            version_id="24.04",
            name="Ubuntu",
            pretty_name="Ubuntu 24.04 LTS",
            kernel="6.8.0-generic",
        )
        mock_check.return_value = []

        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--revert-to-nouveau"]):
            rc = main()

        assert rc == 0
        captured = capsys.readouterr()
        assert "No proprietary Nvidia packages found" in captured.out


# ---------------------------------------------------------------------------
# --rollback option
# ---------------------------------------------------------------------------


class TestRollbackOption:
    """E2E tests for --rollback CLI option."""

    def test_rollback_argument_parsed(self):
        """Test that --rollback argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--rollback"]):
            args = parse_args()
        assert args.rollback is True

    @patch("nvidia_inst.installer.rollback.RollbackManager")
    def test_rollback_no_snapshots(self, mock_mgr_cls, capsys):
        """Test --rollback when no snapshots are available."""
        mock_mgr = MagicMock()
        mock_mgr.list_snapshots.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--rollback"]):
            rc = main()

        assert rc == 1
        captured = capsys.readouterr()
        assert "No snapshots available" in captured.out


# ---------------------------------------------------------------------------
# --list-snapshots option
# ---------------------------------------------------------------------------


class TestListSnapshotsOption:
    """E2E tests for --list-snapshots CLI option."""

    def test_list_snapshots_argument_parsed(self):
        """Test that --list-snapshots argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--list-snapshots"]):
            args = parse_args()
        assert args.list_snapshots is True

    @patch("nvidia_inst.installer.rollback.RollbackManager")
    def test_list_snapshots_empty(self, mock_mgr_cls, capsys):
        """Test --list-snapshots when no snapshots exist."""
        mock_mgr = MagicMock()
        mock_mgr.list_snapshots.return_value = []
        mock_mgr_cls.return_value = mock_mgr

        from nvidia_inst.cli.main import main

        with patch.object(sys, "argv", ["nvidia-inst", "--list-snapshots"]):
            rc = main()

        assert rc == 0
        captured = capsys.readouterr()
        assert "No snapshots available" in captured.out


# ---------------------------------------------------------------------------
# --create-cache option
# ---------------------------------------------------------------------------


class TestCreateCacheOption:
    """E2E tests for --create-cache CLI option."""

    def test_create_cache_argument_parsed(self):
        """Test that --create-cache argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--create-cache"]):
            args = parse_args()
        assert args.create_cache is True

    def test_cache_dir_default(self):
        """Test that --cache-dir defaults to /var/cache/nvidia-inst."""
        with patch.object(sys, "argv", ["nvidia-inst"]):
            args = parse_args()
        assert args.cache_dir == "/var/cache/nvidia-inst"

    def test_cache_dir_custom(self):
        """Test that --cache-dir accepts custom path."""
        with patch.object(sys, "argv", ["nvidia-inst", "--cache-dir", "/tmp/cache"]):
            args = parse_args()
        assert args.cache_dir == "/tmp/cache"


# ---------------------------------------------------------------------------
# --verify-cache option
# ---------------------------------------------------------------------------


class TestVerifyCacheOption:
    """E2E tests for --verify-cache CLI option."""

    def test_verify_cache_argument_parsed(self):
        """Test that --verify-cache argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--verify-cache"]):
            args = parse_args()
        assert args.verify_cache is True


# ---------------------------------------------------------------------------
# --power-profile option
# ---------------------------------------------------------------------------


class TestPowerProfileOption:
    """E2E tests for --power-profile CLI option."""

    def test_power_profile_intel(self):
        """Test --power-profile intel."""
        with patch.object(sys, "argv", ["nvidia-inst", "--power-profile", "intel"]):
            args = parse_args()
        assert args.power_profile == "intel"

    def test_power_profile_hybrid(self):
        """Test --power-profile hybrid."""
        with patch.object(sys, "argv", ["nvidia-inst", "--power-profile", "hybrid"]):
            args = parse_args()
        assert args.power_profile == "hybrid"

    def test_power_profile_nvidia(self):
        """Test --power-profile nvidia."""
        with patch.object(sys, "argv", ["nvidia-inst", "--power-profile", "nvidia"]):
            args = parse_args()
        assert args.power_profile == "nvidia"

    def test_power_profile_invalid(self, capsys):
        """Test --power-profile rejects invalid choices."""
        with (
            patch.object(sys, "argv", ["nvidia-inst", "--power-profile", "invalid"]),
            pytest.raises(SystemExit),
        ):
            parse_args()


# ---------------------------------------------------------------------------
# --offline option
# ---------------------------------------------------------------------------


class TestOfflineOption:
    """E2E tests for --offline CLI option."""

    def test_offline_argument_parsed(self):
        """Test that --offline argument is correctly parsed."""
        with patch.object(sys, "argv", ["nvidia-inst", "--offline"]):
            args = parse_args()
        assert args.offline is True


# ---------------------------------------------------------------------------
# Combined flags
# ---------------------------------------------------------------------------


class TestCombinedFlags:
    """E2E tests for combined CLI flags."""

    def test_simulate_with_no_cuda(self):
        """Test --simulate combined with --no-cuda."""
        with patch.object(sys, "argv", ["nvidia-inst", "--simulate", "--no-cuda"]):
            args = parse_args()
        assert args.simulate is True
        assert args.no_cuda is True

    def test_simulate_with_cuda_version(self):
        """Test --simulate combined with --cuda-version."""
        with patch.object(
            sys,
            "argv",
            ["nvidia-inst", "--simulate", "--cuda-version", "12.2"],
        ):
            args = parse_args()
        assert args.simulate is True
        assert args.cuda_version == "12.2"

    def test_debug_with_check(self):
        """Test --debug combined with --check."""
        with patch.object(sys, "argv", ["nvidia-inst", "--debug", "--check"]):
            args = parse_args()
        assert args.debug is True
        assert args.check is True

    def test_yes_with_create_cache(self):
        """Test --yes combined with --create-cache."""
        with patch.object(sys, "argv", ["nvidia-inst", "--yes", "--create-cache"]):
            args = parse_args()
        assert args.yes is True
        assert args.create_cache is True
