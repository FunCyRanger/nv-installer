"""Integration tests for CLI actions and dry-run scenarios."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.cli.driver_state import DriverOption, DriverState, DriverStatus
from nvidia_inst.distro.detector import DistroInfo
from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.gpu.detector import GPUInfo


@pytest.fixture
def mock_distro():
    """Mock distro info."""
    return DistroInfo(
        id="ubuntu",
        version_id="22.04",
        name="Ubuntu",
        pretty_name="Ubuntu 22.04.3 LTS",
        kernel="5.15.0-91-generic",
    )


@pytest.fixture
def mock_gpu():
    """Mock GPU info."""
    return GPUInfo(model="NVIDIA GeForce RTX 3080", generation="ampere")


@pytest.fixture
def mock_driver_range():
    """Mock driver range."""
    return DriverRange(
        min_version="520.56.06",
        max_version="590.48.01",
        cuda_min="11.0",
        cuda_max="12.8",
        is_eol=False,
        is_limited=False,
        max_branch="590",
        cuda_is_locked=False,
    )


@pytest.fixture
def optimal_state():
    """Mock optimal driver state."""
    return DriverState(
        status=DriverStatus.OPTIMAL,
        current_version="535.154.05",
        is_compatible=True,
        is_optimal=True,
        suggested_packages=["nvidia-driver-535"],
        options=[
            DriverOption(1, "NVIDIA proprietary", "upgrade"),
            DriverOption(2, "Keep current driver", "keep"),
            DriverOption(3, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
        ],
        message="NVIDIA driver is working optimally",
    )


@pytest.fixture
def wrong_branch_state():
    """Mock wrong branch driver state."""
    return DriverState(
        status=DriverStatus.WRONG_BRANCH,
        current_version="580.142",
        is_compatible=False,
        is_optimal=False,
        suggested_packages=["nvidia-driver-535"],
        options=[
            DriverOption(1, "NVIDIA proprietary", "install"),
            DriverOption(2, "Keep current driver", "keep"),
            DriverOption(3, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
        ],
        message="Driver may not be optimal",
    )


@pytest.fixture
def nouveau_state():
    """Mock nouveau active driver state."""
    return DriverState(
        status=DriverStatus.NOUVEAU_ACTIVE,
        current_version=None,
        is_compatible=False,
        is_optimal=False,
        suggested_packages=["nvidia-driver-535"],
        options=[
            DriverOption(1, "NVIDIA proprietary", "install"),
            DriverOption(2, "Keep Nouveau (no CUDA)", "keep"),
        ],
        message="Nouveau (open-source) driver is active",
    )


@pytest.fixture
def nothing_state():
    """Mock no driver installed state."""
    return DriverState(
        status=DriverStatus.NOTHING,
        current_version=None,
        is_compatible=False,
        is_optimal=False,
        suggested_packages=["nvidia-driver-535"],
        options=[
            DriverOption(1, "NVIDIA proprietary", "install"),
            DriverOption(2, "Nouveau (open-source, no CUDA)", "revert_nouveau"),
            DriverOption(3, "Cancel", "cancel"),
        ],
        message="No NVIDIA driver installed",
    )


class TestExecuteDriverChangeSimulate:
    """Test execute_driver_change with simulate=True for all actions."""

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_upgrade_optimal(
        self,
        mock_stdout,
        mock_packages,
        mock_distro,
        mock_gpu,
        mock_driver_range,
        optimal_state,
    ):
        """Test simulate for upgrade action in optimal state."""
        from nvidia_inst.cli.main import execute_driver_change

        option = optimal_state.options[0]  # upgrade
        mock_packages.return_value = ["nvidia-driver-535"]

        result = execute_driver_change(
            option,
            optimal_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
            with_cuda=True,
            cuda_version="12.2",
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "SIMULATE MODE - Driver Change" in output
        assert "nvidia-driver-535" in output

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_install_wrong_branch(
        self,
        mock_stdout,
        mock_packages,
        mock_distro,
        mock_gpu,
        mock_driver_range,
        wrong_branch_state,
    ):
        """Test dry-run for install action in wrong_branch state."""
        from nvidia_inst.cli.main import execute_driver_change

        option = wrong_branch_state.options[0]  # install
        mock_packages.return_value = ["nvidia-driver-535"]

        result = execute_driver_change(
            option,
            wrong_branch_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
            with_cuda=False,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "SIMULATE MODE - Driver Change" in output

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_install_nouveau(
        self,
        mock_stdout,
        mock_packages,
        mock_distro,
        mock_gpu,
        mock_driver_range,
        nouveau_state,
    ):
        """Test dry-run for install action in nouveau state."""
        from nvidia_inst.cli.main import execute_driver_change

        option = nouveau_state.options[0]  # install
        mock_packages.return_value = ["nvidia-driver-535"]

        result = execute_driver_change(
            option,
            nouveau_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
            with_cuda=True,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "SIMULATE MODE - Driver Change" in output

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_install_nothing(
        self,
        mock_stdout,
        mock_packages,
        mock_distro,
        mock_gpu,
        mock_driver_range,
        nothing_state,
    ):
        """Test dry-run for install action in nothing state."""
        from nvidia_inst.cli.main import execute_driver_change

        option = nothing_state.options[0]  # install
        mock_packages.return_value = ["nvidia-driver-535"]

        result = execute_driver_change(
            option,
            nothing_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
            with_cuda=False,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "SIMULATE MODE - Driver Change" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_keep(
        self, mock_stdout, mock_distro, mock_gpu, mock_driver_range, optimal_state
    ):
        """Test dry-run for keep action."""
        from nvidia_inst.cli.main import execute_driver_change

        option = optimal_state.options[1]  # keep

        result = execute_driver_change(
            option,
            optimal_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "No changes made" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_cancel(
        self, mock_stdout, mock_distro, mock_gpu, mock_driver_range, nothing_state
    ):
        """Test dry-run for cancel action."""
        from nvidia_inst.cli.main import execute_driver_change

        option = nothing_state.options[2]  # cancel

        result = execute_driver_change(
            option,
            nothing_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Cancelled" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_revert_nouveau(
        self, mock_stdout, mock_distro, mock_gpu, mock_driver_range, optimal_state
    ):
        """Test dry-run for revert_nouveau action."""
        from nvidia_inst.cli.main import execute_driver_change

        option = optimal_state.options[2]  # revert_nouveau

        result = execute_driver_change(
            option,
            optimal_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Nouveau Installation" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_revert_nouveau_nothing_state(
        self, mock_stdout, mock_distro, mock_gpu, mock_driver_range, nothing_state
    ):
        """Test dry-run for revert_nouveau action in nothing state."""
        from nvidia_inst.cli.main import execute_driver_change

        option = nothing_state.options[1]  # revert_nouveau

        result = execute_driver_change(
            option,
            nothing_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
        )

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Nouveau Installation" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_simulate_unknown_action(
        self, mock_stdout, mock_distro, mock_gpu, mock_driver_range, optimal_state
    ):
        """Test dry-run for unknown action."""
        from nvidia_inst.cli.main import execute_driver_change

        option = DriverOption(99, "Unknown", "unknown_action")

        result = execute_driver_change(
            option,
            optimal_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
        )

        assert result == 1
        output = mock_stdout.getvalue()
        assert "Unknown action" in output


class TestRevertToNouveauCli:
    """Test revert_to_nouveau_cli function."""

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.check_nvidia_packages_installed")
    @patch("sys.stdout", new_callable=StringIO)
    def test_revert_no_packages(
        self, mock_stdout, mock_packages, mock_distro, mock_args
    ):
        """Test revert when no packages installed."""
        from nvidia_inst.cli.main import revert_to_nouveau_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_packages.return_value = []

        result = revert_to_nouveau_cli()

        assert result == 0
        output = mock_stdout.getvalue()
        assert "No proprietary Nvidia packages found" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.check_nvidia_packages_installed")
    @patch("builtins.input", return_value="n")
    @patch("sys.stdout", new_callable=StringIO)
    def test_revert_cancelled(
        self, mock_stdout, mock_input, mock_packages, mock_distro, mock_args
    ):
        """Test revert when user cancels."""
        from nvidia_inst.cli.main import revert_to_nouveau_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_packages.return_value = ["nvidia-driver-535"]

        result = revert_to_nouveau_cli()

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Cancelled" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.check_nvidia_packages_installed")
    @patch("builtins.input", return_value="y")
    @patch("nvidia_inst.cli.main.require_root", return_value=False)
    @patch("sys.stdout", new_callable=StringIO)
    def test_revert_no_root(
        self, mock_stdout, mock_root, mock_input, mock_packages, mock_distro, mock_args
    ):
        """Test revert when no root privileges."""
        from nvidia_inst.cli.main import revert_to_nouveau_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_packages.return_value = ["nvidia-driver-535"]

        result = revert_to_nouveau_cli()

        assert result == 1
        output = mock_stdout.getvalue()
        assert "Root privileges required" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.check_nvidia_packages_installed")
    @patch("builtins.input", return_value="y")
    @patch("nvidia_inst.cli.main.require_root", return_value=True)
    @patch("nvidia_inst.cli.main.revert_to_nouveau")
    @patch("sys.stdout", new_callable=StringIO)
    def test_revert_success(
        self,
        mock_stdout,
        mock_revert,
        mock_root,
        mock_input,
        mock_packages,
        mock_distro,
        mock_args,
    ):
        """Test successful revert."""
        from nvidia_inst.cli.main import revert_to_nouveau_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_packages.return_value = ["nvidia-driver-535"]
        mock_revert.return_value = MagicMock(
            success=True,
            message="Reverted successfully",
            packages_removed=["nvidia-driver-535"],
        )

        result = revert_to_nouveau_cli()

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Reverted successfully" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.check_nvidia_packages_installed")
    @patch("builtins.input", return_value="y")
    @patch("nvidia_inst.cli.main.require_root", return_value=True)
    @patch("nvidia_inst.cli.main.revert_to_nouveau")
    @patch("sys.stdout", new_callable=StringIO)
    def test_revert_failure(
        self,
        mock_stdout,
        mock_revert,
        mock_root,
        mock_input,
        mock_packages,
        mock_distro,
        mock_args,
    ):
        """Test failed revert."""
        from nvidia_inst.cli.main import revert_to_nouveau_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_packages.return_value = ["nvidia-driver-535"]
        mock_revert.return_value = MagicMock(
            success=False, errors=["Package removal failed"]
        )

        result = revert_to_nouveau_cli()

        assert result == 1
        output = mock_stdout.getvalue()
        assert "Revert failed" in output


class TestSetPowerProfileCli:
    """Test set_power_profile_cli function."""

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.get_native_tool")
    @patch("sys.stdout", new_callable=StringIO)
    def test_no_native_tool(self, mock_stdout, mock_native, mock_distro, mock_args):
        """Test when no native tool found."""
        from nvidia_inst.cli.main import set_power_profile_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_native.return_value = (None, None, None)

        result = set_power_profile_cli("intel")

        assert result == 1
        output = mock_stdout.getvalue()
        assert "No native hybrid graphics tool found" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.get_native_tool")
    @patch("nvidia_inst.cli.main.require_root", return_value=False)
    @patch("sys.stdout", new_callable=StringIO)
    def test_no_root(self, mock_stdout, mock_root, mock_native, mock_distro, mock_args):
        """Test when no root privileges."""
        from nvidia_inst.cli.main import set_power_profile_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_native.return_value = ("prime-select", "nvidia", "intel")

        result = set_power_profile_cli("intel")

        assert result == 1
        output = mock_stdout.getvalue()
        assert "Root privileges required" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.get_native_tool")
    @patch("nvidia_inst.cli.main.require_root", return_value=True)
    @patch("nvidia_inst.cli.main.set_power_profile")
    @patch("sys.stdout", new_callable=StringIO)
    def test_success_intel(
        self, mock_stdout, mock_set, mock_root, mock_native, mock_distro, mock_args
    ):
        """Test successful power profile change to intel."""
        from nvidia_inst.cli.main import set_power_profile_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_native.return_value = ("prime-select", "nvidia", "intel")
        mock_set.return_value = True

        result = set_power_profile_cli("intel")

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Power profile set to: intel" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.get_native_tool")
    @patch("nvidia_inst.cli.main.require_root", return_value=True)
    @patch("nvidia_inst.cli.main.set_power_profile")
    @patch("sys.stdout", new_callable=StringIO)
    def test_success_hybrid(
        self, mock_stdout, mock_set, mock_root, mock_native, mock_distro, mock_args
    ):
        """Test successful power profile change to hybrid."""
        from nvidia_inst.cli.main import set_power_profile_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_native.return_value = ("prime-select", "nvidia", "intel")
        mock_set.return_value = True

        result = set_power_profile_cli("hybrid")

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Power profile set to: hybrid" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.get_native_tool")
    @patch("nvidia_inst.cli.main.require_root", return_value=True)
    @patch("nvidia_inst.cli.main.set_power_profile")
    @patch("sys.stdout", new_callable=StringIO)
    def test_success_nvidia(
        self, mock_stdout, mock_set, mock_root, mock_native, mock_distro, mock_args
    ):
        """Test successful power profile change to nvidia."""
        from nvidia_inst.cli.main import set_power_profile_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_native.return_value = ("prime-select", "nvidia", "intel")
        mock_set.return_value = True

        result = set_power_profile_cli("nvidia")

        assert result == 0
        output = mock_stdout.getvalue()
        assert "Power profile set to: nvidia" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.get_native_tool")
    @patch("nvidia_inst.cli.main.require_root", return_value=True)
    @patch("nvidia_inst.cli.main.set_power_profile")
    @patch("sys.stdout", new_callable=StringIO)
    def test_failure(
        self, mock_stdout, mock_set, mock_root, mock_native, mock_distro, mock_args
    ):
        """Test failed power profile change."""
        from nvidia_inst.cli.main import set_power_profile_cli

        mock_args.return_value = MagicMock()
        mock_distro.return_value = MagicMock(id="ubuntu", version_id="22.04")
        mock_native.return_value = ("prime-select", "nvidia", "intel")
        mock_set.return_value = False

        result = set_power_profile_cli("intel")

        assert result == 1
        output = mock_stdout.getvalue()
        assert "Failed to set power profile" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.has_nvidia_gpu")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.detect_gpu")
    @patch("nvidia_inst.cli.main.get_driver_range")
    @patch("nvidia_inst.cli.compatibility.PrerequisitesChecker")
    @patch("nvidia_inst.cli.compatibility.print_compatibility_info")
    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.gpu.compatibility.is_driver_compatible")
    @patch("nvidia_inst.gpu.hybrid.detect_hybrid")
    @patch("sys.stdout", new_callable=StringIO)
    def test_check_with_working_driver(
        self,
        mock_stdout,
        mock_hybrid,
        mock_compat,
        mock_working,
        mock_print,
        mock_prereq,
        mock_range,
        mock_gpu,
        mock_distro,
        mock_has_gpu,
        mock_args,
    ):
        """Test --check with working driver."""
        from nvidia_inst.cli.main import check_compatibility

        mock_args.return_value = MagicMock()
        mock_has_gpu.return_value = True
        mock_distro.return_value = MagicMock(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0",
        )
        mock_gpu.return_value = MagicMock(model="RTX 3080", generation="ampere")
        mock_range.return_value = MagicMock(
            min_version="520.56.06",
            max_version="590.48.01",
            is_eol=False,
            is_limited=False,
        )
        mock_prereq.return_value = MagicMock(
            check_all=MagicMock(return_value=MagicMock(success=True))
        )
        mock_working.return_value = MagicMock(
            is_working=True, driver_version="535.154.05"
        )
        mock_compat.return_value = True
        mock_hybrid.return_value = None

        result = check_compatibility()

        assert result == 0
        output = mock_stdout.getvalue()
        assert "NVIDIA driver is working" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.has_nvidia_gpu")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.detect_gpu")
    @patch("nvidia_inst.cli.main.get_driver_range")
    @patch("nvidia_inst.cli.compatibility.PrerequisitesChecker")
    @patch("nvidia_inst.cli.compatibility.print_compatibility_info")
    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.gpu.compatibility.is_driver_compatible")
    @patch("nvidia_inst.gpu.hybrid.detect_hybrid")
    @patch("sys.stdout", new_callable=StringIO)
    def test_check_with_wrong_branch_eol(
        self,
        mock_stdout,
        mock_hybrid,
        mock_compat,
        mock_working,
        mock_print,
        mock_prereq,
        mock_range,
        mock_gpu,
        mock_distro,
        mock_has_gpu,
        mock_args,
    ):
        """Test --check with wrong branch installed."""
        from nvidia_inst.cli.main import check_compatibility

        mock_args.return_value = MagicMock()
        mock_has_gpu.return_value = True
        mock_distro.return_value = MagicMock(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0",
        )
        mock_gpu.return_value = MagicMock(model="RTX 3080", generation="ampere")
        mock_range.return_value = MagicMock(
            min_version="520.56.06",
            max_version="590.48.01",
            is_eol=False,
            is_limited=False,
        )
        mock_prereq.return_value = MagicMock(
            check_all=MagicMock(return_value=MagicMock(success=True))
        )
        mock_working.return_value = MagicMock(is_working=True, driver_version="580.142")
        mock_compat.return_value = False
        mock_hybrid.return_value = None

        result = check_compatibility()

        assert result == 0
        output = mock_stdout.getvalue()
        assert "may not be optimal" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.has_nvidia_gpu")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.detect_gpu")
    @patch("nvidia_inst.cli.main.get_driver_range")
    @patch("nvidia_inst.cli.compatibility.PrerequisitesChecker")
    @patch("nvidia_inst.cli.compatibility.print_compatibility_info")
    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.gpu.hybrid.detect_hybrid")
    @patch("sys.stdout", new_callable=StringIO)
    def test_check_with_no_driver(
        self,
        mock_stdout,
        mock_hybrid,
        mock_working,
        mock_print,
        mock_prereq,
        mock_range,
        mock_gpu,
        mock_distro,
        mock_has_gpu,
        mock_args,
    ):
        """Test --check with no driver installed."""
        from nvidia_inst.cli.main import check_compatibility

        mock_args.return_value = MagicMock()
        mock_has_gpu.return_value = True
        mock_distro.return_value = MagicMock(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0",
        )
        mock_gpu.return_value = MagicMock(model="RTX 3080", generation="ampere")
        mock_range.return_value = MagicMock(
            min_version="520.56.06",
            max_version="590.48.01",
            is_eol=False,
            is_limited=False,
        )
        mock_prereq.return_value = MagicMock(
            check_all=MagicMock(return_value=MagicMock(success=True))
        )
        mock_working.return_value = MagicMock(is_working=False, gpu_detected=True)
        mock_hybrid.return_value = None

        result = check_compatibility()

        assert result == 0
        output = mock_stdout.getvalue()
        # Check that compatibility check completed successfully
        assert "System Compatibility Check" in output or "Prerequisites Check" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.has_nvidia_gpu")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.detect_gpu")
    @patch("nvidia_inst.cli.main.get_driver_range")
    @patch("nvidia_inst.cli.compatibility.PrerequisitesChecker")
    @patch("nvidia_inst.cli.compatibility.print_compatibility_info")
    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.gpu.hybrid.detect_hybrid")
    @patch("sys.stdout", new_callable=StringIO)
    def test_check_with_hybrid_graphics(
        self,
        mock_stdout,
        mock_hybrid,
        mock_working,
        mock_print,
        mock_prereq,
        mock_range,
        mock_gpu,
        mock_distro,
        mock_has_gpu,
        mock_args,
    ):
        """Test --check with hybrid graphics."""
        from nvidia_inst.cli.main import check_compatibility

        mock_args.return_value = MagicMock()
        mock_has_gpu.return_value = True
        mock_distro.return_value = MagicMock(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0",
        )
        mock_gpu.return_value = MagicMock(model="RTX 3080", generation="ampere")
        mock_range.return_value = MagicMock(
            min_version="520.56.06",
            max_version="590.48.01",
            is_eol=False,
            is_limited=False,
        )
        mock_prereq.return_value = MagicMock(
            check_all=MagicMock(return_value=MagicMock(success=True))
        )
        mock_working.return_value = MagicMock(
            is_working=True, driver_version="535.154.05"
        )
        mock_hybrid.return_value = MagicMock(
            system_type="nvidia_prime",
            igpu_type="intel",
            dgpu_model="RTX 3080",
        )

        result = check_compatibility()

        assert result == 0
        output = mock_stdout.getvalue()
        assert "HYBRID GRAPHICS DETECTED" in output

    @patch("nvidia_inst.cli.main.parse_args")
    @patch("nvidia_inst.cli.main.has_nvidia_gpu")
    @patch("nvidia_inst.cli.main.detect_distro")
    @patch("nvidia_inst.cli.main.detect_gpu")
    @patch("nvidia_inst.cli.main.get_driver_range")
    @patch("nvidia_inst.cli.compatibility.PrerequisitesChecker")
    @patch("nvidia_inst.cli.compatibility.print_compatibility_info")
    @patch("nvidia_inst.installer.validation.is_nvidia_working")
    @patch("nvidia_inst.gpu.hybrid.detect_hybrid")
    @patch("sys.stdout", new_callable=StringIO)
    def test_check_with_eol_gpu(
        self,
        mock_stdout,
        mock_hybrid,
        mock_working,
        mock_print,
        mock_prereq,
        mock_range,
        mock_gpu,
        mock_distro,
        mock_has_gpu,
        mock_args,
    ):
        """Test --check with EOL GPU."""
        from nvidia_inst.cli.main import check_compatibility

        mock_args.return_value = MagicMock()
        mock_has_gpu.return_value = True
        mock_distro.return_value = MagicMock(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0",
        )
        mock_gpu.return_value = MagicMock(model="GTX 780", generation="kepler")
        mock_range.return_value = MagicMock(
            min_version="390.157.0",
            max_version="470.256.02",
            is_eol=True,
            is_limited=True,
            eol_message="Kepler GPUs are end-of-life",
        )
        mock_prereq.return_value = MagicMock(
            check_all=MagicMock(return_value=MagicMock(success=True))
        )
        mock_working.return_value = MagicMock(is_working=False, gpu_detected=True)
        mock_hybrid.return_value = None

        result = check_compatibility()

        assert result == 0
        output = mock_stdout.getvalue()
        # Check that compatibility check completed successfully
        assert "System Compatibility Check" in output or "Prerequisites Check" in output
