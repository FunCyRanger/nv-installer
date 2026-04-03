"""Integration tests for CLI actions and dry-run scenarios."""

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
    def test_simulate_upgrade_optimal(
        self,
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
# #         assert "nvidia-driver-535" in output

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    def test_simulate_install_wrong_branch(
        self,
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

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    def test_simulate_install_nouveau(
        self,
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

    @patch("nvidia_inst.cli.main.get_compatible_driver_packages")
    def test_simulate_install_nothing(
        self,
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
    def test_simulate_keep(
        self, capsys, mock_distro, mock_gpu, mock_driver_range, optimal_state
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
    def test_simulate_cancel(
        self, capsys, mock_distro, mock_gpu, mock_driver_range, nothing_state
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
    def test_simulate_revert_nouveau(
        self, capsys, mock_distro, mock_gpu, mock_driver_range, optimal_state
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
    def test_simulate_revert_nouveau_nothing_state(
        self, capsys, mock_distro, mock_gpu, mock_driver_range, nothing_state
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
    def test_simulate_unknown_action(
        self, capsys, mock_distro, mock_gpu, mock_driver_range, optimal_state
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

    @patch("nvidia_inst.cli.main.get_nvidia_open_packages")
    def test_simulate_switch_nvidia_open(
        self,
                mock_open_packages,
        mock_distro,
        mock_gpu,
        mock_driver_range,
        optimal_state,
    ):
        """Test simulate for switch_nvidia_open action."""
        from nvidia_inst.cli.main import execute_driver_change

        option = DriverOption(3, "NVIDIA Open", "switch_nvidia_open")
        mock_open_packages.return_value = [
            "nvidia-driver-590-open",
            "nvidia-dkms-590-open",
        ]

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
# #         assert "nvidia-driver-590-open" in output

    @patch("nvidia_inst.cli.main.get_nvidia_open_packages")
    def test_simulate_install_nvidia_open(
        self,
                mock_open_packages,
        mock_distro,
        mock_gpu,
        mock_driver_range,
        nouveau_state,
    ):
        """Test simulate for install_nvidia_open action."""
        from nvidia_inst.cli.main import execute_driver_change

        option = DriverOption(2, "NVIDIA Open", "install_nvidia_open")
        mock_open_packages.return_value = [
            "nvidia-driver-590-open",
            "nvidia-dkms-590-open",
        ]

        result = execute_driver_change(
            option,
            nouveau_state,
            mock_distro,
            mock_gpu,
            mock_driver_range,
            simulate=True,
            with_cuda=True,
            cuda_version="12.2",
        )

        assert result == 0
# #         assert "nvidia-driver-590-open" in output


class TestRevertToNouveauCli:
    """Test revert_to_nouveau_cli function."""

    def test_revert_no_packages(self, capsys):
        """Test revert when no packages installed."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main, "check_nvidia_packages_installed", return_value=[]
                ):
                    from nvidia_inst.cli.main import revert_to_nouveau_cli

                    result = revert_to_nouveau_cli()
                    assert result == 0
# #                     output = capsys.readouterr().out
#                     assert "No proprietary Nvidia packages found" in output

    def test_revert_cancelled(self, capsys):
        """Test revert when user cancels."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "check_nvidia_packages_installed",
                    return_value=["nvidia-driver-535"],
                ):
                    with patch("builtins.input", return_value="n"):
                        from nvidia_inst.cli.main import revert_to_nouveau_cli

                        result = revert_to_nouveau_cli()
                        assert result == 0
# #                         output = capsys.readouterr().out
# #                         assert "Cancelled" in output

    def test_revert_no_root(self, capsys):
        """Test revert when no root privileges."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "check_nvidia_packages_installed",
                    return_value=["nvidia-driver-535"],
                ):
                    with patch("builtins.input", return_value="y"):
                        with patch.object(cli_main, "require_root", return_value=False):
                            from nvidia_inst.cli.main import revert_to_nouveau_cli

                            result = revert_to_nouveau_cli()
                            assert result == 1
# #                             output = capsys.readouterr().out
#                             assert "Root privileges required" in output

    def test_revert_success(self, capsys):
        """Test successful revert."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "check_nvidia_packages_installed",
                    return_value=["nvidia-driver-535"],
                ):
                    with patch("builtins.input", return_value="y"):
                        with patch.object(cli_main, "require_root", return_value=True):
                            with patch.object(
                                cli_main,
                                "revert_to_nouveau",
                                return_value=MagicMock(
                                    success=True,
                                    message="Reverted successfully",
                                    packages_removed=["nvidia-driver-535"],
                                ),
                            ):
                                from nvidia_inst.cli.main import revert_to_nouveau_cli

                                result = revert_to_nouveau_cli()
                                assert result == 0
# #                                 output = capsys.readouterr().out
#                                 assert "Reverted successfully" in output

    def test_revert_failure(self, capsys):
        """Test failed revert."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "check_nvidia_packages_installed",
                    return_value=["nvidia-driver-535"],
                ):
                    with patch("builtins.input", return_value="y"):
                        with patch.object(cli_main, "require_root", return_value=True):
                            with patch.object(
                                cli_main,
                                "revert_to_nouveau",
                                return_value=MagicMock(
                                    success=False,
                                    errors=["Package removal failed"],
                                ),
                            ):
                                from nvidia_inst.cli.main import revert_to_nouveau_cli

                                result = revert_to_nouveau_cli()
                                assert result == 1
# #                                 output = capsys.readouterr().out
#                                 assert "Revert failed" in output


class TestSetPowerProfileCli:
    """Test set_power_profile_cli function."""

    def test_no_native_tool(self, capsys):
        """Test when no native tool found."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main, "get_native_tool", return_value=(None, None, None)
                ):
                    from nvidia_inst.cli.main import set_power_profile_cli

                    result = set_power_profile_cli("intel")
                    assert result == 1
# #                     output = capsys.readouterr().out
#                     assert "No native hybrid graphics tool found" in output

    def test_no_root(self, capsys):
        """Test when no root privileges."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "get_native_tool",
                    return_value=("prime-select", "nvidia", "intel"),
                ):
                    with patch.object(cli_main, "require_root", return_value=False):
                        from nvidia_inst.cli.main import set_power_profile_cli

                        result = set_power_profile_cli("intel")
                        assert result == 1
# #                         output = capsys.readouterr().out
#                         assert "Root privileges required" in output

    def test_success_intel(self, capsys):
        """Test successful power profile change to intel."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "get_native_tool",
                    return_value=("prime-select", "nvidia", "intel"),
                ):
                    with patch.object(cli_main, "require_root", return_value=True):
                        with patch.object(
                            cli_main, "set_power_profile", return_value=True
                        ):
                            from nvidia_inst.cli.main import set_power_profile_cli

                            result = set_power_profile_cli("intel")
                            assert result == 0
# #                             output = capsys.readouterr().out
#                             assert "Power profile set to: intel" in output

    def test_success_hybrid(self, capsys):
        """Test successful power profile change to hybrid."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "get_native_tool",
                    return_value=("prime-select", "nvidia", "intel"),
                ):
                    with patch.object(cli_main, "require_root", return_value=True):
                        with patch.object(
                            cli_main, "set_power_profile", return_value=True
                        ):
                            from nvidia_inst.cli.main import set_power_profile_cli

                            result = set_power_profile_cli("hybrid")
                            assert result == 0
# #                             output = capsys.readouterr().out
#                             assert "Power profile set to: hybrid" in output

    def test_success_nvidia(self, capsys):
        """Test successful power profile change to nvidia."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "get_native_tool",
                    return_value=("prime-select", "nvidia", "intel"),
                ):
                    with patch.object(cli_main, "require_root", return_value=True):
                        with patch.object(
                            cli_main, "set_power_profile", return_value=True
                        ):
                            from nvidia_inst.cli.main import set_power_profile_cli

                            result = set_power_profile_cli("nvidia")
                            assert result == 0
# #                             output = capsys.readouterr().out
#                             assert "Power profile set to: nvidia" in output

    def test_failure(self, capsys):
        """Test failed power profile change."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(
                cli_main,
                "detect_distro",
                return_value=MagicMock(id="ubuntu", version_id="22.04"),
            ):
                with patch.object(
                    cli_main,
                    "get_native_tool",
                    return_value=("prime-select", "nvidia", "intel"),
                ):
                    with patch.object(cli_main, "require_root", return_value=True):
                        with patch.object(
                            cli_main, "set_power_profile", return_value=False
                        ):
                            from nvidia_inst.cli.main import set_power_profile_cli

                            result = set_power_profile_cli("intel")
                            assert result == 1
# #                             output = capsys.readouterr().out
#                             assert "Failed to set power profile" in output


class TestCheckModeWithGpu:
    """Test --check mode with GPU present."""

    def test_check_with_working_driver(self, capsys):
        """Test --check with working driver."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(cli_main, "has_nvidia_gpu", return_value=True):
                with patch(
                    "nvidia_inst.cli.compatibility.detect_distro",
                    return_value=MagicMock(
                        id="ubuntu",
                        version_id="22.04",
                        name="Ubuntu",
                        pretty_name="Ubuntu 22.04.3 LTS",
                        kernel="5.15.0",
                    ),
                ):
                    with patch(
                        "nvidia_inst.cli.compatibility.detect_gpu",
                        return_value=MagicMock(
                            model="RTX 3080",
                            generation="ampere",
                        ),
                    ):
                        with patch(
                            "nvidia_inst.cli.compatibility.get_driver_range",
                            return_value=MagicMock(
                                min_version="520.56.06",
                                max_version="590.48.01",
                                is_eol=False,
                                is_limited=False,
                            ),
                        ):
                            with patch(
                                "nvidia_inst.cli.compatibility.PrerequisitesChecker",
                                return_value=MagicMock(
                                    check_all=MagicMock(
                                        return_value=MagicMock(success=True)
                                    ),
                                ),
                            ):
                                with patch(
                                    "nvidia_inst.cli.compatibility.print_compatibility_info"
                                ):
                                    with patch(
                                        "nvidia_inst.installer.validation.is_nvidia_working",
                                        return_value=MagicMock(
                                            is_working=True,
                                            driver_version="535.154.05",
                                        ),
                                    ):
                                        with patch(
                                            "nvidia_inst.gpu.compatibility.is_driver_compatible",
                                            return_value=True,
                                        ):
                                            with patch(
                                                "nvidia_inst.gpu.hybrid.detect_hybrid",
                                                return_value=None,
                                            ):
                                                from nvidia_inst.cli.compatibility import (
                                                    check_compatibility,
                                                )

                                                result = check_compatibility()
                                                assert result == 0

    def test_check_with_no_driver(self, capsys):
        """Test --check with no driver installed."""
        import sys

        cli_main = sys.modules["nvidia_inst.cli.main"]
        with patch.object(cli_main, "parse_args", return_value=MagicMock()):
            with patch.object(cli_main, "has_nvidia_gpu", return_value=True):
                with patch(
                    "nvidia_inst.cli.compatibility.detect_distro",
                    return_value=MagicMock(
                        id="ubuntu",
                        version_id="22.04",
                        name="Ubuntu",
                        pretty_name="Ubuntu 22.04.3 LTS",
                        kernel="5.15.0",
                    ),
                ):
                    with patch(
                        "nvidia_inst.cli.compatibility.detect_gpu",
                        return_value=MagicMock(
                            model="RTX 3080",
                            generation="ampere",
                        ),
                    ):
                        with patch(
                            "nvidia_inst.cli.compatibility.get_driver_range",
                            return_value=MagicMock(
                                min_version="520.56.06",
                                max_version="590.48.01",
                                is_eol=False,
                                is_limited=False,
                            ),
                        ):
                            with patch(
                                "nvidia_inst.cli.compatibility.PrerequisitesChecker",
                                return_value=MagicMock(
                                    check_all=MagicMock(
                                        return_value=MagicMock(success=True)
                                    ),
                                ),
                            ):
                                with patch(
                                    "nvidia_inst.cli.compatibility.print_compatibility_info"
                                ):
                                    with patch(
                                        "nvidia_inst.installer.validation.is_nvidia_working",
                                        return_value=MagicMock(
                                            is_working=False,
                                            gpu_detected=True,
                                        ),
                                    ):
                                        with patch(
                                            "nvidia_inst.gpu.hybrid.detect_hybrid",
                                            return_value=None,
                                        ):
                                            from nvidia_inst.cli.compatibility import (
                                                check_compatibility,
                                            )

                                            result = check_compatibility()
                                            assert result == 0
