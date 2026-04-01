"""Tests for driver state detection and switching."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetCurrentDriverType:
    """Tests for get_current_driver_type function."""

    def test_proprietary_working(self):
        """Test when proprietary driver is working."""
        with patch(
            "nvidia_inst.installer.validation.is_nvidia_working"
        ) as mock_working:
            mock_working.return_value = MagicMock(is_working=True)
            from nvidia_inst.installer.driver import get_current_driver_type

            result = get_current_driver_type()
            assert result == "proprietary"

    def test_nouveau_loaded(self):
        """Test when nouveau driver is loaded."""
        with patch(
            "nvidia_inst.installer.validation.is_nvidia_working"
        ) as mock_working:
            mock_working.return_value = MagicMock(is_working=False)
            with patch("nvidia_inst.installer.driver.check_nouveau", return_value=True):
                from nvidia_inst.installer.driver import get_current_driver_type

                result = get_current_driver_type()
                assert result == "nouveau"

    def test_nothing_installed(self):
        """Test when nothing is installed."""
        with patch(
            "nvidia_inst.installer.validation.is_nvidia_working"
        ) as mock_working:
            mock_working.return_value = MagicMock(is_working=False)
            with patch(
                "nvidia_inst.installer.driver.check_nouveau", return_value=False
            ):
                from nvidia_inst.installer.driver import get_current_driver_type

                result = get_current_driver_type()
                assert result == "none"


class TestDriverStatusEnum:
    """Tests for DriverStatus enum."""

    def test_status_values(self):
        """Test DriverStatus enum values exist."""
        from nvidia_inst.cli import DriverStatus

        assert DriverStatus.OPTIMAL.value == "optimal"
        assert DriverStatus.WRONG_BRANCH.value == "wrong_branch"
        assert DriverStatus.NOUVEAU_ACTIVE.value == "nouveau_active"
        assert DriverStatus.NOTHING.value == "nothing"


class TestDriverOption:
    """Tests for DriverOption dataclass."""

    def test_option_creation(self):
        """Test DriverOption can be created."""
        from nvidia_inst.cli import DriverOption

        opt = DriverOption(number=1, description="Test", action="test")
        assert opt.number == 1
        assert opt.description == "Test"
        assert opt.action == "test"
        assert opt.recommended is False

    def test_option_with_recommended(self):
        """Test DriverOption with recommended=True."""
        from nvidia_inst.cli import DriverOption

        opt = DriverOption(
            number=1, description="Test", action="test", recommended=True
        )
        assert opt.recommended is True


class TestDriverState:
    """Tests for DriverState dataclass."""

    def test_state_creation(self):
        """Test DriverState can be created."""
        from nvidia_inst.cli import DriverOption, DriverState, DriverStatus

        state = DriverState(
            status=DriverStatus.OPTIMAL,
            current_version="590.48.01",
            is_compatible=True,
            is_optimal=True,
            suggested_packages=["nvidia-driver-535"],
            options=[DriverOption(1, "Test", "test")],
            message="Test message",
        )
        assert state.status == DriverStatus.OPTIMAL
        assert state.current_version == "590.48.01"
        assert state.is_compatible is True
        assert len(state.options) == 1


class TestDetectDriverState:
    """Tests for detect_driver_state function."""

    @pytest.fixture
    def mock_gpu(self):
        """Create mock GPU."""
        gpu = MagicMock()
        gpu.model = "NVIDIA GeForce RTX 3080"
        gpu.generation = "ampere"
        return gpu

    @pytest.fixture
    def mock_driver_range(self):
        """Create mock driver range."""
        dr = MagicMock()
        dr.max_version = "590.48.01"
        dr.max_branch = "590"
        dr.is_eol = False
        dr.is_limited = False
        dr.min_version = "520.56.06"
        dr.cuda_min = "11.0"
        dr.cuda_max = "12.2"
        return dr

    def test_optimal_driver(self, mock_gpu, mock_driver_range):
        """Test detection when driver is optimal."""
        with (
            patch("nvidia_inst.cli.main.get_current_driver_type") as mock_type,
            patch("nvidia_inst.cli.main.is_nvidia_working") as mock_working,
            patch("nvidia_inst.cli.main.is_driver_compatible") as mock_compat,
            patch("nvidia_inst.cli.main.get_compatible_driver_packages") as mock_pkgs,
            patch(
                "nvidia_inst.cli.main.check_nvidia_open_available"
            ) as mock_open_avail,
            patch("nvidia_inst.cli.main.check_nonfree_available") as mock_nonfree_avail,
        ):
            mock_type.return_value = "proprietary"
            mock_working.return_value = MagicMock(
                is_working=True, driver_version="590.48.01"
            )
            mock_compat.return_value = True
            mock_pkgs.return_value = ["nvidia-driver-535"]
            mock_open_avail.return_value = True
            mock_nonfree_avail.return_value = True

            from nvidia_inst.cli import detect_driver_state

            state = detect_driver_state(mock_gpu, mock_driver_range, "ubuntu")

            assert state.status.value == "optimal"
            assert state.current_version == "590.48.01"
            assert state.is_optimal is True
            assert len(state.options) == 4
            assert "NVIDIA" in state.options[0].description

    def test_wrong_branch(self, mock_gpu, mock_driver_range):
        """Test detection when wrong branch installed."""
        with (
            patch("nvidia_inst.cli.main.get_current_driver_type") as mock_type,
            patch("nvidia_inst.cli.main.is_nvidia_working") as mock_working,
            patch("nvidia_inst.cli.main.is_driver_compatible") as mock_compat,
            patch("nvidia_inst.cli.main.get_compatible_driver_packages") as mock_pkgs,
        ):
            mock_type.return_value = "proprietary"
            mock_working.return_value = MagicMock(
                is_working=True, driver_version="590.48.01"
            )
            mock_compat.return_value = False
            mock_pkgs.return_value = ["nvidia-driver-580"]

            from nvidia_inst.cli import detect_driver_state

            state = detect_driver_state(mock_gpu, mock_driver_range, "ubuntu")

            assert state.status.value == "wrong_branch"
            assert state.is_compatible is False
            assert state.is_optimal is False

    def test_nouveau_active(self, mock_gpu, mock_driver_range):
        """Test detection when nouveau is active."""
        with (
            patch("nvidia_inst.cli.main.get_current_driver_type") as mock_type,
            patch("nvidia_inst.cli.main.is_nvidia_working") as mock_working,
            patch("nvidia_inst.cli.main.get_compatible_driver_packages") as mock_pkgs,
            patch(
                "nvidia_inst.cli.main.check_nvidia_open_available"
            ) as mock_open_avail,
            patch("nvidia_inst.cli.main.check_nonfree_available") as mock_nonfree_avail,
        ):
            mock_type.return_value = "nouveau"
            mock_working.return_value = MagicMock(is_working=False)
            mock_pkgs.return_value = ["nvidia-driver-535"]
            mock_open_avail.return_value = True
            mock_nonfree_avail.return_value = True

            from nvidia_inst.cli import detect_driver_state

            state = detect_driver_state(mock_gpu, mock_driver_range, "ubuntu")

            assert state.status.value == "nouveau_active"
            assert len(state.options) == 3
            assert state.options[0].action == "install"

    def test_nothing_installed(self, mock_gpu, mock_driver_range):
        """Test detection when nothing installed."""
        with (
            patch("nvidia_inst.cli.main.get_current_driver_type") as mock_type,
            patch("nvidia_inst.cli.main.is_nvidia_working") as mock_working,
            patch("nvidia_inst.cli.main.get_compatible_driver_packages") as mock_pkgs,
            patch(
                "nvidia_inst.cli.main.check_nvidia_open_available"
            ) as mock_open_avail,
            patch("nvidia_inst.cli.main.check_nonfree_available") as mock_nonfree_avail,
        ):
            mock_type.return_value = "none"
            mock_working.return_value = MagicMock(is_working=False)
            mock_pkgs.return_value = ["nvidia-driver-535"]
            mock_open_avail.return_value = True
            mock_nonfree_avail.return_value = True

            from nvidia_inst.cli import detect_driver_state

            state = detect_driver_state(mock_gpu, mock_driver_range, "ubuntu")

            assert state.status.value == "nothing"
            assert len(state.options) == 4
            assert state.options[-1].action == "cancel"
