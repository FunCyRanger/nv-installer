"""Tests for CLI argument parsing and integration."""

import sys
from unittest.mock import MagicMock


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_args(self, monkeypatch):
        """Test default arguments when no flags provided."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.gui is False
        assert args.check is False
        assert args.yes is False
        assert args.debug is False
        assert args.dry_run is False

    def test_gui_flag(self, monkeypatch):
        """Test --gui flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--gui"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.gui is True

    def test_gui_type_tkinter(self, monkeypatch):
        """Test --gui-type tkinter."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--gui-type", "tkinter"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.gui_type == "tkinter"

    def test_gui_type_zenity(self, monkeypatch):
        """Test --gui-type zenity."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--gui-type", "zenity"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.gui_type == "zenity"

    def test_check_mode(self, monkeypatch):
        """Test --check flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--check"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.check is True

    def test_driver_version(self, monkeypatch):
        """Test --driver-version flag."""
        monkeypatch.setattr(
            sys, "argv", ["nvidia-inst", "--driver-version", "535.154.05"]
        )
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.driver_version == "535.154.05"

    def test_no_cuda_flag(self, monkeypatch):
        """Test --no-cuda flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--no-cuda"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.no_cuda is True

    def test_cuda_version(self, monkeypatch):
        """Test --cuda-version flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--cuda-version", "12.2"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.cuda_version == "12.2"

    def test_yes_short_flag(self, monkeypatch):
        """Test -y short flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "-y"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.yes is True

    def test_yes_long_flag(self, monkeypatch):
        """Test --yes long flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--yes"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.yes is True

    def test_debug_flag(self, monkeypatch):
        """Test --debug flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--debug"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.debug is True

    def test_version_flag(self, monkeypatch):
        """Test --version flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--version"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.version is True

    def test_dry_run_flag(self, monkeypatch):
        """Test --dry-run flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--dry-run"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.dry_run is True

    def test_simulate_alias(self, monkeypatch):
        """Test --simulate alias for --dry-run."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--simulate"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.dry_run is True

    def test_revert_to_nouveau_flag(self, monkeypatch):
        """Test --revert-to-nouveau flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--revert-to-nouveau"])
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.revert_to_nouveau is True

    def test_combined_flags(self, monkeypatch):
        """Test multiple flags combined."""
        monkeypatch.setattr(
            sys, "argv", ["nvidia-inst", "--check", "--debug", "--dry-run", "-y"]
        )
        from nvidia_inst.cli import parse_args

        args = parse_args()
        assert args.check is True
        assert args.debug is True
        assert args.dry_run is True
        assert args.yes is True


class TestCLIIntegration:
    """Integration tests for CLI functions."""

    def test_check_compatibility_no_gpu(
        self,
        monkeypatch,
        mock_distro_ubuntu,
    ):
        """Test check_compatibility when no GPU detected."""
        monkeypatch.setattr(
            "nvidia_inst.gpu.detector.has_nvidia_gpu",
            lambda: False,
        )
        from nvidia_inst.cli import check_compatibility

        result = check_compatibility()
        assert result == 0

    def test_install_driver_cli_no_gpu(
        self,
        monkeypatch,
    ):
        """Test install_driver_cli when no GPU detected."""
        monkeypatch.setattr(
            "nvidia_inst.cli.has_nvidia_gpu",
            lambda: False,
        )
        monkeypatch.setattr(
            "nvidia_inst.distro.detector.detect_distro",
            lambda: MagicMock(
                id="ubuntu",
                version_id="22.04",
                name="Ubuntu",
                pretty_name="Ubuntu 22.04.3 LTS",
                kernel="5.15.0-91-generic",
            ),
        )
        from nvidia_inst.cli import install_driver_cli

        result = install_driver_cli()
        assert result == 0

    def test_install_driver_cli_dry_run(
        self,
        monkeypatch,
        mock_distro_ubuntu,
        mock_gpu_detect_rtx3080,
        mock_driver_range,
        mock_nouveau_not_loaded,
        mock_secure_boot_disabled,
    ):
        """Test install_driver_cli in dry-run mode."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        monkeypatch.setattr(
            "nvidia_inst.installer.driver.get_compatible_driver_packages",
            lambda *args: ["nvidia-driver-535"],
        )
        monkeypatch.setattr(
            "nvidia_inst.gpu.compatibility.get_driver_range",
            lambda *args: mock_driver_range,
        )
        monkeypatch.setattr(
            "nvidia_inst.distro.factory.get_package_manager",
            lambda: MagicMock(),
        )
        from nvidia_inst.cli import install_driver_cli

        result = install_driver_cli(dry_run=True)
        assert result == 0


class TestCLIMain:
    """Tests for main() function."""

    def test_main_version_flag(self, monkeypatch, capsys):
        """Test main() with --version flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--version"])
        from nvidia_inst.cli import main

        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "nvidia-inst" in captured.out

    def test_main_check_flag(self, monkeypatch, mock_distro_ubuntu):
        """Test main() with --check flag."""
        monkeypatch.setattr(sys, "argv", ["nvidia-inst", "--check"])
        monkeypatch.setattr(
            "nvidia_inst.gpu.detector.has_nvidia_gpu",
            lambda: False,
        )
        from nvidia_inst.cli import main

        result = main()
        assert result == 0


class TestPrintCompatibilityInfo:
    """Tests for compatibility info printing."""

    def test_print_compatibility_info(self, monkeypatch, capsys):
        """Test print_compatibility_info outputs correct format."""
        from nvidia_inst.cli import print_compatibility_info
        from nvidia_inst.distro.detector import DistroInfo
        from nvidia_inst.gpu.compatibility import DriverRange
        from nvidia_inst.gpu.detector import GPUInfo

        distro = DistroInfo(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0-91-generic",
        )
        gpu = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            compute_capability=8.6,
            vram="10GB",
        )
        driver_range = DriverRange(
            min_version="535.154.05",
            max_version=None,
            cuda_min="11.8",
            cuda_max="12.2",
            is_eol=False,
        )

        print_compatibility_info(distro, gpu, driver_range)
        captured = capsys.readouterr()
        assert "Ubuntu" in captured.out
        assert "RTX 3080" in captured.out
        assert "535.154.05" in captured.out
