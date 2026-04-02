"""CLI integration tests that test real argument parsing and execution.

These tests ensure that the CLI actually works end-to-end without mocking
the argument parser. This catches bugs like:
- Missing args attributes
- Incorrect flag names
- Broken argument parsing
"""

import os
import subprocess
import sys

# Use the local version of nv-install
NV_INSTALL = "./nv-install"

# Get the source directory for subprocess
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")


class TestCliRealArgumentParsing:
    """Test CLI with real argument parsing (no mocking)."""

    def test_help_flag(self):
        """Test --help flag shows usage."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0
        assert "Cross-distribution Nvidia driver installer" in result.stdout
        assert "--simulate" in result.stdout

    def test_version_flag(self):
        """Test --version flag shows version."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0
        assert "nvidia-inst version" in result.stdout

    def test_invalid_flag(self):
        """Test invalid flag returns error."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--invalid-flag"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode != 0

    def test_simulate_flag_exists(self):
        """Test --simulate flag is recognized."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert "--simulate" in result.stdout
        assert "Show what would be installed" in result.stdout

    def test_check_flag_exists(self):
        """Test --check flag is recognized."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert "--check" in result.stdout
        assert "Check compatibility only" in result.stdout

    def test_revert_to_nouveau_flag_exists(self):
        """Test --revert-to-nouveau flag is recognized."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert "--revert-to-nouveau" in result.stdout

    def test_power_profile_flag_exists(self):
        """Test --power-profile flag is recognized."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert "--power-profile" in result.stdout
        assert "intel" in result.stdout
        assert "hybrid" in result.stdout
        assert "nvidia" in result.stdout


class TestParserAttributeValidation:
    """Test that parser creates expected attributes."""

    def test_simulate_attribute_exists(self):
        """Test args has simulate attribute when --simulate is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--simulate"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "simulate")
            assert args.simulate is True
        finally:
            sys.argv = old_argv

    def test_check_attribute_exists(self):
        """Test args has check attribute when --check is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--check"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "check")
            assert args.check is True
        finally:
            sys.argv = old_argv

    def test_revert_to_nouveau_attribute_exists(self):
        """Test args has revert_to_nouveau attribute when --revert-to-nouveau is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--revert-to-nouveau"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "revert_to_nouveau")
            assert args.revert_to_nouveau is True
        finally:
            sys.argv = old_argv

    def test_no_cuda_attribute_exists(self):
        """Test args has no_cuda attribute when --no-cuda is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--no-cuda"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "no_cuda")
            assert args.no_cuda is True
        finally:
            sys.argv = old_argv

    def test_yes_attribute_exists(self):
        """Test args has yes attribute when --yes is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--yes"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "yes")
            assert args.yes is True
        finally:
            sys.argv = old_argv

    def test_debug_attribute_exists(self):
        """Test args has debug attribute when --debug is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--debug"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "debug")
            assert args.debug is True
        finally:
            sys.argv = old_argv

    def test_power_profile_attribute_exists(self):
        """Test args has power_profile attribute when --power-profile is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--power-profile", "intel"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "power_profile")
            assert args.power_profile == "intel"
        finally:
            sys.argv = old_argv

    def test_cuda_version_attribute_exists(self):
        """Test args has cuda_version attribute when --cuda-version is used."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--cuda-version", "12.2"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "cuda_version")
            assert args.cuda_version == "12.2"
        finally:
            sys.argv = old_argv


class TestNvInstallScript:
    """Test the nv-install shell script."""

    def test_nv_install_version(self):
        """Test nv-install --version works."""
        result = subprocess.run(
            [NV_INSTALL, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "nvidia-inst version" in result.stdout

    def test_nv_install_help(self):
        """Test nv-install --help works."""
        result = subprocess.run(
            [NV_INSTALL, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--simulate" in result.stdout

    def test_nv_install_check(self):
        """Test nv-install --check works."""
        result = subprocess.run(
            [NV_INSTALL, "--check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete without error
        assert result.returncode == 0
        assert (
            "System Compatibility Check" in result.stdout
            or "Prerequisites Check" in result.stdout
        )


class TestMainModuleExecution:
    """Test execution via python -m nvidia_inst.cli."""

    def test_main_module_version(self):
        """Test python -m nvidia_inst.cli --version."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0
        assert "nvidia-inst version" in result.stdout

    def test_main_module_help(self):
        """Test python -m nvidia_inst.cli --help."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode == 0
        assert "Cross-distribution Nvidia driver installer" in result.stdout

    def test_main_module_invalid_arg(self):
        """Test python -m nvidia_inst.cli with invalid argument."""
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--invalid-arg"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        assert result.returncode != 0
