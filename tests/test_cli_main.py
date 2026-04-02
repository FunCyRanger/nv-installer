"""Tests for cli/__main__.py module."""

import subprocess
import sys


class TestMainModule:
    """Tests for __main__.py entry point."""

    def test_main_module_can_be_imported(self):
        """Test that __main__.py can be imported."""
        from nvidia_inst.cli import __main__ as cli_main

        assert hasattr(cli_main, "main")

    def test_main_module_has_main_function(self):
        """Test that __main__.py has a main function."""
        from nvidia_inst.cli.__main__ import main

        assert callable(main)

    def test_python_m_nvidia_inst_cli_runs(self):
        """Test that python3 -m nvidia_inst.cli runs successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Cross-distribution Nvidia driver installer" in result.stdout

    def test_python_m_nvidia_inst_cli_version(self):
        """Test that python3 -m nvidia_inst.cli --version works."""
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "nvidia-inst version" in result.stdout

    def test_python_m_nvidia_inst_cli_invalid_arg(self):
        """Test that python3 -m nvidia_inst.cli with invalid arg exits with error."""
        result = subprocess.run(
            [sys.executable, "-m", "nvidia_inst.cli", "--invalid-option"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
