"""Tests for cli/display.py module."""

import pytest
from unittest.mock import patch
from io import StringIO

from nvidia_inst.cli.display import (
    print_row,
    print_section_header,
    print_step,
    print_warning,
    print_error,
    print_info,
    print_success,
    format_package_list,
    print_driver_status,
    print_gpu_info,
    print_distro_info,
)


class TestPrintRow:
    """Tests for print_row function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_row(self, mock_stdout):
        """Test printing a row."""
        print_row("Label", "Value")
        output = mock_stdout.getvalue()
        assert "Label" in output
        assert "Value" in output


class TestPrintSectionHeader:
    """Tests for print_section_header function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_section_header(self, mock_stdout):
        """Test printing a section header."""
        print_section_header("Test Title")
        output = mock_stdout.getvalue()
        assert "Test Title" in output
        assert "=" in output


class TestPrintStep:
    """Tests for print_step function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_step(self, mock_stdout):
        """Test printing a step."""
        print_step(1, "Test step")
        output = mock_stdout.getvalue()
        assert "1." in output
        assert "Test step" in output


class TestPrintWarning:
    """Tests for print_warning function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_warning(self, mock_stdout):
        """Test printing a warning."""
        print_warning("Test warning")
        output = mock_stdout.getvalue()
        assert "[!]" in output
        assert "Test warning" in output


class TestPrintError:
    """Tests for print_error function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_error(self, mock_stdout):
        """Test printing an error."""
        print_error("Test error")
        output = mock_stdout.getvalue()
        assert "[ERROR]" in output
        assert "Test error" in output


class TestPrintInfo:
    """Tests for print_info function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_info(self, mock_stdout):
        """Test printing an info message."""
        print_info("Test info")
        output = mock_stdout.getvalue()
        assert "[INFO]" in output
        assert "Test info" in output


class TestPrintSuccess:
    """Tests for print_success function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_success(self, mock_stdout):
        """Test printing a success message."""
        print_success("Test success")
        output = mock_stdout.getvalue()
        assert "[OK]" in output
        assert "Test success" in output


class TestFormatPackageList:
    """Tests for format_package_list function."""

    def test_format_few_packages(self):
        """Test formatting few packages."""
        result = format_package_list(["pkg1", "pkg2"])
        assert result == "pkg1 pkg2"

    def test_format_many_packages(self):
        """Test formatting many packages."""
        result = format_package_list(["pkg1", "pkg2", "pkg3", "pkg4"], max_display=3)
        assert result == "pkg1 pkg2 pkg3 ..."

    def test_format_exact_limit(self):
        """Test formatting exact limit packages."""
        result = format_package_list(["pkg1", "pkg2", "pkg3"], max_display=3)
        assert result == "pkg1 pkg2 pkg3"


class TestPrintDriverStatus:
    """Tests for print_driver_status function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_driver_installed(self, mock_stdout):
        """Test printing installed driver status."""
        print_driver_status("535.154.05", True, "proprietary")
        output = mock_stdout.getvalue()
        assert "535.154.05" in output
        assert "Working" in output
        assert "proprietary" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_driver_not_working(self, mock_stdout):
        """Test printing not working driver status."""
        print_driver_status("535.154.05", False)
        output = mock_stdout.getvalue()
        assert "Not working" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_no_driver(self, mock_stdout):
        """Test printing no driver status."""
        print_driver_status(None, False)
        output = mock_stdout.getvalue()
        assert "No NVIDIA driver" in output


class TestPrintGpuInfo:
    """Tests for print_gpu_info function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_gpu_info_full(self, mock_stdout):
        """Test printing full GPU info."""
        print_gpu_info("RTX 3080", "ampere", 8.6)
        output = mock_stdout.getvalue()
        assert "RTX 3080" in output
        assert "ampere" in output
        assert "8.6" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_gpu_info_minimal(self, mock_stdout):
        """Test printing minimal GPU info."""
        print_gpu_info("RTX 3080")
        output = mock_stdout.getvalue()
        assert "RTX 3080" in output


class TestPrintDistroInfo:
    """Tests for print_distro_info function."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_distro_info_full(self, mock_stdout):
        """Test printing full distro info."""
        print_distro_info("ubuntu", "22.04", "5.15.0")
        output = mock_stdout.getvalue()
        assert "ubuntu" in output
        assert "22.04" in output
        assert "5.15.0" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_distro_info_minimal(self, mock_stdout):
        """Test printing minimal distro info."""
        print_distro_info("ubuntu")
        output = mock_stdout.getvalue()
        assert "ubuntu" in output
