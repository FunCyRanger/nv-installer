"""Tests for system utilities."""

from unittest.mock import MagicMock, patch

import pytest


class TestFindNVCC:
    """Test find_nvcc function."""

    @patch("shutil.which")
    def test_find_in_path(self, mock_which):
        """Test finding nvcc in PATH."""
        mock_which.return_value = "/usr/bin/nvcc"

        from nvidia_inst.utils.system import find_nvcc

        result = find_nvcc()
        assert result == "/usr/bin/nvcc"
        mock_which.assert_called_once_with("nvcc")

    @patch("glob.glob")
    @patch("os.path.isfile")
    @patch("os.access")
    @patch("shutil.which")
    def test_find_in_common_directory(
        self, mock_which, mock_access, mock_isfile, mock_glob
    ):
        """Test finding nvcc in common CUDA directory."""
        mock_which.return_value = None
        mock_glob.return_value = ["/usr/local/cuda-12.2/bin"]
        mock_isfile.return_value = True
        mock_access.return_value = True

        from nvidia_inst.utils.system import find_nvcc

        result = find_nvcc()
        assert result == "/usr/local/cuda-12.2/bin/nvcc"

    @patch("glob.glob")
    @patch("shutil.which")
    def test_not_found(self, mock_which, mock_glob):
        """Test when nvcc is not found."""
        mock_which.return_value = None
        mock_glob.return_value = []

        from nvidia_inst.utils.system import find_nvcc

        result = find_nvcc()
        assert result is None

    @patch("glob.glob")
    @patch("os.path.isfile")
    @patch("os.access")
    @patch("shutil.which")
    def test_find_in_opt_cuda(self, mock_which, mock_access, mock_isfile, mock_glob):
        """Test finding nvcc in /opt/cuda directory."""
        mock_which.return_value = None
        mock_glob.side_effect = [
            [],
            ["/opt/cuda-11.8/bin"],
        ]  # First pattern returns [], second finds it
        mock_isfile.return_value = True
        mock_access.return_value = True

        from nvidia_inst.utils.system import find_nvcc

        result = find_nvcc()
        assert result == "/opt/cuda-11.8/bin/nvcc"

    @patch("glob.glob")
    @patch("os.path.isfile")
    @patch("os.access")
    @patch("shutil.which")
    def test_non_executable_file(self, mock_which, mock_access, mock_isfile, mock_glob):
        """Test that non-executable files are skipped."""
        mock_which.return_value = None
        mock_glob.return_value = ["/usr/local/cuda/bin"]
        mock_isfile.return_value = True
        mock_access.return_value = False  # Not executable

        from nvidia_inst.utils.system import find_nvcc

        result = find_nvcc()
        assert result is None
