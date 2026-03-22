"""Tests for system utilities."""

from unittest.mock import MagicMock, patch

import pytest


class TestParseCudaVersionFromPackage:
    """Test _parse_cuda_version_from_package function."""

    def test_cuda_toolkit_rpm_style(self):
        """Test parsing cuda-toolkit-12-6-12.6.3-1.x86_64."""
        from nvidia_inst.installer.cuda import _parse_cuda_version_from_package

        result = _parse_cuda_version_from_package("cuda-toolkit-12-6-12.6.3-1.x86_64")
        assert result == "12.6"

    def test_cuda_toolkit_rpm_style_13_2(self):
        """Test parsing cuda-toolkit-13-2-13.2.0-1.x86_64."""
        from nvidia_inst.installer.cuda import _parse_cuda_version_from_package

        result = _parse_cuda_version_from_package("cuda-toolkit-13-2-13.2.0-1.x86_64")
        assert result == "13.2"

    def test_cuda_rpm_style(self):
        """Test parsing cuda-12.2-12.2.0-1.x86_64."""
        from nvidia_inst.installer.cuda import _parse_cuda_version_from_package

        result = _parse_cuda_version_from_package("cuda-12.2-12.2.0-1.x86_64")
        assert result == "12.2"

    def test_cuda_deb_style(self):
        """Test parsing cuda-12.2."""
        from nvidia_inst.installer.cuda import _parse_cuda_version_from_package

        result = _parse_cuda_version_from_package("cuda-12.2")
        assert result == "12.2"

    def test_non_cuda_package(self):
        """Test non-CUDA package returns None."""
        from nvidia_inst.installer.cuda import _parse_cuda_version_from_package

        result = _parse_cuda_version_from_package("nvidia-driver-535")
        assert result is None


class TestDetectViaRpm:
    """Test _detect_via_rpm function."""

    @patch("subprocess.run")
    def test_find_cuda_toolkit(self, mock_run):
        """Test finding cuda-toolkit via rpm."""
        from nvidia_inst.installer.cuda import _detect_via_rpm

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(
                [
                    "nvidia-driver-535-3:535.216.03-1.fc39.x86_64",
                    "cuda-toolkit-12-6-12.6.3-1.x86_64",
                    "libnvidia-535-3:535.216.03-1.fc39.x86_64",
                ]
            ),
        )

        result = _detect_via_rpm()
        assert result == "12.6"

    @patch("subprocess.run")
    def test_find_cuda_12_2(self, mock_run):
        """Test finding cuda-12.2 via rpm."""
        from nvidia_inst.installer.cuda import _detect_via_rpm

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(
                [
                    "cuda-12.2-12.2.0-1.x86_64",
                ]
            ),
        )

        result = _detect_via_rpm()
        assert result == "12.2"

    @patch("subprocess.run")
    def test_no_cuda_found(self, mock_run):
        """Test when no CUDA packages found."""
        from nvidia_inst.installer.cuda import _detect_via_rpm

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(
                [
                    "nvidia-driver-535-3:535.216.03-1.fc39.x86_64",
                    "libnvidia-535-3:535.216.03-1.fc39.x86_64",
                ]
            ),
        )

        result = _detect_via_rpm()
        assert result is None

    @patch("subprocess.run")
    def test_rpm_not_found(self, mock_run):
        """Test when rpm command not found."""
        from nvidia_inst.installer.cuda import _detect_via_rpm

        mock_run.side_effect = FileNotFoundError()

        result = _detect_via_rpm()
        assert result is None


class TestDetectViaDpkg:
    """Test _detect_via_dpkg function."""

    @patch("subprocess.run")
    def test_find_cuda_toolkit(self, mock_run):
        """Test finding cuda-toolkit via dpkg."""
        from nvidia_inst.installer.cuda import _detect_via_dpkg

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(
                [
                    "ii  cuda-toolkit-12-6  12.6.3-1  amd64",
                    "ii  nvidia-driver-535  535.216.03-1  amd64",
                ]
            ),
        )

        result = _detect_via_dpkg()
        assert result == "12.6"

    @patch("subprocess.run")
    def test_find_cuda_12_2(self, mock_run):
        """Test finding cuda-12.2 via dpkg."""
        from nvidia_inst.installer.cuda import _detect_via_dpkg

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(
                [
                    "ii  cuda-12.2  12.2.0-1  amd64",
                ]
            ),
        )

        result = _detect_via_dpkg()
        assert result == "12.2"

    @patch("subprocess.run")
    def test_no_cuda_found(self, mock_run):
        """Test when no CUDA packages found."""
        from nvidia_inst.installer.cuda import _detect_via_dpkg

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(
                [
                    "ii  nvidia-driver-535  535.216.03-1  amd64",
                ]
            ),
        )

        result = _detect_via_dpkg()
        assert result is None

    @patch("subprocess.run")
    def test_dpkg_not_found(self, mock_run):
        """Test when dpkg command not found."""
        from nvidia_inst.installer.cuda import _detect_via_dpkg

        mock_run.side_effect = FileNotFoundError()

        result = _detect_via_dpkg()
        assert result is None


class TestDetectInstalledCudaVersion:
    """Test detect_installed_cuda_version function."""

    @patch(
        "nvidia_inst.installer.cuda._detect_via_rpm",
        return_value="12.6",
    )
    def test_uses_rpm_first(self, mock_rpm):
        """Test that RPM is checked first."""
        from nvidia_inst.installer.cuda import detect_installed_cuda_version

        result = detect_installed_cuda_version()
        assert result == "12.6"
        mock_rpm.assert_called_once()

    @patch("nvidia_inst.installer.cuda._detect_via_rpm", return_value=None)
    @patch(
        "nvidia_inst.installer.cuda._detect_via_dpkg",
        return_value="11.8",
    )
    def test_falls_back_to_dpkg(self, mock_dpkg, mock_rpm):
        """Test fallback to DPKG when RPM fails."""
        from nvidia_inst.installer.cuda import detect_installed_cuda_version

        result = detect_installed_cuda_version()
        assert result == "11.8"
        mock_rpm.assert_called_once()
        mock_dpkg.assert_called_once()

    @patch("nvidia_inst.installer.cuda._detect_via_rpm", return_value=None)
    @patch("nvidia_inst.installer.cuda._detect_via_dpkg", return_value=None)
    @patch(
        "nvidia_inst.installer.cuda._detect_via_pacman",
        return_value="12.4",
    )
    def test_falls_back_to_pacman(self, mock_pacman, mock_dpkg, mock_rpm):
        """Test fallback to pacman when RPM and DPKG fail."""
        from nvidia_inst.installer.cuda import detect_installed_cuda_version

        result = detect_installed_cuda_version()
        assert result == "12.4"
        mock_rpm.assert_called_once()
        mock_dpkg.assert_called_once()
        mock_pacman.assert_called_once()

    @patch("nvidia_inst.installer.cuda._detect_via_rpm", return_value=None)
    @patch("nvidia_inst.installer.cuda._detect_via_dpkg", return_value=None)
    @patch("nvidia_inst.installer.cuda._detect_via_pacman", return_value=None)
    def test_returns_none_when_not_found(self, mock_pacman, mock_dpkg, mock_rpm):
        """Test returns None when no CUDA found."""
        from nvidia_inst.installer.cuda import detect_installed_cuda_version

        result = detect_installed_cuda_version()
        assert result is None
