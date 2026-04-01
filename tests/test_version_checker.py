"""Tests for installer/version_checker.py module."""

import pytest
from unittest.mock import patch, MagicMock

from nvidia_inst.installer.version_checker import (
    VersionCheckResult,
    VersionChecker,
)


class TestVersionCheckResult:
    """Tests for VersionCheckResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = VersionCheckResult()
        assert result.success is False
        assert result.repo_versions == []
        assert result.official_versions == []
        assert result.installed_driver_version is None
        assert result.compatible is False
        assert result.compatible_versions == []
        assert result.incompatible_versions == []
        assert result.warnings == []
        assert result.errors == []


class TestVersionChecker:
    """Tests for VersionChecker class."""

    def test_extract_branch(self):
        """Test extracting branch from version."""
        checker = VersionChecker()
        assert checker._extract_branch("535.154.05") == "535"
        assert checker._extract_branch("580.126.18") == "580"
        assert checker._extract_branch("invalid") is None

    def test_version_sort_key(self):
        """Test version sort key."""
        checker = VersionChecker()
        assert checker._version_sort_key("535.154.05") == (535, 154, 5)
        assert checker._version_sort_key("580.126.18") == (580, 126, 18)
        assert checker._version_sort_key("invalid") == (0, 0, 0)

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_get_repo_versions_success(self, mock_get_pm):
        """Test getting repo versions successfully."""
        mock_pm = MagicMock()
        mock_pm.get_all_versions.return_value = ["535.154.05", "535.126.18"]
        mock_get_pm.return_value = mock_pm

        checker = VersionChecker()
        versions = checker.get_repo_versions("fedora", "akmod-nvidia")

        assert versions == ["535.154.05", "535.126.18"]

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_get_repo_versions_failure(self, mock_get_pm):
        """Test getting repo versions failure."""
        mock_get_pm.side_effect = Exception("PM error")

        checker = VersionChecker()
        versions = checker.get_repo_versions("fedora", "akmod-nvidia")

        assert versions == []

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_check_installed_driver_fedora(self, mock_get_pm):
        """Test checking installed driver on Fedora."""
        mock_pm = MagicMock()
        mock_get_pm.return_value = mock_pm

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="akmod-nvidia.x86_64    3:535.154.05-1.fc43    @cuda\n",
            )

            checker = VersionChecker()
            version = checker.check_installed_driver("fedora")

            assert version == "535.154.05"

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_check_installed_driver_ubuntu(self, mock_get_pm):
        """Test checking installed driver on Ubuntu."""
        mock_pm = MagicMock()
        mock_get_pm.return_value = mock_pm

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="535.154.05-0ubuntu0.22.04.1\n"
            )

            checker = VersionChecker()
            version = checker.check_installed_driver("ubuntu")

            assert version == "535.154.05"

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_check_installed_driver_arch(self, mock_get_pm):
        """Test checking installed driver on Arch."""
        mock_pm = MagicMock()
        mock_pm.get_installed_version.return_value = "535.154.05"
        mock_get_pm.return_value = mock_pm

        checker = VersionChecker()
        version = checker.check_installed_driver("arch")

        assert version == "nvidia"

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_check_installed_driver_opensuse(self, mock_get_pm):
        """Test checking installed driver on openSUSE."""
        mock_pm = MagicMock()
        mock_pm.get_installed_version.return_value = "535.154.05"
        mock_get_pm.return_value = mock_pm

        checker = VersionChecker()
        version = checker.check_installed_driver("opensuse")

        assert version == "G05"

    @patch("nvidia_inst.installer.version_checker.get_package_manager")
    def test_check_installed_driver_none(self, mock_get_pm):
        """Test checking installed driver when none installed."""
        mock_pm = MagicMock()
        mock_get_pm.return_value = mock_pm

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")

            checker = VersionChecker()
            version = checker.check_installed_driver("fedora")

            assert version is None


class TestFetchOfficialVersions:
    """Tests for fetch_official_versions method."""

    @patch("urllib.request.urlopen")
    def test_fetch_versions_success(self, mock_urlopen):
        """Test fetching official versions successfully."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'<a href="535.154.05/">535.154.05</a>\n<a href="580.126.18/">580.126.18</a>'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        checker = VersionChecker()
        versions = checker.fetch_official_versions()

        assert "535.154.05" in versions
        assert "580.126.18" in versions

    @patch("urllib.request.urlopen")
    def test_fetch_versions_with_branch(self, mock_urlopen):
        """Test fetching official versions with branch filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'<a href="535.154.05/">535.154.05</a>\n<a href="580.126.18/">580.126.18</a>'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        checker = VersionChecker()
        versions = checker.fetch_official_versions(branch="535")

        assert "535.154.05" in versions
        assert "580.126.18" not in versions

    @patch("urllib.request.urlopen")
    def test_fetch_versions_failure(self, mock_urlopen):
        """Test fetching official versions failure."""
        mock_urlopen.side_effect = Exception("Network error")

        checker = VersionChecker()
        versions = checker.fetch_official_versions()

        assert versions == []
