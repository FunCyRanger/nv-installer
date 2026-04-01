"""Tests for installer/prerequisites.py module."""

import pytest
from unittest.mock import patch, MagicMock

from nvidia_inst.installer.prerequisites import (
    PrerequisitesResult,
    PrerequisitesChecker,
)


class TestPrerequisitesResult:
    """Tests for PrerequisitesResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = PrerequisitesResult(success=True)
        assert result.success is True
        assert result.package_manager_available is False
        assert result.repos_configured == []
        assert result.repos_missing == []
        assert result.driver_packages_available is False
        assert result.driver_packages == []
        assert result.version_check is None
        assert result.fix_commands == []
        assert result.warnings == []
        assert result.errors == []


class TestPrerequisitesChecker:
    """Tests for PrerequisitesChecker class."""

    def test_init(self):
        """Test initialization."""
        checker = PrerequisitesChecker()
        assert checker._pm is None

    def test_get_cuda_repo_version_no_cuda(self):
        """Test getting repo version without CUDA major."""
        result = PrerequisitesChecker.get_cuda_repo_version("43")
        assert result == "43"

    def test_get_cuda_repo_version_cuda12(self):
        """Test getting repo version for CUDA 12."""
        result = PrerequisitesChecker.get_cuda_repo_version("43", "12")
        assert result == "41"

    def test_get_cuda_repo_version_cuda13(self):
        """Test getting repo version for CUDA 13."""
        result = PrerequisitesChecker.get_cuda_repo_version("43", "13")
        assert result == "42"

    def test_get_cuda_repo_version_older_distro(self):
        """Test getting repo version for older distro."""
        result = PrerequisitesChecker.get_cuda_repo_version("40", "12")
        assert result == "40"

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_package_manager"
    )
    def test_check_all_no_pm(self, mock_check_pm):
        """Test check_all when package manager not available."""
        mock_check_pm.return_value = (False, "")
        checker = PrerequisitesChecker()
        result = checker.check_all("ubuntu")
        assert result.success is False
        assert "Package manager not available" in result.errors

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_package_manager"
    )
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_repositories"
    )
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_driver_packages"
    )
    def test_check_all_success(self, mock_check_pkgs, mock_check_repos, mock_check_pm):
        """Test check_all success."""
        mock_check_pm.return_value = (True, "apt")
        mock_check_repos.return_value = {
            "configured": ["cuda"],
            "missing": [],
            "fix_commands": [],
        }
        mock_check_pkgs.return_value = {
            "available": True,
            "packages": ["nvidia-driver-535"],
        }

        checker = PrerequisitesChecker()
        result = checker.check_all("ubuntu", "22.04")

        assert result.success is True
        assert result.package_manager_available is True
        assert result.driver_packages_available is True


class TestGetCudaRepoFixCommands:
    """Tests for get_cuda_repo_fix_commands method."""

    @patch("nvidia_inst.installer.prerequisites.PrerequisitesChecker._repo_exists")
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._get_cuda_repo_version_from_file"
    )
    def test_fedora_repo_missing(self, mock_get_version, mock_repo_exists):
        """Test Fedora repo missing returns fix command."""
        mock_repo_exists.return_value = False
        mock_get_version.return_value = None

        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("fedora", "43")

        assert len(commands) > 0
        assert any("dnf config-manager" in cmd for cmd in commands)

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._package_installed"
    )
    def test_ubuntu_keyring_missing(self, mock_pkg_installed):
        """Test Ubuntu keyring missing returns fix command."""
        mock_pkg_installed.return_value = False

        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("ubuntu", "22.04")

        assert len(commands) > 0
        assert any("cuda-keyring" in cmd for cmd in commands)

    def test_arch_no_fix_needed(self):
        """Test Arch doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("arch", "")

        assert commands == []

    def test_opensuse_no_fix_needed(self):
        """Test openSUSE doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("opensuse", "15.5")

        assert commands == []
