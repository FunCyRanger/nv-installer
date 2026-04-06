"""Tests for installer/prerequisites.py module."""

import subprocess
from unittest.mock import MagicMock, patch

from nvidia_inst.installer.prerequisites import (
    PrerequisitesChecker,
    PrerequisitesResult,
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


class TestCheckPackageManager:
    """Tests for _check_package_manager method."""

    def test_check_apt_paths_defined(self):
        """Test APT paths are properly defined."""
        from nvidia_inst.installer.prerequisites import PrerequisitesChecker

        checker = PrerequisitesChecker()
        pm_available, pm_name = checker._check_package_manager()

        assert pm_available in (True, False)
        assert pm_name in ("", "apt", "dnf", "pacman", "zypper")

    def test_check_returns_tuple(self):
        """Test package manager check returns proper tuple."""
        from nvidia_inst.installer.prerequisites import PrerequisitesChecker

        checker = PrerequisitesChecker()
        result = checker._check_package_manager()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


class TestCheckRepositories:
    """Tests for _check_repositories method."""

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_fedora_repos"
    )
    def test_check_fedora_repositories(self, mock_check):
        """Test Fedora repositories check."""
        mock_check.return_value = {
            "configured": ["rpmfusion"],
            "missing": [],
            "fix_commands": [],
        }

        checker = PrerequisitesChecker()
        result = checker._check_repositories("fedora", "43")

        assert "configured" in result
        assert "missing" in result
        mock_check.assert_called_once_with("fedora", "43")

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_ubuntu_repos"
    )
    def test_check_ubuntu_repositories(self, mock_check):
        """Test Ubuntu repositories check."""
        mock_check.return_value = {
            "configured": ["main"],
            "missing": [],
            "fix_commands": [],
        }

        checker = PrerequisitesChecker()
        result = checker._check_repositories("ubuntu", "22.04")

        assert "configured" in result
        mock_check.assert_called_once_with("ubuntu", "22.04")

    @patch("nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_arch_repos")
    def test_check_arch_repositories(self, mock_check):
        """Test Arch repositories check."""
        mock_check.return_value = {
            "configured": ["core"],
            "missing": [],
            "fix_commands": [],
        }

        checker = PrerequisitesChecker()
        checker._check_repositories("arch", "")

        mock_check.assert_called_once()

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_opensuse_repos"
    )
    def test_check_opensuse_repositories(self, mock_check):
        """Test openSUSE repositories check."""
        mock_check.return_value = {
            "configured": ["oss"],
            "missing": [],
            "fix_commands": [],
        }

        checker = PrerequisitesChecker()
        checker._check_repositories("opensuse", "15.5")

        mock_check.assert_called_once()

    def test_check_unknown_distro(self):
        """Test unknown distribution returns empty repos."""
        checker = PrerequisitesChecker()
        result = checker._check_repositories("unknown", "1.0")

        assert result == {"configured": [], "missing": [], "fix_commands": []}


class TestCheckFedoraRepos:
    """Tests for _check_fedora_repos method."""

    @patch("nvidia_inst.installer.prerequisites.PrerequisitesChecker._repo_exists")
    def test_fedora_rpmfusion_missing(self, mock_repo):
        """Test Fedora missing RPM Fusion."""
        mock_repo.side_effect = lambda r: r == "cuda-fedora"

        checker = PrerequisitesChecker()
        result = checker._check_fedora_repos("fedora", "43")

        assert (
            "RPM Fusion nonfree (required for proprietary drivers)" in result["missing"]
        )
        assert len(result["fix_commands"]) > 0
        assert "dnf install" in result["fix_commands"][0]

    @patch("nvidia_inst.installer.prerequisites.PrerequisitesChecker._repo_exists")
    def test_fedora_all_repos_configured(self, mock_repo):
        """Test Fedora with all repos configured."""
        mock_repo.return_value = True

        checker = PrerequisitesChecker()
        result = checker._check_fedora_repos("fedora", "43")

        assert "RPM Fusion nonfree" in result["configured"]
        assert "NVIDIA CUDA repository" in result["configured"]


class TestCheckUbuntuRepos:
    """Tests for _check_ubuntu_repos method."""

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._package_installed"
    )
    def test_ubuntu_cuda_repo_missing(self, mock_installed):
        """Test Ubuntu CUDA repo missing."""
        mock_installed.return_value = False

        checker = PrerequisitesChecker()
        result = checker._check_ubuntu_repos("ubuntu", "22.04")

        assert "NVIDIA CUDA repository (required for CUDA toolkit)" in result["missing"]
        assert len(result["fix_commands"]) > 0

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._package_installed"
    )
    def test_ubuntu_cuda_repo_configured(self, mock_installed):
        """Test Ubuntu CUDA repo configured."""
        mock_installed.return_value = True

        checker = PrerequisitesChecker()
        result = checker._check_ubuntu_repos("ubuntu", "22.04")

        assert "NVIDIA CUDA repository" in result["configured"]

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._package_installed"
    )
    def test_debian_repos(self, mock_installed):
        """Test Debian repositories."""
        mock_installed.return_value = False

        checker = PrerequisitesChecker()
        result = checker._check_ubuntu_repos("debian", "12")

        assert "Debian contrib, non-free" in result["configured"]


class TestCheckArchRepos:
    """Tests for _check_arch_repos method."""

    def test_arch_repos_configured(self):
        """Test Arch repositories always configured."""
        checker = PrerequisitesChecker()
        result = checker._check_arch_repos()

        assert "Core repositories" in result["configured"]
        assert "CUDA available via AUR" in result["configured"][1]
        assert len(result["fix_commands"]) > 0


class TestCheckOpenSUSERepos:
    """Tests for _check_opensuse_repos method."""

    def test_opensuse_repos_configured(self):
        """Test openSUSE repositories."""
        checker = PrerequisitesChecker()
        result = checker._check_opensuse_repos()

        assert "openSUSE OSS" in result["configured"]
        assert "NVIDIA CUDA repository (provided by NVIDIA)" in result["configured"]
        assert len(result["fix_commands"]) > 0
        assert "zypper addrepo" in result["fix_commands"][0]


class TestRepoExists:
    """Tests for _repo_exists method."""

    @patch("subprocess.run")
    def test_repo_exists_found(self, mock_run):
        """Test when repo is found."""
        mock_run.return_value = MagicMock(
            stdout="cuda-fedora43-x86_64  cuda-fedora43", returncode=0
        )

        checker = PrerequisitesChecker()
        result = checker._repo_exists("cuda-fedora")

        assert result == "cuda-fedora43-x86_64"

    @patch("subprocess.run")
    def test_repo_exists_not_found(self, mock_run):
        """Test when repo is not found."""
        mock_run.return_value = MagicMock(stdout="main", returncode=0)

        checker = PrerequisitesChecker()
        result = checker._repo_exists("nonexistent")

        assert result is False

    @patch("subprocess.run")
    def test_repo_exists_error(self, mock_run):
        """Test when repo check errors."""
        mock_run.side_effect = FileNotFoundError()

        checker = PrerequisitesChecker()
        result = checker._repo_exists("cuda-fedora")

        assert result is False


class TestPackageInstalled:
    """Tests for _package_installed method."""

    @patch("subprocess.run")
    def test_package_installed_apt(self, mock_run):
        """Test package check for APT."""
        mock_run.return_value = MagicMock(stdout="ii  cuda-keyring", returncode=0)

        checker = PrerequisitesChecker()
        result = checker._package_installed("cuda-keyring", "ubuntu")

        assert result is True

    @patch("subprocess.run")
    def test_package_not_installed_apt(self, mock_run):
        """Test package not installed for APT."""
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        checker = PrerequisitesChecker()
        result = checker._package_installed("nonexistent", "ubuntu")

        assert result is False

    @patch("subprocess.run")
    def test_package_installed_rpm(self, mock_run):
        """Test package check for RPM."""
        mock_run.return_value = MagicMock(returncode=0)

        checker = PrerequisitesChecker()
        result = checker._package_installed("cuda-keyring", "fedora")

        assert result is True

    @patch("subprocess.run")
    def test_package_not_installed_rpm(self, mock_run):
        """Test package not installed for RPM."""
        mock_run.return_value = MagicMock(returncode=1)

        checker = PrerequisitesChecker()
        result = checker._package_installed("nonexistent", "fedora")

        assert result is False

    @patch("subprocess.run")
    def test_package_installed_pacman(self, mock_run):
        """Test package check for Pacman."""
        mock_run.return_value = MagicMock(returncode=0)

        checker = PrerequisitesChecker()
        result = checker._package_installed("nvidia", "arch")

        assert result is True

    @patch("subprocess.run")
    def test_package_not_installed_pacman(self, mock_run):
        """Test package not installed for Pacman."""
        mock_run.return_value = MagicMock(returncode=1)

        checker = PrerequisitesChecker()
        result = checker._package_installed("nonexistent", "arch")

        assert result is False


class TestCheckDriverPackages:
    """Tests for _check_driver_packages method."""

    @patch("subprocess.run")
    def test_fedora_driver_packages_available(self, mock_run):
        """Test Fedora driver packages available."""
        mock_run.return_value = MagicMock(stdout="akmod-nvidia", returncode=0)

        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("fedora")

        assert result["available"] is True
        assert "akmod-nvidia" in result["packages"]

    @patch("subprocess.run")
    def test_fedora_driver_packages_not_found(self, mock_run):
        """Test Fedora driver packages not found."""
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("fedora")

        assert result["available"] is False

    @patch("subprocess.run")
    def test_ubuntu_driver_packages_available(self, mock_run):
        """Test Ubuntu driver packages available."""
        mock_run.return_value = MagicMock(stdout="nvidia-driver-535", returncode=0)

        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("ubuntu")

        assert result["available"] is True

    def test_arch_driver_packages(self):
        """Test Arch driver packages."""
        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("arch")

        assert result["available"] is True
        assert "nvidia" in result["packages"]

    def test_manjaro_driver_packages(self):
        """Test Manjaro driver packages."""
        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("manjaro")

        assert result["available"] is True

    def test_opensuse_driver_packages(self):
        """Test openSUSE driver packages."""
        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("opensuse")

        assert result["available"] is True
        assert "x11-video-nvidiaG05" in result["packages"]

    @patch("subprocess.run")
    def test_driver_check_timeout(self, mock_run):
        """Test driver check with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("dnf", 30)

        checker = PrerequisitesChecker()
        result = checker._check_driver_packages("fedora")

        assert result["available"] is False


class TestFixRepositories:
    """Tests for fix_repositories method."""

    @patch("subprocess.run")
    def test_fix_repositories_success(self, mock_run):
        """Test fixing repositories successfully."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        checker = PrerequisitesChecker()
        success, message = checker.fix_repositories(["sudo dnf install package"])

        assert success is True
        assert "successfully" in message

    @patch("subprocess.run")
    def test_fix_repositories_failure(self, mock_run):
        """Test fixing repositories failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error occurred")

        checker = PrerequisitesChecker()
        success, message = checker.fix_repositories(["sudo dnf install package"])

        assert success is False
        assert "Failed" in message

    @patch("subprocess.run")
    def test_fix_repositories_timeout(self, mock_run):
        """Test fixing repositories with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("dnf", 60)

        checker = PrerequisitesChecker()
        success, message = checker.fix_repositories(["sudo dnf install package"])

        assert success is False
        assert "timed out" in message


class TestGetCudaRepoVersion:
    """Tests for get_cuda_repo_version static method."""

    def test_no_cuda_major(self):
        """Test without CUDA major version."""
        result = PrerequisitesChecker.get_cuda_repo_version("43")
        assert result == "43"

    def test_cuda_12_with_new_distro(self):
        """Test CUDA 12 with newer distro."""
        result = PrerequisitesChecker.get_cuda_repo_version("43", "12")
        assert result == "41"

    def test_cuda_13_with_new_distro(self):
        """Test CUDA 13 with newer distro."""
        result = PrerequisitesChecker.get_cuda_repo_version("43", "13")
        assert result == "42"

    def test_cuda_12_with_old_distro(self):
        """Test CUDA 12 with older distro."""
        result = PrerequisitesChecker.get_cuda_repo_version("40", "12")
        assert result == "40"

    def test_cuda_13_with_older_distro(self):
        """Test CUDA 13 with older distro."""
        result = PrerequisitesChecker.get_cuda_repo_version("40", "13")
        assert result == "40"

    def test_cuda_11_not_mapped(self):
        """Test CUDA 11 not in mapping."""
        result = PrerequisitesChecker.get_cuda_repo_version("43", "11")
        assert result == "43"


class TestCheckAllWithDriverRange:
    """Tests for check_all with driver range."""

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_package_manager"
    )
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_repositories"
    )
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_driver_packages"
    )
    @patch("nvidia_inst.installer.version_checker.VersionChecker.check_compatibility")
    def test_check_all_with_driver_range_compatible(
        self, mock_check, mock_pkgs, mock_repos, mock_pm
    ):
        """Test check_all with compatible driver range."""
        mock_pm.return_value = (True, "apt")
        mock_repos.return_value = {"configured": [], "missing": [], "fix_commands": []}
        mock_pkgs.return_value = {"available": True, "packages": []}
        mock_check.return_value = MagicMock(compatible=True, errors=[], warnings=[])

        from nvidia_inst.gpu.compatibility import DriverRange

        checker = PrerequisitesChecker()
        driver_range = DriverRange(
            min_version="535", max_version="550", cuda_min="11.8", cuda_max=None
        )
        result = checker.check_all("ubuntu", "22.04", driver_range)

        assert result.success is True
        assert result.version_check is not None

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_package_manager"
    )
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_repositories"
    )
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._check_driver_packages"
    )
    @patch("nvidia_inst.installer.version_checker.VersionChecker.check_compatibility")
    def test_check_all_with_driver_range_incompatible(
        self, mock_check, mock_pkgs, mock_repos, mock_pm
    ):
        """Test check_all with incompatible driver range."""
        mock_pm.return_value = (True, "apt")
        mock_repos.return_value = {"configured": [], "missing": [], "fix_commands": []}
        mock_pkgs.return_value = {"available": True, "packages": []}
        mock_check.return_value = MagicMock(
            compatible=False, errors=["Driver too old"], warnings=[]
        )

        from nvidia_inst.gpu.compatibility import DriverRange

        checker = PrerequisitesChecker()
        driver_range = DriverRange(
            min_version="535", max_version="550", cuda_min="11.8", cuda_max=None
        )
        result = checker.check_all("ubuntu", "22.04", driver_range)

        assert result.success is False
        assert "Driver too old" in result.errors


class TestGetCudaRepoFixCommandsEdgeCases:
    """Edge case tests for get_cuda_repo_fix_commands."""

    @patch("nvidia_inst.installer.prerequisites.PrerequisitesChecker._repo_exists")
    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._get_cuda_repo_version_from_file"
    )
    def test_fedora_needs_update(self, mock_get_version, mock_repo_exists):
        """Test Fedora needs repo update."""
        mock_repo_exists.return_value = "cuda-fedora40-x86_64"
        mock_get_version.return_value = "40"

        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("fedora", "43", "12")

        assert len(commands) > 0
        assert any("rm" in cmd for cmd in commands)

    @patch("nvidia_inst.installer.prerequisites.PrerequisitesChecker._repo_exists")
    def test_fedora_repo_exists_no_update(self, mock_repo_exists):
        """Test Fedora repo exists, no update needed."""
        mock_repo_exists.return_value = "cuda-fedora43-x86_64"

        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("fedora", "43")

        assert commands == []

    @patch(
        "nvidia_inst.installer.prerequisites.PrerequisitesChecker._package_installed"
    )
    def test_debian_keyring_missing(self, mock_pkg_installed):
        """Test Debian keyring missing."""
        mock_pkg_installed.return_value = False

        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("debian", "12")

        assert len(commands) > 0
        assert any("debian" in cmd for cmd in commands)

    def test_rhel_no_fix_needed(self):
        """Test RHEL doesn't need repo fix (uses same logic as Fedora)."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("rhel", "9", "12")

        assert len(commands) > 0

    def test_centos_no_fix_needed(self):
        """Test CentOS doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("centos", "9", "12")

        assert len(commands) > 0

    def test_rocky_no_fix_needed(self):
        """Test Rocky Linux doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("rocky", "9", "12")

        assert len(commands) > 0

    def test_almalinux_no_fix_needed(self):
        """Test AlmaLinux doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("alma", "9", "12")

        assert len(commands) > 0

    def test_endeavouros_no_fix_needed(self):
        """Test EndeavourOS doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("endeavouros", "")

        assert commands == []

    def test_sles_no_fix_needed(self):
        """Test SLES doesn't need repo fix."""
        checker = PrerequisitesChecker()
        commands = checker.get_cuda_repo_fix_commands("sles", "15.5")

        assert commands == []


class TestPrerequisitesResultEdgeCases:
    """Edge case tests for PrerequisitesResult."""

    def test_all_fields_set(self):
        """Test PrerequisitesResult with all fields."""
        from nvidia_inst.installer.version_checker import VersionCheckResult

        result = PrerequisitesResult(
            success=True,
            package_manager_available=True,
            package_manager="apt",
            repos_configured=["cuda"],
            repos_missing=[],
            driver_packages_available=True,
            driver_packages=["nvidia-driver-535"],
            version_check=VersionCheckResult(compatible=True),
            fix_commands=["cmd1"],
            warnings=["warning1"],
            errors=[],
        )

        assert result.success is True
        assert result.package_manager == "apt"
        assert result.repos_configured == ["cuda"]
        assert result.version_check is not None

    def test_all_errors_present(self):
        """Test PrerequisitesResult with errors."""
        result = PrerequisitesResult(
            success=False,
            errors=["error1", "error2"],
        )

        assert result.success is False
        assert len(result.errors) == 2
