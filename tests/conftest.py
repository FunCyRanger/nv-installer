"""Shared pytest fixtures for package manager tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for package manager tests."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )
        yield mock_run


@pytest.fixture
def mock_subprocess_popen():
    """Mock subprocess.Popen for package manager tests."""
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "Installation complete"
        mock_popen.return_value = mock_proc
        yield mock_popen


@pytest.fixture
def mock_shutil_which():
    """Mock shutil.which for package manager availability checks."""
    with patch("shutil.which") as mock_which:
        mock_which.return_value = "/usr/bin/test"
        yield mock_which


@pytest.fixture
def mock_open():
    """Mock built-in open for file operations."""
    with patch("builtins.open", create=True) as mock_file:
        mock_file_instance = MagicMock()
        mock_file.return_value.__enter__.return_value = mock_file_instance
        yield mock_file


@pytest.fixture
def apt_success_response():
    """Standard successful apt output."""
    return MagicMock(
        returncode=0,
        stdout="",
        stderr="",
    )


@pytest.fixture
def apt_package_search_output():
    """Sample apt-cache search output."""
    return """nvidia-driver-535 - NVIDIA driver metapackage
nvidia-driver-540 - NVIDIA driver metapackage
nvidia-driver-550 - NVIDIA driver metapackage"""


@pytest.fixture
def apt_policy_output():
    """Sample apt-cache policy output."""
    return """nvidia-driver-535:
  Installed: 535.154.05-0ubuntu1
  Candidate: 535.154.05-0ubuntu2
  Version table:
     535.154.05-0ubuntu2 500
        500 http://security.ubuntu.com focal-security/main amd64 Packages"""


@pytest.fixture
def apt_madison_output():
    """Sample apt-cache madison output."""
    return """| nvidia-driver-535 | 535.154.05-0ubuntu1 | http://archive.ubuntu.com focal/multiverse amd64 Packages
| nvidia-driver-535 | 535.54.06-0ubuntu1 | http://archive.ubuntu.com focal/multiverse amd64 Packages
| nvidia-driver-535 | 535.43.02-0ubuntu1 | http://archive.ubuntu.com focal/multiverse amd64 Packages"""


@pytest.fixture
def dnf_info_output():
    """Sample dnf info output."""
    return """Name         : akmod-nvidia
Version      : 535.154.05
Release      : 1.fc38
Architecture : x86_64"""


@pytest.fixture
def dnf_list_output():
    """Sample dnf list --showduplicates output."""
    return """Installed Packages
akmod-nvidia.x86_64        535.154.05-1.fc38        @rpmfusion-nvidia-driver
akmod-nvidia.x86_64        535.54.06-1.fc38        rpmfusion-nvidia-driver
akmod-nvidia.x86_64        535.43.02-1.fc38        rpmfusion-nvidia-driver"""


@pytest.fixture
def pacman_si_output():
    """Sample pacman -Si output for nvidia."""
    return """Repository      : extra
Name            : nvidia
Version         : 535.154.05-14
Description     : NVIDIA drivers for linux-hardened"""


@pytest.fixture
def pacman_q_output():
    """Sample pacman -Q output."""
    return "nvidia 535.154.05-14"


@pytest.fixture
def pacman_ss_output():
    """Sample pacman -Ss output."""
    return """extra/nvidia 535.154.05-14 [installed]
extra/nvidia-470xx 470.223.02-2
extra/nvidia-535xx 535.154.05-14"""


@pytest.fixture
def zypper_info_output():
    """Sample zypper info output."""
    return """Name        : x11-video-nvidiaG05
Version: 535.154.05
Repository  : rpmfusion-nvidia-driver"""


@pytest.fixture
def zypper_packages_output():
    """Sample zypper packages output."""
    return """| x11-video-nvidiaG05 | 535.154.05 | x86_64 | rpmfusion-nvidia-driver
| x11-video-nvidiaG05 | 535.54.06 | x86_64 | rpmfusion-nvidia-driver"""


@pytest.fixture
def dpkg_query_output():
    """Sample dpkg-query output."""
    return "535.154.05-0ubuntu1"
