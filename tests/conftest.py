"""Shared pytest fixtures for package manager tests."""

from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.gpu.compatibility import DriverRange


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


@pytest.fixture
def mock_gpu():
    """Mock GPU info for testing."""
    return {
        "model": "NVIDIA GeForce RTX 3080",
        "compute_capability": 8.6,
        "driver_version": "535.154.05",
        "cuda_version": "12.2",
        "vram": "10GB",
    }


@pytest.fixture
def mock_distro():
    """Mock distribution info for testing."""
    return {
        "id": "ubuntu",
        "version_id": "22.04",
        "name": "Ubuntu 22.04.3 LTS",
        "kernel": "5.15.0-91-generic",
        "pretty_name": "Ubuntu 22.04.3 LTS (Jammy Jellyfish)",
    }


@pytest.fixture
def mock_driver_range():
    """Mock driver range for Ampere GPU."""
    return DriverRange(
        min_version="535.154.05",
        max_version=None,
        max_branch="590",
        cuda_min="11.8",
        cuda_max="12.2",
        is_eol=False,
        is_limited=False,
    )


@pytest.fixture
def mock_driver_range_eol():
    """Mock driver range for Kepler (EOL) GPU."""
    return DriverRange(
        min_version="470.256.02",
        max_version="470.256.02",
        max_branch="470",
        cuda_min="9.0",
        cuda_max="9.0",
        is_eol=True,
        is_limited=True,
        eol_message="Kepler GPUs are end-of-life. Limited to legacy driver 470.xx.",
    )


@pytest.fixture
def mock_user_yes(monkeypatch):
    """Mock user input to return 'yes'."""
    monkeypatch.setattr("builtins.input", lambda _: "y")


@pytest.fixture
def mock_user_no(monkeypatch):
    """Mock user input to return 'no'."""
    monkeypatch.setattr("builtins.input", lambda _: "n")


@pytest.fixture
def mock_user_cancel(monkeypatch):
    """Mock user input to return empty string (cancel)."""
    monkeypatch.setattr("builtins.input", lambda _: "")


@pytest.fixture
def mock_has_nvidia_gpu_true(monkeypatch):
    """Mock has_nvidia_gpu to return True."""
    monkeypatch.setattr(
        "nvidia_inst.gpu.detector.has_nvidia_gpu",
        lambda: True,
    )


@pytest.fixture
def mock_has_nvidia_gpu_false(monkeypatch):
    """Mock has_nvidia_gpu to return False."""
    monkeypatch.setattr(
        "nvidia_inst.gpu.detector.has_nvidia_gpu",
        lambda: False,
    )


@pytest.fixture
def mock_distro_ubuntu(monkeypatch):
    """Mock detect_distro to return Ubuntu."""
    from nvidia_inst.distro.detector import DistroInfo

    monkeypatch.setattr(
        "nvidia_inst.distro.detector.detect_distro",
        lambda: DistroInfo(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04.3 LTS",
            kernel="5.15.0-91-generic",
        ),
    )


@pytest.fixture
def mock_distro_fedora(monkeypatch):
    """Mock detect_distro to return Fedora."""
    from nvidia_inst.distro.detector import DistroInfo

    monkeypatch.setattr(
        "nvidia_inst.distro.detector.detect_distro",
        lambda: DistroInfo(
            id="fedora",
            version_id="39",
            name="Fedora",
            pretty_name="Fedora Linux 39",
            kernel="6.5.6-200.fc39.x86_64",
        ),
    )


@pytest.fixture
def mock_gpu_detect_rtx3080(monkeypatch):
    """Mock detect_gpu to return RTX 3080."""
    from nvidia_inst.gpu.detector import GPUInfo

    monkeypatch.setattr(
        "nvidia_inst.gpu.detector.detect_gpu",
        lambda: GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            compute_capability=8.6,
            driver_version="535.154.05",
            cuda_version="12.2",
            vram="10GB",
        ),
    )


@pytest.fixture
def mock_nouveau_loaded(monkeypatch):
    """Mock check_nouveau to return True (loaded)."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_nouveau", lambda: True)


@pytest.fixture
def mock_nouveau_not_loaded(monkeypatch):
    """Mock check_nouveau to return False (not loaded)."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_nouveau", lambda: False)


@pytest.fixture
def mock_secure_boot_enabled(monkeypatch):
    """Mock check_secure_boot to return True (enabled)."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_secure_boot", lambda: True)


@pytest.fixture
def mock_secure_boot_disabled(monkeypatch):
    """Mock check_secure_boot to return False (disabled)."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_secure_boot", lambda: False)


@pytest.fixture
def mock_is_root(monkeypatch):
    """Mock is_root to return True in all relevant modules."""
    monkeypatch.setattr("nvidia_inst.utils.permissions.is_root", lambda: True)


@pytest.fixture
def mock_kmod_only_setup(monkeypatch):
    """Mock a kmod-only setup (no akmod installed)."""
    monkeypatch.setattr(
        "nvidia_inst.installer.uninstaller.check_nvidia_packages_installed",
        lambda distro: [
            "kmod-nvidia-6.19.8-200.fc43.x86_64-580.126.18-1.fc43.x86_64",
            "xorg-x11-drv-nvidia-580.126.18-1.fc43.x86_64",
            "xorg-x11-drv-nvidia-cuda-580.126.18-1.fc43.x86_64",
            "nvidia-persistenced-580.126.18-1.fc43.x86_64",
        ],
    )


@pytest.fixture
def mock_akmod_setup(monkeypatch):
    """Mock an akmod setup (with akmod installed)."""
    monkeypatch.setattr(
        "nvidia_inst.installer.uninstaller.check_nvidia_packages_installed",
        lambda distro: [
            "akmod-nvidia-580.126.18-1.fc43.x86_64",
            "kmod-nvidia-6.19.8-200.fc43.x86_64-580.126.18-1.fc43.x86_64",
            "xorg-x11-drv-nvidia-580.126.18-1.fc43.x86_64",
            "xorg-x11-drv-nvidia-cuda-580.126.18-1.fc43.x86_64",
        ],
    )


@pytest.fixture
def mock_nvidia_working_kmod_only(monkeypatch):
    """Mock nvidia_working for kmod-only setup."""
    monkeypatch.setattr(
        "nvidia_inst.installer.validation.is_nvidia_working",
        lambda: {
            "is_working": True,
            "driver_version": "580.126.18",
            "kernel_module_loaded": True,
            "gpu_detected": True,
        },
    )


# Fixtures for real package manager integration tests


@pytest.fixture
def dnf_manager():
    """Create DnfManager instance for real integration tests."""
    from nvidia_inst.distro.dnf import DnfManager

    return DnfManager()


@pytest.fixture
def apt_manager():
    """Create AptManager instance for real integration tests."""
    from nvidia_inst.distro.apt import AptManager

    return AptManager()


@pytest.fixture
def pacman_manager():
    """Create PacmanManager instance for real integration tests."""
    from nvidia_inst.distro.pacman import PacmanManager

    return PacmanManager()


@pytest.fixture
def zypper_manager():
    """Create ZypperManager instance for real integration tests."""
    from nvidia_inst.distro.zypper import ZypperManager

    return ZypperManager()


# Additional fixtures for installation/reinstall/driver switch tests


@pytest.fixture
def mock_nvidia_open_installed(monkeypatch):
    """Mock check_nvidia_open_installed to return True."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_nvidia_open_installed", lambda: True)


@pytest.fixture
def mock_nvidia_open_not_installed(monkeypatch):
    """Mock check_nvidia_open_installed to return False."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_nvidia_open_installed", lambda: False)


@pytest.fixture
def mock_nonfree_available(monkeypatch):
    """Mock check_nonfree_available to return True."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_nonfree_available", lambda: True)


@pytest.fixture
def mock_nonfree_not_available(monkeypatch):
    """Mock check_nonfree_available to return False."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.check_nonfree_available", lambda: False)


@pytest.fixture
def mock_current_driver_type(monkeypatch):
    """Mock get_current_driver_type. Parametrize with driver_type value."""

    def _mock(driver_type="none"):
        from nvidia_inst.installer import driver

        monkeypatch.setattr(
            f"{driver.__name__}.get_current_driver_type", lambda: driver_type
        )

    return _mock


@pytest.fixture
def mock_cuda_version(monkeypatch):
    """Mock detect_installed_cuda_version to return a version string."""

    def _mock(version="12.2"):
        import nvidia_inst.installer.cuda as cuda_module

        monkeypatch.setattr(
            f"{cuda_module.__name__}.detect_installed_cuda_version",
            lambda: version,
        )

    return _mock


@pytest.fixture
def mock_no_cuda(monkeypatch):
    """Mock detect_installed_cuda_version to return None."""
    import nvidia_inst.installer.cuda as cuda_module

    monkeypatch.setattr(
        f"{cuda_module.__name__}.detect_installed_cuda_version",
        lambda: None,
    )


@pytest.fixture
def mock_pkg_manager():
    """Create a fully mocked package manager with all methods."""
    mock = MagicMock()
    mock.update.return_value = MagicMock(returncode=0)
    mock.install.return_value = MagicMock(returncode=0)
    mock.remove.return_value = MagicMock(returncode=0)
    mock.pin_version.return_value = MagicMock(returncode=0)
    mock.is_installed.return_value = False
    return mock


@pytest.fixture
def mock_detect_gpu_multiple():
    """Parameterized GPU info fixture. Use with different GPU models."""

    def _mock(
        model="NVIDIA GeForce RTX 3080",
        compute_capability=8.6,
        generation="ampere",
        driver_version="535.154.05",
        cuda_version="12.2",
        vram="10GB",
    ):
        from nvidia_inst.gpu.detector import GPUInfo

        return GPUInfo(
            model=model,
            compute_capability=compute_capability,
            generation=generation,
            driver_version=driver_version,
            cuda_version=cuda_version,
            vram=vram,
        )

    return _mock


@pytest.fixture
def mock_distro_all():
    """Parameterized distro fixture. Use with different distro IDs."""

    def _mock(
        distro_id="ubuntu",
        version_id="22.04",
        name="Ubuntu",
        pretty_name="Ubuntu 22.04 LTS",
        kernel="5.15.0-generic",
    ):
        from nvidia_inst.distro.detector import DistroInfo

        return DistroInfo(
            id=distro_id,
            version_id=version_id,
            name=name,
            pretty_name=pretty_name,
            kernel=kernel,
        )

    return _mock


@pytest.fixture
def mock_driver_range_pascal():
    """Mock driver range for Pascal (limited) GPU."""
    return DriverRange(
        min_version="450.191.0",
        max_version="580.142",
        max_branch="580",
        cuda_min="8.0",
        cuda_max="12.x",
        cuda_is_locked=True,
        cuda_locked_major="12",
        is_eol=False,
        is_limited=True,
    )


@pytest.fixture
def mock_driver_range_blackwell():
    """Mock driver range for Blackwell GPU."""
    return DriverRange(
        min_version="550.127.05",
        max_version=None,
        max_branch="590",
        cuda_min="12.4",
        cuda_max="13.x",
        is_eol=False,
        is_limited=False,
    )


@pytest.fixture
def mock_disable_nouveau_success(monkeypatch):
    """Mock disable_nouveau to return True."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.disable_nouveau", lambda: True)


@pytest.fixture
def mock_disable_nouveau_failure(monkeypatch):
    """Mock disable_nouveau to return False."""
    from nvidia_inst.installer import driver

    monkeypatch.setattr(f"{driver.__name__}.disable_nouveau", lambda: False)
