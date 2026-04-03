"""E2E CUDA installation workflow tests.

These tests verify CUDA installation with real package managers,
correct package resolution for each distro, and CUDA installer
instantiation.
"""

import os

import pytest

from nvidia_inst.installer.cuda import (
    ArchCUDAInstaller,
    FedoraCUDAInstaller,
    UbuntuCUDAInstaller,
    check_cuda_driver_compatibility,
    get_cuda_installer,
    get_cuda_packages_for_version,
    get_uninstall_cuda_packages,
)


def is_fedora_container() -> bool:
    """Check if running in a Fedora container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            return "fedora" in f.read().lower()
    return False


def is_ubuntu_container() -> bool:
    """Check if running in an Ubuntu container."""
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            content = f.read().lower()
            return "ubuntu" in content
    return False


def has_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0


# ---------------------------------------------------------------------------
# CUDA installer instantiation
# ---------------------------------------------------------------------------


class TestCudaInstallerInstantiation:
    """Test CUDA installer instantiation for each distro."""

    def test_ubuntu_installer_type(self):
        """Test Ubuntu returns UbuntuCUDAInstaller."""
        installer = get_cuda_installer("ubuntu")
        assert isinstance(installer, UbuntuCUDAInstaller)

    def test_debian_installer_type(self):
        """Test Debian returns UbuntuCUDAInstaller."""
        installer = get_cuda_installer("debian")
        assert isinstance(installer, UbuntuCUDAInstaller)

    def test_fedora_installer_type(self):
        """Test Fedora returns FedoraCUDAInstaller."""
        installer = get_cuda_installer("fedora")
        assert isinstance(installer, FedoraCUDAInstaller)

    def test_rhel_installer_type(self):
        """Test RHEL returns FedoraCUDAInstaller."""
        installer = get_cuda_installer("rhel")
        assert isinstance(installer, FedoraCUDAInstaller)

    def test_arch_installer_type(self):
        """Test Arch returns ArchCUDAInstaller."""
        installer = get_cuda_installer("arch")
        assert isinstance(installer, ArchCUDAInstaller)

    def test_manjaro_installer_type(self):
        """Test Manjaro returns ArchCUDAInstaller."""
        installer = get_cuda_installer("manjaro")
        assert isinstance(installer, ArchCUDAInstaller)

    def test_unknown_installer_fallback(self):
        """Test unknown distro falls back to UbuntuCUDAInstaller."""
        installer = get_cuda_installer("unknown")
        assert isinstance(installer, UbuntuCUDAInstaller)


# ---------------------------------------------------------------------------
# get_cuda_packages() correctness
# ---------------------------------------------------------------------------


class TestGetCudaPackages:
    """Test get_cuda_packages() returns correct packages for each distro."""

    def test_ubuntu_default_packages(self):
        """Test Ubuntu default CUDA packages."""
        installer = UbuntuCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert packages == ["cuda-toolkit"]

    def test_ubuntu_versioned_packages(self):
        """Test Ubuntu versioned CUDA packages (still uses meta-package)."""
        installer = UbuntuCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert packages == ["cuda-toolkit"]

    def test_fedora_default_packages(self):
        """Test Fedora default CUDA packages."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert packages == ["cuda-toolkit"]

    def test_fedora_versioned_packages(self):
        """Test Fedora versioned CUDA packages."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert any("cuda-toolkit-12" in pkg for pkg in packages)

    def test_fedora_version_13_packages(self):
        """Test Fedora CUDA 13 packages."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages("13.0")
        assert any("cuda-toolkit-13" in pkg for pkg in packages)

    def test_arch_default_packages(self):
        """Test Arch default CUDA packages."""
        installer = ArchCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert packages == ["cuda"]

    def test_arch_versioned_packages(self):
        """Test Arch versioned CUDA packages (still uses meta-package)."""
        installer = ArchCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert packages == ["cuda"]


# ---------------------------------------------------------------------------
# get_cuda_packages_for_version() correctness
# ---------------------------------------------------------------------------


class TestGetCudaPackagesForVersion:
    """Test get_cuda_packages_for_version() for each distro."""

    def test_ubuntu_versioned(self):
        """Test Ubuntu versioned CUDA packages."""
        packages = get_cuda_packages_for_version("ubuntu", "12.2")
        assert "cuda-12.2" in packages
        assert "cuda-toolkit-12.2" in packages

    def test_fedora_versioned(self):
        """Test Fedora versioned CUDA packages."""
        packages = get_cuda_packages_for_version("fedora", "12.2")
        assert "cuda-toolkit-12.2" in packages

    def test_arch_versioned(self):
        """Test Arch versioned CUDA packages."""
        packages = get_cuda_packages_for_version("arch", "12.2")
        assert "cuda-12.2" in packages

    def test_opensuse_versioned(self):
        """Test openSUSE versioned CUDA packages."""
        packages = get_cuda_packages_for_version("opensuse", "12.2")
        assert "cuda-12.2" in packages

    def test_unknown_distro(self):
        """Test unknown distro returns empty list."""
        packages = get_cuda_packages_for_version("unknown", "12.2")
        assert packages == []


# ---------------------------------------------------------------------------
# get_uninstall_cuda_packages() correctness
# ---------------------------------------------------------------------------


class TestGetUninstallCudaPackages:
    """Test get_uninstall_cuda_packages() for each distro."""

    def test_ubuntu_uninstall_all(self):
        """Test Ubuntu uninstall all CUDA packages."""
        packages = get_uninstall_cuda_packages("ubuntu")
        assert "cuda-*" in packages
        assert "cuda-toolkit-*" in packages

    def test_ubuntu_uninstall_versioned(self):
        """Test Ubuntu uninstall specific CUDA version."""
        packages = get_uninstall_cuda_packages("ubuntu", "12.2")
        assert "cuda-12.2*" in packages
        assert "cuda-toolkit-12.2*" in packages

    def test_fedora_uninstall_all(self):
        """Test Fedora uninstall all CUDA packages."""
        packages = get_uninstall_cuda_packages("fedora")
        assert "cuda-toolkit*" in packages
        assert "cuda-runtime*" in packages

    def test_fedora_uninstall_versioned(self):
        """Test Fedora uninstall specific CUDA version."""
        packages = get_uninstall_cuda_packages("fedora", "12.2")
        assert "cuda-toolkit-12.2*" in packages

    def test_arch_uninstall_all(self):
        """Test Arch uninstall all CUDA packages."""
        packages = get_uninstall_cuda_packages("arch")
        assert "cuda*" in packages

    def test_unknown_distro_uninstall(self):
        """Test unknown distro uninstall returns empty list."""
        packages = get_uninstall_cuda_packages("unknown")
        assert packages == []


# ---------------------------------------------------------------------------
# CUDA driver compatibility checks
# ---------------------------------------------------------------------------


class TestCudaDriverCompatibility:
    """Test CUDA version compatibility with driver versions."""

    def test_cuda_12_with_driver_535(self):
        """Test CUDA 12.x is compatible with driver 535."""
        compatible, msg = check_cuda_driver_compatibility("12.2", "535.154.05")
        assert compatible is True

    def test_cuda_12_with_driver_525(self):
        """Test CUDA 12.x is compatible with driver 525 (minimum)."""
        compatible, msg = check_cuda_driver_compatibility("12.0", "525.60.13")
        assert compatible is True

    def test_cuda_12_with_driver_520(self):
        """Test CUDA 12.x is NOT compatible with driver 520."""
        compatible, msg = check_cuda_driver_compatibility("12.0", "520.56.06")
        assert compatible is False
        assert "requires driver 525" in msg

    def test_cuda_11_with_driver_450(self):
        """Test CUDA 11.x is compatible with driver 450 (minimum)."""
        compatible, msg = check_cuda_driver_compatibility("11.8", "450.191.0")
        assert compatible is True

    def test_cuda_11_with_driver_440(self):
        """Test CUDA 11.x is NOT compatible with driver 440."""
        compatible, msg = check_cuda_driver_compatibility("11.0", "440.100")
        assert compatible is False
        assert "requires driver 450" in msg

    def test_cuda_10_with_driver_410(self):
        """Test CUDA 10.x is compatible with driver 410 (minimum)."""
        compatible, msg = check_cuda_driver_compatibility("10.2", "410.104")
        assert compatible is True

    def test_cuda_10_with_driver_400(self):
        """Test CUDA 10.x is NOT compatible with driver 400."""
        compatible, msg = check_cuda_driver_compatibility("10.0", "400.0")
        assert compatible is False
        assert "requires driver 410" in msg

    def test_invalid_version_handling(self):
        """Test invalid version strings are handled gracefully."""
        compatible, msg = check_cuda_driver_compatibility("invalid", "invalid")
        assert compatible is True  # Falls back to "unable to validate"


# ---------------------------------------------------------------------------
# Real distro CUDA package search
# ---------------------------------------------------------------------------


class TestRealCudaPackageSearch:
    """Test CUDA package search on real repos (container-specific)."""

    @pytest.mark.skipif("not is_ubuntu_container()")
    def test_ubuntu_cuda_package_search(self):
        """Test that CUDA packages are searchable on Ubuntu."""
        import subprocess

        result = subprocess.run(
            ["apt-cache", "search", "cuda-toolkit"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # May or may not find packages depending on repos configured
        # Just verify the command doesn't crash
        assert result.returncode in (0, 100)

    @pytest.mark.skipif("not is_fedora_container()")
    def test_fedora_cuda_package_search(self):
        """Test that CUDA packages are searchable on Fedora."""
        import subprocess

        result = subprocess.run(
            ["dnf", "search", "cuda-toolkit"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # May or may not find packages depending on repos configured
        assert result.returncode in (0, 1)

    @pytest.mark.skipif("not is_ubuntu_container()")
    def test_ubuntu_installer_is_cuda_installed(self):
        """Test is_cuda_installed on Ubuntu (should be False in container)."""
        installer = UbuntuCUDAInstaller()
        # In a clean container, CUDA should not be installed
        result = installer.is_cuda_installed()
        assert isinstance(result, bool)

    @pytest.mark.skipif("not is_fedora_container()")
    def test_fedora_installer_is_cuda_installed(self):
        """Test is_cuda_installed on Fedora (should be False in container)."""
        installer = FedoraCUDAInstaller()
        result = installer.is_cuda_installed()
        assert isinstance(result, bool)

    @pytest.mark.skipif("not is_ubuntu_container()")
    def test_ubuntu_installer_get_installed_version(self):
        """Test get_installed_cuda_version on Ubuntu."""
        installer = UbuntuCUDAInstaller()
        version = installer.get_installed_cuda_version()
        # Should be None in clean container
        assert version is None or isinstance(version, str)

    @pytest.mark.skipif("not is_fedora_container()")
    def test_fedora_installer_get_installed_version(self):
        """Test get_installed_cuda_version on Fedora."""
        installer = FedoraCUDAInstaller()
        version = installer.get_installed_cuda_version()
        assert version is None or isinstance(version, str)
