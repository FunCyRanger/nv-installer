"""Tests for installer/cuda.py module."""

import pytest
from unittest.mock import patch, MagicMock

from nvidia_inst.installer.cuda import (
    CUDAInstaller,
    UbuntuCUDAInstaller,
    FedoraCUDAInstaller,
    ArchCUDAInstaller,
    get_cuda_installer,
    _parse_cuda_version_from_package,
    get_cuda_packages_tool_based,
)
from nvidia_inst.distro.tools import PackageContext


class TestUbuntuCUDAInstaller:
    """Tests for UbuntuCUDAInstaller."""

    def test_get_cuda_packages(self):
        """Test getting CUDA packages."""
        installer = UbuntuCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert "cuda-toolkit" in packages

    def test_get_cuda_packages_with_version(self):
        """Test getting CUDA packages with version."""
        installer = UbuntuCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert "cuda-toolkit" in packages


class TestFedoraCUDAInstaller:
    """Tests for FedoraCUDAInstaller."""

    def test_get_cuda_packages_no_version(self):
        """Test getting CUDA packages without version."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert "cuda-toolkit" in packages

    def test_get_cuda_packages_with_version(self):
        """Test getting CUDA packages with version."""
        installer = FedoraCUDAInstaller()
        packages = installer.get_cuda_packages("12.0")
        assert any("cuda-toolkit-12" in pkg for pkg in packages)


class TestArchCUDAInstaller:
    """Tests for ArchCUDAInstaller."""

    def test_get_cuda_packages(self):
        """Test getting CUDA packages."""
        installer = ArchCUDAInstaller()
        packages = installer.get_cuda_packages()
        assert "cuda" in packages


class TestGetCudaInstaller:
    """Tests for get_cuda_installer function."""

    def test_ubuntu_installer(self):
        """Test getting Ubuntu installer."""
        installer = get_cuda_installer("ubuntu")
        assert isinstance(installer, UbuntuCUDAInstaller)

    def test_fedora_installer(self):
        """Test getting Fedora installer."""
        installer = get_cuda_installer("fedora")
        assert isinstance(installer, FedoraCUDAInstaller)

    def test_arch_installer(self):
        """Test getting Arch installer."""
        installer = get_cuda_installer("arch")
        assert isinstance(installer, ArchCUDAInstaller)

    def test_unknown_distro_fallback(self):
        """Test unknown distro falls back to Ubuntu installer."""
        installer = get_cuda_installer("unknown")
        assert isinstance(installer, UbuntuCUDAInstaller)


class TestParseCudaVersionFromPackage:
    """Tests for _parse_cuda_version_from_package function."""

    def test_parse_cuda_toolkit_12_6(self):
        """Test parsing cuda-toolkit-12-6 package."""
        version = _parse_cuda_version_from_package("cuda-toolkit-12-6-12.6.3-1.x86_64")
        assert version == "12.6"

    def test_parse_cuda_toolkit_13(self):
        """Test parsing cuda-toolkit-13 package."""
        version = _parse_cuda_version_from_package("cuda-toolkit-13-0")
        assert version == "13.0"

    def test_parse_cuda_12_2(self):
        """Test parsing cuda-12.2 package."""
        version = _parse_cuda_version_from_package("cuda-12.2-12.2.2-1.x86_64")
        assert version == "12.2"

    def test_parse_invalid_package(self):
        """Test parsing invalid package name."""
        version = _parse_cuda_version_from_package("not-a-cuda-package")
        assert version is None


class TestGetCudaPackagesToolBased:
    """Tests for get_cuda_packages_tool_based function."""

    def test_apt_no_version(self):
        """Test APT packages without version."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )
        packages = get_cuda_packages_tool_based(ctx)
        assert "cuda-toolkit" in packages

    def test_dnf_no_version(self):
        """Test DNF packages without version."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        packages = get_cuda_packages_tool_based(ctx)
        assert "cuda-toolkit" in packages

    def test_dnf_with_version(self):
        """Test DNF packages with version."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        packages = get_cuda_packages_tool_based(ctx, version="12.0")
        # DNF tool-based returns cuda-toolkit (version handled by package manager)
        assert "cuda-toolkit" in packages

    def test_pacman_no_version(self):
        """Test Pacman packages without version."""
        ctx = PackageContext(
            tool="pacman", distro_id="arch", distro_family="arch", version_id=""
        )
        packages = get_cuda_packages_tool_based(ctx)
        assert "cuda" in packages
