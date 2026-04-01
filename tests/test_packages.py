"""Tests for package name maps."""


from nvidia_inst.distro.packages import (
    CUDA_MAJOR_PACKAGES,
    CUDA_PACKAGES,
    DRIVER_OPEN_PACKAGES,
    DRIVER_PACKAGES,
    NOUVEAU_REMOVE_PACKAGES,
    _get_package_from_map,
    format_package_name,
    get_cuda_major_packages,
    get_cuda_packages,
    get_driver_open_packages,
    get_driver_packages,
    get_nouveau_remove_packages,
)
from nvidia_inst.distro.tools import PackageContext


class TestPackageMaps:
    """Tests for package map dictionaries."""

    def test_driver_packages_has_tools(self):
        """Test DRIVER_PACKAGES has expected tools."""
        assert "apt" in DRIVER_PACKAGES
        assert "dnf" in DRIVER_PACKAGES
        assert "pacman" in DRIVER_PACKAGES
        assert "zypper" in DRIVER_PACKAGES

    def test_driver_open_packages_has_tools(self):
        """Test DRIVER_OPEN_PACKAGES has expected tools."""
        assert "apt" in DRIVER_OPEN_PACKAGES
        assert "dnf" in DRIVER_OPEN_PACKAGES
        assert "pacman" in DRIVER_OPEN_PACKAGES

    def test_cuda_packages_has_tools(self):
        """Test CUDA_PACKAGES has expected tools."""
        assert "apt" in CUDA_PACKAGES
        assert "dnf" in CUDA_PACKAGES
        assert "pacman" in CUDA_PACKAGES

    def test_cuda_major_packages_has_tools(self):
        """Test CUDA_MAJOR_PACKAGES has expected tools."""
        assert "apt" in CUDA_MAJOR_PACKAGES
        assert "dnf" in CUDA_MAJOR_PACKAGES
        assert "pacman" in CUDA_MAJOR_PACKAGES

    def test_nouveau_remove_packages_has_tools(self):
        """Test NOUVEAU_REMOVE_PACKAGES has expected tools."""
        assert "apt" in NOUVEAU_REMOVE_PACKAGES
        assert "dnf" in NOUVEAU_REMOVE_PACKAGES
        assert "pacman" in NOUVEAU_REMOVE_PACKAGES


class TestGetPackageFromMap:
    """Tests for _get_package_from_map function."""

    def test_exact_match(self):
        """Test returns packages for exact distro match."""
        pkg_map = {"ubuntu": ["pkg1", "pkg2"], "_default": ["default"]}
        result = _get_package_from_map(pkg_map, "ubuntu")
        assert result == ["pkg1", "pkg2"]

    def test_default_fallback(self):
        """Test returns default packages when no exact match."""
        pkg_map = {"ubuntu": ["pkg1"], "_default": ["default"]}
        result = _get_package_from_map(pkg_map, "fedora")
        assert result == ["default"]

    def test_empty_map(self):
        """Test returns empty list for empty map."""
        result = _get_package_from_map({}, "fedora")
        assert result == []

    def test_format_kwargs(self):
        """Test formats package names with kwargs."""
        pkg_map = {"_default": ["pkg-{branch}"]}
        result = _get_package_from_map(pkg_map, "fedora", branch="535")
        assert result == ["pkg-535"]

    def test_missing_format_arg(self):
        """Test uses package name as-is when format arg missing."""
        pkg_map = {"_default": ["pkg-{branch}"]}
        result = _get_package_from_map(pkg_map, "fedora")
        assert result == ["pkg-{branch}"]


class TestGetDriverPackages:
    """Tests for get_driver_packages function."""

    def test_ubuntu_packages(self):
        """Test returns Ubuntu driver packages."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )
        result = get_driver_packages(ctx, branch="535")
        assert "nvidia-driver-535" in result

    def test_fedora_packages(self):
        """Test returns Fedora driver packages."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        result = get_driver_packages(ctx)
        assert "akmod-nvidia" in result

    def test_arch_packages(self):
        """Test returns Arch driver packages."""
        ctx = PackageContext(
            tool="pacman", distro_id="arch", distro_family="arch", version_id=""
        )
        result = get_driver_packages(ctx)
        assert "nvidia" in result


class TestGetDriverOpenPackages:
    """Tests for get_driver_open_packages function."""

    def test_ubuntu_open_packages(self):
        """Test returns Ubuntu open driver packages."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )
        result = get_driver_open_packages(ctx, branch="535")
        assert any("nvidia" in pkg for pkg in result)

    def test_fedora_open_packages(self):
        """Test returns Fedora open driver packages."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        result = get_driver_open_packages(ctx)
        assert "akmod-nvidia" in result


class TestGetCudaPackages:
    """Tests for get_cuda_packages function."""

    def test_ubuntu_cuda_packages(self):
        """Test returns Ubuntu CUDA packages."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )
        result = get_cuda_packages(ctx, "12.2")
        assert "cuda-toolkit" in result

    def test_fedora_cuda_packages(self):
        """Test returns Fedora CUDA packages."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        result = get_cuda_packages(ctx, "12.2")
        assert "cuda-toolkit" in result

    def test_arch_cuda_packages(self):
        """Test returns Arch CUDA packages."""
        ctx = PackageContext(
            tool="pacman", distro_id="arch", distro_family="arch", version_id=""
        )
        result = get_cuda_packages(ctx, "12.2")
        assert "cuda" in result


class TestGetCudaMajorPackages:
    """Tests for get_cuda_major_packages function."""

    def test_ubuntu_cuda_major_packages(self):
        """Test returns Ubuntu CUDA major version packages."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )
        result = get_cuda_major_packages(ctx, "12")
        assert any("cuda-toolkit-12" in pkg for pkg in result)

    def test_fedora_cuda_major_packages(self):
        """Test returns Fedora CUDA major version packages."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        result = get_cuda_major_packages(ctx, "12")
        assert any("cuda-toolkit-12" in pkg for pkg in result)


class TestGetNouveauRemovePackages:
    """Tests for get_nouveau_remove_packages function."""

    def test_ubuntu_nouveau_packages(self):
        """Test returns Ubuntu nouveau packages."""
        ctx = PackageContext(
            tool="apt", distro_id="ubuntu", distro_family="debian", version_id="22.04"
        )
        result = get_nouveau_remove_packages(ctx)
        assert "xserver-xorg-video-nouveau" in result

    def test_fedora_nouveau_packages(self):
        """Test returns Fedora nouveau packages."""
        ctx = PackageContext(
            tool="dnf", distro_id="fedora", distro_family="fedora", version_id="39"
        )
        result = get_nouveau_remove_packages(ctx)
        assert "xorg-x11-drv-nouveau" in result


class TestFormatPackageName:
    """Tests for format_package_name function."""

    def test_format_with_kwargs(self):
        """Test formats package name with kwargs."""
        result = format_package_name("pkg-{branch}", branch="535")
        assert result == "pkg-535"

    def test_format_missing_arg(self):
        """Test returns template when arg missing."""
        result = format_package_name("pkg-{branch}")
        assert result == "pkg-{branch}"

    def test_format_no_placeholder(self):
        """Test returns name when no placeholder."""
        result = format_package_name("simple-package")
        assert result == "simple-package"
