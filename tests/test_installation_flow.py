"""Integration tests for installation flow with tool detection."""

import pytest

from nvidia_inst.cli.installer import get_packages_to_remove
from nvidia_inst.distro.factory import get_package_manager
from nvidia_inst.distro.tools import get_install_command, get_remove_command


class TestToolDetection:
    """Test tool detection and usage."""

    def test_package_manager_has_tool_property(self):
        """Verify package manager has tool property."""
        pkg_mgr = get_package_manager()
        assert hasattr(pkg_mgr, "tool")
        assert isinstance(pkg_mgr.tool, str)
        assert len(pkg_mgr.tool) > 0

    def test_tool_property_returns_valid_tool(self):
        """Verify tool property returns a known tool."""
        pkg_mgr = get_package_manager()
        valid_tools = ["apt", "dnf", "dnf4", "dnf5", "pacman", "zypper", "pamac"]
        assert pkg_mgr.tool in valid_tools, f"Unexpected tool: {pkg_mgr.tool}"

    def test_tool_works_with_get_install_command(self):
        """Verify tool works with get_install_command."""
        pkg_mgr = get_package_manager()
        cmd = get_install_command(pkg_mgr.tool)
        assert isinstance(cmd, list)
        assert len(cmd) > 1

    def test_tool_works_with_get_remove_command(self):
        """Verify tool works with get_remove_command."""
        pkg_mgr = get_package_manager()
        cmd = get_remove_command(pkg_mgr.tool)
        assert isinstance(cmd, list)
        assert len(cmd) > 1


class TestPackagesToRemove:
    """Test get_packages_to_remove function."""

    def test_get_packages_to_remove_with_apt(self):
        """Test get_packages_to_remove with apt tool."""
        packages = get_packages_to_remove("apt")
        assert isinstance(packages, list)
        assert len(packages) > 0
        assert "nvidia-driver-*" in packages

    def test_get_packages_to_remove_with_dnf(self):
        """Test get_packages_to_remove with dnf tool."""
        packages = get_packages_to_remove("dnf")
        assert isinstance(packages, list)
        assert len(packages) > 0
        assert "akmod-nvidia" in packages

    def test_get_packages_to_remove_with_pacman(self):
        """Test get_packages_to_remove with pacman tool."""
        packages = get_packages_to_remove("pacman")
        assert isinstance(packages, list)
        assert len(packages) > 0
        assert "nvidia" in packages

    def test_get_packages_to_remove_with_zypper(self):
        """Test get_packages_to_remove with zypper tool."""
        packages = get_packages_to_remove("zypper")
        assert isinstance(packages, list)
        assert len(packages) > 0

    def test_get_packages_to_remove_with_unknown_returns_empty(self):
        """Test get_packages_to_remove with unknown tool returns empty list."""
        packages = get_packages_to_remove("unknown_tool")
        assert isinstance(packages, list)
        assert len(packages) == 0


class TestToolVsDistroId:
    """Test that tool names work but distro IDs don't (regression test)."""

    def test_tool_names_work(self):
        """Verify tool names work with get_packages_to_remove."""
        tool_names = ["apt", "dnf", "pacman", "zypper"]
        for tool in tool_names:
            packages = get_packages_to_remove(tool)
            assert isinstance(packages, list), f"Expected list for tool={tool}"

    def test_distro_ids_return_empty(self):
        """Verify distro IDs return empty list (not valid tool names).

        This documents the expected behavior: get_packages_to_remove expects
        tool names (dnf, apt, pacman) not distro IDs (fedora, ubuntu, arch).
        """
        distro_ids = ["fedora", "ubuntu", "arch", "debian", "opensuse"]
        for distro_id in distro_ids:
            packages = get_packages_to_remove(distro_id)
            assert packages == [], f"Expected empty list for distro_id={distro_id}"

    def test_install_command_with_tool(self):
        """Verify install command works with tool name."""
        cmd = get_install_command("apt")
        assert "apt-get" in cmd or "apt" in cmd

    def test_install_command_unknown_tool_raises(self):
        """Verify install command raises ValueError for unknown tool."""
        with pytest.raises(ValueError, match="Unknown package tool"):
            get_install_command("not_a_real_tool")
