"""Tests for distro/factory.py module."""

from unittest.mock import patch

import pytest

from nvidia_inst.distro.apt import AptManager
from nvidia_inst.distro.dnf import DnfManager
from nvidia_inst.distro.factory import (
    _get_manager_for_family,
    get_manager_for_tool,
    get_package_manager,
    get_supported_tools,
    is_tool_supported,
)
from nvidia_inst.distro.pacman import PacmanManager
from nvidia_inst.distro.zypper import ZypperManager


class TestGetPackageManager:
    """Tests for get_package_manager function."""

    @patch("nvidia_inst.distro.factory.detect_package_tool", return_value="apt")
    def test_get_apt_manager(self, mock_detect):
        """Test getting APT manager."""
        manager = get_package_manager()
        assert isinstance(manager, AptManager)

    @patch("nvidia_inst.distro.factory.detect_package_tool", return_value="dnf")
    def test_get_dnf_manager(self, mock_detect):
        """Test getting DNF manager."""
        manager = get_package_manager()
        assert isinstance(manager, DnfManager)

    @patch("nvidia_inst.distro.factory.detect_package_tool", return_value="pacman")
    def test_get_pacman_manager(self, mock_detect):
        """Test getting Pacman manager."""
        manager = get_package_manager()
        assert isinstance(manager, PacmanManager)

    @patch("nvidia_inst.distro.factory.detect_package_tool", return_value="zypper")
    def test_get_zypper_manager(self, mock_detect):
        """Test getting Zypper manager."""
        manager = get_package_manager()
        assert isinstance(manager, ZypperManager)

    @patch("nvidia_inst.distro.factory.detect_package_tool", return_value=None)
    def test_get_manager_no_tool(self, mock_detect):
        """Test RuntimeError when no tool found."""
        with pytest.raises(RuntimeError, match="No supported package manager"):
            get_package_manager()


class TestGetManagerForTool:
    """Tests for get_manager_for_tool function."""

    def test_get_apt_manager(self):
        """Test getting APT manager by tool name."""
        manager = get_manager_for_tool("apt")
        assert isinstance(manager, AptManager)

    def test_get_dnf_manager(self):
        """Test getting DNF manager by tool name."""
        manager = get_manager_for_tool("dnf")
        assert isinstance(manager, DnfManager)

    def test_get_pacman_manager(self):
        """Test getting Pacman manager by tool name."""
        manager = get_manager_for_tool("pacman")
        assert isinstance(manager, PacmanManager)

    def test_get_zypper_manager(self):
        """Test getting Zypper manager by tool name."""
        manager = get_manager_for_tool("zypper")
        assert isinstance(manager, ZypperManager)

    def test_get_manager_unknown_tool(self):
        """Test getting manager for unknown tool returns None."""
        manager = get_manager_for_tool("unknown")
        assert manager is None


class TestIsToolSupported:
    """Tests for is_tool_supported function."""

    def test_apt_supported(self):
        """Test APT is supported."""
        assert is_tool_supported("apt") is True

    def test_dnf_supported(self):
        """Test DNF is supported."""
        assert is_tool_supported("dnf") is True

    def test_pacman_supported(self):
        """Test Pacman is supported."""
        assert is_tool_supported("pacman") is True

    def test_zypper_supported(self):
        """Test Zypper is supported."""
        assert is_tool_supported("zypper") is True

    def test_unknown_not_supported(self):
        """Test unknown tool is not supported."""
        assert is_tool_supported("unknown") is False


class TestGetSupportedTools:
    """Tests for get_supported_tools function."""

    def test_returns_list(self):
        """Test returns a list."""
        tools = get_supported_tools()
        assert isinstance(tools, list)

    def test_contains_apt(self):
        """Test list contains APT."""
        tools = get_supported_tools()
        assert "apt" in tools

    def test_contains_dnf(self):
        """Test list contains DNF."""
        tools = get_supported_tools()
        assert "dnf" in tools

    def test_contains_pacman(self):
        """Test list contains Pacman."""
        tools = get_supported_tools()
        assert "pacman" in tools

    def test_contains_zypper(self):
        """Test list contains Zypper."""
        tools = get_supported_tools()
        assert "zypper" in tools


class TestGetManagerForFamily:
    """Tests for _get_manager_for_family function."""

    def test_debian_family(self):
        """Test debian family returns AptManager."""
        manager_class = _get_manager_for_family("debian")
        assert manager_class == AptManager

    def test_fedora_family(self):
        """Test fedora family returns DnfManager."""
        manager_class = _get_manager_for_family("fedora")
        assert manager_class == DnfManager

    def test_arch_family(self):
        """Test arch family returns PacmanManager."""
        manager_class = _get_manager_for_family("arch")
        assert manager_class == PacmanManager

    def test_suse_family(self):
        """Test suse family returns ZypperManager."""
        manager_class = _get_manager_for_family("suse")
        assert manager_class == ZypperManager

    def test_unknown_family(self):
        """Test unknown family returns None."""
        manager_class = _get_manager_for_family("unknown")
        assert manager_class is None
