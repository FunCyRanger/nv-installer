"""Tests for tool-based package manager detection."""

import pytest
from unittest.mock import patch, MagicMock

from nvidia_inst.distro.tools import (
    PackageContext,
    TOOL_FAMILIES,
    TOOL_PRIORITY,
    DISTRO_FAMILIES,
    detect_package_tool,
    get_tool_family,
    get_distro_family,
    detect_package_context,
    get_install_command,
    get_remove_command,
    get_update_command,
    is_aur_tool,
    is_gui_tool,
)


class TestPackageContext:
    """Tests for PackageContext dataclass."""

    def test_create_context(self):
        """Test creating PackageContext."""
        ctx = PackageContext(
            tool="apt",
            distro_id="ubuntu",
            distro_family="debian",
            version_id="22.04",
        )
        assert ctx.tool == "apt"
        assert ctx.distro_id == "ubuntu"
        assert ctx.distro_family == "debian"
        assert ctx.version_id == "22.04"


class TestToolFamilies:
    """Tests for TOOL_FAMILIES dict."""

    def test_has_apt(self):
        """Test apt is mapped to debian."""
        assert TOOL_FAMILIES["apt"] == "debian"

    def test_has_dnf(self):
        """Test dnf is mapped to fedora."""
        assert TOOL_FAMILIES["dnf"] == "fedora"

    def test_has_pacman(self):
        """Test pacman is mapped to arch."""
        assert TOOL_FAMILIES["pacman"] == "arch"

    def test_has_zypper(self):
        """Test zypper is mapped to suse."""
        assert TOOL_FAMILIES["zypper"] == "suse"


class TestDistroFamilies:
    """Tests for DISTRO_FAMILIES dict."""

    def test_ubuntu_is_debian(self):
        """Test ubuntu is in debian family."""
        assert DISTRO_FAMILIES["ubuntu"] == "debian"

    def test_fedora_is_fedora(self):
        """Test fedora is in fedora family."""
        assert DISTRO_FAMILIES["fedora"] == "fedora"

    def test_arch_is_arch(self):
        """Test arch is in arch family."""
        assert DISTRO_FAMILIES["arch"] == "arch"

    def test_opensuse_is_suse(self):
        """Test opensuse is in suse family."""
        assert DISTRO_FAMILIES["opensuse"] == "suse"


class TestDetectPackageTool:
    """Tests for detect_package_tool function."""

    @patch("shutil.which")
    def test_detects_apt(self, mock_which):
        """Test detects apt when available."""
        mock_which.side_effect = lambda x: "/usr/bin/apt" if x == "apt" else None
        result = detect_package_tool()
        assert result == "apt"

    @patch("shutil.which")
    def test_detects_dnf(self, mock_which):
        """Test detects dnf when available."""
        mock_which.side_effect = lambda x: "/usr/bin/dnf" if x == "dnf" else None
        result = detect_package_tool()
        assert result == "dnf"

    @patch("shutil.which")
    def test_detects_pacman(self, mock_which):
        """Test detects pacman when available."""
        mock_which.side_effect = lambda x: "/usr/bin/pacman" if x == "pacman" else None
        result = detect_package_tool()
        assert result == "pacman"

    @patch("shutil.which", return_value=None)
    def test_returns_none_when_no_tool(self, mock_which):
        """Test returns None when no tool found."""
        result = detect_package_tool()
        assert result is None


class TestGetToolFamily:
    """Tests for get_tool_family function."""

    def test_apt_family(self):
        """Test apt returns debian."""
        assert get_tool_family("apt") == "debian"

    def test_dnf_family(self):
        """Test dnf returns fedora."""
        assert get_tool_family("dnf") == "fedora"

    def test_pacman_family(self):
        """Test pacman returns arch."""
        assert get_tool_family("pacman") == "arch"

    def test_zypper_family(self):
        """Test zypper returns suse."""
        assert get_tool_family("zypper") == "suse"

    def test_unknown_tool(self):
        """Test unknown tool returns unknown."""
        assert get_tool_family("unknown-tool") == "unknown"


class TestGetDistroFamily:
    """Tests for get_distro_family function."""

    def test_ubuntu_family(self):
        """Test ubuntu returns debian."""
        assert get_distro_family("ubuntu") == "debian"

    def test_fedora_family(self):
        """Test fedora returns fedora."""
        assert get_distro_family("fedora") == "fedora"

    def test_arch_family(self):
        """Test arch returns arch."""
        assert get_distro_family("arch") == "arch"

    def test_opensuse_family(self):
        """Test opensuse returns suse."""
        assert get_distro_family("opensuse") == "suse"

    def test_unknown_distro(self):
        """Test unknown distro returns distro_id."""
        assert get_distro_family("my-custom-distro") == "my-custom-distro"


class TestDetectPackageContext:
    """Tests for detect_package_context function."""

    @patch("nvidia_inst.distro.tools.detect_package_tool", return_value="apt")
    def test_detect_context_ubuntu(self, mock_detect):
        """Test detects context for Ubuntu."""
        ctx = detect_package_context("ubuntu", "22.04")
        assert ctx.tool == "apt"
        assert ctx.distro_id == "ubuntu"
        assert ctx.distro_family == "debian"
        assert ctx.version_id == "22.04"

    @patch("nvidia_inst.distro.tools.detect_package_tool", return_value="dnf")
    def test_detect_context_fedora(self, mock_detect):
        """Test detects context for Fedora."""
        ctx = detect_package_context("fedora", "39")
        assert ctx.tool == "dnf"
        assert ctx.distro_id == "fedora"
        assert ctx.distro_family == "fedora"

    @patch("nvidia_inst.distro.tools.detect_package_tool", return_value=None)
    def test_detect_context_no_tool(self, mock_detect):
        """Test raises RuntimeError when no tool found."""
        with pytest.raises(RuntimeError):
            detect_package_context("ubuntu", "22.04")


class TestGetInstallCommand:
    """Tests for get_install_command function."""

    def test_apt_install(self):
        """Test apt install command."""
        cmd = get_install_command("apt")
        assert cmd == ["apt-get", "install", "-y"]

    def test_dnf_install(self):
        """Test dnf install command."""
        cmd = get_install_command("dnf")
        assert cmd == ["dnf", "install", "-y"]

    def test_pacman_install(self):
        """Test pacman install command."""
        cmd = get_install_command("pacman")
        assert cmd == ["pacman", "-S", "--noconfirm"]

    def test_zypper_install(self):
        """Test zypper install command."""
        cmd = get_install_command("zypper")
        assert cmd == ["zypper", "install", "-y"]

    def test_apt_upgrade(self):
        """Test apt upgrade command."""
        cmd = get_install_command("apt", upgrade=True)
        assert cmd == ["apt-get", "upgrade", "-y"]

    def test_dnf_upgrade(self):
        """Test dnf upgrade command."""
        cmd = get_install_command("dnf", upgrade=True)
        assert cmd == ["dnf", "upgrade", "-y"]

    def test_pacman_upgrade(self):
        """Test pacman upgrade command."""
        cmd = get_install_command("pacman", upgrade=True)
        assert cmd == ["pacman", "-Syu", "--noconfirm"]

    def test_unknown_tool(self):
        """Test unknown tool raises ValueError."""
        with pytest.raises(ValueError):
            get_install_command("unknown-tool")


class TestGetRemoveCommand:
    """Tests for get_remove_command function."""

    def test_apt_remove(self):
        """Test apt remove command."""
        cmd = get_remove_command("apt")
        assert cmd == ["apt-get", "remove", "-y", "--purge"]

    def test_dnf_remove(self):
        """Test dnf remove command."""
        cmd = get_remove_command("dnf")
        assert cmd == ["dnf", "remove", "-y"]

    def test_pacman_remove(self):
        """Test pacman remove command."""
        cmd = get_remove_command("pacman")
        assert cmd == ["pacman", "-Rns", "--noconfirm"]

    def test_unknown_tool(self):
        """Test unknown tool raises ValueError."""
        with pytest.raises(ValueError):
            get_remove_command("unknown-tool")


class TestGetUpdateCommand:
    """Tests for get_update_command function."""

    def test_apt_update(self):
        """Test apt update command."""
        cmd = get_update_command("apt")
        assert cmd == ["apt-get", "update"]

    def test_dnf_update(self):
        """Test dnf update command."""
        cmd = get_update_command("dnf")
        assert cmd == ["dnf", "makecache"]

    def test_pacman_update(self):
        """Test pacman update command."""
        cmd = get_update_command("pacman")
        assert cmd == ["pacman", "-Sy"]

    def test_zypper_update(self):
        """Test zypper update command."""
        cmd = get_update_command("zypper")
        assert cmd == ["zypper", "refresh"]

    def test_unknown_tool(self):
        """Test unknown tool raises ValueError."""
        with pytest.raises(ValueError):
            get_update_command("unknown-tool")


class TestIsAurTool:
    """Tests for is_aur_tool function."""

    def test_paru_is_aur(self):
        """Test paru is AUR tool."""
        assert is_aur_tool("paru") is True

    def test_yay_is_aur(self):
        """Test yay is AUR tool."""
        assert is_aur_tool("yay") is True

    def test_pacman_not_aur(self):
        """Test pacman is not AUR tool."""
        assert is_aur_tool("pacman") is False

    def test_apt_not_aur(self):
        """Test apt is not AUR tool."""
        assert is_aur_tool("apt") is False


class TestIsGuiTool:
    """Tests for is_gui_tool function."""

    def test_pamac_is_gui(self):
        """Test pamac is GUI tool."""
        assert is_gui_tool("pamac") is True

    def test_pacman_not_gui(self):
        """Test pacman is not GUI tool."""
        assert is_gui_tool("pacman") is False

    def test_apt_not_gui(self):
        """Test apt is not GUI tool."""
        assert is_gui_tool("apt") is False
