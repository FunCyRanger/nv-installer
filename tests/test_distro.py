"""Tests for distro detection."""

from unittest.mock import patch

from nvidia_inst.distro.detector import (
    DistroInfo,
    detect_distro,
    is_arch,
    is_debian,
    is_fedora,
    is_ubuntu,
)
from nvidia_inst.distro.factory import get_package_manager


class TestDistroDetection:
    """Test distribution detection."""

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists")
    def test_detect_from_os_release_ubuntu(self, mock_exists, mock_read_text):
        """Test detection from /etc/os-release for Ubuntu."""
        mock_exists.return_value = True
        mock_read_text.return_value = '''NAME="Ubuntu"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
ID=ubuntu
VERSION_ID="22.04"
PRETTY_NAME="Ubuntu 22.04.3 LTS"
'''

        with patch("nvidia_inst.distro.detector._get_kernel_version", return_value="5.15.0-91-generic"):
            distro = detect_distro()

        assert distro.id == "ubuntu"
        assert distro.version_id == "22.04"
        assert "Ubuntu" in distro.name

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists")
    def test_detect_from_os_release_fedora(self, mock_exists, mock_read_text):
        """Test detection from /etc/os-release for Fedora."""
        mock_exists.return_value = True
        mock_read_text.return_value = '''NAME="Fedora Linux"
VERSION="38 (Workstation Edition)"
ID=fedora
VERSION_ID="38"
PRETTY_NAME="Fedora Linux 38 (Workstation Edition)"
'''

        with patch("nvidia_inst.distro.detector._get_kernel_version", return_value="6.2.9-300.fc38.x86_64"):
            distro = detect_distro()

        assert distro.id == "fedora"
        assert distro.version_id == "38"


class TestDistroDetectionHelpers:
    """Test distro detection helper functions."""

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_ubuntu_true(self, mock_detect):
        """Test is_ubuntu returns True for Ubuntu."""
        mock_detect.return_value = DistroInfo(
            id="ubuntu",
            version_id="22.04",
            name="Ubuntu",
            pretty_name="Ubuntu 22.04",
            kernel="5.15.0",
        )
        assert is_ubuntu() is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_ubuntu_false(self, mock_detect):
        """Test is_ubuntu returns False for Fedora."""
        mock_detect.return_value = DistroInfo(
            id="fedora",
            version_id="38",
            name="Fedora",
            pretty_name="Fedora 38",
            kernel="6.2.9",
        )
        assert is_ubuntu() is False

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_fedora_true(self, mock_detect):
        """Test is_fedora returns True for Fedora."""
        mock_detect.return_value = DistroInfo(
            id="fedora",
            version_id="38",
            name="Fedora",
            pretty_name="Fedora 38",
            kernel="6.2.9",
        )
        assert is_fedora() is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_fedora_rhel(self, mock_detect):
        """Test is_fedora returns True for RHEL variants."""
        mock_detect.return_value = DistroInfo(
            id="rhel",
            version_id="9.1",
            name="Red Hat Enterprise Linux",
            pretty_name="RHEL 9.1",
            kernel="5.14.0",
        )
        assert is_fedora() is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_arch_true(self, mock_detect):
        """Test is_arch returns True for Arch."""
        mock_detect.return_value = DistroInfo(
            id="arch",
            version_id="rolling",
            name="Arch Linux",
            pretty_name="Arch Linux",
            kernel="6.2.9",
        )
        assert is_arch() is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_arch_manjaro(self, mock_detect):
        """Test is_arch returns True for Manjaro."""
        mock_detect.return_value = DistroInfo(
            id="manjaro",
            version_id="23.0",
            name="Manjaro Linux",
            pretty_name="Manjaro Linux 23.0",
            kernel="6.1.0",
        )
        assert is_arch() is True

    @patch("nvidia_inst.distro.detector.detect_distro")
    def test_is_debian_true(self, mock_detect):
        """Test is_debian returns True for Debian."""
        mock_detect.return_value = DistroInfo(
            id="debian",
            version_id="12",
            name="Debian GNU/Linux",
            pretty_name="Debian GNU/Linux 12 (bookworm)",
            kernel="6.1.0",
        )
        assert is_debian() is True


class TestPackageManager:
    """Test package manager detection."""

    @patch("nvidia_inst.distro.factory.is_debian")
    @patch("nvidia_inst.distro.factory.is_ubuntu")
    def test_get_package_manager_apt(self, mock_is_ubuntu, mock_is_debian):
        """Test getting APT package manager."""
        mock_is_ubuntu.return_value = True
        mock_is_debian.return_value = False
        pm = get_package_manager()
        assert pm.__class__.__name__ == "AptManager"

    @patch("nvidia_inst.distro.factory.is_ubuntu")
    @patch("nvidia_inst.distro.factory.is_arch")
    @patch("nvidia_inst.distro.factory.is_debian")
    @patch("nvidia_inst.distro.factory.is_fedora")
    def test_get_package_manager_dnf(self, mock_is_fedora, mock_is_debian, mock_is_arch, mock_is_ubuntu):
        """Test getting DNF package manager."""
        mock_is_ubuntu.return_value = False
        mock_is_fedora.return_value = True
        mock_is_debian.return_value = False
        mock_is_arch.return_value = False
        pm = get_package_manager()
        assert pm.__class__.__name__ == "DnfManager"

    @patch("nvidia_inst.distro.factory.is_ubuntu")
    @patch("nvidia_inst.distro.factory.is_opensuse")
    @patch("nvidia_inst.distro.factory.is_fedora")
    @patch("nvidia_inst.distro.factory.is_debian")
    @patch("nvidia_inst.distro.factory.is_arch")
    def test_get_package_manager_pacman(self, mock_is_arch, mock_is_debian, mock_is_fedora, mock_is_opensuse, mock_is_ubuntu):
        """Test getting Pacman package manager."""
        mock_is_ubuntu.return_value = False
        mock_is_arch.return_value = True
        mock_is_debian.return_value = False
        mock_is_fedora.return_value = False
        mock_is_opensuse.return_value = False
        pm = get_package_manager()
        assert pm.__class__.__name__ == "PacmanManager"
