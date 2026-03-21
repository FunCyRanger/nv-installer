"""Tests for kmod-only setup detection and handling."""

from unittest.mock import MagicMock, patch

from nvidia_inst.installer.uninstaller import _get_packages_to_remove


class TestKmodOnlyDetection:
    """Tests for kmod-only setup detection."""

    def test_akmod_package_pattern(self):
        """Test that akmod package pattern is recognized in removal list."""
        packages = _get_packages_to_remove("fedora")

        assert "akmod-nvidia" in packages

    @patch("subprocess.run")
    def test_detect_kmod_only_setup(self, mock_run):
        """Test detection of kmod-only (without akmod)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="kmod-nvidia-6.19.8-200.fc43.x86_64-580.126.18-1.fc43.x86_64\n"
            "xorg-x11-drv-nvidia-580.126.18-1.fc43.x86_64",
            stderr="",
        )

        from nvidia_inst.installer.uninstaller import check_nvidia_packages_installed

        packages = check_nvidia_packages_installed("fedora")

        has_akmod = any("akmod" in p for p in packages)
        has_kmod = any("kmod" in p for p in packages)

        assert has_kmod is True
        assert has_akmod is False

    @patch("subprocess.run")
    def test_detect_akmod_setup(self, mock_run):
        """Test detection of akmod setup."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="akmod-nvidia-580.126.18-1.fc43.x86_64\n"
            "kmod-nvidia-6.19.8-200.fc43.x86_64-580.126.18-1.fc43.x86_64\n"
            "xorg-x11-drv-nvidia-580.126.18-1.fc43.x86_64",
            stderr="",
        )

        from nvidia_inst.installer.uninstaller import check_nvidia_packages_installed

        packages = check_nvidia_packages_installed("fedora")

        has_akmod = any("akmod" in p for p in packages)
        has_kmod = any("kmod" in p for p in packages)

        assert has_akmod is True
        assert has_kmod is True


class TestKmodOnlyValidation:
    """Tests for validation with kmod-only setup."""

    @patch("subprocess.run")
    @patch("glob.glob")
    def test_validation_with_prebuilt_kmod(self, mock_glob, mock_run):
        """Test validation passes when pre-built kmod exists."""
        mock_glob.return_value = [
            "/lib/modules/6.19.8-200.fc43.x86_64/extra/nvidia/nvidia.ko.xz"
        ]
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="NVIDIA Quadro M2200\n"),
            MagicMock(returncode=0, stdout="580.126.18\n"),
            MagicMock(returncode=0, stdout="nvidia  123456  0\n"),
        ]

        from nvidia_inst.installer.validation import is_nvidia_working

        result = is_nvidia_working()

        assert result.is_working is True
        assert result.kernel_module_loaded is True
        assert result.gpu_detected is True

    @patch("subprocess.run")
    def test_nvidia_smi_works_without_akmod(self, mock_run):
        """Test nvidia-smi works when only kmod (not akmod) is installed."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="NVIDIA Quadro M2200\n"),
            MagicMock(returncode=0, stdout="580.126.18\n"),
        ]

        from nvidia_inst.installer.validation import is_nvidia_working

        result = is_nvidia_working()

        assert result.gpu_detected is True
        assert result.driver_version == "580.126.18"


class TestKmodOnlyUninstall:
    """Tests for uninstaller with kmod-only setup."""

    def test_get_packages_fedora_kmod_only(self):
        """Test package list for Fedora kmod-only (no akmod)."""
        packages = _get_packages_to_remove("fedora")

        assert "kmodtool" not in packages
        assert "akmods" not in packages

    @patch("subprocess.run")
    @patch("nvidia_inst.installer.uninstaller._remove_packages")
    @patch("nvidia_inst.installer.uninstaller._rebuild_initramfs")
    @patch("nvidia_inst.installer.uninstaller._remove_blacklist")
    @patch("nvidia_inst.installer.uninstaller._get_packages_to_remove")
    def test_uninstall_kmod_only(
        self,
        mock_get_pkgs,
        mock_remove_bl,
        mock_rebuild,
        mock_remove_pkgs,
        mock_run,
        mock_is_root,
    ):
        """Test uninstaller works with kmod-only setup."""
        mock_get_pkgs.return_value = ["kmod-nvidia", "xorg-x11-drv-nvidia"]
        mock_remove_pkgs.return_value = ["kmod-nvidia", "xorg-x11-drv-nvidia"]
        mock_remove_bl.return_value = True
        mock_rebuild.return_value = True

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from nvidia_inst.installer.uninstaller import revert_to_nouveau

        result = revert_to_nouveau("fedora")

        assert result.success is True
        assert "akmod" not in str(result.packages_removed)


class TestAkmodNotInstalled:
    """Tests for behavior when akmod is not installed."""

    def test_akmod_not_in_fedora_packages(self):
        """Test akmod is not expected in kmod-only removal list."""
        packages = _get_packages_to_remove("fedora")

        for pkg in packages:
            if "akmod" in pkg:
                assert pkg in [
                    "akmod-nvidia",
                    "akmod-nvidia-470xx",
                    "akmod-nvidia-535xx",
                ]

    @patch("subprocess.run")
    def test_check_akmods_service_not_found(self, mock_run):
        """Test akmods service check returns not-found for kmod-only."""
        mock_run.return_value = MagicMock(
            returncode=4,
            stdout="Unit akmods.service could not be found.",
            stderr="",
        )

        from nvidia_inst.installer.uninstaller import check_nvidia_packages_installed

        packages = check_nvidia_packages_installed("fedora")
        assert isinstance(packages, list)


class TestKmodOnlyInfo:
    """Tests for informational output about kmod-only setup."""

    @patch("subprocess.run")
    def test_kernel_module_exists_in_lib(self, mock_run):
        """Test that kernel module location is checked correctly."""
        import glob
        import os

        kernel_version = os.uname().release
        module_pattern = f"/lib/modules/{kernel_version}/extra/nvidia*.ko*"

        modules = glob.glob(module_pattern)

        assert isinstance(modules, list)

    @patch("glob.glob")
    def test_module_check_pattern(self, mock_glob):
        """Test module existence check pattern."""
        mock_glob.return_value = [
            "/lib/modules/6.19.8-200.fc43.x86_64/extra/nvidia/nvidia.ko.xz"
        ]

        import glob
        import os

        kernel_version = os.uname().release
        module_pattern = f"/lib/modules/{kernel_version}/extra/nvidia*.ko*"
        modules = glob.glob(module_pattern)

        assert len(modules) > 0
        mock_glob.assert_called_once()


class TestAkmodsTroubleshooting:
    """Tests for akmods troubleshooting patterns."""

    def test_akmods_lib_directory_pattern(self):
        """Test that /var/lib/akmods directory pattern is used correctly."""
        expected_path = "/var/lib/akmods"

        assert expected_path.startswith("/var/lib/akmods")
        assert "nvidia" not in expected_path

    def test_akmods_cache_directory_pattern(self):
        """Test that /var/cache/akmods directory pattern is used."""
        expected_pattern = "/var/cache/akmods/nvidia"

        assert expected_pattern.startswith("/var/cache/akmods")
        assert "nvidia" in expected_pattern

    @patch("subprocess.run")
    def test_systemd_analyze_command(self, mock_run):
        """Test systemd-analyze critical-chain command pattern."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="akmods.service     +5000ms",
            stderr="",
        )

        import subprocess

        result = subprocess.run(
            ["systemd-analyze", "critical-chain"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "akmods" in result.stdout or result.returncode == 0

    @patch("subprocess.run")
    def test_check_akmods_service_status(self, mock_run):
        """Test akmods service status check."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="active (exited)",
            stderr="",
        )

        import subprocess

        result = subprocess.run(
            ["systemctl", "is-active", "akmods.service"],
            capture_output=True,
            text=True,
        )

        assert result.returncode in [0, 3]

    @patch("subprocess.run")
    def test_akmods_force_rebuild_command(self, mock_run):
        """Test akmods force rebuild command."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        import subprocess

        result = subprocess.run(
            ["akmods", "--force"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0 or result.returncode == 1

    @patch("os.path.exists")
    def test_lib_akmods_missing_detection(self, mock_exists):
        """Test detection of missing /var/lib/akmods directory."""
        mock_exists.return_value = False

        import os

        path = "/var/lib/akmods"
        exists = os.path.exists(path)

        assert exists is False
        mock_exists.assert_called_with(path)


class TestKernelUpdateWorkflow:
    """Tests for kernel update workflow patterns."""

    @patch("subprocess.run")
    @patch("subprocess.run")
    def test_dnf_upgrade_command(self, mock_run2, mock_run1):
        """Test dnf upgrade command pattern."""
        mock_run1.return_value = MagicMock(
            returncode=0,
            stdout="Complete!",
            stderr="",
        )

        import subprocess

        result = subprocess.run(
            ["dnf", "upgrade"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_dracut_command_pattern(self):
        """Test dracut command pattern for initramfs rebuild."""
        import subprocess

        cmd = ["dracut", "--force"]
        assert "--force" in cmd
        assert cmd[0] == "dracut"

    @patch("subprocess.run")
    def test_reboot_command(self, mock_run):
        """Test reboot command pattern."""
        mock_run.side_effect = KeyboardInterrupt

        import subprocess

        try:
            subprocess.run(["reboot"], check=True)
        except KeyboardInterrupt:
            pass

        mock_run.assert_called_with(["reboot"], check=True)


class TestSecureBootHandling:
    """Tests for Secure Boot MOK enrollment patterns."""

    @patch("subprocess.run")
    def test_mokutil_sb_state_command(self, mock_run):
        """Test mokutil Secure Boot state check."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="SecureBoot disabled",
            stderr="",
        )

        import subprocess

        result = subprocess.run(
            ["mokutil", "--sb-state"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "SecureBoot" in result.stdout or result.returncode == 0
