"""Tests for Secure Boot support module (installer/secureboot.py).

Covers:
- Secure Boot state detection
- MOK key generation and enrollment
- Module signing
- DKMS and pacman hooks
- Auto-signing setup
- disable_secure_boot_validation()
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.installer.secureboot import (
    MokKeyPaths,
    MokutilNotFoundError,
    SecureBootError,
    SecureBootResult,
    SecureBootState,
    check_mokutil_available,
    disable_secure_boot_validation,
    enroll_mok_key,
    find_nvidia_modules,
    generate_mok_key,
    get_mok_key_paths,
    get_secure_boot_state,
    get_sign_file_path,
    install_signing_script,
    is_mok_enrolled,
    setup_auto_signing,
    setup_dkms_hook,
    setup_pacman_hook,
    sign_module,
    sign_nvidia_modules,
)


class TestSecureBootState:
    """Tests for SecureBootState enum."""

    def test_enabled_state(self):
        """Test ENABLED state."""
        assert SecureBootState.ENABLED.value == "enabled"

    def test_disabled_state(self):
        """Test DISABLED state."""
        assert SecureBootState.DISABLED.value == "disabled"

    def test_setup_mode_state(self):
        """Test SETUP_MODE state."""
        assert SecureBootState.SETUP_MODE.value == "setup_mode"

    def test_unknown_state(self):
        """Test UNKNOWN state."""
        assert SecureBootState.UNKNOWN.value == "unknown"


class TestSecureBootResult:
    """Tests for SecureBootResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = SecureBootResult(
            success=True,
            message="Success",
            requires_reboot=False,
        )
        assert result.success is True
        assert result.requires_reboot is False

    def test_reboot_required_result(self):
        """Test result requiring reboot."""
        result = SecureBootResult(
            success=True,
            message="Reboot needed",
            requires_reboot=True,
            reboot_instructions="Reboot now",
        )
        assert result.requires_reboot is True
        assert result.reboot_instructions == "Reboot now"

    def test_default_values(self):
        """Test default values."""
        result = SecureBootResult(success=False, message="Error")
        assert result.requires_reboot is False
        assert result.reboot_instructions is None


class TestMokKeyPaths:
    """Tests for MokKeyPaths dataclass."""

    def test_paths_creation(self, tmp_path):
        """Test MokKeyPaths creation."""
        priv = tmp_path / "MOK.priv"
        pub = tmp_path / "MOK.der"
        paths = MokKeyPaths(private_key=priv, public_cert=pub)
        assert paths.private_key == priv
        assert paths.public_cert == pub


class TestExceptions:
    """Tests for exception classes."""

    def test_secure_boot_error(self):
        """Test SecureBootError can be raised."""
        with pytest.raises(SecureBootError):
            raise SecureBootError("Test error")

    def test_mokutil_not_found_error(self):
        """Test MokutilNotFoundError can be raised."""
        with pytest.raises(MokutilNotFoundError):
            raise MokutilNotFoundError("mokutil not found")


class TestGetSecureBootState:
    """Tests for get_secure_boot_state()."""

    @patch("subprocess.run")
    def test_enabled(self, mock_run):
        """Test Secure Boot enabled detection."""
        mock_run.return_value = MagicMock(
            stdout="SecureBoot enabled\n",
            returncode=0,
        )
        assert get_secure_boot_state() == SecureBootState.ENABLED

    @patch("subprocess.run")
    def test_disabled(self, mock_run):
        """Test Secure Boot disabled detection."""
        mock_run.return_value = MagicMock(
            stdout="SecureBoot disabled\n",
            returncode=0,
        )
        assert get_secure_boot_state() == SecureBootState.DISABLED

    @patch("subprocess.run")
    def test_setup_mode(self, mock_run):
        """Test Setup Mode detection."""
        mock_run.return_value = MagicMock(
            stdout="SecureBoot disabled\nPlatform is in Setup Mode\n",
            returncode=0,
        )
        assert get_secure_boot_state() == SecureBootState.SETUP_MODE

    @patch("subprocess.run")
    def test_mokutil_not_found(self, mock_run):
        """Test mokutil not found."""
        mock_run.side_effect = FileNotFoundError()
        assert get_secure_boot_state() == SecureBootState.UNKNOWN

    @patch("subprocess.run")
    def test_timeout(self, mock_run):
        """Test mokutil timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("mokutil", 10)
        assert get_secure_boot_state() == SecureBootState.UNKNOWN

    @patch("subprocess.run")
    def test_generic_error(self, mock_run):
        """Test generic error handling."""
        mock_run.side_effect = Exception("Unknown error")
        assert get_secure_boot_state() == SecureBootState.UNKNOWN


class TestCheckMokutilAvailable:
    """Tests for check_mokutil_available()."""

    @patch("subprocess.run")
    def test_available(self, mock_run):
        """Test mokutil is available."""
        mock_run.return_value = MagicMock(returncode=0)
        assert check_mokutil_available() is True

    @patch("subprocess.run")
    def test_not_available(self, mock_run):
        """Test mokutil is not available."""
        mock_run.side_effect = FileNotFoundError()
        assert check_mokutil_available() is False

    @patch("subprocess.run")
    def test_generic_error(self, mock_run):
        """Test generic error handling."""
        mock_run.side_effect = Exception("Error")
        assert check_mokutil_available() is False


class TestGetMokKeyPaths:
    """Tests for get_mok_key_paths()."""

    def test_ubuntu_paths(self):
        """Test Ubuntu MOK key paths."""
        paths = get_mok_key_paths("ubuntu")
        assert paths.private_key == Path("/var/lib/shim-signed/mok/MOK.priv")
        assert paths.public_cert == Path("/var/lib/shim-signed/mok/MOK.der")

    def test_linuxmint_paths(self):
        """Test Linux Mint MOK key paths."""
        paths = get_mok_key_paths("linuxmint")
        assert paths.private_key == Path("/var/lib/shim-signed/mok/MOK.priv")

    def test_fedora_paths(self):
        """Test Fedora MOK key paths."""
        paths = get_mok_key_paths("fedora")
        assert paths.private_key == Path("/etc/pki/akmods/private/private_key.priv")
        assert paths.public_cert == Path("/etc/pki/akmods/certs/public_key.der")

    def test_rhel_paths(self):
        """Test RHEL MOK key paths."""
        paths = get_mok_key_paths("rhel")
        assert paths.private_key == Path("/etc/pki/akmods/private/private_key.priv")

    def test_arch_paths(self):
        """Test Arch MOK key paths."""
        paths = get_mok_key_paths("arch")
        assert paths.private_key == Path("/etc/secureboot/keys/MOK.priv")
        assert paths.public_cert == Path("/etc/secureboot/keys/MOK.der")

    def test_manjaro_paths(self):
        """Test Manjaro MOK key paths."""
        paths = get_mok_key_paths("manjaro")
        assert paths.private_key == Path("/etc/secureboot/keys/MOK.priv")

    def test_debian_paths(self):
        """Test Debian MOK key paths."""
        paths = get_mok_key_paths("debian")
        assert paths.private_key == Path("/var/lib/dkms/mok.key")
        assert paths.public_cert == Path("/var/lib/dkms/mok.pub")

    def test_unknown_paths(self):
        """Test unknown distro fallback paths."""
        paths = get_mok_key_paths("unknown")
        assert paths.private_key == Path("/etc/secureboot/mok/MOK.priv")
        assert paths.public_cert == Path("/etc/secureboot/mok/MOK.der")


class TestIsMokEnrolled:
    """Tests for is_mok_enrolled()."""

    @patch("subprocess.run")
    def test_enrolled(self, mock_run, tmp_path):
        """Test MOK is enrolled."""
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="KEY is already enrolled",
        )
        assert is_mok_enrolled(cert) is True

    @patch("subprocess.run")
    def test_not_enrolled(self, mock_run, tmp_path):
        """Test MOK is not enrolled."""
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="KEY is not enrolled",
        )
        assert is_mok_enrolled(cert) is False

    def test_cert_not_exists(self, tmp_path):
        """Test when cert doesn't exist."""
        cert = tmp_path / "nonexistent.der"
        assert is_mok_enrolled(cert) is False

    @patch("subprocess.run")
    def test_error_handling(self, mock_run, tmp_path):
        """Test error handling."""
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")
        mock_run.side_effect = Exception("Error")
        assert is_mok_enrolled(cert) is False


class TestGenerateMokKey:
    """Tests for generate_mok_key()."""

    def test_generate_success(self, tmp_path):
        """Test successful key generation."""
        paths = generate_mok_key(tmp_path)
        assert paths.private_key.exists()
        assert paths.public_cert.exists()
        assert paths.private_key.stat().st_mode & 0o777 == 0o600
        assert paths.public_cert.stat().st_mode & 0o777 == 0o644

    def test_generate_custom_name(self, tmp_path):
        """Test key generation with custom name."""
        paths = generate_mok_key(tmp_path, key_name="CustomKey")
        assert paths.private_key.name == "CustomKey.priv"
        assert paths.public_cert.name == "CustomKey.der"

    def test_generate_custom_params(self, tmp_path):
        """Test key generation with custom parameters."""
        paths = generate_mok_key(tmp_path, key_bits=4096, validity_days=3650)
        assert paths.private_key.exists()

    @patch("subprocess.run")
    def test_openssl_not_found(self, mock_run, tmp_path):
        """Test openssl not found error."""
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(SecureBootError, match="openssl not found"):
            generate_mok_key(tmp_path)

    @patch("subprocess.run")
    def test_openssl_failure(self, mock_run, tmp_path):
        """Test openssl failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Key generation failed",
        )
        with pytest.raises(SecureBootError):
            generate_mok_key(tmp_path)


class TestEnrollMokKey:
    """Tests for enroll_mok_key()."""

    def test_cert_not_found(self, tmp_path):
        """Test enrollment when cert doesn't exist."""
        cert = tmp_path / "nonexistent.der"
        result = enroll_mok_key(cert)
        assert result.success is False
        assert "not found" in result.message

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    def test_secure_boot_disabled(self, mock_state, tmp_path):
        """Test enrollment when Secure Boot is disabled."""
        mock_state.return_value = SecureBootState.DISABLED
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is True
        assert "no MOK enrollment needed" in result.message

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    def test_secure_boot_unknown(self, mock_state, tmp_path):
        """Test enrollment when Secure Boot state is unknown."""
        mock_state.return_value = SecureBootState.UNKNOWN
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is False
        assert "Cannot determine" in result.message

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    @patch("subprocess.run")
    def test_enrollment_setup_mode(self, mock_run, mock_state, tmp_path):
        """Test enrollment in Setup Mode."""
        mock_state.return_value = SecureBootState.SETUP_MODE
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is True
        assert result.requires_reboot is False
        assert "Setup Mode" in result.message

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    @patch("subprocess.run")
    def test_enrollment_enabled(self, mock_run, mock_state, tmp_path):
        """Test enrollment when Secure Boot is enabled."""
        mock_state.return_value = SecureBootState.ENABLED
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is True
        assert result.requires_reboot is True
        assert result.reboot_instructions is not None
        assert "Enroll MOK" in result.reboot_instructions

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    @patch("subprocess.run")
    def test_enrollment_already_enrolled(self, mock_run, mock_state, tmp_path):
        """Test enrollment when already enrolled."""
        mock_state.return_value = SecureBootState.ENABLED
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="already enrolled",
            stderr="",
        )
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is True

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    @patch("subprocess.run")
    def test_enrollment_failure(self, mock_run, mock_state, tmp_path):
        """Test enrollment failure."""
        mock_state.return_value = SecureBootState.ENABLED
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Enrollment failed",
        )
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is False
        assert "failed" in result.message.lower()

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    @patch("subprocess.run")
    def test_enrollment_timeout(self, mock_run, mock_state, tmp_path):
        """Test enrollment timeout."""
        import subprocess

        mock_state.return_value = SecureBootState.ENABLED
        mock_run.side_effect = subprocess.TimeoutExpired("mokutil", 60)
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is False
        assert "timed out" in result.message.lower()

    @patch("nvidia_inst.installer.secureboot.get_secure_boot_state")
    @patch("subprocess.run")
    def test_enrollment_mokutil_not_found(self, mock_run, mock_state, tmp_path):
        """Test enrollment when mokutil not found."""
        mock_state.return_value = SecureBootState.ENABLED
        mock_run.side_effect = FileNotFoundError()
        cert = tmp_path / "MOK.der"
        cert.write_bytes(b"cert")

        result = enroll_mok_key(cert)
        assert result.success is False
        assert "mokutil not found" in result.message.lower()


class TestGetSignFilePath:
    """Tests for get_sign_file_path()."""

    @patch("pathlib.Path.exists")
    def test_find_in_linux_headers(self, mock_exists):
        """Test finding sign-file in linux-headers."""
        mock_exists.return_value = True
        path = get_sign_file_path("6.8.0")
        assert path == Path("/usr/src/linux-headers-6.8.0/scripts/sign-file")

    @patch("pathlib.Path.exists")
    def test_find_in_modules_build(self, mock_exists):
        """Test finding sign-file in modules build dir."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return call_count[0] == 2

        mock_exists.side_effect = side_effect
        path = get_sign_file_path("6.8.0")
        assert path == Path("/lib/modules/6.8.0/build/scripts/sign-file")

    @patch("pathlib.Path.exists")
    def test_not_found(self, mock_exists):
        """Test sign-file not found."""
        mock_exists.return_value = False
        path = get_sign_file_path("6.8.0")
        assert path is None

    @patch("pathlib.Path.exists")
    @patch("os.uname")
    def test_default_kernel_version(self, mock_uname, mock_exists):
        """Test with default kernel version."""
        mock_uname.return_value = MagicMock(release="6.8.0-generic")
        mock_exists.return_value = True
        path = get_sign_file_path()
        assert path is not None


class TestFindNvidiaModules:
    """Tests for find_nvidia_modules()."""

    @patch("glob.glob")
    def test_find_modules_dkms(self, mock_glob):
        """Test finding modules in DKMS dir."""
        mock_glob.return_value = [
            "/lib/modules/6.8.0/updates/dkms/nvidia.ko",
            "/lib/modules/6.8.0/updates/dkms/nvidia-modeset.ko",
        ]
        modules = find_nvidia_modules("6.8.0")
        assert len(modules) == 2
        assert all(isinstance(m, Path) for m in modules)

    @patch("glob.glob")
    def test_find_modules_extra(self, mock_glob):
        """Test finding modules in extra dir."""
        mock_glob.side_effect = [
            [],
            ["/lib/modules/6.8.0/extra/nvidia.ko"],
            [],
        ]
        modules = find_nvidia_modules("6.8.0")
        assert len(modules) == 1

    @patch("glob.glob")
    def test_no_modules(self, mock_glob):
        """Test no modules found."""
        mock_glob.return_value = []
        modules = find_nvidia_modules("6.8.0")
        assert len(modules) == 0

    @patch("glob.glob")
    @patch("os.uname")
    def test_default_kernel_version(self, mock_uname, mock_glob):
        """Test with default kernel version."""
        mock_uname.return_value = MagicMock(release="6.8.0-generic")
        mock_glob.return_value = []
        modules = find_nvidia_modules()
        assert isinstance(modules, list)


class TestSignModule:
    """Tests for sign_module()."""

    def test_sign_success(self, tmp_path):
        """Test successful module signing."""
        module = tmp_path / "nvidia.ko"
        module.write_bytes(b"module")
        priv = tmp_path / "MOK.priv"
        priv.write_bytes(b"priv")
        pub = tmp_path / "MOK.der"
        pub.write_bytes(b"pub")
        sign_file = tmp_path / "sign-file"
        sign_file.write_text("#!/bin/bash")
        sign_file.chmod(0o755)

        with (
            patch(
                "nvidia_inst.installer.secureboot.get_sign_file_path",
                return_value=sign_file,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = sign_module(module, priv, pub)
            assert result is True

    def test_sign_file_not_found(self, tmp_path):
        """Test when sign-file not found."""
        module = tmp_path / "nvidia.ko"
        module.write_bytes(b"module")
        priv = tmp_path / "MOK.priv"
        priv.write_bytes(b"priv")
        pub = tmp_path / "MOK.der"
        pub.write_bytes(b"pub")

        with patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=None,
        ):
            result = sign_module(module, priv, pub)
            assert result is False

    def test_module_not_exists(self, tmp_path):
        """Test when module doesn't exist."""
        priv = tmp_path / "MOK.priv"
        priv.write_bytes(b"priv")
        pub = tmp_path / "MOK.der"
        pub.write_bytes(b"pub")

        with patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=tmp_path / "sign-file",
        ):
            result = sign_module(tmp_path / "nonexistent.ko", priv, pub)
            assert result is False

    def test_private_key_not_exists(self, tmp_path):
        """Test when private key doesn't exist."""
        module = tmp_path / "nvidia.ko"
        module.write_bytes(b"module")
        pub = tmp_path / "MOK.der"
        pub.write_bytes(b"pub")

        with patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=tmp_path / "sign-file",
        ):
            result = sign_module(module, tmp_path / "nonexistent.priv", pub)
            assert result is False

    def test_public_cert_not_exists(self, tmp_path):
        """Test when public cert doesn't exist."""
        module = tmp_path / "nvidia.ko"
        module.write_bytes(b"module")
        priv = tmp_path / "MOK.priv"
        priv.write_bytes(b"priv")

        with patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=tmp_path / "sign-file",
        ):
            result = sign_module(module, priv, tmp_path / "nonexistent.der")
            assert result is False


class TestSignNvidiaModules:
    """Tests for sign_nvidia_modules()."""

    def test_sign_multiple(self, tmp_path):
        """Test signing multiple modules."""
        priv = tmp_path / "MOK.priv"
        priv.write_bytes(b"priv")
        pub = tmp_path / "MOK.der"
        pub.write_bytes(b"pub")

        modules = [tmp_path / "nvidia.ko", tmp_path / "nvidia-modeset.ko"]
        for m in modules:
            m.write_bytes(b"module")

        sign_file = tmp_path / "sign-file"
        sign_file.write_text("#!/bin/bash")
        sign_file.chmod(0o755)

        with (
            patch(
                "nvidia_inst.installer.secureboot.find_nvidia_modules",
                return_value=modules,
            ),
            patch(
                "nvidia_inst.installer.secureboot.get_sign_file_path",
                return_value=sign_file,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            signed, failed = sign_nvidia_modules(priv, pub)
            assert signed == 2
            assert failed == 0

    def test_sign_no_modules(self, tmp_path):
        """Test signing when no modules exist."""
        priv = tmp_path / "MOK.priv"
        priv.write_bytes(b"priv")
        pub = tmp_path / "MOK.der"
        pub.write_bytes(b"pub")

        with patch(
            "nvidia_inst.installer.secureboot.find_nvidia_modules",
            return_value=[],
        ):
            signed, failed = sign_nvidia_modules(priv, pub)
            assert signed == 0
            assert failed == 0


class TestSetupDkmsHook:
    """Tests for setup_dkms_hook()."""

    def test_hook_created(self, tmp_path):
        """Test DKMS hook creation requires root (returns False in test env)."""
        script = tmp_path / "sign-nvidia-modules"
        script.write_text("#!/bin/bash")
        script.chmod(0o755)
        result = setup_dkms_hook(script, "ubuntu")
        assert result is False  # Requires root

    def test_hook_not_needed(self, tmp_path):
        """Test DKMS hook not needed for non-Debian distros."""
        script = tmp_path / "sign-nvidia-modules"
        result = setup_dkms_hook(script, "fedora")
        assert result is True  # No hook needed

    def test_hook_permission_error(self, tmp_path):
        """Test DKMS hook permission error."""
        script = tmp_path / "sign-nvidia-modules"
        script.write_text("#!/bin/bash")

        with patch("pathlib.Path.mkdir", side_effect=PermissionError()):
            result = setup_dkms_hook(script, "ubuntu")
            assert result is False


class TestSetupPacmanHook:
    """Tests for setup_pacman_hook()."""

    def test_hook_created(self, tmp_path):
        """Test pacman hook creation requires root (returns False in test env)."""
        script = tmp_path / "sign-nvidia-modules"
        script.write_text("#!/bin/bash")
        script.chmod(0o755)
        result = setup_pacman_hook(script)
        assert result is False  # Requires root

    def test_hook_permission_error(self, tmp_path):
        """Test pacman hook permission error."""
        script = tmp_path / "sign-nvidia-modules"
        script.write_text("#!/bin/bash")

        with patch("pathlib.Path.mkdir", side_effect=PermissionError()):
            result = setup_pacman_hook(script)
            assert result is False


class TestInstallSigningScript:
    """Tests for install_signing_script()."""

    def test_script_installed(self, tmp_path):
        """Test signing script installation."""
        script = tmp_path / "sign-nvidia-modules"
        priv = tmp_path / "MOK.priv"
        pub = tmp_path / "MOK.der"

        result = install_signing_script(script, priv, pub)
        assert result is True
        assert script.exists()

    def test_script_permission_error(self, tmp_path):
        """Test signing script permission error."""
        script = tmp_path / "readonly" / "sign-nvidia-modules"

        with patch("pathlib.Path.mkdir", side_effect=PermissionError()):
            result = install_signing_script(script, tmp_path / "priv", tmp_path / "pub")
            assert result is False


class TestSetupAutoSigning:
    """Tests for setup_auto_signing()."""

    def test_auto_signing_ubuntu(self, tmp_path):
        """Test auto-signing setup on Ubuntu."""
        priv = tmp_path / "MOK.priv"
        pub = tmp_path / "MOK.der"

        result = setup_auto_signing(priv, pub, "ubuntu", script_dir=tmp_path)
        assert result.success is True

    def test_auto_signing_arch(self, tmp_path):
        """Test auto-signing setup on Arch."""
        priv = tmp_path / "MOK.priv"
        pub = tmp_path / "MOK.der"

        result = setup_auto_signing(priv, pub, "arch", script_dir=tmp_path)
        assert result.success is True

    def test_auto_signing_fedora(self, tmp_path):
        """Test auto-signing setup on Fedora (no hooks)."""
        priv = tmp_path / "MOK.priv"
        pub = tmp_path / "MOK.der"

        result = setup_auto_signing(priv, pub, "fedora", script_dir=tmp_path)
        assert result.success is True


class TestDisableSecureBootValidation:
    """Tests for disable_secure_boot_validation()."""

    @patch("subprocess.run")
    def test_disable_success(self, mock_run):
        """Test successful validation disable."""
        mock_run.return_value = MagicMock(returncode=0)
        assert disable_secure_boot_validation() is True

    @patch("subprocess.run")
    def test_disable_failure(self, mock_run):
        """Test validation disable failure."""
        mock_run.return_value = MagicMock(returncode=1)
        assert disable_secure_boot_validation() is False

    @patch("subprocess.run")
    def test_disable_error(self, mock_run):
        """Test validation disable error handling."""
        mock_run.side_effect = Exception("Error")
        assert disable_secure_boot_validation() is False
