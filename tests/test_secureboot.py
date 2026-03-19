"""Tests for Secure Boot support functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvidia_inst.installer.secureboot import (
    SecureBootError,
    SecureBootResult,
    SecureBootState,
    check_mokutil_available,
    enroll_mok_key,
    find_nvidia_modules,
    generate_mok_key,
    get_mok_key_paths,
    get_secure_boot_state,
    get_sign_file_path,
    is_mok_enrolled,
    sign_module,
    sign_nvidia_modules,
)


class TestGetSecureBootState:
    """Tests for Secure Boot state detection."""

    def test_secure_boot_enabled(self):
        """Test detection when Secure Boot is enabled."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="SecureBoot enabled",
                returncode=0,
            )
            state = get_secure_boot_state()
            assert state == SecureBootState.ENABLED

    def test_secure_boot_disabled(self):
        """Test detection when Secure Boot is disabled."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="SecureBoot disabled",
                returncode=0,
            )
            state = get_secure_boot_state()
            assert state == SecureBootState.DISABLED

    def test_secure_boot_setup_mode(self):
        """Test detection when in Setup Mode."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="SecureBoot disabled\nPlatform is in Setup Mode",
                returncode=0,
            )
            state = get_secure_boot_state()
            assert state == SecureBootState.SETUP_MODE

    def test_mokutil_not_found(self):
        """Test when mokutil is not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            state = get_secure_boot_state()
            assert state == SecureBootState.UNKNOWN

    def test_mokutil_timeout(self):
        """Test when mokutil times out."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("mokutil", 10)
            state = get_secure_boot_state()
            assert state == SecureBootState.UNKNOWN


class TestCheckMokutilAvailable:
    """Tests for mokutil availability check."""

    def test_mokutil_available(self):
        """Test when mokutil is installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert check_mokutil_available() is True

    def test_mokutil_not_available(self):
        """Test when mokutil is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert check_mokutil_available() is False


class TestGetMokKeyPaths:
    """Tests for MOK key path generation."""

    def test_ubuntu_paths(self):
        """Test Ubuntu MOK key paths."""
        paths = get_mok_key_paths("ubuntu")
        assert paths.private_key == Path("/var/lib/shim-signed/mok/MOK.priv")
        assert paths.public_cert == Path("/var/lib/shim-signed/mok/MOK.der")

    def test_fedora_paths(self):
        """Test Fedora MOK key paths."""
        paths = get_mok_key_paths("fedora")
        assert paths.private_key == Path("/etc/pki/akmods/private/private_key.priv")
        assert paths.public_cert == Path("/etc/pki/akmods/certs/public_key.der")

    def test_arch_paths(self):
        """Test Arch Linux MOK key paths."""
        paths = get_mok_key_paths("arch")
        assert paths.private_key == Path("/etc/secureboot/keys/MOK.priv")
        assert paths.public_cert == Path("/etc/secureboot/keys/MOK.der")

    def test_debian_paths(self):
        """Test Debian MOK key paths."""
        paths = get_mok_key_paths("debian")
        assert paths.private_key == Path("/var/lib/dkms/mok.key")
        assert paths.public_cert == Path("/var/lib/dkms/mok.pub")

    def test_opensuse_paths(self):
        """Test openSUSE MOK key paths."""
        paths = get_mok_key_paths("opensuse")
        assert paths.private_key == Path("/etc/pki/akmods/private/private_key.priv")
        assert paths.public_cert == Path("/etc/pki/akmods/certs/public_key.der")

    def test_unknown_distro_paths(self):
        """Test fallback paths for unknown distributions."""
        paths = get_mok_key_paths("unknown")
        assert paths.private_key == Path("/etc/secureboot/mok/MOK.priv")
        assert paths.public_cert == Path("/etc/secureboot/mok/MOK.der")


class TestIsMokEnrolled:
    """Tests for MOK enrollment checking."""

    def test_enrolled(self, tmp_path):
        """Test when key is already enrolled."""
        cert_path = tmp_path / "MOK.der"
        cert_path.write_bytes(b"cert data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="KEY is already enrolled",
            )
            assert is_mok_enrolled(cert_path) is True

    def test_not_enrolled(self, tmp_path):
        """Test when key is not enrolled."""
        cert_path = tmp_path / "MOK.der"
        cert_path.write_bytes(b"cert data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="KEY is not enrolled",
            )
            assert is_mok_enrolled(cert_path) is False

    def test_cert_not_exists(self, tmp_path):
        """Test when certificate file does not exist."""
        cert_path = tmp_path / "nonexistent.der"
        assert is_mok_enrolled(cert_path) is False


class TestGenerateMokKey:
    """Tests for MOK key generation."""

    def test_generate_key_success(self, tmp_path):
        """Test successful key generation."""
        key_paths = generate_mok_key(tmp_path)

        assert key_paths.private_key.exists()
        assert key_paths.public_cert.exists()
        assert key_paths.private_key.stat().st_mode & 0o777 == 0o600
        assert key_paths.public_cert.stat().st_mode & 0o777 == 0o644

    def test_generate_key_with_custom_name(self, tmp_path):
        """Test key generation with custom name."""
        key_paths = generate_mok_key(tmp_path, key_name="CustomKey")

        assert key_paths.private_key.name == "CustomKey.priv"
        assert key_paths.public_cert.name == "CustomKey.der"

    def test_generate_key_openssl_not_found(self, tmp_path):
        """Test error when openssl is not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with pytest.raises(SecureBootError, match="openssl not found"):
                generate_mok_key(tmp_path)

    def test_generate_key_openssl_failure(self, tmp_path):
        """Test error when openssl fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Key generation failed"
            )
            with pytest.raises(SecureBootError):
                generate_mok_key(tmp_path)


class TestEnrollMokKey:
    """Tests for MOK enrollment."""

    def test_enroll_disabled_state(self, tmp_path):
        """Test enrollment when Secure Boot is disabled."""
        cert_path = tmp_path / "MOK.der"
        cert_path.write_bytes(b"cert data")

        with patch.object(
            __import__(
                "nvidia_inst.installer.secureboot", fromlist=["get_secure_boot_state"]
            ),
            "get_secure_boot_state",
            return_value=SecureBootState.DISABLED,
        ):

            result = enroll_mok_key(cert_path)
            assert result.success is True
            assert "no MOK enrollment needed" in result.message

    def test_enroll_setup_mode(self, tmp_path):
        """Test enrollment in Setup Mode (no reboot required)."""
        cert_path = tmp_path / "MOK.der"
        cert_path.write_bytes(b"cert data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch(
                "nvidia_inst.installer.secureboot.get_secure_boot_state",
                return_value=SecureBootState.SETUP_MODE,
            ):
                result = enroll_mok_key(cert_path)
                assert result.success is True
                assert result.requires_reboot is False
                assert "Setup Mode" in result.message

    def test_enroll_enabled(self, tmp_path):
        """Test enrollment when Secure Boot is enabled (reboot required)."""
        cert_path = tmp_path / "MOK.der"
        cert_path.write_bytes(b"cert data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch(
                "nvidia_inst.installer.secureboot.get_secure_boot_state",
                return_value=SecureBootState.ENABLED,
            ):
                result = enroll_mok_key(cert_path)
                assert result.success is True
                assert result.requires_reboot is True
                assert result.reboot_instructions is not None
                assert "Enroll MOK" in result.reboot_instructions

    def test_enroll_cert_not_found(self, tmp_path):
        """Test enrollment when certificate does not exist."""
        cert_path = tmp_path / "nonexistent.der"

        result = enroll_mok_key(cert_path)
        assert result.success is False
        assert "not found" in result.message

    def test_enroll_already_enrolled(self, tmp_path):
        """Test enrollment when key is already enrolled."""
        cert_path = tmp_path / "MOK.der"
        cert_path.write_bytes(b"cert data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="already enrolled",
                stderr="",
            )
            with patch(
                "nvidia_inst.installer.secureboot.get_secure_boot_state",
                return_value=SecureBootState.ENABLED,
            ):
                result = enroll_mok_key(cert_path)
                assert result.success is True


class TestGetSignFilePath:
    """Tests for sign-file path detection."""

    def test_find_in_linux_headers(self):
        """Test finding sign-file in linux-headers directory."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            path = get_sign_file_path("6.8.0")
            assert path == Path("/usr/src/linux-headers-6.8.0/scripts/sign-file")

    def test_sign_file_not_found(self):
        """Test when sign-file is not found."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            path = get_sign_file_path("6.8.0")
            assert path is None


class TestFindNvidiaModules:
    """Tests for NVIDIA module discovery."""

    def test_find_modules_in_dkms(self):
        """Test finding modules in updates/dkms directory."""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = [
                "/lib/modules/6.8.0/updates/dkms/nvidia.ko",
                "/lib/modules/6.8.0/updates/dkms/nvidia-modeset.ko",
            ]
            modules = find_nvidia_modules("6.8.0")
            assert len(modules) == 2
            assert all(isinstance(m, Path) for m in modules)

    def test_find_modules_in_extra(self):
        """Test finding modules in extra directory."""
        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = [
                [],
                [
                    "/lib/modules/6.8.0/extra/nvidia.ko",
                ],
                [],
            ]
            modules = find_nvidia_modules("6.8.0")
            assert len(modules) == 1

    def test_no_modules_found(self):
        """Test when no modules are found."""
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []
            modules = find_nvidia_modules("6.8.0")
            assert len(modules) == 0


class TestSignModule:
    """Tests for individual module signing."""

    def test_sign_module_success(self, tmp_path):
        """Test successful module signing."""
        module = tmp_path / "nvidia.ko"
        module.write_bytes(b"module data")
        priv_key = tmp_path / "MOK.priv"
        priv_key.write_bytes(b"private key")
        pub_cert = tmp_path / "MOK.der"
        pub_cert.write_bytes(b"public cert")

        sign_file = tmp_path / "sign-file"
        sign_file.write_text("#!/bin/bash")
        sign_file.chmod(0o755)

        with patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=sign_file,
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = sign_module(module, priv_key, pub_cert)
            assert result is True

    def test_sign_module_missing_sign_file(self, tmp_path):
        """Test signing when sign-file is not available."""
        module = tmp_path / "nvidia.ko"
        module.write_bytes(b"module data")
        priv_key = tmp_path / "MOK.priv"
        priv_key.write_bytes(b"private key")
        pub_cert = tmp_path / "MOK.der"
        pub_cert.write_bytes(b"public cert")

        with patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=None,
        ):
            result = sign_module(module, priv_key, pub_cert)
            assert result is False


class TestSignNvidiaModules:
    """Tests for signing all NVIDIA modules."""

    def test_sign_multiple_modules(self, tmp_path):
        """Test signing multiple modules."""
        priv_key = tmp_path / "MOK.priv"
        priv_key.write_bytes(b"private key")
        pub_cert = tmp_path / "MOK.der"
        pub_cert.write_bytes(b"public cert")

        modules = [
            tmp_path / "nvidia.ko",
            tmp_path / "nvidia-modeset.ko",
        ]
        for m in modules:
            m.write_bytes(b"module data")

        sign_file = tmp_path / "sign-file"
        sign_file.write_text("#!/bin/bash")
        sign_file.chmod(0o755)

        with patch(
            "nvidia_inst.installer.secureboot.find_nvidia_modules",
            return_value=modules,
        ), patch(
            "nvidia_inst.installer.secureboot.get_sign_file_path",
            return_value=sign_file,
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            signed, failed = sign_nvidia_modules(priv_key, pub_cert)
            assert signed == 2
            assert failed == 0

    def test_sign_no_modules(self, tmp_path):
        """Test signing when no modules exist."""
        priv_key = tmp_path / "MOK.priv"
        priv_key.write_bytes(b"private key")
        pub_cert = tmp_path / "MOK.der"
        pub_cert.write_bytes(b"public cert")

        with patch(
            "nvidia_inst.installer.secureboot.find_nvidia_modules",
            return_value=[],
        ):
            signed, failed = sign_nvidia_modules(priv_key, pub_cert)
            assert signed == 0
            assert failed == 0


class TestSecureBootResult:
    """Tests for SecureBootResult dataclass."""

    def test_result_creation(self):
        """Test creating a SecureBootResult."""
        result = SecureBootResult(
            success=True,
            message="Test message",
            requires_reboot=True,
            reboot_instructions="Reboot now",
        )
        assert result.success is True
        assert result.message == "Test message"
        assert result.requires_reboot is True
        assert result.reboot_instructions == "Reboot now"

    def test_result_defaults(self):
        """Test default values for SecureBootResult."""
        result = SecureBootResult(success=False, message="Error")
        assert result.requires_reboot is False
        assert result.reboot_instructions is None
