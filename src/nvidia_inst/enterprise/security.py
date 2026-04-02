"""Security enhancements for nvidia-inst.

This module provides security features including:
- Package signature verification
- Secure Boot management
- Audit logging
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuditLogEntry:
    """Audit log entry."""

    timestamp: str = ""
    action: str = ""
    user: str = ""
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class PackageVerifier:
    """Verify package signatures."""

    def __init__(
        self,
        keyring_path: str = "/etc/nvidia-inst/trustedkeys.gpg",
    ):
        """Initialize package verifier.

        Args:
            keyring_path: Path to GPG keyring
        """
        self.keyring_path = Path(keyring_path)

    def verify_package_signature(self, package_path: str) -> bool:
        """Verify GPG signature of a package.

        Args:
            package_path: Path to package file

        Returns:
            True if signature is valid
        """
        try:
            # For DEB packages
            if package_path.endswith(".deb"):
                return self._verify_deb_signature(package_path)

            # For RPM packages
            if package_path.endswith(".rpm"):
                return self._verify_rpm_signature(package_path)

            # Generic GPG verification
            return self._verify_gpg_signature(package_path)

        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False

    def _verify_deb_signature(self, package_path: str) -> bool:
        """Verify DEB package signature."""
        try:
            result = subprocess.run(
                ["dpkg-sig", "--verify", package_path],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception:
            return False

    def _verify_rpm_signature(self, package_path: str) -> bool:
        """Verify RPM package signature."""
        try:
            result = subprocess.run(
                ["rpm", "--checksig", package_path],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0 and "OK" in result.stdout

        except Exception:
            return False

    def _verify_gpg_signature(self, file_path: str) -> bool:
        """Verify GPG signature."""
        try:
            sig_file = f"{file_path}.sig"

            if not Path(sig_file).exists():
                logger.warning(f"Signature file not found: {sig_file}")
                return False

            result = subprocess.run(
                [
                    "gpg",
                    "--verify",
                    sig_file,
                    file_path,
                ],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception:
            return False

    def calculate_checksum(
        self,
        file_path: str,
        algorithm: str = "sha256",
    ) -> str:
        """Calculate file checksum.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)

        Returns:
            Hex digest of checksum
        """
        hash_func = getattr(hashlib, algorithm)()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def verify_checksum(
        self,
        file_path: str,
        expected_checksum: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify file checksum.

        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            algorithm: Hash algorithm

        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_checksum(file_path, algorithm)
        return actual_checksum == expected_checksum


class SecureBootManager:
    """Enhanced Secure Boot management."""

    def __init__(self):
        """Initialize Secure Boot manager."""
        self.mok_dir = Path("/var/lib/nvidia-inst/mok")

    def get_secure_boot_status(self) -> dict[str, Any]:
        """Get Secure Boot status.

        Returns:
            Dictionary with Secure Boot information
        """
        status = {
            "enabled": False,
            "setup_mode": False,
            "mok_enrolled": False,
            "platform_key": None,
        }

        try:
            # Check Secure Boot state
            result = subprocess.run(
                ["mokutil", "--sb-state"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                status["enabled"] = "enabled" in result.stdout.lower()
                status["setup_mode"] = "setup" in result.stdout.lower()

            # Check if MOK is enrolled
            result = subprocess.run(
                ["mokutil", "--list-enrolled"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                status["mok_enrolled"] = "nvidia" in result.stdout.lower()

        except Exception as e:
            logger.error(f"Failed to get Secure Boot status: {e}")

        return status

    def generate_mok_key(
        self,
        output_dir: str | None = None,
    ) -> dict[str, str]:
        """Generate MOK key for module signing.

        Args:
            output_dir: Directory to save keys

        Returns:
            Dictionary with key paths
        """
        if output_dir is None:
            output_dir = str(self.mok_dir)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        key_paths = {
            "private_key": str(output_path / "MOK.priv"),
            "public_key": str(output_path / "MOK.der"),
            "certificate": str(output_path / "MOK.crt"),
        }

        try:
            # Generate private key
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-new",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-keyout",
                    key_paths["private_key"],
                    "-outform",
                    "der",
                    "-out",
                    key_paths["public_key"],
                    "-nodes",
                    "-days",
                    "36500",
                    "-subj",
                    "/CN=NVIDIA Module Signing Key/",
                ],
                check=True,
            )

            # Generate certificate
            subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-inform",
                    "der",
                    "-in",
                    key_paths["public_key"],
                    "-out",
                    key_paths["certificate"],
                ],
                check=True,
            )

            logger.info("MOK key generated successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate MOK key: {e}")
            raise

        return key_paths

    def enroll_mok_key(
        self,
        key_path: str,
        password: str,
    ) -> bool:
        """Enroll MOK key with password protection.

        Args:
            key_path: Path to MOK key
            password: Enrollment password

        Returns:
            True if enrollment initiated successfully
        """
        try:
            # Import MOK key
            result = subprocess.run(
                ["mokutil", "--import", key_path],
                input=password,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("MOK key enrollment initiated")
                logger.info("System reboot required to complete enrollment")
                return True
            else:
                logger.error(f"Failed to enroll MOK key: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to enroll MOK key: {e}")
            return False

    def backup_mok_key(
        self,
        backup_dir: str,
    ) -> bool:
        """Backup MOK key for disaster recovery.

        Args:
            backup_dir: Directory to save backup

        Returns:
            True if backup created successfully
        """
        try:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)

            # Copy MOK files
            if self.mok_dir.exists():
                import shutil

                for file in self.mok_dir.iterdir():
                    dest = backup_path / file.name
                    shutil.copy2(file, dest)
                    logger.info(f"Backed up {file.name} to {dest}")

                return True
            else:
                logger.warning("No MOK keys found to backup")
                return False

        except Exception as e:
            logger.error(f"Failed to backup MOK key: {e}")
            return False

    def restore_mok_key(
        self,
        backup_dir: str,
    ) -> bool:
        """Restore MOK key from backup.

        Args:
            backup_dir: Directory containing backup

        Returns:
            True if restore successful
        """
        try:
            backup_path = Path(backup_dir)
            self.mok_dir.mkdir(parents=True, exist_ok=True)

            import shutil

            for file in backup_path.iterfile():
                dest = self.mok_dir / file.name
                shutil.copy2(file, dest)
                logger.info(f"Restored {file.name} to {dest}")

            return True

        except Exception as e:
            logger.error(f"Failed to restore MOK key: {e}")
            return False


class AuditLogger:
    """Audit logging for compliance."""

    def __init__(
        self,
        log_file: str = "/var/log/nvidia-inst/audit.log",
    ):
        """Initialize audit logger.

        Args:
            log_file: Path to audit log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        action: str,
        success: bool = True,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Log audit event.

        Args:
            action: Action performed
            success: Whether action succeeded
            details: Additional details
            error: Error message if any
        """
        import getpass

        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            user=getpass.getuser(),
            success=success,
            details=details or {},
            error=error,
        )

        self._write_entry(entry)

    def _write_entry(self, entry: AuditLogEntry) -> None:
        """Write audit entry to log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def read_logs(
        self,
        limit: int | None = None,
    ) -> list[AuditLogEntry]:
        """Read audit logs.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries
        """
        entries = []

        try:
            if not self.log_file.exists():
                return entries

            with open(self.log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = AuditLogEntry(**data)
                        entries.append(entry)

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")

        if limit:
            entries = entries[-limit:]

        return entries

    def get_failed_events(self) -> list[AuditLogEntry]:
        """Get all failed audit events.

        Returns:
            List of failed audit entries
        """
        all_entries = self.read_logs()
        return [entry for entry in all_entries if not entry.success]

    def clear_logs(self) -> None:
        """Clear audit logs."""
        try:
            if self.log_file.exists():
                self.log_file.unlink()
                logger.info("Audit logs cleared")

        except Exception as e:
            logger.error(f"Failed to clear audit logs: {e}")
