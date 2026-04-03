"""Monitoring and reporting for NVIDIA driver installations.

This module provides functionality for monitoring driver status
and reporting to central systems.
"""

import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from nvidia_inst.distro.detector import detect_distro
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemHealth:
    """System health status."""

    hostname: str = ""
    timestamp: str = ""
    driver_version: str | None = None
    cuda_version: str | None = None
    gpu_model: str | None = None
    gpu_temperature: float | None = None
    gpu_memory_used: int | None = None
    gpu_memory_total: int | None = None
    gpu_utilization: float | None = None
    driver_loaded: bool = False
    cuda_working: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class InstallationReport:
    """Installation report for central server."""

    hostname: str = ""
    timestamp: str = ""
    distro_id: str = ""
    distro_version: str = ""
    kernel_version: str = ""
    gpu_model: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    installation_success: bool = False
    installation_time: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """Compliance report for fleet management."""

    hostname: str = ""
    timestamp: str = ""
    driver_installed: bool = False
    driver_version: str | None = None
    driver_compliant: bool = False
    required_version: str | None = None
    cuda_installed: bool = False
    cuda_version: str | None = None
    cuda_compliant: bool = False
    required_cuda: str | None = None
    issues: list[str] = field(default_factory=list)


class HealthChecker:
    """Check system health after installation."""

    def __init__(self):
        """Initialize health checker."""
        self.hostname = socket.gethostname()

    def check_driver_loaded(self) -> bool:
        """Verify nvidia kernel module is loaded.

        Returns:
            True if driver is loaded
        """
        try:
            import subprocess

            result = subprocess.run(
                ["lsmod"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return "nvidia" in result.stdout

            return False

        except Exception as e:
            logger.error(f"Failed to check driver loaded: {e}")
            return False

    def check_cuda_working(self) -> bool:
        """Verify CUDA is functional.

        Returns:
            True if CUDA is working
        """
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to check CUDA: {e}")
            return False

    def get_gpu_temperature(self) -> float | None:
        """Get GPU temperature for monitoring.

        Returns:
            GPU temperature in Celsius or None
        """
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())

            return None

        except Exception as e:
            logger.error(f"Failed to get GPU temperature: {e}")
            return None

    def get_gpu_memory(self) -> tuple[int | None, int | None]:
        """Get GPU memory usage.

        Returns:
            Tuple of (used_mb, total_mb)
        """
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) == 2:
                    used = int(parts[0].strip())
                    total = int(parts[1].strip())
                    return (used, total)

            return (None, None)

        except Exception as e:
            logger.error(f"Failed to get GPU memory: {e}")
            return (None, None)

    def get_gpu_utilization(self) -> float | None:
        """Get GPU utilization percentage.

        Returns:
            GPU utilization percentage or None
        """
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())

            return None

        except Exception as e:
            logger.error(f"Failed to get GPU utilization: {e}")
            return None

    def generate_health_report(self) -> SystemHealth:
        """Generate comprehensive health report.

        Returns:
            SystemHealth report
        """
        driver_loaded = self.check_driver_loaded()
        cuda_working = self.check_cuda_working()

        memory_used, memory_total = self.get_gpu_memory()

        return SystemHealth(
            hostname=self.hostname,
            timestamp=datetime.now().isoformat(),
            driver_loaded=driver_loaded,
            cuda_working=cuda_working,
            gpu_temperature=self.get_gpu_temperature(),
            gpu_memory_used=memory_used,
            gpu_memory_total=memory_total,
            gpu_utilization=self.get_gpu_utilization(),
            errors=[]
            if (driver_loaded and cuda_working)
            else [
                "Driver not loaded" if not driver_loaded else "",
                "CUDA not working" if not cuda_working else "",
            ],
        )

    def save_health_report(
        self,
        report: SystemHealth,
        output_path: str | None = None,
    ) -> str:
        """Save health report to file.

        Args:
            report: Health report to save
            output_path: Path to save report

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = f"/tmp/nvidia-health-{self.hostname}.json"

        data = asdict(report)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Health report saved to {output_path}")
        return output_path


class InstallationReporter:
    """Report installation status to central server."""

    def __init__(self, server_url: str | None = None):
        """Initialize reporter.

        Args:
            server_url: Central server URL (optional)
        """
        self.server_url = server_url
        self.hostname = socket.gethostname()

    def create_report(
        self,
        success: bool,
        driver_version: str | None = None,
        cuda_version: str | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        installation_time: float = 0.0,
    ) -> InstallationReport:
        """Create installation report.

        Args:
            success: Whether installation succeeded
            driver_version: Installed driver version
            cuda_version: Installed CUDA version
            errors: List of errors
            warnings: List of warnings
            installation_time: Installation duration in seconds

        Returns:
            InstallationReport
        """
        try:
            distro = detect_distro()
            distro_id = distro.id
            distro_version = distro.version_id if hasattr(distro, "version_id") else ""
        except Exception:
            distro_id = "unknown"
            distro_version = ""

        import platform

        return InstallationReport(
            hostname=self.hostname,
            timestamp=datetime.now().isoformat(),
            distro_id=distro_id,
            distro_version=distro_version,
            kernel_version=platform.release(),
            driver_version=driver_version,
            cuda_version=cuda_version,
            installation_success=success,
            installation_time=installation_time,
            errors=errors or [],
            warnings=warnings or [],
        )

    def report_to_server(self, report: InstallationReport) -> bool:
        """Report installation result to central server.

        Args:
            report: Installation report

        Returns:
            True if report sent successfully
        """
        if not self.server_url:
            logger.debug("No server URL configured, skipping report")
            return False

        try:
            import requests

            data = asdict(report)
            response = requests.post(
                f"{self.server_url}/api/installation-report",
                json=data,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("Report sent successfully")
                return True
            else:
                logger.warning(f"Failed to send report: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send report: {e}")
            return False

    def save_report_locally(
        self,
        report: InstallationReport,
        output_dir: str = "/var/log/nvidia-inst",
    ) -> str:
        """Save report to local file.

        Args:
            report: Installation report
            output_dir: Directory to save report

        Returns:
            Path to saved report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"installation-{timestamp}.json"
        file_path = output_path / filename

        data = asdict(report)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Report saved to {file_path}")
        return str(file_path)


class ComplianceChecker:
    """Check if system meets compliance requirements."""

    def __init__(
        self,
        required_driver_version: str | None = None,
        required_cuda_version: str | None = None,
    ):
        """Initialize compliance checker.

        Args:
            required_driver_version: Required driver version
            required_cuda_version: Required CUDA version
        """
        self.required_driver_version = required_driver_version
        self.required_cuda_version = required_cuda_version
        self.hostname = socket.gethostname()

    def get_installed_driver_version(self) -> str | None:
        """Get installed driver version.

        Returns:
            Driver version string or None
        """
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            return None

        except Exception:
            return None

    def get_installed_cuda_version(self) -> str | None:
        """Get installed CUDA version.

        Returns:
            CUDA version string or None
        """
        try:
            import subprocess

            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line:
                        parts = line.split("release")
                        if len(parts) > 1:
                            version = parts[1].strip().split(",")[0].strip()
                            return version

            return None

        except Exception:
            return None

    def check_compliance(self) -> ComplianceReport:
        """Check system compliance.

        Returns:
            ComplianceReport
        """
        driver_version = self.get_installed_driver_version()
        cuda_version = self.get_installed_cuda_version()

        driver_installed = driver_version is not None
        cuda_installed = cuda_version is not None

        driver_compliant = True
        cuda_compliant = True
        issues = []

        if self.required_driver_version:
            if driver_version:
                if driver_version != self.required_driver_version:
                    driver_compliant = False
                    issues.append(
                        f"Driver version {driver_version} does not match "
                        f"required {self.required_driver_version}"
                    )
            else:
                driver_compliant = False
                issues.append("No driver installed")

        if self.required_cuda_version:
            if cuda_version:
                if not cuda_version.startswith(self.required_cuda_version):
                    cuda_compliant = False
                    issues.append(
                        f"CUDA version {cuda_version} does not match "
                        f"required {self.required_cuda_version}"
                    )
            else:
                cuda_compliant = False
                issues.append("CUDA not installed")

        return ComplianceReport(
            hostname=self.hostname,
            timestamp=datetime.now().isoformat(),
            driver_installed=driver_installed,
            driver_version=driver_version,
            driver_compliant=driver_compliant,
            required_version=self.required_driver_version,
            cuda_installed=cuda_installed,
            cuda_version=cuda_version,
            cuda_compliant=cuda_compliant,
            required_cuda=self.required_cuda_version,
            issues=issues,
        )

    def save_compliance_report(
        self,
        report: ComplianceReport,
        output_path: str | None = None,
    ) -> str:
        """Save compliance report to file.

        Args:
            report: Compliance report
            output_path: Path to save report

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = f"/tmp/nvidia-compliance-{self.hostname}.json"

        data = asdict(report)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Compliance report saved to {output_path}")
        return output_path
