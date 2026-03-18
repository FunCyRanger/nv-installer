"""Tests for GPU detection."""

import pytest
from unittest.mock import patch, MagicMock
from nvidia_inst.gpu.detector import (
    detect_gpu,
    GPUInfo,
    has_nvidia_gpu,
    get_current_driver_version,
    _get_gpu_generation,
    _get_compute_capability,
)


class TestGPUDetection:
    """Test GPU detection."""

    @patch("subprocess.run")
    def test_detect_gpu_rtx_3080(self, mock_run):
        """Test detection of RTX 3080."""
        mock_run.return_value = MagicMock(
            stdout="NVIDIA GeForce RTX 3080, 10.0 GB, 8.6, 535.154.05, 12.2\n",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            gpu = detect_gpu()

        assert gpu is not None
        assert "RTX 3080" in gpu.model
        assert gpu.compute_capability == 8.6
        assert gpu.driver_version == "535.154.05"
        assert gpu.cuda_version == "12.2"

    @patch("subprocess.run")
    def test_detect_gpu_rtx_4090(self, mock_run):
        """Test detection of RTX 4090."""
        mock_run.return_value = MagicMock(
            stdout="NVIDIA GeForce RTX 4090, 24.0 GB, 8.9, 550.40, 12.3\n",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            gpu = detect_gpu()

        assert gpu is not None
        assert "RTX 4090" in gpu.model
        assert gpu.compute_capability == 8.9

    @patch("subprocess.run")
    def test_detect_gpu_gtx_1080(self, mock_run):
        """Test detection of GTX 1080 (Pascal)."""
        mock_run.return_value = MagicMock(
            stdout="NVIDIA GeForce GTX 1080, 8.0 GB, 6.1, 525.147.05, 11.8\n",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            gpu = detect_gpu()

        assert gpu is not None
        assert "GTX 1080" in gpu.model
        assert gpu.compute_capability == 6.1
        assert gpu.generation == "pascal"

    @patch("subprocess.run")
    def test_detect_gpu_gtx_980(self, mock_run):
        """Test detection of GTX 980 (Maxwell)."""
        mock_run.return_value = MagicMock(
            stdout="NVIDIA GeForce GTX 980, 4.0 GB, 5.2, 470.256.02, 11.7\n",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            gpu = detect_gpu()

        assert gpu is not None
        assert "GTX 980" in gpu.model
        assert gpu.compute_capability == 5.2
        assert gpu.generation == "maxwell"


class TestGPUGeneration:
    """Test GPU generation detection."""

    def test_rtx_40xx_ada(self):
        """Test RTX 40xx is Ada generation."""
        assert _get_gpu_generation("NVIDIA GeForce RTX 4090") == "ada"

    def test_rtx_30xx_ampere(self):
        """Test RTX 30xx is Ampere generation."""
        assert _get_gpu_generation("NVIDIA GeForce RTX 3080") == "ampere"

    def test_rtx_20xx_turing(self):
        """Test RTX 20xx is Turing generation."""
        assert _get_gpu_generation("NVIDIA GeForce RTX 2080") == "turing"

    def test_gtx_10xx_pascal(self):
        """Test GTX 10xx is Pascal generation."""
        assert _get_gpu_generation("NVIDIA GeForce GTX 1080") == "pascal"

    def test_gtx_9xx_maxwell(self):
        """Test GTX 9xx is Maxwell generation."""
        assert _get_gpu_generation("NVIDIA GeForce GTX 980") == "maxwell"

    def test_gtx_6xx_kepler(self):
        """Test GTX 6xx is Kepler generation."""
        assert _get_gpu_generation("NVIDIA GeForce GTX 680") == "kepler"

    def test_tesla_v100(self):
        """Test Tesla V100 is Volta generation."""
        assert _get_gpu_generation("Tesla V100") == "volta"

    def test_a100_ampere(self):
        """Test A100 is Ampere generation."""
        assert _get_gpu_generation("NVIDIA A100") == "ampere"

    def test_unknown_generation(self):
        """Test unknown GPU returns unknown."""
        assert _get_gpu_generation("Some Unknown GPU") == "unknown"


class TestComputeCapability:
    """Test compute capability lookup."""

    def test_ada_compute_capability(self):
        """Test Ada compute capability."""
        assert _get_compute_capability("ada") == 8.9

    def test_ampere_compute_capability(self):
        """Test Ampere compute capability."""
        assert _get_compute_capability("ampere") == 8.6

    def test_turing_compute_capability(self):
        """Test Turing compute capability."""
        assert _get_compute_capability("turing") == 7.5

    def test_pascal_compute_capability(self):
        """Test Pascal compute capability."""
        assert _get_compute_capability("pascal") == 6.1

    def test_maxwell_compute_capability(self):
        """Test Maxwell compute capability."""
        assert _get_compute_capability("maxwell") == 5.2

    def test_kepler_compute_capability(self):
        """Test Kepler compute capability."""
        assert _get_compute_capability("kepler") == 3.7

    def test_unknown_compute_capability(self):
        """Test unknown returns None."""
        assert _get_compute_capability("unknown") is None


class TestHasNvidiaGPU:
    """Test Nvidia GPU detection."""

    @patch("subprocess.run")
    def test_has_nvidia_gpu_true(self, mock_run):
        """Test has_nvidia_gpu returns True."""
        mock_run.return_value = MagicMock(
            stdout="GPU 0: NVIDIA GeForce RTX 3080",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            assert has_nvidia_gpu() is True

    @patch("subprocess.run")
    def test_has_nvidia_gpu_false(self, mock_run):
        """Test has_nvidia_gpu returns False."""
        mock_run.side_effect = Exception("not found")

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=False):
            with patch("subprocess.run") as lspci_mock:
                lspci_mock.return_value = MagicMock(
                    stdout="00:02.0 VGA compatible controller: Intel",
                    returncode=0,
                )
                assert has_nvidia_gpu() is False


class TestCurrentDriverVersion:
    """Test current driver version detection."""

    @patch("subprocess.run")
    def test_get_current_driver_version(self, mock_run):
        """Test getting current driver version."""
        mock_run.return_value = MagicMock(
            stdout="535.154.05\n",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            version = get_current_driver_version()

        assert version == "535.154.05"

    @patch("subprocess.run")
    def test_get_current_driver_version_na(self, mock_run):
        """Test getting driver version when nvidia-smi returns N/A."""
        mock_run.return_value = MagicMock(
            stdout="N/A\n",
            returncode=0,
        )

        with patch("nvidia_inst.gpu.detector._nvidia_smi_available", return_value=True):
            version = get_current_driver_version()

        assert version is None
