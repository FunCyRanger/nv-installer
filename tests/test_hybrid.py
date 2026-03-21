"""Tests for hybrid graphics detection and management."""

from unittest.mock import MagicMock, patch


class TestDetectAllGpus:
    """Tests for detect_all_gpus function."""

    @patch("subprocess.run")
    def test_nvidia_smi_available(self, mock_run):
        """Test detection when nvidia-smi is available."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-1234)\n"
            "GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-5678)\n",
        )

        from nvidia_inst.gpu.hybrid import detect_all_gpus

        gpus = detect_all_gpus()
        assert len(gpus) == 2
        assert "RTX 3080" in gpus[0].name
        assert gpus[0].index == 0
        assert "RTX 3090" in gpus[1].name
        assert gpus[1].index == 1

    @patch("subprocess.run")
    def test_nvidia_smi_not_available(self, mock_run):
        """Test detection when nvidia-smi fails."""
        mock_run.side_effect = FileNotFoundError()

        from nvidia_inst.gpu.hybrid import detect_all_gpus

        gpus = detect_all_gpus()
        assert len(gpus) == 0

    @patch("subprocess.run")
    def test_single_gpu(self, mock_run):
        """Test detection with single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-1234)\n",
        )

        from nvidia_inst.gpu.hybrid import detect_all_gpus

        gpus = detect_all_gpus()
        assert len(gpus) == 1
        assert "RTX 4090" in gpus[0].name


class TestDetectIntegratedGpu:
    """Tests for detect_integrated_gpu function."""

    @patch("subprocess.run")
    def test_intel_igpu(self, mock_run):
        """Test detection of Intel integrated GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620\n"
            "01:00.0 3D controller: NVIDIA Corporation TU117M [GeForce GTX 1650 Mobile]\n",
        )

        from nvidia_inst.gpu.hybrid import detect_integrated_gpu

        igpu_type, model = detect_integrated_gpu()
        assert igpu_type == "intel"
        assert model is not None

    @patch("subprocess.run")
    def test_amd_igpu(self, mock_run):
        """Test detection of AMD integrated GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0d:00.0 VGA compatible controller: Advanced Micro Devices [AMD] Radeon Graphics\n"
            "0000:01:00.0 3D controller: NVIDIA Corporation GA104 [GeForce RTX 3070]\n",
        )

        from nvidia_inst.gpu.hybrid import detect_integrated_gpu

        igpu_type, model = detect_integrated_gpu()
        assert igpu_type == "amd"

    @patch("subprocess.run")
    def test_no_igpu(self, mock_run):
        """Test when no integrated GPU is present."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="01:00.0 VGA compatible controller: NVIDIA Corporation GA104 [GeForce RTX 3070]\n",
        )

        from nvidia_inst.gpu.hybrid import detect_integrated_gpu

        igpu_type, model = detect_integrated_gpu()
        assert igpu_type is None


class TestDetectSystemType:
    """Tests for detect_system_type function."""

    def test_laptop_with_battery(self, tmp_path):
        """Test laptop detection with battery present."""
        power_supply = tmp_path / "sys" / "class" / "power_supply"
        power_supply.mkdir(parents=True)
        (power_supply / "BAT0").mkdir()

        with patch("nvidia_inst.gpu.hybrid.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.iterdir.return_value = [power_supply / "BAT0"]

            from nvidia_inst.gpu.hybrid import detect_system_type

            system_type = detect_system_type()
            assert system_type == "laptop"

    def test_desktop_no_battery(self):
        """Test desktop detection without battery."""
        with patch("nvidia_inst.gpu.hybrid.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.iterdir.return_value = iter([])

            from nvidia_inst.gpu.hybrid import detect_system_type

            system_type = detect_system_type()
            assert system_type == "desktop"


class TestIsHybridSystem:
    """Tests for is_hybrid_system function."""

    @patch("nvidia_inst.gpu.hybrid.detect_all_gpus")
    @patch("nvidia_inst.gpu.hybrid.detect_integrated_gpu")
    def test_hybrid_system(self, mock_igpu, mock_gpus):
        """Test hybrid system detection."""
        mock_gpus.return_value = [MagicMock(index=0, name="RTX 3080")]
        mock_igpu.return_value = ("intel", "Intel UHD Graphics")

        from nvidia_inst.gpu.hybrid import is_hybrid_system

        assert is_hybrid_system() is True

    @patch("nvidia_inst.gpu.hybrid.detect_all_gpus")
    @patch("nvidia_inst.gpu.hybrid.detect_integrated_gpu")
    def test_desktop_single_gpu(self, mock_igpu, mock_gpus):
        """Test desktop with single NVIDIA GPU (not hybrid)."""
        mock_gpus.return_value = [MagicMock(index=0, name="RTX 3080")]
        mock_igpu.return_value = (None, None)

        from nvidia_inst.gpu.hybrid import is_hybrid_system

        assert is_hybrid_system() is False

    @patch("nvidia_inst.gpu.hybrid.detect_all_gpus")
    @patch("nvidia_inst.gpu.hybrid.detect_integrated_gpu")
    def test_no_nvidia_gpu(self, mock_igpu, mock_gpus):
        """Test system without NVIDIA GPU."""
        mock_gpus.return_value = []
        mock_igpu.return_value = ("intel", "Intel UHD Graphics")

        from nvidia_inst.gpu.hybrid import is_hybrid_system

        assert is_hybrid_system() is False


class TestGetNativeTool:
    """Tests for get_native_tool function."""

    def test_ubuntu(self):
        """Test Ubuntu detection."""
        with patch("nvidia_inst.gpu.hybrid._command_exists") as mock_cmd:
            mock_cmd.return_value = True

            from nvidia_inst.gpu.hybrid import get_native_tool

            tool, method, needs_install = get_native_tool("ubuntu")
            assert tool == "nvidia-prime"
            assert needs_install is False

    def test_fedora(self):
        """Test Fedora detection."""
        with (
            patch("nvidia_inst.gpu.hybrid._command_exists") as mock_cmd,
            patch("nvidia_inst.gpu.hybrid.is_service_installed") as mock_service,
        ):
            mock_cmd.return_value = True
            mock_service.return_value = True

            from nvidia_inst.gpu.hybrid import get_native_tool

            tool, method, needs_install = get_native_tool("fedora")
            assert tool == "switcherooctl"
            assert needs_install is False

    def test_arch_no_tool(self):
        """Test Arch with no native tool."""
        with patch("nvidia_inst.gpu.hybrid._command_exists") as mock_cmd:
            mock_cmd.return_value = False

            from nvidia_inst.gpu.hybrid import get_native_tool

            tool, method, needs_install = get_native_tool("arch")
            assert tool is None
            assert needs_install is False

    def test_cachyos_with_settings(self):
        """Test CachyOS with cachyos-settings."""
        with patch("nvidia_inst.gpu.hybrid._command_exists") as mock_cmd:
            mock_cmd.return_value = True

            from nvidia_inst.gpu.hybrid import get_native_tool

            tool, method, needs_install = get_native_tool("cachyos")
            assert tool == "cachyos-settings"
            assert needs_install is False


class TestDetectHybrid:
    """Tests for detect_hybrid function."""

    @patch("nvidia_inst.gpu.hybrid.is_hybrid_system")
    @patch("nvidia_inst.gpu.hybrid.detect_all_gpus")
    @patch("nvidia_inst.gpu.hybrid.detect_integrated_gpu")
    @patch("nvidia_inst.gpu.hybrid.detect_system_type")
    @patch("nvidia_inst.gpu.hybrid.get_native_tool")
    def test_full_hybrid_detection(
        self, mock_tool, mock_sys, mock_igpu, mock_gpus, mock_hybrid
    ):
        """Test complete hybrid detection."""
        from nvidia_inst.gpu.hybrid import GPUDevice

        mock_hybrid.return_value = True
        mock_gpus.return_value = [
            GPUDevice(
                index=0,
                name="NVIDIA GeForce RTX 3080",
                gpu_type="discrete",
                vendor="NVIDIA",
            ),
        ]
        mock_igpu.return_value = ("intel", "Intel Corporation UHD Graphics")
        mock_sys.return_value = "laptop"
        mock_tool.return_value = ("nvidia-prime", "prime-select", False)

        from nvidia_inst.gpu.hybrid import detect_hybrid

        hybrid_info = detect_hybrid("ubuntu")

        assert hybrid_info is not None
        assert hybrid_info.is_hybrid is True
        assert hybrid_info.igpu_type == "intel"
        assert hybrid_info.dgpu_model == "NVIDIA GeForce RTX 3080"
        assert hybrid_info.dgpu_count == 1
        assert hybrid_info.system_type == "laptop"
        assert hybrid_info.native_tool == "nvidia-prime"
        assert hybrid_info.needs_install is False
        assert "on-demand" in hybrid_info.available_modes

    @patch("nvidia_inst.gpu.hybrid.is_hybrid_system")
    def test_no_hybrid(self, mock_hybrid):
        """Test when no hybrid system present."""
        mock_hybrid.return_value = False

        from nvidia_inst.gpu.hybrid import detect_hybrid

        hybrid_info = detect_hybrid("ubuntu")
        assert hybrid_info is None


class TestToolServicePackages:
    """Tests for TOOL_SERVICE_PACKAGES constant."""

    def test_tool_service_packages(self):
        """Test TOOL_SERVICE_PACKAGES mapping."""
        from nvidia_inst.gpu.hybrid import TOOL_SERVICE_PACKAGES

        assert TOOL_SERVICE_PACKAGES["switcherooctl"] == "switcheroo-control"
        assert TOOL_SERVICE_PACKAGES["nvidia-prime"] is None
        assert TOOL_SERVICE_PACKAGES["system76-power"] is None


class TestIsServiceInstalled:
    """Tests for is_service_installed function."""

    @patch("subprocess.run")
    @patch("nvidia_inst.gpu.hybrid.Path")
    def test_rpm_installed(self, mock_path, mock_run):
        """Test package check with rpm (installed)."""
        mock_path.return_value.exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0)

        from nvidia_inst.gpu.hybrid import is_service_installed

        result = is_service_installed("switcheroo-control")
        assert result is True
        mock_run.assert_called_once_with(
            ["rpm", "-q", "switcheroo-control"],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("subprocess.run")
    @patch("nvidia_inst.gpu.hybrid.Path")
    def test_rpm_not_installed(self, mock_path, mock_run):
        """Test package check with rpm (not installed)."""
        mock_path.return_value.exists.return_value = True
        mock_run.return_value = MagicMock(returncode=1)

        from nvidia_inst.gpu.hybrid import is_service_installed

        result = is_service_installed("nonexistent-package")
        assert result is False


class TestHybridPackages:
    """Tests for hybrid package management."""

    def test_fedora_packages(self):
        """Test Fedora hybrid packages."""
        from nvidia_inst.installer.hybrid import get_hybrid_packages

        packages = get_hybrid_packages("fedora")
        assert "switcheroo-control" in packages

    def test_ubuntu_packages(self):
        """Test Ubuntu (no extra packages needed)."""
        from nvidia_inst.installer.hybrid import get_hybrid_packages

        packages = get_hybrid_packages("ubuntu")
        assert packages == []

    def test_arch_packages(self):
        """Test Arch (no packages - uses env file)."""
        from nvidia_inst.installer.hybrid import get_hybrid_packages

        packages = get_hybrid_packages("arch")
        assert packages == []


class TestPrimeEnvConfig:
    """Tests for PRIME environment configuration."""

    def test_env_file_path(self):
        """Test environment file path constant."""
        from nvidia_inst.installer.hybrid import HYBRID_ENV_FILE

        assert HYBRID_ENV_FILE == "/etc/environment.d/90-nvidia-hybrid.conf"

    def test_env_content(self):
        """Test environment file content."""
        from nvidia_inst.installer.hybrid import HYBRID_ENV_CONTENT

        assert "__NV_PRIME_RENDER_OFFLOAD=1" in HYBRID_ENV_CONTENT
        assert "__GLX_VENDOR_LIBRARY_NAME=nvidia" in HYBRID_ENV_CONTENT


class TestGetAvailableModes:
    """Tests for _get_available_modes function."""

    def test_nvidia_prime_modes(self):
        """Test NVIDIA Prime available modes."""
        from nvidia_inst.gpu.hybrid import _get_available_modes

        modes = _get_available_modes("nvidia-prime", "ubuntu")
        assert "on-demand" in modes
        assert "intel" in modes
        assert "nvidia" in modes

    def test_switcherooctl_modes(self):
        """Test switcherooctl available modes."""
        from nvidia_inst.gpu.hybrid import _get_available_modes

        modes = _get_available_modes("switcherooctl", "fedora")
        assert "hybrid" not in modes
        assert "intel" in modes
        assert "nvidia" in modes

    def test_system76_power_modes(self):
        """Test system76-power available modes."""
        from nvidia_inst.gpu.hybrid import _get_available_modes

        modes = _get_available_modes("system76-power", "pop")
        assert "hybrid" in modes
        assert "integrated" in modes
        assert "nvidia" in modes
        assert "compute" in modes

    def test_no_native_tool(self):
        """Test modes when no native tool available."""
        from nvidia_inst.gpu.hybrid import _get_available_modes

        modes = _get_available_modes(None, "arch")
        assert "hybrid" in modes
        assert "intel" in modes
        assert "nvidia" in modes
