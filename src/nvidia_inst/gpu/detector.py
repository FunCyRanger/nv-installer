"""GPU detection for Nvidia graphics cards."""

import re
import subprocess
from dataclasses import dataclass

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """Information about an Nvidia GPU."""

    model: str
    vram: str | None = None
    compute_capability: float | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    generation: str | None = None

    def __str__(self) -> str:
        return self.model


class GPUDetectionError(Exception):
    """Raised when GPU detection fails."""
    pass


GPU_PATTERNS = [
    (r"RTX\s*50\d{2}", "blackwell", 9.0),
    (r"RTX\s*40\d{2}", "ada", 8.9),
    (r"RTX\s*30\d{2}", "ampere", 8.6),
    (r"RTX\s*20\d{2}", "turing", 7.5),
    (r"GTX\s*16\d{2}", "turing", 7.5),
    (r"GTX\s*10\d{2}", "pascal", 6.1),
    (r"GTX\s*9\d{2}", "maxwell", 5.2),
    (r"GTX\s*6\d{2}", "kepler", 3.7),
    (r"GTX\s*7\d{2,3}", "kepler", 3.5),
    (r"GTX\s*750", "maxwell", 5.0),
    (r"GM200", "maxwell", 5.2),
    (r"GM204", "maxwell", 5.2),
    (r"GM206", "maxwell", 5.2),
    (r"GM20[0-8]", "maxwell", 5.2),
    (r"GM10[0-8]", "maxwell", 5.2),
    (r"GP10[0-8]", "pascal", 6.1),
    (r"TU1[0-2][0-9]", "turing", 7.5),
    (r"GA10[0-9]", "ampere", 8.6),
    (r"AD10[0-9]", "ada", 8.9),
    (r"Tesla\s*V100", "volta", 7.0),
    (r"\bV100\b", "volta", 7.0),
    (r"Tesla\s*K\d+", "kepler", 3.7),
    (r"Tesla\s*M\d+", "maxwell", 5.2),
    (r"Tesla\s*P100", "pascal", 6.0),
    (r"Tesla\s*T4", "turing", 7.5),
    (r"A100", "ampere", 8.0),
    (r"L40", "ada", 8.9),
    (r"Quadro\s*RTX", "turing", 7.5),
    (r"Quadro\s*RM5000", "maxwell", 5.2),
    (r"Quadro\s*M\d+", "maxwell", 5.2),
    (r"Quadro\s*P\d+", "pascal", 6.1),
    (r"Quadro\s*[^R]", "pascal", 6.0),
]


def detect_gpu() -> GPUInfo | None:
    """Detect Nvidia GPU using nvidia-smi.

    Returns:
        GPUInfo if GPU found, None otherwise.
    """
    if not _nvidia_smi_available():
        logger.warning("nvidia-smi not available, trying lspci fallback")
        return _detect_gpu_lspci()

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout.strip()
        if not output:
            return None

        parts = [p.strip() for p in output.split(",")]

        gpu = GPUInfo(
            model=parts[0],
            vram=parts[1] if len(parts) > 1 else None,
            compute_capability=float(parts[2]) if len(parts) > 2 and parts[2] != "N/A" else None,
            driver_version=parts[3] if len(parts) > 3 and parts[3] != "N/A" else None,
            cuda_version=parts[4] if len(parts) > 4 and parts[4] != "N/A" else None,
        )

        gpu.generation = _get_gpu_generation(gpu.model)

        logger.info(f"Detected GPU: {gpu.model}")
        return gpu

    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi failed: {e.stderr}")
        return _detect_gpu_lspci()


def _nvidia_smi_available() -> bool:
    """Check if nvidia-smi is available."""
    import shutil
    return shutil.which("nvidia-smi") is not None


def _detect_gpu_lspci() -> GPUInfo | None:
    """Detect Nvidia GPU using lspci fallback."""
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.splitlines():
            if "nvidia" in line.lower() and "vga" in line.lower():
                gpu_model = _parse_lspci_gpu(line)
                if gpu_model:
                    gpu = GPUInfo(
                        model=gpu_model,
                        generation=_get_gpu_generation(gpu_model),
                    )
                    gpu.compute_capability = _get_compute_capability(gpu.generation)
                    logger.info(f"Detected GPU via lspci: {gpu.model}")
                    return gpu

        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"lspci failed: {e.stderr}")
        raise GPUDetectionError("Cannot detect GPU") from e


def _parse_lspci_gpu(line: str) -> str | None:
    """Parse GPU model from lspci output."""
    match = re.search(r"NVIDIA Corporation ([A-Za-z0-9]+)", line, re.IGNORECASE)
    if match:
        model_code = match.group(1).strip()
        friendly_name = _get_friendly_gpu_name(model_code)
        return friendly_name

    match = re.search(r"\[([^\]]+)\]", line)
    if match:
        return match.group(1).strip()

    match = re.search(r"(?:VGA compatible controller|NVIDIA)[:\s]+(.+)", line, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    parts = line.split(":")
    if len(parts) > 2:
        return parts[2].strip()

    return line.split(":")[-1].strip() if ":" in line else None


def _get_friendly_gpu_name(model_code: str) -> str:
    """Convert GPU model code to friendly name."""
    gpu_names = {
        "GM206GLM": "Nvidia Quadro M2200 (Maxwell)",
        "GM204GLM": "Nvidia Quadro M2000 (Maxwell)",
        "GM203GLM": "Nvidia Quadro M1000 (Maxwell)",
        "GM202GLM": "Nvidia Quadro M600 (Maxwell)",
        "GM108M": "Nvidia GeForce 940M (Maxwell)",
        "GM107M": "Nvidia GeForce GTX 960M (Maxwell)",
        "GM106M": "Nvidia GeForce GTX 960M (Maxwell)",
        "GM105M": "Nvidia GeForce GTX 1050M (Maxwell)",
        "GM206": "Nvidia GM206 (Maxwell)",
        "GM204": "Nvidia GM204 (Maxwell)",
        "GM200": "Nvidia GM200 (Maxwell)",
        "GM108": "Nvidia GeForce GT 1030 (Maxwell)",
        "GM107": "Nvidia GeForce GTX 750 (Maxwell)",
        "GP108M": "Nvidia GeForce GT 1030M (Pascal)",
        "GP107M": "Nvidia GeForce GTX 1050M (Pascal)",
        "GP106M": "Nvidia GeForce GTX 1060M (Pascal)",
        "GP104M": "Nvidia GeForce GTX 1070M (Pascal)",
        "GP102M": "Nvidia GeForce GTX 1080M (Pascal)",
        "GP108": "Nvidia GeForce GT 1030 (Pascal)",
        "GP107": "Nvidia GeForce GTX 1050 (Pascal)",
        "GP106": "Nvidia GeForce GTX 1060 (Pascal)",
        "GP104": "Nvidia GeForce GTX 1080 (Pascal)",
        "GP102": "Nvidia GeForce GTX 1080 Ti (Pascal)",
        "TU117M": "Nvidia GeForce GTX 1650M (Turing)",
        "TU116M": "Nvidia GeForce GTX 1660M (Turing)",
        "TU106M": "Nvidia GeForce RTX 2070M (Turing)",
        "TU104M": "Nvidia GeForce RTX 2080M (Turing)",
        "TU117": "Nvidia GeForce GTX 1650 (Turing)",
        "TU116": "Nvidia GeForce GTX 1660 (Turing)",
        "TU106": "Nvidia GeForce RTX 2070 (Turing)",
        "TU104": "Nvidia GeForce RTX 2080 (Turing)",
        "TU102": "Nvidia GeForce RTX 2080 Ti (Turing)",
        "GA102M": "Nvidia GeForce RTX 3090M (Ampere)",
        "GA104M": "Nvidia GeForce RTX 3070M (Ampere)",
        "GA106M": "Nvidia GeForce RTX 3060M (Ampere)",
        "GA102": "Nvidia GeForce RTX 3090 (Ampere)",
        "GA104": "Nvidia GeForce RTX 3070 (Ampere)",
        "GA106": "Nvidia GeForce RTX 3060 (Ampere)",
        "GA107": "Nvidia GeForce RTX 3050 (Ampere)",
        "GA108": "Nvidia GeForce RTX 3050 Ti (Ampere)",
        "AD102M": "Nvidia GeForce RTX 4090M (Ada)",
        "AD103M": "Nvidia GeForce RTX 4080M (Ada)",
        "AD106M": "Nvidia GeForce RTX 4060M (Ada)",
        "AD107M": "Nvidia GeForce RTX 4060 TiM (Ada)",
        "AD102": "Nvidia GeForce RTX 4090 (Ada)",
        "AD103": "Nvidia GeForce RTX 4080 (Ada)",
        "AD106": "Nvidia GeForce RTX 4060 (Ada)",
        "AD107": "Nvidia GeForce RTX 4060 Ti (Ada)",
        "AD108": "Nvidia GeForce RTX 4070 (Ada)",
    }
    return gpu_names.get(model_code, f"Nvidia {model_code}")


def _get_gpu_generation(gpu_model: str) -> str | None:
    """Determine GPU generation from model name."""
    for pattern, generation, _ in GPU_PATTERNS:
        if re.search(pattern, gpu_model, re.IGNORECASE):
            return generation

    return "unknown"


def _get_compute_capability(generation: str | None) -> float | None:
    """Get compute capability for a GPU generation."""
    if not generation:
        return None

    for _, gen, cc in GPU_PATTERNS:
        if gen == generation:
            return cc

    return None


def has_nvidia_gpu() -> bool:
    """Check if system has an Nvidia GPU.

    Returns:
        True if Nvidia GPU is present, False otherwise.
    """
    if _nvidia_smi_available():
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "nvidia" in result.stdout.lower()
    except subprocess.CalledProcessError:
        return False


def get_current_driver_version() -> str | None:
    """Get the currently installed Nvidia driver version.

    Returns:
        Driver version string if installed, None otherwise.
    """
    if not _nvidia_smi_available():
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        version = result.stdout.strip()
        return version if version != "N/A" else None
    except subprocess.CalledProcessError:
        return None
