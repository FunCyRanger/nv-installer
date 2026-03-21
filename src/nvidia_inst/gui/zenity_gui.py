"""Zenity GUI implementation."""

import subprocess

from nvidia_inst.cli import install_driver_cli
from nvidia_inst.distro.detector import DistroDetectionError, detect_distro
from nvidia_inst.gpu.compatibility import get_driver_range
from nvidia_inst.gpu.detector import detect_gpu, has_nvidia_gpu
from nvidia_inst.utils.logger import get_logger
from nvidia_inst.utils.permissions import require_root

logger = get_logger(__name__)


def zenity_info(title: str, text: str) -> None:
    """Show info dialog."""
    subprocess.run(
        ["zenity", "--info", f"--title={title}", f"--text={text}"],
        capture_output=True,
    )


def zenity_error(title: str, text: str) -> None:
    """Show error dialog."""
    subprocess.run(
        ["zenity", "--error", f"--title={title}", f"--text={text}"],
        capture_output=True,
    )


def zenity_warning(title: str, text: str) -> None:
    """Show warning dialog."""
    subprocess.run(
        ["zenity", "--warning", f"--title={title}", f"--text={text}"],
        capture_output=True,
    )


def zenity_question(title: str, text: str) -> bool:
    """Show question dialog.

    Returns:
        True if user clicked Yes, False otherwise.
    """
    result = subprocess.run(
        ["zenity", "--question", f"--title={title}", f"--text={text}"],
        capture_output=True,
    )
    return result.returncode == 0


def zenity_progress(
    title: str,
    text: str,
    percentage: int = 0,
) -> subprocess.Popen:
    """Show progress dialog.

    Returns:
        Popen process handle.
    """
    return subprocess.Popen(
        [
            "zenity",
            "--progress",
            f"--title={title}",
            f"--text={text}",
            f"--percentage={percentage}",
            "--auto-close",
            "--no-cancel",
        ],
        stdin=subprocess.PIPE,
    )


def zenity_entry(title: str, text: str, hidden: bool = False) -> str | None:
    """Show entry dialog.

    Returns:
        User input or None if cancelled.
    """
    cmd = ["zenity", "--entry", f"--title={title}", f"--text={text}"]
    if hidden:
        cmd.append("--hide-text")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def detect_gui_type() -> bool:
    """Check if Zenity is available.

    Returns:
        True if Zenity is available.
    """
    import shutil

    return shutil.which("zenity") is not None


def run_gui(args) -> int:
    """Run the Zenity GUI.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    if not detect_gui_type():
        zenity_error("Error", "Zenity is not installed")
        return 1

    try:
        distro = detect_distro()
    except DistroDetectionError as e:
        zenity_error("Error", f"Failed to detect distribution: {e}")
        return 1

    if not has_nvidia_gpu():
        zenity_info("No GPU", "No Nvidia GPU detected")
        return 0

    try:
        gpu = detect_gpu()
    except Exception as e:
        zenity_error("Error", f"Failed to detect GPU: {e}")
        return 1

    assert gpu is not None
    driver_range = get_driver_range(gpu)

    info_text = f"""Distribution: {distro}

GPU: {gpu.model}
Compute Capability: {gpu.compute_capability or "Unknown"}
VRAM: {gpu.vram or "Unknown"}

Driver: {driver_range.min_version}"""
    if driver_range.max_version:
        info_text += f" - {driver_range.max_version}"

    info_text += f"""

CUDA: {driver_range.cuda_min}"""
    if driver_range.cuda_max:
        info_text += f" - {driver_range.cuda_max}"

    if driver_range.is_eol:
        info_text += f"\n\nWARNING: {driver_range.eol_message}"

    zenity_info("nvidia-inst - System Information", info_text)

    if driver_range.is_eol:
        zenity_warning("EOL GPU", f"{driver_range.eol_message}\n\nContinue anyway?")

    confirmed = zenity_question(
        "Install Driver",
        f"Install Nvidia driver?\n\nGPU: {gpu.model}",
    )

    if not confirmed:
        return 0

    if not require_root(interactive=True):
        zenity_error("Error", "Root privileges required to install drivers")
        return 1

    progress = zenity_progress("Installing", "Installing driver...", 0)

    try:
        install_driver_cli(
            driver_version=driver_range.max_version,
            with_cuda=True,
            skip_confirmation=True,
        )

        if progress.stdin:
            progress.stdin.write("100\n")
            progress.stdin.close()
        progress.wait()

        zenity_info(
            "Success",
            "Driver installed successfully!\nPlease reboot your system.",
        )

    except Exception as e:
        if progress.stdin:
            progress.stdin.close()
        progress.wait()
        zenity_error("Error", f"Installation failed: {e}")
        return 1

    return 0
