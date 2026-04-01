"""Zenity GUI implementation."""

import subprocess

from nvidia_inst.cli import (
    DriverOption,
    DriverState,
    detect_driver_state,
    execute_driver_change,
)
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


def zenity_show_options(state: DriverState) -> DriverOption | None:
    """Show driver options using zenity dialog.

    Args:
        state: Current driver state with available options.

    Returns:
        Selected DriverOption or None if cancelled.
    """
    # Prepare options for zenity list
    options_text = []
    for opt in state.options:
        option_line = f"{opt.number}. {opt.description}"
        if opt.recommended:
            option_line += " [RECOMMENDED]"
        options_text.append(option_line)

    # Join options with newline for zenity --list
    options_data = "\n".join(options_text)

    # Show zenity list dialog
    try:
        result = subprocess.run(
            [
                "zenity",
                "--list",
                "--title=Driver Management Options",
                f"--text={state.message}",
                "--column=Option",
                "--width=500",
                "--height=400",
            ],
            input=options_data,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # User cancelled or error
            return None

        selected_description = result.stdout.strip()
        if not selected_description:
            return None

        # Find the option matching the selected description
        for opt in state.options:
            desc = f"{opt.number}. {opt.description}"
            if opt.recommended:
                desc += " [RECOMMENDED]"
            if desc == selected_description:
                return opt

        return None  # Should not happen

    except Exception:
        return None


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
        zenity_warning(
            "EOL GPU",
            f"{driver_range.eol_message}\n\nContinue anyway?",
        )

    # Detect driver state and get available options
    if not gpu or not distro or not driver_range:
        zenity_error("Error", "Could not detect system information")
        return 1

    state = detect_driver_state(gpu, driver_range, distro.id)

    # Show options to user
    selected_option = zenity_show_options(state)
    if selected_option is None:  # User cancelled
        return 0

    if not require_root(interactive=True):
        zenity_error("Error", "Root privileges required for driver operations")
        return 1

    # Execute the selected option
    try:
        result = execute_driver_change(
            selected_option, state, distro, gpu, driver_range, dry_run=False
        )

        if result == 0:
            zenity_info(
                "Success",
                "Operation completed successfully!\nPlease reboot your system if required.",
            )
        else:
            zenity_error("Error", "Operation failed")
        return result

    except Exception as e:
        zenity_error("Error", f"Operation failed: {e}")
        return 1
