"""GUI implementations for nvidia-inst.

This module provides GUI launcher functionality for the installer.
"""

import argparse
import shutil

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def launch_gui(args: argparse.Namespace) -> int:
    """Launch GUI interface.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    gui_type = args.gui_type

    if not gui_type:
        gui_type = detect_gui_type()

    if gui_type == "tkinter":
        try:
            from nvidia_inst.gui.tkinter_gui import run_gui

            return run_gui(args)
        except ImportError as e:
            logger.error(f"Failed to import Tkinter: {e}")
            print("Tkinter not available. Trying Zenity...")
            gui_type = "zenity"

    if gui_type == "zenity":
        try:
            from nvidia_inst.gui.zenity_gui import run_gui

            return run_gui(args)
        except ImportError as e:
            logger.error(f"Failed to import Zenity: {e}")
            print("Neither Tkinter nor Zenity is available.")
            return 1

    return 1


def detect_gui_type() -> str | None:
    """Detect available GUI type.

    Returns:
        'tkinter', 'zenity', or None.
    """
    if shutil.which("zenity"):
        return "zenity"

    try:
        import importlib.util

        if importlib.util.find_spec("tkinter"):
            return "tkinter"
    except ImportError:
        pass
    return None
