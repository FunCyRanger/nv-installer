"""Tkinter GUI implementation."""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from nvidia_inst.cli import install_driver_cli
from nvidia_inst.distro.detector import DistroDetectionError, DistroInfo, detect_distro
from nvidia_inst.gpu.compatibility import DriverRange, get_driver_range
from nvidia_inst.gpu.detector import GPUInfo, detect_gpu, has_nvidia_gpu
from nvidia_inst.utils.logger import get_logger
from nvidia_inst.utils.permissions import require_root

logger = get_logger(__name__)


class NvidiaInstGUI:
    """Main GUI application using Tkinter."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("nvidia-inst")
        self.root.geometry("600x550")

        self.distro: DistroInfo | None = None
        self.gpu: GPUInfo | None = None
        self.driver_range: DriverRange | None = None

        self._setup_ui()
        self._detect_hardware()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="wens")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self._add_distro_section(main_frame, 0)
        self._add_gpu_section(main_frame, 2)
        self._add_compat_section(main_frame, 4)
        self._add_buttons(main_frame, 6)
        self._add_log_section(main_frame, 8)

    def _add_distro_section(self, parent: ttk.Frame, row: int) -> None:
        """Add distribution info section."""
        ttk.Label(parent, text="Distribution", font=("Arial", 12, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5)
        )

        self.distro_label = ttk.Label(parent, text="Detecting...")
        self.distro_label.grid(row=row + 1, column=0, sticky=tk.W)

    def _add_gpu_section(self, parent: ttk.Frame, row: int) -> None:
        """Add GPU info section."""
        ttk.Label(parent, text="GPU", font=("Arial", 12, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5)
        )

        self.gpu_model_label = ttk.Label(parent, text="Detecting...")
        self.gpu_model_label.grid(row=row + 1, column=0, sticky=tk.W)

        self.gpu_cc_label = ttk.Label(parent, text="")
        self.gpu_cc_label.grid(row=row + 2, column=0, sticky=tk.W)

    def _add_compat_section(self, parent: ttk.Frame, row: int) -> None:
        """Add compatibility info section."""
        ttk.Label(parent, text="Compatibility", font=("Arial", 12, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5)
        )

        self.driver_range_label = ttk.Label(parent, text="Checking...")
        self.driver_range_label.grid(row=row + 1, column=0, sticky=tk.W)

        self.cuda_range_label = ttk.Label(parent, text="")
        self.cuda_range_label.grid(row=row + 2, column=0, sticky=tk.W)

        self.status_label = ttk.Label(parent, text="", foreground="green")
        self.status_label.grid(row=row + 3, column=0, sticky=tk.W)

    def _add_buttons(self, parent: ttk.Frame, row: int) -> None:
        """Add action buttons."""
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row, column=0, pady=15)

        self.install_btn = ttk.Button(
            btn_frame,
            text="Install Driver",
            command=self._on_install,
        )
        self.install_btn.pack(side=tk.LEFT, padx=5)

        self.check_btn = ttk.Button(
            btn_frame,
            text="Refresh",
            command=self._detect_hardware,
        )
        self.check_btn.pack(side=tk.LEFT, padx=5)

    def _add_log_section(self, parent: ttk.Frame, row: int) -> None:
        """Add log viewer section."""
        ttk.Label(parent, text="Logs", font=("Arial", 12, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5)
        )

        self.log_text = scrolledtext.ScrolledText(
            parent,
            height=8,
            width=70,
            wrap=tk.WORD,
            font=("Courier", 9),
        )
        self.log_text.grid(row=row + 1, column=0, sticky="we")

        parent.rowconfigure(row + 1, weight=1)

    def _detect_hardware(self) -> None:
        """Detect system hardware and update UI."""
        try:
            self.distro = detect_distro()
            self.distro_label.config(text=str(self.distro))
            self.log(f"Detected distro: {self.distro}")
        except DistroDetectionError as e:
            self.distro_label.config(text=f"Error: {e}")
            self.log(f"Error detecting distro: {e}")

        if not has_nvidia_gpu():
            self.gpu_model_label.config(text="No Nvidia GPU detected")
            self.log("No Nvidia GPU detected")
            return

        try:
            self.gpu = detect_gpu()
            if self.gpu:
                self.gpu_model_label.config(text=self.gpu.model)
                cc_text = f"Compute Capability: {self.gpu.compute_capability}"
                if self.gpu.vram:
                    cc_text += f" | VRAM: {self.gpu.vram}"
                self.gpu_cc_label.config(text=cc_text)
                self.log(f"Detected GPU: {self.gpu.model}")

                self.driver_range = get_driver_range(self.gpu)
                driver_text = f"Driver: {self.driver_range.min_version}"
                if self.driver_range.max_version:
                    driver_text += f" - {self.driver_range.max_version}"
                self.driver_range_label.config(text=driver_text)

                cuda_text = f"CUDA: {self.driver_range.cuda_min}"
                if self.driver_range.cuda_max:
                    cuda_text += f" - {self.driver_range.cuda_max}"
                self.cuda_range_label.config(text=cuda_text)

                status = (
                    "Limited (EOL GPU)" if self.driver_range.is_eol else "Compatible"
                )
                status_color = "orange" if self.driver_range.is_eol else "green"
                self.status_label.config(
                    text=f"Status: {status}", foreground=status_color
                )

                if self.driver_range.is_eol:
                    self.log(f"WARNING: {self.driver_range.eol_message}")

        except Exception as e:
            self.log(f"Error detecting GPU: {e}")

    def _on_install(self) -> None:
        """Handle install button click."""
        if not self.gpu:
            messagebox.showerror("Error", "No GPU detected")
            return

        if not self.driver_range:
            messagebox.showerror("Error", "Could not determine driver compatibility")
            return

        confirm = messagebox.askyesno(
            "Confirm Installation",
            f"Install Nvidia driver?\n\n"
            f"GPU: {self.gpu.model}\n"
            f"Driver: {self.driver_range.min_version}",
        )

        if not confirm:
            return

        if not require_root(interactive=True):
            messagebox.showerror("Error", "Root privileges required to install drivers")
            return

        self.install_btn.config(state=tk.DISABLED)
        self.log("Starting installation...")

        try:
            driver_version = self.driver_range.max_version
            if self.driver_range.is_eol and driver_version:
                self.log(f"Using EOL driver version: {driver_version}")

            result = install_driver_cli(
                driver_version=driver_version,
                with_cuda=True,
                skip_confirmation=True,
            )

            if result == 0:
                messagebox.showinfo(
                    "Success", "Driver installed successfully!\nPlease reboot."
                )
                self.log("Installation completed successfully")
            else:
                messagebox.showerror("Error", "Installation failed")
                self.log("Installation failed")

        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error", str(e))

        finally:
            self.install_btn.config(state=tk.NORMAL)

    def log(self, message: str) -> None:
        """Add message to log viewer."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)


def run_gui(args) -> int:
    """Run the Tkinter GUI.

    Args:
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    root = tk.Tk()
    NvidiaInstGUI(root)
    root.mainloop()
    return 0
