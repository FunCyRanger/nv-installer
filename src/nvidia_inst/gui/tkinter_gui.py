"""Tkinter GUI implementation."""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from nvidia_inst.cli import (
    DriverOption,
    DriverState,
    detect_driver_state,
    execute_driver_change,
)
from nvidia_inst.distro.detector import DistroDetectionError, DistroInfo, detect_distro
from nvidia_inst.gpu.compatibility import DriverRange, get_driver_range
from nvidia_inst.gpu.detector import GPUDetectionError, GPUInfo, detect_gpu
from nvidia_inst.utils.logger import get_logger
from nvidia_inst.utils.permissions import require_root

logger = get_logger(__name__)


class NvidiaInstGUI:
    """Main GUI application using Tkinter."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("nvidia-inst")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # System Info Tab
        info_frame = ttk.Frame(notebook, padding="10")
        notebook.add(info_frame, text="System Information")

        # Driver Info Tab
        driver_frame = ttk.Frame(notebook, padding="10")
        notebook.add(driver_frame, text="Driver Information")

        # Log Tab
        log_frame = ttk.Frame(notebook, padding="10")
        notebook.add(log_frame, text="Log")

        # System Info Tab Content
        ttk.Label(
            info_frame, text="System Information:", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        self.info_text = scrolledtext.ScrolledText(
            info_frame, wrap=tk.WORD, width=70, height=20
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_text.config(state=tk.DISABLED)

        # Driver Info Tab Content
        ttk.Label(
            driver_frame, text="Driver Information:", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        self.driver_text = scrolledtext.ScrolledText(
            driver_frame, wrap=tk.WORD, width=70, height=20
        )
        self.driver_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.driver_text.config(state=tk.DISABLED)

        # Log Tab Content
        ttk.Label(log_frame, text="Installation Log:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 5)
        )
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=70, height=20
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.action_btn = ttk.Button(
            button_frame, text="Select Action", command=self._on_select_action
        )
        self.action_btn.pack(side=tk.LEFT, padx=5)

        self.refresh_btn = ttk.Button(
            button_frame, text="Refresh", command=self._refresh_info
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        self.close_btn = ttk.Button(
            button_frame, text="Close", command=self.root.destroy
        )
        self.close_btn.pack(side=tk.RIGHT, padx=5)

        # Initialize variables
        self.gpu: GPUInfo | None = None
        self.driver_range: DriverRange | None = None
        self.distro: DistroInfo | None = None
        self.state: DriverState | None = None

        # Initial refresh
        self._refresh_info()

    def _refresh_info(self) -> None:
        """Refresh all information displayed in the GUI."""
        self.log("Refreshing system information...")

        # Clear existing content
        self._clear_text_widgets()

        try:
            self.distro = detect_distro()
        except DistroDetectionError as e:
            self.distro = None
            self.log(f"Failed to detect distribution: {e}")

        try:
            self.gpu = detect_gpu()
        except GPUDetectionError as e:
            self.gpu = None
            self.log(f"Failed to detect GPU: {e}")

        # Update system info tab
        if self.distro:
            self._append_to_text_widget(
                self.info_text, f"Distribution: {self.distro}\n"
            )
        else:
            self._append_to_text_widget(self.info_text, "Distribution: Not detected\n")

        if self.gpu:
            self._append_to_text_widget(self.info_text, f"GPU: {self.gpu.model}\n")
            if self.gpu.compute_capability:
                self._append_to_text_widget(
                    self.info_text,
                    f"Compute Capability: {self.gpu.compute_capability}\n",
                )
            if self.gpu.vram:
                self._append_to_text_widget(self.info_text, f"VRAM: {self.gpu.vram}\n")
        else:
            self._append_to_text_widget(self.info_text, "GPU: Not detected\n")

        # Update driver info tab
        if self.gpu and self.distro:
            try:
                self.driver_range = get_driver_range(self.gpu)
                if self.driver_range:
                    self._append_to_text_widget(
                        self.driver_text,
                        f"Compatible Driver Range: {self.driver_range.min_version}",
                    )
                    if self.driver_range.max_version:
                        self._append_to_text_widget(
                            self.driver_text, f" - {self.driver_range.max_version}"
                        )
                    else:
                        self._append_to_text_widget(self.driver_text, " or later")
                    self._append_to_text_widget(self.driver_text, "\n")

                    if self.driver_range.cuda_min:
                        self._append_to_text_widget(
                            self.driver_text,
                            f"CUDA Support: {self.driver_range.cuda_min}",
                        )
                        if self.driver_range.cuda_max:
                            self._append_to_text_widget(
                                self.driver_text, f" - {self.driver_range.cuda_max}"
                            )
                        else:
                            self._append_to_text_widget(self.driver_text, " or later")
                        self._append_to_text_widget(self.driver_text, "\n")

                    if self.driver_range.is_eol:
                        self._append_to_text_widget(
                            self.driver_text,
                            f"WARNING: {self.driver_range.eol_message}\n",
                        )
                else:
                    self._append_to_text_widget(
                        self.driver_text, "No compatible driver range found\n"
                    )
            except Exception as e:
                self._append_to_text_widget(
                    self.driver_text, f"Error getting driver range: {e}\n"
                )
        else:
            self._append_to_text_widget(
                self.driver_text,
                "Cannot determine driver range without GPU and distribution\n",
            )

        # Detect driver state if we have GPU and distro
        if self.gpu and self.distro and self.driver_range:
            try:
                self.state = detect_driver_state(
                    self.gpu, self.driver_range, self.distro.id
                )
                self.log(f"Driver state detected: {self.state.status.value}")
            except Exception as e:
                self.state = None
                self.log(f"Failed to detect driver state: {e}")
        else:
            self.state = None

        self.log("System information refreshed")

    def _clear_text_widgets(self) -> None:
        """Clear all text widgets."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.config(state=tk.DISABLED)

        self.driver_text.config(state=tk.NORMAL)
        self.driver_text.delete(1.0, tk.END)
        self.driver_text.config(state=tk.DISABLED)

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _append_to_text_widget(
        self, widget: scrolledtext.ScrolledText, text: str
    ) -> None:
        """Append text to a scrolledtext widget."""
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)
        widget.see(tk.END)

    def _log_to_gui(self, message: str) -> None:
        """Log a message to the GUI log tab."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def _show_options_window(self) -> DriverOption | None:
        """Show a window to select driver action.

        Returns:
            Selected DriverOption or None if cancelled.
        """
        if not self.state:
            messagebox.showerror("Error", "Driver state not available")
            return None

        # Capture value for type narrowing
        state = self.state

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Driver Action")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=state.message).pack(padx=10, pady=10)

        # Frame for radio buttons
        frame = ttk.Frame(dialog)
        frame.pack(padx=10, pady=10, fill="both", expand=True)

        var = tk.IntVar(value=-1)  # No selection by default

        for opt in state.options:
            text = f"[{opt.number}] {opt.description}"
            if opt.recommended:
                text += " [RECOMMENDED]"
            ttk.Radiobutton(frame, text=text, variable=var, value=opt.number).pack(
                anchor="w"
            )

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(padx=10, pady=10, fill="x")

        selected_option: DriverOption | None = None

        def on_ok():
            nonlocal selected_option
            selected_num = var.get()
            if selected_num == -1:
                messagebox.showwarning("Warning", "Please select an option")
                return
            # Find the option with the selected number
            for opt in state.options:
                if opt.number == selected_num:
                    selected_option = opt
                    break
            dialog.destroy()

        def on_cancel():
            nonlocal selected_option
            selected_option = None
            dialog.destroy()

        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right")

        # Wait for the window to be destroyed
        self.root.wait_window(dialog)

        return selected_option

    def _on_select_action(self) -> None:
        """Handle action button click."""
        if not self.gpu:
            messagebox.showerror("Error", "No GPU detected")
            return

        if not self.driver_range:
            messagebox.showerror("Error", "Could not determine driver compatibility")
            return

        if not self.state:
            messagebox.showerror("Error", "Could not determine driver state")
            return

        if not self.distro:
            messagebox.showerror("Error", "Could not determine distribution")
            return

        if not require_root(interactive=True):
            messagebox.showerror(
                "Error", "Root privileges required for driver operations"
            )
            return

        selected_option = self._show_options_window()
        if selected_option is None:
            return  # User cancelled

        self.action_btn.config(state=tk.DISABLED)
        self._log_to_gui(f"Selected option: {selected_option.description}")

        try:
            # Use local variables for type narrowing
            distro = self.distro
            gpu = self.gpu
            driver_range = self.driver_range
            state = self.state

            result = execute_driver_change(
                selected_option,
                state,
                distro,
                gpu,
                driver_range,
                dry_run=False,
            )
            if result == 0:
                messagebox.showinfo(
                    "Success",
                    "Operation completed successfully!\nPlease reboot your system if required.",
                )
                self._log_to_gui("Operation completed successfully")
            else:
                messagebox.showerror("Error", "Operation failed")
                self._log_to_gui("Operation failed")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log_to_gui(f"Error: {e}")
        finally:
            self.action_btn.config(state=tk.NORMAL)

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
