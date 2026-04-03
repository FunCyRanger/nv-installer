"""CLI module for nvidia-inst.

This module provides command-line interface functionality including:
- Argument parsing
- Driver state detection
- Installation orchestration
- Compatibility checking
- GUI launcher functionality
"""

# Re-export from parser
# Re-export from commands
from nvidia_inst.cli.commands import (
    detect_dnf_path,
    format_install_command,
    format_remove_command,
    format_update_command,
    get_cuda_lock_command,
    get_cuda_unlock_command,
    get_driver_lock_command,
    get_driver_unlock_command,
    get_initramfs_command,
    get_nouveau_remove_command,
    sudo_path,
)

# Re-export from compatibility
from nvidia_inst.cli.compatibility import (
    check_compatibility,
    check_prerequisites,
    print_compatibility_info,
    print_version_check,
)

# Re-export from display
from nvidia_inst.cli.display import (
    format_package_list,
    print_distro_info,
    print_driver_status,
    print_error,
    print_gpu_info,
    print_info,
    print_row,
    print_section_header,
    print_step,
    print_success,
    print_warning,
)

# Re-export from driver_state
from nvidia_inst.cli.driver_state import (
    DriverOption,
    DriverState,
    DriverStatus,
    detect_driver_state,
    show_driver_options,
)

# Re-export from installer
from nvidia_inst.cli.installer import (
    InstallResult,
    get_packages_to_remove,
    install_cuda_packages,
    install_driver_packages,
    prompt_reboot,
    rebuild_initramfs,
    remove_packages,
)

# Re-export from main.py for backward compatibility
from nvidia_inst.cli.main import (
    execute_driver_change,
    handle_secure_boot,
    install_driver_cli,
    main,
    revert_to_nouveau_cli,
    set_power_profile_cli,
)
from nvidia_inst.cli.parser import parse_args

# Re-export from simulate
from nvidia_inst.cli.simulate import (
    simulate_change,
    simulate_nouveau_install,
    simulate_nvidia_open_install,
    simulate_revert,
)

# Re-export from gpu.detector for tests
from nvidia_inst.gpu.detector import has_nvidia_gpu

# Re-export from gui
from nvidia_inst.gui import detect_gui_type, launch_gui

# Re-export from installer.driver for tests
from nvidia_inst.installer.driver import get_current_driver_type

__all__ = [
    # From parser
    "parse_args",
    # From driver_state
    "DriverStatus",
    "DriverOption",
    "DriverState",
    "detect_driver_state",
    "show_driver_options",
    # From compatibility
    "check_compatibility",
    "check_prerequisites",
    "print_compatibility_info",
    "print_version_check",
    # From installer
    "InstallResult",
    "get_packages_to_remove",
    "remove_packages",
    "rebuild_initramfs",
    "install_driver_packages",
    "install_cuda_packages",
    "prompt_reboot",
    # From commands
    "detect_dnf_path",
    "sudo_path",
    "get_nouveau_remove_command",
    "get_initramfs_command",
    "get_driver_lock_command",
    "get_driver_unlock_command",
    "get_cuda_lock_command",
    "get_cuda_unlock_command",
    "format_install_command",
    "format_update_command",
    "format_remove_command",
    # From simulate
    "simulate_generic",
    "simulate_change",
    "simulate_nvidia_open_install",
    "simulate_nouveau_install",
    "simulate_revert",
    # From display
    "print_row",
    "print_section_header",
    "print_step",
    "print_warning",
    "print_error",
    "print_info",
    "print_success",
    "format_package_list",
    "print_driver_status",
    "print_gpu_info",
    "print_distro_info",
    # From gui
    "launch_gui",
    "detect_gui_type",
    # From gpu.detector
    "has_nvidia_gpu",
    # From installer.driver
    "get_current_driver_type",
    # From main.py
    "main",
    "install_driver_cli",
    "execute_driver_change",
    "revert_to_nouveau_cli",
    "set_power_profile_cli",
    "handle_secure_boot",
]
