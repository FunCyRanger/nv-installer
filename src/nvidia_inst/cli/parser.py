"""Argument parsing for nvidia-inst CLI.

This module provides command-line argument parsing functionality.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Cross-distribution Nvidia driver installer with CUDA support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI (Tkinter or Zenity)",
    )

    parser.add_argument(
        "--gui-type",
        choices=["tkinter", "zenity"],
        help="Force specific GUI type",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check compatibility only, don't install",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Install driver without CUDA",
    )

    parser.add_argument(
        "--cuda-version",
        help="Specific CUDA version to install",
    )

    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Show what would be installed without actually installing",
    )

    parser.add_argument(
        "--revert-to-nouveau",
        action="store_true",
        help="Switch from proprietary driver to Nouveau (open-source)",
    )

    parser.add_argument(
        "--power-profile",
        choices=["intel", "hybrid", "nvidia"],
        help="Set hybrid graphics power profile",
    )

    # Branch selection for Blackwell GPUs
    parser.add_argument(
        "--branch",
        choices=["470", "580", "590", "595"],
        help="Force specific driver branch (useful for Blackwell GPUs choosing between 590 and 595)",
    )

    # Rollback options
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous driver state",
    )

    parser.add_argument(
        "--list-snapshots",
        action="store_true",
        help="List available system snapshots for rollback",
    )

    # Offline installation options
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline package cache for installation",
    )

    parser.add_argument(
        "--create-cache",
        action="store_true",
        help="Download and cache packages for offline installation",
    )

    parser.add_argument(
        "--cache-dir",
        default="/var/cache/nvidia-inst",
        help="Directory for offline package cache (default: /var/cache/nvidia-inst)",
    )

    parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="Verify integrity of offline package cache",
    )

    return parser.parse_args()
