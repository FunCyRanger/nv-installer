"""System utilities for nvidia-inst."""

import glob
import os
import shutil

from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


def find_nvcc() -> str | None:
    """Find nvcc binary in system.

    Searches in order:
    1. PATH via shutil.which
    2. Common CUDA installation directories:
       - /usr/local/cuda*/bin/nvcc
       - /opt/cuda*/bin/nvcc
       - /usr/lib/cuda/bin/nvcc
       - /usr/local/lib/cuda/bin/nvcc

    Returns:
        Absolute path to nvcc if found, None otherwise.
    """
    # 1. Try PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        logger.debug(f"Found nvcc in PATH: {nvcc_path}")
        return nvcc_path

    # 2. Search common directories
    search_dirs = [
        "/usr/local/cuda*/bin",
        "/opt/cuda*/bin",
        "/usr/lib/cuda/bin",
        "/usr/local/lib/cuda/bin",
    ]

    for pattern in search_dirs:
        for bin_dir in glob.glob(pattern):
            candidate = os.path.join(bin_dir, "nvcc")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                logger.debug(f"Found nvcc at: {candidate}")
                return candidate

    logger.debug("nvcc not found in any known location")
    return None
