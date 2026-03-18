"""Nvidia driver compatibility matrix.

This package provides GPU/driver compatibility data sourced from official
Nvidia documentation, with support for online updates and offline fallback.

Usage:
    from nvidia_inst.gpu.matrix import MatrixManager

    manager = MatrixManager()
    info = manager.get_generation_info("ampere")
"""

from nvidia_inst.gpu.matrix.data import (
    ComputeCapability,
    CUDARange,
    DriverBranchInfo,
    GPUGenerationInfo,
    MatrixMeta,
    SupportStatus,
)
from nvidia_inst.gpu.matrix.manager import MatrixManager

__all__ = [
    "CUDARange",
    "ComputeCapability",
    "GPUGenerationInfo",
    "DriverBranchInfo",
    "MatrixMeta",
    "SupportStatus",
    "MatrixManager",
]
