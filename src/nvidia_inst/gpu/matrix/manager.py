"""Compatibility matrix manager with online update support."""

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

from nvidia_inst.gpu.matrix.data import (
    ComputeCapability,
    CUDARange,
    DriverBranchInfo,
    GPUGenerationInfo,
    SupportStatus,
)
from nvidia_inst.gpu.matrix.data import (
    get_branch_info as get_br_info,
)
from nvidia_inst.gpu.matrix.data import (
    get_generation_info as get_gen_info,
)
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)

NVIDIA_ARCHIVE_URL = "https://download.nvidia.com/XFree86/Linux-x86_64/"

CACHE_DIR = Path.home() / ".cache" / "nvidia-inst"
CACHE_FILE = CACHE_DIR / "matrix_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


class MatrixUpdateError(Exception):
    """Raised when matrix update fails."""


class MatrixManager:
    """Manages compatibility matrix with online/offline support."""

    def __init__(self, force_update: bool = False) -> None:
        """Initialize matrix manager.

        Args:
            force_update: If True, bypass cache and fetch fresh data.
        """
        self._force_update = force_update
        self._matrix_data: dict | None = None
        self._is_online: bool = False
        self._update_performed: bool = False

    def check_for_updates(self) -> tuple[bool, str]:
        """Check for matrix updates (non-blocking).

        Returns:
            Tuple of (updated, message).
        """
        try:
            self._ensure_matrix_loaded()
            if self._update_performed:
                if self._is_online:
                    return True, "Matrix updated from online sources"
                else:
                    return False, "Using fallback matrix (offline)"
            else:
                return False, "Matrix is up to date"
        except Exception as e:
            logger.warning(f"Matrix update check failed: {e}")
            return False, f"Using cached/fallback matrix: {e}"

    @property
    def is_online_data(self) -> bool:
        """Check if using online/fresh data."""
        return self._is_online

    @property
    def is_fallback(self) -> bool:
        """Check if using fallback data."""
        return self._matrix_data is not None and self._matrix_data.get("_meta", {}).get("is_fallback", False)

    def get_last_update_time(self) -> str | None:
        """Get when matrix was last updated."""
        if self._matrix_data:
            return self._matrix_data.get("_meta", {}).get("last_updated")
        return None

    def get_version(self) -> str:
        """Get matrix version."""
        if self._matrix_data:
            return self._matrix_data.get("_meta", {}).get("version", "unknown")
        return "unknown"

    def get_generation_info(self, generation: str) -> GPUGenerationInfo | None:
        """Get compatibility info for a GPU generation.

        Args:
            generation: GPU generation name (e.g., "ampere", "turing").

        Returns:
            GPUGenerationInfo or None if not found.
        """
        self._ensure_matrix_loaded()
        if not self._matrix_data:
            return get_gen_info(generation)

        gen_data = self._matrix_data.get("generations", {}).get(generation.lower())
        if not gen_data:
            return get_gen_info(generation)

        return _parse_generation_info(gen_data)

    def get_branch_info(self, branch: str) -> DriverBranchInfo | None:
        """Get info for a driver branch.

        Args:
            branch: Branch number (e.g., "590", "580").

        Returns:
            DriverBranchInfo or None if not found.
        """
        self._ensure_matrix_loaded()
        if not self._matrix_data:
            return get_br_info(branch)

        br_data = self._matrix_data.get("branches", {}).get(branch)
        if not br_data:
            return get_br_info(branch)

        return _parse_branch_info(branch, br_data)

    def get_all_branches(self) -> dict[str, DriverBranchInfo]:
        """Get all driver branches."""
        self._ensure_matrix_loaded()
        result = {}

        if self._matrix_data:
            for branch, data in self._matrix_data.get("branches", {}).items():
                result[branch] = _parse_branch_info(branch, data)

        if not result:
            for _, info in get_br_info.__module__.__dict__.items():
                if isinstance(info, DriverBranchInfo):
                    result[info.number] = info

        return result

    def get_all_generations(self) -> dict[str, GPUGenerationInfo]:
        """Get all GPU generations."""
        self._ensure_matrix_loaded()
        result = {}

        if self._matrix_data:
            for name, data in self._matrix_data.get("generations", {}).items():
                info = _parse_generation_info(data)
                if info:
                    result[name] = info

        if not result:
            for _, info in get_gen_info.__module__.__dict__.items():
                if isinstance(info, GPUGenerationInfo):
                    result[info.name] = info

        return result

    def _ensure_matrix_loaded(self) -> None:
        """Ensure matrix data is loaded."""
        if self._matrix_data is None:
            self._load_matrix()

    def _load_matrix(self) -> None:
        """Load matrix with fallback logic."""
        if self._force_update:
            self._invalidate_cache()

        if not self._force_update:
            cached = self._load_from_cache()
            if cached:
                self._matrix_data = cached
                self._is_online = False
                return

        try:
            online_data = self._fetch_online_matrix()
            if online_data:
                self._matrix_data = online_data
                self._is_online = True
                self._update_performed = True
                self._save_to_cache(online_data)
                return
        except Exception as e:
            logger.debug(f"Online fetch failed: {e}")

        fallback = self._load_fallback_matrix()
        self._matrix_data = fallback
        self._is_online = False
        self._update_performed = False

    def _fetch_online_matrix(self) -> dict | None:
        """Fetch fresh matrix data from Nvidia archive.

        Returns:
            Matrix dict or None if fetch fails.
        """
        logger.info("Fetching compatibility matrix from Nvidia archive...")

        try:
            import urllib.request

            request = urllib.request.Request(
                NVIDIA_ARCHIVE_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8")

            versions = re.findall(r'href="(\d+)\.(\d+)\.(\d+)/"', html)

            branch_versions: dict[str, str] = {}
            for version in versions:
                major = version[0]
                full_version = f"{version[0]}.{version[1]}.{version[2]}"
                if major not in branch_versions:
                    branch_versions[major] = full_version
                else:
                    if _compare_versions(full_version, branch_versions[major]) > 0:
                        branch_versions[major] = full_version

            fallback = self._load_fallback_matrix()
            updated = False

            for branch, new_version in branch_versions.items():
                old_version = fallback.get("branches", {}).get(branch, {}).get("latest")
                if old_version and new_version != old_version:
                    logger.info(f"New version for branch {branch}: {old_version} -> {new_version}")
                    updated = True

            if updated or self._force_update:
                for branch, version in branch_versions.items():
                    if branch in fallback.get("branches", {}):
                        fallback["branches"][branch]["latest"] = version

                fallback["_meta"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
                fallback["_meta"]["is_fallback"] = False

                content = json.dumps(fallback, sort_keys=True)
                fallback["_meta"]["version"] = hashlib.sha1(content.encode()).hexdigest()[:8]

                logger.info("Matrix updated with new driver versions")
                return fallback
            else:
                logger.info("Matrix is up to date")
                fallback["_meta"]["is_fallback"] = False
                return fallback

        except Exception as e:
            raise MatrixUpdateError(f"Failed to fetch online matrix: {e}") from e

    def _load_from_cache(self) -> dict | None:
        """Load matrix from local cache."""
        if not CACHE_FILE.exists():
            return None

        cache_age = time.time() - CACHE_FILE.stat().st_mtime
        if cache_age > CACHE_TTL_SECONDS:
            logger.debug(f"Cache expired (age: {cache_age / 3600:.1f}h)")
            return None

        try:
            with open(CACHE_FILE) as f:
                data = json.load(f)
            data["_meta"]["is_fallback"] = False
            logger.debug("Using cached matrix")
            return data
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, data: dict) -> None:
        """Save matrix to cache file."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_data = {k: v for k, v in data.items() if not k.startswith("_")}

        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Matrix saved to cache: {CACHE_FILE}")

    def _load_fallback_matrix(self) -> dict:
        """Load fallback matrix from committed JSON."""
        fallback_path = Path(__file__).parent / "compatibility_data.json"

        try:
            with open(fallback_path) as f:
                data = json.load(f)
            data["_meta"]["is_fallback"] = True
            return data
        except Exception as e:
            logger.error(f"Failed to load fallback matrix: {e}")
            return {"_meta": {}, "branches": {}, "generations": {}}

    def _invalidate_cache(self) -> None:
        """Clear the matrix cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        logger.debug("Matrix cache invalidated")


def _parse_generation_info(data: dict) -> GPUGenerationInfo | None:
    """Parse generation data into GPUGenerationInfo."""
    try:
        cuda_data = data.get("cuda", {})
        cc_data = data.get("compute_capability", {})

        status_str = data.get("status", "full")
        try:
            status = SupportStatus(status_str)
        except ValueError:
            status = SupportStatus.FULL

        return GPUGenerationInfo(
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            compute_cap=ComputeCapability(
                min=cc_data.get("min", 0.0),
                max=cc_data.get("max", 0.0),
            ),
            cuda=CUDARange(
                min_version=cuda_data.get("min", "11.0"),
                max_version=cuda_data.get("max"),
                recommended=cuda_data.get("recommended", "12.2"),
            ),
            branches=data.get("branches", []),
            status=status,
            min_driver=data.get("min_driver", "520.56.06"),
            max_driver=data.get("max_driver"),
            eol_message=data.get("eol_message"),
        )
    except Exception as e:
        logger.warning(f"Failed to parse generation info: {e}")
        return None


def _parse_branch_info(branch: str, data: dict) -> DriverBranchInfo | None:
    """Parse branch data into DriverBranchInfo."""
    try:
        return DriverBranchInfo(
            number=branch,
            name=data.get("name", f"Branch {branch}"),
            latest_version=data.get("latest", "0.0.0"),
            release_date=data.get("release_date", ""),
            eol_date=data.get("eol_date"),
            gpu_generations=data.get("gpu_generations", []),
        )
    except Exception as e:
        logger.warning(f"Failed to parse branch info: {e}")
        return None


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Returns:
        1 if v1 > v2, -1 if v1 < v2, 0 if equal.
    """
    def parse(v: str) -> tuple:
        parts = re.findall(r"\d+", v)
        return tuple(int(p) for p in parts[:3])

    p1, p2 = parse(v1), parse(v2)
    if p1 > p2:
        return 1
    if p1 < p2:
        return -1
    return 0


def update_matrix(force: bool = False) -> tuple[bool, str]:
    """Update the compatibility matrix.

    Args:
        force: Force update even if cache is fresh.

    Returns:
        Tuple of (success, message).
    """
    manager = MatrixManager(force_update=force)
    updated, message = manager.check_for_updates()
    return updated, message
