#!/usr/bin/env python3
"""Update Nvidia driver compatibility matrix from online sources.

This script fetches the latest driver versions from Nvidia's official archive
and updates the compatibility matrix data file.

Usage:
    python scripts/update-matrix.py [--dry-run] [--check] [--verify]
    python scripts/update-matrix.py --output FILE

Options:
    --dry-run     Show what would be updated without writing files
    --check       Check matrix status without updating
    --verify      Verify matrix data integrity
    --output      Output file path (default: compatibility_data.json)
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

NVIDIA_ARCHIVE_URL = "https://download.nvidia.com/XFree86/Linux-x86_64/"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update Nvidia driver compatibility matrix from online sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing files",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check matrix status without updating",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify matrix data integrity",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path",
    )

    return parser.parse_args()


def fetch_driver_versions() -> dict[str, str]:
    """Fetch latest version for each driver branch from Nvidia archive.

    Returns:
        Dictionary mapping branch number to latest version.
    """
    print(f"Fetching driver versions from {NVIDIA_ARCHIVE_URL}...")

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
            elif _compare_versions(full_version, branch_versions[major]) > 0:
                branch_versions[major] = full_version

        print(f"Found {len(branch_versions)} driver branches")
        return branch_versions

    except Exception as e:
        print(f"Error: Failed to fetch versions: {e}")
        return {}


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings."""
    def parse(v: str) -> tuple:
        parts = re.findall(r"\d+", v)
        return tuple(int(p) for p in parts[:3])

    p1, p2 = parse(v1), parse(v2)
    if p1 > p2:
        return 1
    if p1 < p2:
        return -1
    return 0


def verify_matrix(data: dict) -> list[str]:
    """Verify matrix data integrity.

    Args:
        data: Matrix data to verify.

    Returns:
        List of verification issues (empty if all OK).
    """
    issues = []

    if "_meta" not in data:
        issues.append("Missing _meta section")
    else:
        if "version" not in data["_meta"]:
            issues.append("Missing version in _meta")
        if "last_updated" not in data["_meta"]:
            issues.append("Missing last_updated in _meta")

    if "branches" not in data:
        issues.append("Missing branches section")
    else:
        for branch, info in data["branches"].items():
            if "latest" not in info:
                issues.append(f"Branch {branch} missing 'latest' version")
            if "name" not in info:
                issues.append(f"Branch {branch} missing 'name'")

    if "generations" not in data:
        issues.append("Missing generations section")
    else:
        required_gen_fields = ["name", "display_name", "branches", "status", "cuda"]
        for gen, info in data["generations"].items():
            for field in required_gen_fields:
                if field not in info:
                    issues.append(f"Generation '{gen}' missing field: {field}")

    return issues


def check_matrix_status(matrix_path: Path) -> None:
    """Check matrix status."""
    print(f"Checking matrix at {matrix_path}...")

    try:
        with open(matrix_path) as f:
            data = json.load(f)

        issues = verify_matrix(data)
        if issues:
            print("Verification FAILED:")
            for issue in issues:
                print(f"  - {issue}")
            return

        meta = data.get("_meta", {})
        print("\nMatrix Status:")
        print(f"  Version: {meta.get('version', 'unknown')}")
        print(f"  Last Updated: {meta.get('last_updated', 'unknown')}")
        print(f"  Generator: {meta.get('generator', 'unknown')}")

        if meta.get("sources"):
            print(f"  Sources: {len(meta['sources'])}")

        branches = data.get("branches", {})
        print(f"\n  Branches: {len(branches)}")
        for branch in sorted(branches.keys()):
            info = branches[branch]
            print(f"    {branch}: {info.get('latest', 'unknown')}")

        generations = data.get("generations", {})
        print(f"\n  GPU Generations: {len(generations)}")
        for gen in sorted(generations.keys()):
            info = generations[gen]
            status = info.get("status", "unknown")
            cuda = info.get("cuda", {}).get("min", "?")
            print(f"    {gen}: {info.get('display_name', gen)} ({status}, CUDA {cuda}+)")

        print("\nVerification PASSED")

    except FileNotFoundError:
        print(f"Error: Matrix file not found at {matrix_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")


def update_matrix(
    matrix_path: Path,
    dry_run: bool = False,
) -> bool:
    """Update matrix with latest driver versions.

    Args:
        matrix_path: Path to matrix data file.
        dry_run: If True, don't write changes.

    Returns:
        True if updated, False otherwise.
    """
    print(f"Loading matrix from {matrix_path}...")

    try:
        with open(matrix_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Matrix file not found at {matrix_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return False

    branch_versions = fetch_driver_versions()
    if not branch_versions:
        print("No versions fetched, keeping existing data")
        return False

    updated = False
    print("\nBranch comparison:")

    for branch, new_version in sorted(branch_versions.items()):
        old_info = data.get("branches", {}).get(branch, {})
        old_version = old_info.get("latest", "N/A")

        if old_version != new_version:
            print(f"  {branch}: {old_version} -> {new_version} [UPDATE]")
            updated = True
        else:
            print(f"  {branch}: {old_version} [same]")

    if not updated:
        print("\nMatrix is up to date, no changes needed")
        return False

    if dry_run:
        print("\n[DRY RUN] Would update matrix with changes above")
        return True

    for branch, new_version in branch_versions.items():
        if branch in data.get("branches", {}):
            data["branches"][branch]["latest"] = new_version

    data["_meta"]["last_updated"] = datetime.utcnow().isoformat() + "Z"

    content = json.dumps(data, sort_keys=True)
    data["_meta"]["version"] = hashlib.sha1(content.encode()).hexdigest()[:8]

    with open(matrix_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nMatrix updated successfully!")
    print(f"New version: {data['_meta']['version']}")
    return True


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.output:
        matrix_path = args.output
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        matrix_path = project_root / "src" / "nvidia_inst" / "gpu" / "matrix" / "compatibility_data.json"

    if args.verify:
        check_matrix_status(matrix_path)
        return 0

    if args.check:
        check_matrix_status(matrix_path)
        return 0

    if args.dry_run:
        updated = update_matrix(matrix_path, dry_run=True)
        return 0 if updated else 1

    updated = update_matrix(matrix_path, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
