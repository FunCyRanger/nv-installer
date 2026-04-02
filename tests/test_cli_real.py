"""CLI integration tests that test real argument parsing.

These tests validate that the CLI parser works correctly with real arguments.
These tests do NOT run subprocess calls to avoid CI timeouts.
"""

import sys

import pytest


class TestParserAttributeValidation:
    """Test that parser creates expected attributes."""

    def test_simulate_attribute_exists(self):
        """Test args has simulate attribute when --simulate is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--simulate"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "simulate")
            assert args.simulate is True
        finally:
            sys.argv = old_argv

    def test_check_attribute_exists(self):
        """Test args has check attribute when --check is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--check"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "check")
            assert args.check is True
        finally:
            sys.argv = old_argv

    def test_revert_to_nouveau_attribute_exists(self):
        """Test args has revert_to_nouveau attribute when --revert-to-nouveau is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--revert-to-nouveau"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "revert_to_nouveau")
            assert args.revert_to_nouveau is True
        finally:
            sys.argv = old_argv

    def test_no_cuda_attribute_exists(self):
        """Test args has no_cuda attribute when --no-cuda is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--no-cuda"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "no_cuda")
            assert args.no_cuda is True
        finally:
            sys.argv = old_argv

    def test_yes_attribute_exists(self):
        """Test args has yes attribute when --yes is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--yes"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "yes")
            assert args.yes is True
        finally:
            sys.argv = old_argv

    def test_debug_attribute_exists(self):
        """Test args has debug attribute when --debug is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--debug"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "debug")
            assert args.debug is True
        finally:
            sys.argv = old_argv

    def test_power_profile_attribute_exists(self):
        """Test args has power_profile attribute when --power-profile is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--power-profile", "intel"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "power_profile")
            assert args.power_profile == "intel"
        finally:
            sys.argv = old_argv

    def test_cuda_version_attribute_exists(self):
        """Test args has cuda_version attribute when --cuda-version is used."""
        old_argv = sys.argv
        try:
            sys.argv = ["nv-install", "--cuda-version", "12.2"]
            from nvidia_inst.cli.parser import parse_args

            args = parse_args()
            assert hasattr(args, "cuda_version")
            assert args.cuda_version == "12.2"
        finally:
            sys.argv = old_argv
