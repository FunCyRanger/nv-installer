"""Tests for cli/__main__.py module."""


class TestMainModule:
    """Tests for __main__.py entry point."""

    def test_main_module_can_be_imported(self):
        """Test that __main__.py can be imported."""
        from nvidia_inst.cli import __main__ as cli_main

        assert hasattr(cli_main, "main")

    def test_main_module_has_main_function(self):
        """Test that __main__.py has a main function."""
        from nvidia_inst.cli.__main__ import main

        assert callable(main)
