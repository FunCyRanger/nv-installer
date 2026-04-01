"""Tests for permission utilities."""

from unittest.mock import MagicMock, patch

from nvidia_inst.utils import permissions


class TestIsRoot:
    """Tests for is_root function."""

    @patch("os.geteuid", return_value=0)
    def test_is_root_true(self, mock_geteuid):
        """Test is_root returns True when running as root."""
        # Reset the cached value
        permissions._sudo_cached = None
        assert permissions.is_root() is True

    @patch("os.geteuid", return_value=1000)
    def test_is_root_false(self, mock_geteuid):
        """Test is_root returns False when not running as root."""
        assert permissions.is_root() is False


class TestHaveSudo:
    """Tests for have_sudo function."""

    def setup_method(self):
        """Reset cached sudo value before each test."""
        permissions._sudo_cached = None

    @patch("subprocess.run")
    def test_have_sudo_cached_true(self, mock_run):
        """Test have_sudo returns cached value when True."""
        permissions._sudo_cached = True
        assert permissions.have_sudo() is True
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_have_sudo_cached_false(self, mock_run):
        """Test have_sudo returns cached value when False."""
        permissions._sudo_cached = False
        assert permissions.have_sudo() is False
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_have_sudo_succeeds(self, mock_run):
        """Test have_sudo returns True when sudo succeeds."""
        mock_run.return_value = MagicMock(returncode=0)
        assert permissions.have_sudo() is True

    @patch("subprocess.run")
    def test_have_sudo_fails(self, mock_run):
        """Test have_sudo returns False when sudo fails."""
        mock_run.return_value = MagicMock(returncode=1)
        assert permissions.have_sudo() is False

    @patch("subprocess.run")
    def test_have_sudo_exception(self, mock_run):
        """Test have_sudo returns False on exception."""
        mock_run.side_effect = Exception("Error")
        assert permissions.have_sudo() is False


class TestRequireRoot:
    """Tests for require_root function."""

    def setup_method(self):
        """Reset cached sudo value before each test."""
        permissions._sudo_cached = None

    @patch("nvidia_inst.utils.permissions.is_root", return_value=True)
    def test_require_root_already_root(self, mock_is_root):
        """Test require_root returns True when already root."""
        assert permissions.require_root() is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    def test_require_root_cached(self, mock_is_root):
        """Test require_root returns True when sudo is cached."""
        permissions._sudo_cached = True
        assert permissions.require_root() is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    def test_require_root_non_interactive(self, mock_is_root):
        """Test require_root returns False when non-interactive and not root."""
        assert permissions.require_root(interactive=False) is False

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    @patch("subprocess.run")
    def test_require_root_sudo_succeeds(self, mock_run, mock_is_root):
        """Test require_root returns True when sudo succeeds."""
        mock_run.return_value = MagicMock(returncode=0)
        assert permissions.require_root() is True

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    @patch("subprocess.run")
    def test_require_root_sudo_fails(self, mock_run, mock_is_root):
        """Test require_root returns False when sudo fails."""
        mock_run.return_value = MagicMock(returncode=1)
        assert permissions.require_root() is False

    @patch("nvidia_inst.utils.permissions.is_root", return_value=False)
    @patch("subprocess.run")
    def test_require_root_sudo_exception(self, mock_run, mock_is_root):
        """Test require_root returns False on sudo exception."""
        mock_run.side_effect = Exception("Error")
        assert permissions.require_root() is False


class TestCheckRootRequired:
    """Tests for check_root_required function."""

    @patch("nvidia_inst.utils.permissions.require_root", return_value=True)
    def test_check_root_required_has_root(self, mock_require_root):
        """Test returns 0 when root is available."""
        result = permissions.check_root_required("test operation")
        assert result == 0

    @patch("nvidia_inst.utils.permissions.require_root", return_value=False)
    def test_check_root_required_no_root(self, mock_require_root, capsys):
        """Test returns 1 and prints message when no root."""
        result = permissions.check_root_required("test operation")
        assert result == 1
        captured = capsys.readouterr()
        assert "Root privileges required" in captured.out
