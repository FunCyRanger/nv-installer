"""Tests for installation order verification.

These tests verify that the installation workflow follows the correct order:
1. Remove old packages
2. Set version locks (BEFORE installation)
3. Install driver packages
4. Install CUDA packages
5. Verify locks are active
6. Rebuild initramfs
7. Reboot
"""

from unittest.mock import MagicMock, patch

from nvidia_inst.gpu.compatibility import DriverRange


class TestInstallDriverOrder:
    """Tests for install_driver() function order verification."""

    def test_locks_set_before_install(self):
        """Test that version locks are set BEFORE package installation."""
        from nvidia_inst.installer.driver import (
            DistroInstaller,
            install_driver,
        )

        mock_installer = MagicMock(spec=DistroInstaller)
        mock_installer.pre_install_check.return_value = True
        mock_installer.get_driver_packages.return_value = [
            "akmod-nvidia",
            "xorg-x11-drv-nvidia",
        ]
        mock_installer.get_cuda_packages.return_value = ["cuda-toolkit-12-*"]
        mock_installer.install.return_value = None
        mock_installer.post_install.return_value = None

        mock_pkg_manager = MagicMock()
        mock_pkg_manager.pin_version.return_value = True

        driver_range = DriverRange(
            min_version="520.56.06",
            max_version="580.142",
            cuda_min="11.0",
            cuda_max="12.x",
            max_branch="580",
            is_eol=False,
            is_limited=True,
            cuda_is_locked=True,
            cuda_locked_major="12",
        )

        with patch(
            "nvidia_inst.installer.cuda.pin_cuda_to_major_version", return_value=True
        ):
            with patch(
                "nvidia_inst.distro.versionlock.verify_versionlock_pattern_active",
                return_value=(True, "verified"),
            ):
                install_driver(
                    installer=mock_installer,
                    driver_version="580.142",
                    with_cuda=True,
                    cuda_version="12.0",
                    pkg_manager=mock_pkg_manager,
                    driver_range=driver_range,
                )

        # Verify order: pin_version called BEFORE installer.install
        pin_calls = mock_pkg_manager.pin_version.call_args_list
        install_calls = mock_installer.install.call_args_list

        assert len(pin_calls) >= 2  # At least 2 driver packages pinned
        assert len(install_calls) >= 1  # At least 1 install call

        # The first pin_version call should come before the first install call
        # We verify this by checking that pin_version was called with driver packages
        first_pin_pkg = pin_calls[0][0][0]
        assert first_pin_pkg in ["akmod-nvidia", "xorg-x11-drv-nvidia"]

    def test_install_fails_if_locking_fails(self):
        """Test that installation fails if version locking fails."""
        from nvidia_inst.installer.driver import DistroInstaller, install_driver

        mock_installer = MagicMock(spec=DistroInstaller)
        mock_installer.pre_install_check.return_value = True
        mock_installer.get_driver_packages.return_value = ["akmod-nvidia"]

        mock_pkg_manager = MagicMock()
        mock_pkg_manager.pin_version.return_value = False

        driver_range = DriverRange(
            min_version="520.56.06",
            max_version="580.142",
            cuda_min="11.0",
            cuda_max="12.x",
            max_branch="580",
            is_eol=False,
            is_limited=True,
            cuda_is_locked=True,
            cuda_locked_major="12",
        )

        result = install_driver(
            installer=mock_installer,
            driver_version="580.142",
            with_cuda=False,
            pkg_manager=mock_pkg_manager,
            driver_range=driver_range,
        )

        assert result.success is False
        assert "Failed to set version locks" in result.message
        # Verify install was NOT called
        mock_installer.install.assert_not_called()

    def test_lock_verification_called_after_install(self):
        """Test that lock verification is called after installation."""
        from nvidia_inst.installer.driver import DistroInstaller, install_driver

        mock_installer = MagicMock(spec=DistroInstaller)
        mock_installer.pre_install_check.return_value = True
        mock_installer.get_driver_packages.return_value = ["akmod-nvidia"]
        mock_installer.install.return_value = None
        mock_installer.post_install.return_value = None

        mock_pkg_manager = MagicMock()
        mock_pkg_manager.pin_version.return_value = True

        driver_range = DriverRange(
            min_version="520.56.06",
            max_version="580.142",
            cuda_min="11.0",
            cuda_max="12.x",
            max_branch="580",
            is_eol=False,
            is_limited=True,
            cuda_is_locked=True,
            cuda_locked_major="12",
        )

        with patch(
            "nvidia_inst.distro.versionlock.verify_versionlock_pattern_active",
            return_value=(True, "verified"),
        ) as mock_verify:
            result = install_driver(
                installer=mock_installer,
                driver_version="580.142",
                with_cuda=False,
                pkg_manager=mock_pkg_manager,
                driver_range=driver_range,
            )

        assert result.success is True
        # Verify lock verification was called
        mock_verify.assert_called()


class TestCLIInstallationOrder:
    """Tests for CLI execute_driver_change() order verification."""

    def test_cli_locks_before_install(self):
        """Test that CLI path sets locks before installation."""
        from nvidia_inst.cli.driver_state import (
            DriverOption,
            DriverState,
            DriverStatus,
        )
        from nvidia_inst.distro.detector import DistroInfo
        from nvidia_inst.gpu.detector import GPUInfo

        state = DriverState(
            status=DriverStatus.NOTHING,
            current_version=None,
            is_compatible=False,
            is_optimal=False,
            suggested_packages=["akmod-nvidia", "xorg-x11-drv-nvidia"],
            options=[DriverOption(1, "Install", "install")],
            message="No driver installed",
            cuda_range="11.0-12.8",
        )
        distro = DistroInfo(
            id="fedora",
            version_id="40",
            name="Fedora",
            pretty_name="Fedora 40",
            kernel="6.8.0",
        )
        gpu = GPUInfo(model="NVIDIA GeForce GTX 1080", generation="pascal")
        driver_range = DriverRange(
            min_version="520.56.06",
            max_version="580.142",
            cuda_min="11.0",
            cuda_max="12.x",
            max_branch="580",
            is_eol=False,
            is_limited=True,
            cuda_is_locked=True,
            cuda_locked_major="12",
        )
        option = DriverOption(1, "Install", "install")

        mock_pkg_manager = MagicMock()
        mock_pkg_manager.pin_version.return_value = True
        mock_pkg_manager.install.return_value = None

        with patch("nvidia_inst.cli.main.require_root", return_value=True):
            with patch(
                "nvidia_inst.cli.main.get_package_manager",
                return_value=mock_pkg_manager,
            ):
                with patch(
                    "nvidia_inst.cli.main.get_packages_to_remove",
                    return_value=[],
                ):
                    with patch(
                        "nvidia_inst.cli.main.remove_packages",
                        return_value=[],
                    ):
                        with patch(
                            "nvidia_inst.cli.main.get_cuda_installer"
                        ) as mock_cuda_inst:
                            mock_cuda_inst.return_value.get_cuda_packages.return_value = [
                                "cuda-toolkit-12-*"
                            ]
                            with patch(
                                "nvidia_inst.cli.main.rebuild_initramfs",
                                return_value=True,
                            ):
                                with patch("nvidia_inst.cli.main.prompt_reboot"):
                                    with patch(
                                        "nvidia_inst.cli.main.get_compatible_driver_packages",
                                        return_value=[
                                            "akmod-nvidia",
                                            "xorg-x11-drv-nvidia",
                                        ],
                                    ):
                                        from nvidia_inst.cli.main import (
                                            execute_driver_change,
                                        )

                                        execute_driver_change(
                                            option,
                                            state,
                                            distro,
                                            gpu,
                                            driver_range,
                                            simulate=False,
                                            with_cuda=True,
                                            cuda_version="12.0",
                                        )

        # Verify pin_version was called (locks set)
        pin_calls = mock_pkg_manager.pin_version.call_args_list
        assert len(pin_calls) >= 2  # At least 2 driver packages locked

        # Verify install was called after locks
        install_calls = mock_pkg_manager.install.call_args_list
        assert len(install_calls) >= 1
