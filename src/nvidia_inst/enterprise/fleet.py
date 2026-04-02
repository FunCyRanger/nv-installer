"""Fleet management for enterprise deployments.

This module provides functionality for managing NVIDIA drivers across
multiple machines using Ansible, Puppet, or other configuration management tools.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nvidia_inst.distro.detector import detect_distro
from nvidia_inst.gpu.compatibility import DriverRange
from nvidia_inst.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FleetConfig:
    """Configuration for fleet deployment."""

    driver_branch: str = "590"
    cuda_version: str | None = None
    install_cuda: bool = True
    hosts: list[str] = field(default_factory=list)
    ssh_user: str = "root"
    ssh_port: int = 22
    sudo_password: str | None = None
    extra_vars: dict[str, Any] = field(default_factory=dict)


class FleetManager:
    """Manage NVIDIA drivers across multiple machines."""

    def __init__(self, config: FleetConfig | None = None):
        """Initialize fleet manager.

        Args:
            config: Fleet configuration
        """
        self.config = config or FleetConfig()

    def generate_ansible_playbook(
        self,
        output_path: str | None = None,
    ) -> str:
        """Generate Ansible playbook for fleet deployment.

        Args:
            output_path: Path to save playbook (optional)

        Returns:
            Playbook YAML string
        """
        playbook = [
            {
                "name": "NVIDIA Driver Installation",
                "hosts": self.config.hosts if self.config.hosts else "all",
                "become": True,
                "vars": {
                    "nvidia_driver_branch": self.config.driver_branch,
                    "nvidia_cuda_version": self.config.cuda_version,
                    "nvidia_install_cuda": self.config.install_cuda,
                },
                "tasks": [
                    {
                        "name": "Install nvidia-inst",
                        "ansible.builtin.pip": {
                            "name": "nvidia-inst",
                            "state": "present",
                        },
                    },
                    {
                        "name": "Run nvidia-inst check",
                        "ansible.builtin.command": "nvidia-inst --check",
                        "register": "check_result",
                        "changed_when": False,
                    },
                    {
                        "name": "Display check results",
                        "ansible.builtin.debug": {
                            "var": "check_result.stdout",
                        },
                    },
                    {
                        "name": "Install NVIDIA driver",
                        "ansible.builtin.command": {
                            "cmd": "nvidia-inst --yes",
                            "creates": "/usr/bin/nvidia-smi",
                        },
                        "when": "check_result.rc == 0",
                    },
                    {
                        "name": "Verify driver installation",
                        "ansible.builtin.command": "nvidia-smi",
                        "register": "nvidia_smi_result",
                        "changed_when": False,
                        "failed_when": "nvidia_smi_result.rc != 0",
                    },
                ],
            }
        ]

        import yaml  # type: ignore[import-untyped]

        playbook_yaml = yaml.dump(playbook, default_flow_style=False)

        if output_path:
            Path(output_path).write_text(playbook_yaml)
            logger.info(f"Playbook saved to {output_path}")

        return str(playbook_yaml)

    def generate_ansible_inventory(
        self,
        hosts: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> str:
        """Generate Ansible inventory file.

        Args:
            hosts: List of host configurations
            output_path: Path to save inventory (optional)

        Returns:
            Inventory INI string
        """
        inventory_lines = ["[nvidia_hosts]"]

        for host in hosts:
            line = host.get("hostname", host.get("name", "unknown"))
            if "ansible_host" in host:
                line += f" ansible_host={host['ansible_host']}"
            if "ansible_user" in host:
                line += f" ansible_user={host['ansible_user']}"
            if "ansible_port" in host:
                line += f" ansible_port={host['ansible_port']}"
            inventory_lines.append(line)

        inventory = "\n".join(inventory_lines)

        if output_path:
            Path(output_path).write_text(inventory)
            logger.info(f"Inventory saved to {output_path}")

        return inventory

    def generate_puppet_manifest(
        self,
        output_path: str | None = None,
    ) -> str:
        """Generate Puppet manifest for fleet deployment.

        Args:
            output_path: Path to save manifest (optional)

        Returns:
            Puppet manifest string
        """
        manifest = f"""# NVIDIA Driver Installation Manifest
# Generated by nvidia-inst

class nvidia_driver {{
  $driver_branch = '{self.config.driver_branch}'
  $cuda_version = '{self.config.cuda_version or "latest"}'

  # Install nvidia-inst
  package {{ 'nvidia-inst':
    ensure   => present,
    provider => 'pip',
  }}

  # Run nvidia-inst
  exec {{ 'install-nvidia-driver':
    command     => 'nvidia-inst --yes',
    path        => ['/usr/bin', '/usr/local/bin'],
    unless      => 'nvidia-smi',
    require     => Package['nvidia-inst'],
    environment => [
      "NVIDIA_DRIVER_BRANCH=$driver_branch",
      "NVIDIA_CUDA_VERSION=$cuda_version",
    ],
  }}

  # Verify installation
  exec {{ 'verify-nvidia-driver':
    command => 'nvidia-smi',
    path    => ['/usr/bin', '/usr/local/bin'],
    unless  => 'nvidia-smi',
    require => Exec['install-nvidia-driver'],
  }}
}}

# Apply to all nodes with nvidia_gpu fact
node default {{
  if $facts['nvidia_gpu'] {{
    include nvidia_driver
  }}
}}
"""

        if output_path:
            Path(output_path).write_text(manifest)
            logger.info(f"Manifest saved to {output_path}")

        return manifest

    def generate_salt_state(
        self,
        output_path: str | None = None,
    ) -> str:
        """Generate Salt state file for fleet deployment.

        Args:
            output_path: Path to save state file (optional)

        Returns:
            Salt state YAML string
        """
        state = f"""# NVIDIA Driver Installation State
# Generated by nvidia-inst

nvidia-driver:
  pip.installed:
    - name: nvidia-inst

install-nvidia-driver:
  cmd.run:
    - name: nvidia-inst --yes
    - unless: nvidia-smi
    - require:
      - pip: nvidia-driver
    - env:
      - NVIDIA_DRIVER_BRANCH: '{self.config.driver_branch}'
      - NVIDIA_CUDA_VERSION: '{self.config.cuda_version or "latest"}'

verify-nvidia-driver:
  cmd.run:
    - name: nvidia-smi
    - require:
      - cmd: install-nvidia-driver
"""

        if output_path:
            Path(output_path).write_text(state)
            logger.info(f"Salt state saved to {output_path}")

        return state

    def generate_cloud_init(
        self,
        output_path: str | None = None,
    ) -> str:
        """Generate cloud-init configuration for fleet deployment.

        Args:
            output_path: Path to save cloud-init config (optional)

        Returns:
            Cloud-init YAML string
        """
        cloud_init = f"""#cloud-config
# NVIDIA Driver Installation
# Generated by nvidia-inst

package_update: true
package_upgrade: true

packages:
  - python3-pip

runcmd:
  - pip3 install nvidia-inst
  - nvidia-inst --yes --driver-branch {self.config.driver_branch}
  - nvidia-smi

final_message: "NVIDIA driver installation completed"
"""

        if output_path:
            Path(output_path).write_text(cloud_init)
            logger.info(f"Cloud-init config saved to {output_path}")

        return cloud_init

    def generate_terraform_config(
        self,
        output_path: str | None = None,
    ) -> str:
        """Generate Terraform configuration for fleet deployment.

        Args:
            output_path: Path to save Terraform config (optional)

        Returns:
            Terraform HCL string
        """
        terraform = f"""# NVIDIA Driver Installation Terraform Config
# Generated by nvidia-inst

resource "null_resource" "nvidia_driver" {{
  count = length(var.instances)

  connection {{
    type        = "ssh"
    host        = var.instances[count.index]
    user        = "{self.config.ssh_user}"
    port        = {self.config.ssh_port}
    private_key = file(var.ssh_private_key_path)
  }}

  provisioner "remote-exec" {{
    inline = [
      "pip3 install nvidia-inst",
      "nvidia-inst --yes --driver-branch {self.config.driver_branch}",
      "nvidia-smi",
    ]
  }}
}}

variable "instances" {{
  description = "List of instance IPs"
  type        = list(string)
}}

variable "ssh_private_key_path" {{
  description = "Path to SSH private key"
  type        = string
}}
"""

        if output_path:
            Path(output_path).write_text(terraform)
            logger.info(f"Terraform config saved to {output_path}")

        return terraform

    def generate_deployment_package(
        self,
        output_dir: str,
        formats: list[str] | None = None,
    ) -> dict[str, str]:
        """Generate deployment package with multiple formats.

        Args:
            output_dir: Directory to save files
            formats: List of formats to generate (default: all)

        Returns:
            Dictionary mapping format to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["ansible", "puppet", "salt", "cloud-init", "terraform"]

        files = {}

        if "ansible" in formats:
            playbook_path = output_path / "nvidia-driver.yml"
            inventory_path = output_path / "inventory.ini"
            self.generate_ansible_playbook(str(playbook_path))
            self.generate_ansible_inventory([], str(inventory_path))
            files["ansible_playbook"] = str(playbook_path)
            files["ansible_inventory"] = str(inventory_path)

        if "puppet" in formats:
            manifest_path = output_path / "nvidia_driver.pp"
            self.generate_puppet_manifest(str(manifest_path))
            files["puppet_manifest"] = str(manifest_path)

        if "salt" in formats:
            state_path = output_path / "nvidia-driver.sls"
            self.generate_salt_state(str(state_path))
            files["salt_state"] = str(state_path)

        if "cloud-init" in formats:
            cloud_init_path = output_path / "cloud-init.yml"
            self.generate_cloud_init(str(cloud_init_path))
            files["cloud_init"] = str(cloud_init_path)

        if "terraform" in formats:
            terraform_path = output_path / "nvidia-driver.tf"
            self.generate_terraform_config(str(terraform_path))
            files["terraform"] = str(terraform_path)

        logger.info(f"Deployment package created in {output_dir}")
        return files
