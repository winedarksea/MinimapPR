"""Shared adb plumbing for the chirp-lab scripts (standard library only).

Every adb call names its device with ``-s``: two phones are attached at
once, and an unqualified adb command fails or hits the wrong one. Serials,
Wi-Fi addresses and the app's component names live in ``devices.json``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIRECTORY.parent.parent
DEVICES_FILE = SCRIPT_DIRECTORY / "devices.json"
DEFAULT_ADB_TIMEOUT_S = 60
STATUS_LINE_PATTERN = re.compile(r"^(CHIRP_LAB_STATUS|SYNC_STATUS) (\{.*\})\s*$")


@dataclass(frozen=True)
class Device:
    name: str
    serial: str
    platform_name: str
    wifi_ip: str


@dataclass(frozen=True)
class LabConfig:
    package: str
    activity: str
    service: str
    debug_receiver: str
    adb: str
    devices: dict[str, Device]
    roles: dict[str, str]
    session_roles: dict[str, str]

    def device_for_role(self, role: str) -> Device:
        return self.devices[self.roles[role]]

    def session_device(self, role: str) -> Device:
        return self.devices[self.session_roles[role]]


def load_lab_config(use_wifi: bool = False) -> LabConfig:
    raw = json.loads(DEVICES_FILE.read_text())
    port = raw["wifi_adb_port"]
    devices = {
        name: Device(
            name=name,
            serial=f"{entry['wifi_ip']}:{port}" if use_wifi else entry["serial"],
            platform_name=entry["platform_name"],
            wifi_ip=entry["wifi_ip"],
        )
        for name, entry in raw["devices"].items()
    }
    return LabConfig(
        package=raw["package"],
        activity=raw["activity"],
        service=raw["service"],
        debug_receiver=raw["debug_receiver"],
        adb=os.path.expanduser(os.environ.get("ADB", raw["adb"])),
        devices=devices,
        roles=raw["roles"],
        session_roles=raw["session_roles"],
    )


def adb(config: LabConfig, device: Device, *arguments: str, check: bool = True, timeout_s: float = DEFAULT_ADB_TIMEOUT_S) -> str:
    command = [config.adb, "-s", device.serial, *arguments]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    if check and completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}")
    return completed.stdout


def shell(config: LabConfig, device: Device, command: str, check: bool = True) -> str:
    return adb(config, device, "shell", command, check=check)


def is_connected(config: LabConfig, device: Device) -> bool:
    completed = subprocess.run([config.adb, "-s", device.serial, "get-state"], capture_output=True, text=True, timeout=10)
    return completed.returncode == 0 and completed.stdout.strip() == "device"


def start_activity(config: LabConfig, device: Device, extras: dict[str, object]) -> str:
    """``am start`` with typed extras: str → --es, bool → --ez, int → --ei, float → --ef."""
    parts = ["am", "start", "-n", f"{config.package}/{config.activity}"]
    for key, value in extras.items():
        if value is None:
            continue
        if isinstance(value, bool):
            parts += ["--ez", key, "true" if value else "false"]
        elif isinstance(value, int):
            parts += ["--ei", key, str(value)]
        elif isinstance(value, float):
            parts += ["--ef", key, repr(value)]
        else:
            parts += ["--es", key, str(value)]
    return shell(config, device, " ".join(parts))


def broadcast(config: LabConfig, device: Device, action: str, extras: dict[str, str] | None = None) -> str:
    parts = ["am", "broadcast", "-a", f"{config.package}.{action}", "-n", f"{config.package}/{config.debug_receiver}"]
    for key, value in (extras or {}).items():
        parts += ["--es", key, value]
    return shell(config, device, " ".join(parts))


def service_status(config: LabConfig, device: Device) -> dict[str, dict]:
    """Parse ``CHIRP_LAB_STATUS``/``SYNC_STATUS`` JSON lines from dumpsys."""
    output = shell(config, device, f"dumpsys activity service {config.package}/{config.service}", check=False)
    statuses: dict[str, dict] = {}
    for line in output.splitlines():
        match = STATUS_LINE_PATTERN.match(line.strip())
        if match:
            try:
                statuses[match.group(1)] = json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
    return statuses


def external_files_directory(config: LabConfig, run_id: str) -> str:
    return f"/sdcard/Android/data/{config.package}/files/chirp_lab/{run_id}"
