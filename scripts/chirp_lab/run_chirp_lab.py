#!/usr/bin/env python3
"""Drive a two-phone Phase 1 chirp-lab run from the laptop.

  prep                 device check, media volume, auto-rotate off, sound
                       effects off, audio-hardening dump (dumpsys audio)
  start                launch chirp lab on both phones (A at k·P, B at k·P + P/2)
  mark <label>         marker on both phones (foam | hand | table | free text)
  lock / unlock        screen off / on on both phones (+ hardening dump on lock)
  status               print each phone's CHIRP_LAB_STATUS
  stop                 stop chirp lab on both phones
  pull                 pull both bundles, write run_manifest.json, validate
  analyse <run_dir>    run the offline analyser

The run in progress is remembered in data/field_phase1/.current_run.json so
mark/lock/stop/pull need no run id. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from chirp_lab_devices import REPO_ROOT, LabConfig, adb, broadcast, external_files_directory, is_connected, load_lab_config, service_status, shell, start_activity

DATA_ROOT = REPO_ROOT / "data" / "field_phase1"
CURRENT_RUN_FILE = DATA_ROOT / ".current_run.json"
MUSIC_STREAM = 3
DEFAULT_MEDIA_VOLUME_INDEX = 12
BUNDLE_FILES = ("capture.wav", "session.json", "events.jsonl")


def both_devices(config: LabConfig):
    return [("A", config.device_for_role("A")), ("B", config.device_for_role("B"))]


def dump_hardening(config: LabConfig, tag: str, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for role, device in both_devices(config):
        text = shell(config, device, "dumpsys audio", check=False)
        (output_directory / f"dumpsys_audio_{device.name}_{tag}.txt").write_text(text)
        print(f"{role} {device.name}: saved dumpsys audio ({tag}, {len(text)} bytes)")


def command_prep(config: LabConfig, arguments: argparse.Namespace) -> int:
    missing = [device.name for _, device in both_devices(config) if not is_connected(config, device)]
    if missing:
        print(f"not connected: {', '.join(missing)}", file=sys.stderr)
        return 1
    for role, device in both_devices(config):
        shell(config, device, "settings put system accelerometer_rotation 0")
        shell(config, device, "settings put system user_rotation 0")
        shell(config, device, "settings put system sound_effects_enabled 0")
        shell(config, device, f"cmd media_session volume --stream {MUSIC_STREAM} --set {arguments.volume}", check=False)
        version = shell(config, device, "getprop ro.build.version.sdk").strip()
        print(f"{role} {device.name} ({device.serial}): SDK {version}, rotation locked, sound effects off, media volume {arguments.volume}")
    dump_hardening(config, "prep", DATA_ROOT / "prep" / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    return 0


def command_start(config: LabConfig, arguments: argparse.Namespace) -> int:
    run_id = arguments.run_id or f"{dt.datetime.now():%Y%m%d_%H%M%S}_{arguments.label}"
    common = {
        "mode": "chirp_lab",
        "run_id": run_id,
        "period_ms": arguments.period_ms,
        "waveform": arguments.waveform,
        "band_low_hz": float(arguments.band[0]),
        "band_high_hz": float(arguments.band[1]),
        "duration_s": arguments.duration_s,
        "level_dbfs": arguments.level_dbfs,
        "lead_ms": arguments.lead_ms,
    }
    per_role = {
        "A": {"role": "A", "channel": arguments.channel_a, "source": arguments.source_a},
        "B": {"role": "B", "channel": arguments.channel_b, "source": arguments.source_b},
    }
    started_wall = time.time()
    for role, device in both_devices(config):
        output = start_activity(config, device, {**common, **per_role[role]})
        print(f"{role} {device.name}: {output.strip().splitlines()[-1] if output.strip() else 'started'}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    CURRENT_RUN_FILE.write_text(json.dumps({"run_id": run_id, "label": arguments.label, "started_wall_s": started_wall, "settings": {**common, "per_role": per_role}}, indent=2))
    print(f"run {run_id} started")
    return 0


def current_run() -> dict:
    if not CURRENT_RUN_FILE.exists():
        raise SystemExit("no run in progress (start one first)")
    return json.loads(CURRENT_RUN_FILE.read_text())


def command_mark(config: LabConfig, arguments: argparse.Namespace) -> int:
    for role, device in both_devices(config):
        broadcast(config, device, "CHIRP_LAB_MARK", {"label": arguments.label})
        print(f"{role} {device.name}: marker {arguments.label!r}")
    run = current_run()
    run.setdefault("markers", []).append({"label": arguments.label, "wall_s": time.time()})
    CURRENT_RUN_FILE.write_text(json.dumps(run, indent=2))
    return 0


def command_lock(config: LabConfig, arguments: argparse.Namespace) -> int:
    keycode = "KEYCODE_SLEEP" if arguments.lock else "KEYCODE_WAKEUP"
    for role, device in both_devices(config):
        shell(config, device, f"input keyevent {keycode}")
        print(f"{role} {device.name}: {keycode}")
    if arguments.lock:
        time.sleep(2)
        dump_hardening(config, "locked", DATA_ROOT / "prep" / current_run()["run_id"])
    return 0


def command_status(config: LabConfig, arguments: argparse.Namespace) -> int:
    for role, device in both_devices(config):
        print(f"{role} {device.name}: {json.dumps(service_status(config, device).get('CHIRP_LAB_STATUS'))}")
    return 0


def command_stop(config: LabConfig, arguments: argparse.Namespace) -> int:
    for role, device in both_devices(config):
        broadcast(config, device, "CHIRP_LAB_STOP")
        print(f"{role} {device.name}: stop sent")
    return 0


def parse_marker_surfaces(pairs: list[str]) -> dict[str, str]:
    surfaces = {}
    for pair in pairs:
        label, _, surface = pair.partition("=")
        surfaces[label] = surface
    return surfaces


def run_sync_spike(*arguments: str) -> int:
    command = ["cargo", "run", "--quiet", "--release", "-p", "minimappr-sync-spike", "--bin", "sync_spike", "--", *arguments]
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def command_pull(config: LabConfig, arguments: argparse.Namespace) -> int:
    run = current_run()
    run_id = arguments.run_id or run["run_id"]
    run_directory = DATA_ROOT / f"{dt.date.today():%Y%m%d}_{arguments.label or run.get('label', run_id)}"
    roles = {}
    for role, device in both_devices(config):
        phone_directory = run_directory / device.name
        phone_directory.mkdir(parents=True, exist_ok=True)
        remote = external_files_directory(config, run_id)
        for name in BUNDLE_FILES:
            adb(config, device, "pull", f"{remote}/{name}", str(phone_directory / name), timeout_s=600)
        print(f"{role} {device.name}: pulled {remote} → {phone_directory}")
        roles[role] = {
            "serial": device.serial,
            "dir": device.name,
            "self_path_m": getattr(arguments, f"self_path_{role.lower()}"),
            "pose": getattr(arguments, f"pose_{role.lower()}") or "",
        }
    manifest = {
        "label": arguments.label or run.get("label", run_id),
        "matrix_run": arguments.matrix_run,
        "tape_distance_m": arguments.tape_distance_m,
        "temperature_c": arguments.temperature_c,
        "surface": arguments.surface,
        "marker_surfaces": parse_marker_surfaces(arguments.marker_surface),
        "roles": roles,
        "notes": arguments.notes,
        "run_id": run_id,
        "laptop_markers": run.get("markers", []),
    }
    (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest → {run_directory / 'run_manifest.json'}")
    return run_sync_spike("validate", str(run_directory))


def command_analyse(config: LabConfig, arguments: argparse.Namespace) -> int:
    return run_sync_spike("analyse", str(Path(arguments.run_directory).resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wifi", action="store_true", help="use <ip>:5555 serials (after adb_wifi.sh)")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("--volume", type=int, default=DEFAULT_MEDIA_VOLUME_INDEX)
    start = sub.add_parser("start")
    start.add_argument("--label", default="lab")
    start.add_argument("--run-id")
    start.add_argument("--period-ms", type=int, default=4_000)
    start.add_argument("--waveform", choices=["updown_hfm", "hfm", "lfm"], default="updown_hfm")
    start.add_argument("--band", type=float, nargs=2, default=[13_000.0, 16_000.0], metavar=("LOW_HZ", "HIGH_HZ"))
    start.add_argument("--duration-s", type=float, default=0.05)
    start.add_argument("--level-dbfs", type=float, default=-6.0)
    start.add_argument("--lead-ms", type=int, default=300)
    for role in ("a", "b"):
        start.add_argument(f"--channel-{role}", choices=["left", "right", "both"], default="right")
        start.add_argument(f"--source-{role}", choices=["unprocessed", "voice_recognition", "mic", "camcorder", "default"], default="voice_recognition")
    mark = sub.add_parser("mark")
    mark.add_argument("label")
    sub.add_parser("lock").set_defaults(lock=True)
    sub.add_parser("unlock").set_defaults(lock=False)
    sub.add_parser("status")
    sub.add_parser("stop")
    pull = sub.add_parser("pull")
    pull.add_argument("--run-id")
    pull.add_argument("--label")
    pull.add_argument("--matrix-run", default="")
    pull.add_argument("--tape-distance-m", type=float)
    pull.add_argument("--temperature-c", type=float)
    pull.add_argument("--surface", default="")
    pull.add_argument("--pose-a")
    pull.add_argument("--pose-b")
    pull.add_argument("--self-path-a", type=float)
    pull.add_argument("--self-path-b", type=float)
    pull.add_argument("--marker-surface", action="append", default=[], help="label=surface, repeatable")
    pull.add_argument("--notes", default="")
    analyse = sub.add_parser("analyse")
    analyse.add_argument("run_directory")
    return parser


COMMANDS = {
    "prep": command_prep,
    "start": command_start,
    "mark": command_mark,
    "lock": command_lock,
    "unlock": command_lock,
    "status": command_status,
    "stop": command_stop,
    "pull": command_pull,
    "analyse": command_analyse,
}


def main() -> int:
    arguments = build_parser().parse_args()
    config = load_lab_config(use_wifi=arguments.wifi)
    return COMMANDS[arguments.command](config, arguments)


if __name__ == "__main__":
    sys.exit(main())
