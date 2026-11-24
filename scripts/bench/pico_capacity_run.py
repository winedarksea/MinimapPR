#!/usr/bin/env python3
"""Stage 2/3 driver: production sirith_tetrahedral firmware, synthetic audio,
judged by scripts/bench/mmb3_capacity_sink.py.

For each config it builds ``build-bench-<id>-ble<0|1>[-<tag>]``, flashes over
USB (picotool), resets the sink after a warm-up, captures serial, polls the
node's ``:8082/api/v1/stats`` and writes ``<out_dir>/<id>-ble<b>[-<tag>].json``
plus a Markdown row. The sink runs locally (spawned here) unless ``--sink-ssh``
points at a host where it is already running.

    python scripts/bench/pico_capacity_run.py --configs C1,C3 --ble 0,1 \\
        --minutes 10 --sink-host 192.168.8.199 --out-dir bench_results/2026-11-23

PASS = sink PASS (0 seq/sample gaps, bit-exact, no timestamp jumps, no node
drops/overflows) + queue high-water < 50 % + publish latency EWMA not climbing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import serial

REPO = Path(__file__).resolve().parents[2]
NODE_DIR = REPO / "firmware" / "nodes" / "sirith_tetrahedral"
PICO_SDK = Path.home() / ".pico-sdk"
CMAKE = PICO_SDK / "cmake" / "v3.31.5" / "bin" / "cmake"
NINJA_DIR = PICO_SDK / "ninja" / "v1.12.1"
PICOTOOL = PICO_SDK / "picotool" / "2.2.0-a4" / "picotool" / "picotool"
PYTHON = Path.home() / "mambaforge" / "envs" / "minimappr" / "bin" / "python"
SERIAL_GLOB = "/dev/cu.usbmodem*"
STATS_POLL_INTERVAL_S = 5.0
QUEUE_HIGH_WATER_LIMIT = 0.5
# EWMA "climbing": last third of the run averages > this factor of the first third.
LATENCY_CLIMB_FACTOR = 1.5

CONFIGS = {
    "C1": (44100, 4),
    "C2": (48000, 4),
    "C3": (48000, 5),
    "C4": (48000, 6),
    "C5": (48000, 8),
}


def payload_mbps(sample_rate: int, channels: int) -> float:
    return sample_rate * channels * 16 / 1e6


def build(config_id: str, ble: int, sink_url: str, extra_defines: list[str], tag: str, input_mode: int = 3) -> Path:
    sample_rate, channels = CONFIGS[config_id]
    build_dir = NODE_DIR / f"build-bench-{config_id}-ble{ble}{'-' + tag if tag else ''}"
    env = dict(os.environ, PATH=f"{CMAKE.parent}:{NINJA_DIR}:{os.environ['PATH']}")
    defines = [
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DMMPR_NODECFG_AUDIO_INPUT_MODE={input_mode}",
        f"-DMMPR_NODECFG_AUDIO_SAMPLE_RATE_HZ={sample_rate}",
        f"-DMMPR_NODECFG_AUDIO_CHANNELS={channels}",
        f"-DMMPR_NODECFG_ENABLE_BLE_SCAN={ble}",
        f"-DMMPR_NODECFG_SERVER_BASE_URL={sink_url}",
        *[f"-D{d}" for d in extra_defines],
    ]
    for command in (
        [str(CMAKE), "-S", str(NODE_DIR), "-B", str(build_dir), "-G", "Ninja", *defines],
        [str(NINJA_DIR / "ninja"), "-C", str(build_dir)],
    ):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"build failed: {' '.join(command[:3])}\n{result.stdout[-3000:]}{result.stderr[-3000:]}")
    return build_dir / "sirith_tetrahedral.uf2"


def flash(uf2: Path, attempts: int = 4) -> None:
    # picotool's forced reboot into BOOTSEL is sometimes missed (e.g. a node mid
    # watchdog reset); retry rather than benchmark a stale image.
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                [str(PICOTOOL), "load", str(uf2), "-fx"], capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and "rebooted" in result.stdout:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    raise RuntimeError(f"flashing {uf2} failed after {attempts} attempts")


class SerialCapture(threading.Thread):
    def __init__(self, log_path: Path) -> None:
        super().__init__(daemon=True)
        self.log_path = log_path
        self.stop_event = threading.Event()
        self.lines: list[str] = []

    def run(self) -> None:
        port = None
        with self.log_path.open("w", encoding="utf-8") as log:
            while not self.stop_event.is_set():
                if port is None:
                    for candidate in sorted(glob.glob(SERIAL_GLOB)):
                        try:
                            port = serial.Serial(candidate, 115200, timeout=0.5)
                            break
                        except (OSError, serial.SerialException):
                            pass
                    if port is None:
                        time.sleep(0.5)
                        continue
                try:
                    raw = port.readline()
                except (OSError, serial.SerialException):
                    port = None
                    continue
                if raw:
                    line = f"{time.time():.3f} {raw.decode(errors='replace').rstrip()}"
                    self.lines.append(line)
                    log.write(line + "\n")
                    log.flush()
        if port is not None:
            port.close()


def http_json(url: str, method: str = "GET", timeout: float = 10.0) -> dict:
    request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class LocalSink:
    def __init__(self, port: int, log_path: Path) -> None:
        self.log = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(PYTHON), str(REPO / "scripts" / "bench" / "mmb3_capacity_sink.py"), "--port", str(port)],
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        time.sleep(2.0)

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)
        self.log.close()


def wait_for_stream(sink_base: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        summary = http_json(f"{sink_base}/bench/summary")
        if summary["frames"] > 0 and summary["peers"]:
            return summary
        time.sleep(2.0)
    raise TimeoutError("node never streamed to the sink")


def poll_stats(node_ip: str, duration_s: float, samples: list[dict]) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        try:
            stats = http_json(f"http://{node_ip}:8082/api/v1/stats", timeout=4.0)
            stats["t"] = time.time()
            samples.append(stats)
        except Exception as exc:  # node busy or briefly unreachable; record and continue
            samples.append({"t": time.time(), "error": repr(exc)})
        time.sleep(STATS_POLL_INTERVAL_S)


def judge(sink_summary: dict, stats_samples: list[dict]) -> tuple[str, list[str], dict]:
    failures = list(sink_summary["failures"])
    good = [s for s in stats_samples if "error" not in s]
    derived: dict = {"stats_samples": len(good), "stats_errors": len(stats_samples) - len(good)}
    if good:
        last = good[-1]
        capacity = max(last.get("queue_slots_capacity", 0), 1)
        high_water = last.get("queue_slots_high_water", 0)
        derived["queue_high_water"] = f"{high_water}/{capacity}"
        derived["queue_high_water_ratio"] = round(high_water / capacity, 3)
        if high_water / capacity >= QUEUE_HIGH_WATER_LIMIT:
            failures.append(f"queue high-water {high_water}/{capacity} >= 50%")
        ewma = [s.get("publish_latency_ewma_ms", 0) for s in good]
        third = max(len(ewma) // 3, 1)
        first_mean, last_mean = statistics.mean(ewma[:third]), statistics.mean(ewma[-third:])
        derived["latency_ewma_first_third_ms"] = round(first_mean, 1)
        derived["latency_ewma_last_third_ms"] = round(last_mean, 1)
        if len(ewma) >= 6 and last_mean > LATENCY_CLIMB_FACTOR * max(first_mean, 5.0):
            failures.append(f"publish latency EWMA climbing {first_mean:.0f}->{last_mean:.0f} ms")
        for key in (
            "publish_latency_max_ms", "publish_upload_ewma_ms", "publish_upload_max_ms",
            "publish_server_wait_ewma_ms", "publish_server_wait_max_ms", "wifi_rssi_dbm",
            "publish_errors", "discarded_batches", "queue_overflows", "frames_dropped",
        ):
            derived[key] = last.get(key)
        first = good[0]
        derived["node_goodput_mbps"] = round(
            (last.get("bytes_sent", 0) - first.get("bytes_sent", 0)) * 8 / max(last["t"] - first["t"], 1e-9) / 1e6, 3
        )
        for key in ("discarded_batches", "publish_errors"):
            derived[f"{key}_delta"] = last.get(key, 0) - first.get(key, 0)
        if derived["discarded_batches_delta"]:
            failures.append(f"discarded_batches+={derived['discarded_batches_delta']}")
    else:
        failures.append("no :8082 stats collected")
    return ("PASS" if not failures else "FAIL"), failures, derived


def markdown_row(record: dict) -> str:
    sink = record["sink"]
    derived = record["derived"]
    return (
        f"| {record['run_id']} | {record['payload_mbps']:.2f} | {'on' if record['ble'] else 'off'} "
        f"| {record['minutes']} | **{record['verdict']}** | {sink['goodput_mbps']:.2f} "
        f"| {sink['frames_missing']} | {sink['sample_index_gaps']} | {sink['bit_errors']} | {sink['timestamp_jumps']} "
        f"| {derived.get('queue_high_water', '-')} | {derived.get('latency_ewma_last_third_ms', '-')} "
        f"| {derived.get('publish_upload_ewma_ms', '-')}/{derived.get('publish_upload_max_ms', '-')} "
        f"| {derived.get('publish_server_wait_ewma_ms', '-')}/{derived.get('publish_server_wait_max_ms', '-')} "
        f"| {derived.get('publish_errors_delta', '-')} | {derived.get('wifi_rssi_dbm', '-')} "
        f"| {'; '.join(record['failures'])[:120]} |"
    )


MARKDOWN_HEADER = (
    "| run | payload Mbit/s | BLE | min | verdict | sink Mbit/s | frames lost | sample gaps | bit errs | ts jumps "
    "| queue hw | lat EWMA ms | upload ewma/max ms | server-wait ewma/max ms | publish errs | RSSI | failures |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def run_config(args, config_id: str, ble: int, sink_base: str) -> dict:
    sample_rate, channels = CONFIGS[config_id]
    run_id = f"{config_id}-ble{ble}{'-' + args.tag if args.tag else ''}"
    print(f"== {run_id}: {sample_rate} Hz x {channels} ch, {payload_mbps(sample_rate, channels):.2f} Mbit/s", flush=True)
    uf2 = build(config_id, ble, args.node_sink_url, args.define, args.tag, args.input_mode)
    capture = SerialCapture(args.out_dir / f"{run_id}.serial.log")
    flash(uf2)
    capture.start()
    http_json(f"{sink_base}/bench/reset", method="POST")
    first = wait_for_stream(sink_base, 90)
    node_ip = first["peers"][0]
    print(f"   streaming from {node_ip}; warm-up {args.warmup_s}s", flush=True)
    time.sleep(args.warmup_s)
    http_json(f"{sink_base}/bench/reset", method="POST")
    stats_samples: list[dict] = []
    poll_stats(node_ip, args.minutes * 60, stats_samples)
    sink_summary = http_json(f"{sink_base}/bench/summary", timeout=60)
    capture.stop_event.set()
    capture.join(timeout=5)
    verdict, failures, derived = judge(sink_summary, stats_samples)
    record = {
        "run_id": run_id,
        "config": config_id,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "payload_mbps": payload_mbps(sample_rate, channels),
        "ble": ble,
        "minutes": args.minutes,
        "tag": args.tag,
        "extra_defines": args.define,
        "verdict": verdict,
        "failures": failures,
        "derived": derived,
        "sink": sink_summary,
        "stats_samples": stats_samples,
        "finished_at": time.time(),
    }
    (args.out_dir / f"{run_id}.json").write_text(json.dumps(record, indent=1))
    row = markdown_row(record)
    table = args.out_dir / "stage2_table.md"
    if not table.exists():
        table.write_text(MARKDOWN_HEADER + "\n")
    with table.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")
    print(f"   {verdict}: {failures}", flush=True)
    print("   " + json.dumps({k: derived.get(k) for k in (
        "node_goodput_mbps", "queue_high_water", "publish_upload_ewma_ms", "publish_upload_max_ms",
        "publish_server_wait_ewma_ms", "publish_server_wait_max_ms", "publish_errors_delta", "wifi_rssi_dbm")}),
        flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default="C1")
    parser.add_argument("--ble", default="1", help="comma list of 0/1")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--sink-host", default="192.168.8.199", help="address the node posts to")
    parser.add_argument("--sink-port", type=int, default=8091)
    parser.add_argument("--remote-sink", action="store_true", help="sink already running at --sink-host")
    parser.add_argument("--define", action="append", default=[], help="extra -D for cmake, e.g. MMPR_NODECFG_HTTP_TIMEOUT_MS=2000")
    parser.add_argument(
        "--input-mode", type=int, default=3,
        help="3 = synthetic (bit-exact checks apply); 0 = real TDM (Stage 5: ignore bit_errors, 4 ch only)",
    )
    parser.add_argument("--tag", default="", help="suffix for build dir / result names (Stage 3 iterations)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "bench_results" / time.strftime("%Y-%m-%d"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.node_sink_url = f"http://{args.sink_host}:{args.sink_port}"
    sink_base = f"http://{args.sink_host if args.remote_sink else '127.0.0.1'}:{args.sink_port}"

    local_sink = None if args.remote_sink else LocalSink(args.sink_port, args.out_dir / "sink.log")
    try:
        for config_id in args.configs.split(","):
            for ble in (int(b) for b in args.ble.split(",")):
                run_config(args, config_id, ble, sink_base)
    finally:
        if local_sink is not None:
            local_sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
