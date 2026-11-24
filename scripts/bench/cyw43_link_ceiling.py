#!/usr/bin/env python3
"""Stage 1 driver: raw CYW43439 TCP goodput ceiling on a USB-attached Pico 2 W.

Drives ``firmware/nodes/bench_cyw43_iperf`` over USB serial against
``scripts/bench/tcp_discard_sink.py`` running on the receiver, sweeping link
variants (sys clock, gSPI divider, BLE scan). One JSON record per run is
appended to ``--out``; ``--summary`` prints a Markdown table from that file.

    python scripts/bench/cyw43_link_ceiling.py --sink 192.168.8.103:5201 \\
        --variants 150000:2:0,150000:2:1 --runs 3 --seconds 60 \\
        --position near --out bench_results/2026-11-23/stage1.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
import time
from pathlib import Path

import serial

SERIAL_GLOB = "/dev/cu.usbmodem*"
BOOT_TIMEOUT_S = 40.0
LINE_RE = re.compile(r"(\w+)=([-\w.:]+)")


def open_serial(timeout_s: float = BOOT_TIMEOUT_S) -> serial.Serial:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for port in sorted(glob.glob(SERIAL_GLOB)):
            try:
                return serial.Serial(port, 115200, timeout=1)
            except (OSError, serial.SerialException):
                pass
        time.sleep(0.5)
    raise RuntimeError("Pico serial port did not appear")


def parse_fields(line: str) -> dict[str, str]:
    return dict(LINE_RE.findall(line))


def read_until(port: serial.Serial, marker: str, timeout_s: float, echo: bool = True) -> list[str]:
    lines: list[str] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").rstrip()
        lines.append(line)
        if echo:
            print(f"  | {line}", flush=True)
        if marker in line:
            return lines
    raise TimeoutError(f"no '{marker}' within {timeout_s}s")


def query_info(port: serial.Serial) -> dict[str, str]:
    port.reset_input_buffer()
    port.write(b"info\n")
    for line in read_until(port, "[bench] info", 10, echo=False):
        if "[bench] info" in line:
            return parse_fields(line)
    return {}


def apply_variant(port: serial.Serial, sys_khz: int, pio_div: int, ble: int) -> serial.Serial:
    info = query_info(port)
    if (
        info.get("sys_khz") == str(sys_khz)
        and info.get("pio_div") == str(pio_div)
        and info.get("ble") == str(ble)
    ):
        return port
    port.write(f"cfg {sys_khz} {pio_div} {ble}\n".encode())
    time.sleep(1.5)
    port.close()
    time.sleep(2.0)
    port = open_serial()
    # Ready may already have printed before the port reopened; info is authoritative.
    for _ in range(int(BOOT_TIMEOUT_S)):
        info = query_info_safe(port)
        if info.get("ip"):
            break
        time.sleep(1)
    if info.get("pio_div") != str(pio_div) or info.get("ble") != str(ble):
        raise RuntimeError(f"variant not applied: {info}")
    return port


def query_info_safe(port: serial.Serial) -> dict[str, str]:
    try:
        return query_info(port)
    except (TimeoutError, serial.SerialException):
        return {}


# wl_cnt_t word indices (version 10 header seen on the CYW43439 firmware).
# Only the leading fields, whose order is stable across wl_cnt_t versions.
WL_COUNTER_WORDS = {
    "txframe": 1, "txbyte": 2, "txretrans": 3, "txerror": 4, "txserr": 7,
    "txnobuf": 8, "txnoassoc": 9, "rxframe": 16, "rxerror": 18, "rxnobuf": 20,
}


def read_wl_counters(port: serial.Serial) -> dict[str, int]:
    port.reset_input_buffer()
    port.write(b"counters\n")
    for line in read_until(port, "[cnt]", 10, echo=False):
        if line.startswith("[cnt]"):
            words = [int(w, 16) for w in line.split("words=")[1].strip(",").split(",") if w]
            return {name: words[index] for name, index in WL_COUNTER_WORDS.items()}
    return {}


def run_once(
    port: serial.Serial, sink: str, seconds: int, copy_writes: int, rate_kbps: int, cap_bytes: int = 0
) -> dict:
    host, sink_port = sink.split(":")
    before = read_wl_counters(port)
    port.reset_input_buffer()
    port.write(f"run {host} {sink_port} {seconds} {copy_writes} {rate_kbps} {cap_bytes}\n".encode())
    lines = read_until(port, "[tput] done", seconds + 30)
    per_second = [parse_fields(l) for l in lines if l.startswith("[tput] sec=")]
    done = parse_fields(next(l for l in lines if l.startswith("[tput] done")))
    mbps_series = [float(s["mbps"]) for s in per_second]
    after = read_wl_counters(port)
    wl_delta = {k: (after[k] - before[k]) & 0xFFFFFFFF for k in after if k in before}
    return {
        "wl_delta": wl_delta,
        "inflight_cap": cap_bytes,
        "per_second": per_second,
        "done": done,
        "mbps": float(done["mbps"]),
        "median_sec_mbps": statistics.median(mbps_series) if mbps_series else 0.0,
        "p10_sec_mbps": sorted(mbps_series)[len(mbps_series) // 10] if mbps_series else 0.0,
        "zero_seconds": sum(1 for v in mbps_series if v == 0.0),
        "rto_events": int(done.get("rto_ev", 0)),
        "fast_rexmit": int(done.get("fast_rx", 0)),
        "rssi": int(done.get("rssi", 0)),
        "driver_waits": int(done.get("drv_waits", 0)),
        "max_stall_ms": max((int(s.get("max_stall_ms", 0)) for s in per_second), default=0),
    }


def summarize(path: Path) -> str:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["position"], row["arch"], row["sys_khz"], row["pio_div"], row["ble"], row["copy"],
               row.get("rate_kbps", 0), row.get("inflight_cap", 0), row.get("retry_limit", 0))
        groups.setdefault(key, []).append(row)
    out = [
        "| position | arch | sys MHz | gSPI div | BLE | paced kbit/s | inflight cap | retry lim | runs | mean Mbit/s | worst run | p10 sec | zero-s | TCP rexmit | worst ACK stall ms | MAC retry % | chip rxnobuf | RSSI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, group in sorted(groups.items()):
        position, arch, sys_khz, pio_div, ble, copy_writes, rate_kbps, cap, retry_limit = key
        tx_frames = sum(g.get("wl_delta", {}).get("txframe", 0) for g in group)
        tx_retrans = sum(g.get("wl_delta", {}).get("txretrans", 0) for g in group)
        rx_nobuf = sum(g.get("wl_delta", {}).get("rxnobuf", 0) for g in group)
        means = [g["mbps"] for g in group]
        out.append(
            f"| {position} | {arch} | {sys_khz // 1000} | {pio_div} | {'on' if ble else 'off'} "
            f"| {rate_kbps or 'sat'} | {cap or 'sndbuf'} | {retry_limit or 'default'} | {len(group)} "
            f"| {statistics.mean(means):.2f} | {min(means):.2f} "
            f"| {min(g['p10_sec_mbps'] for g in group):.2f} | {sum(g['zero_seconds'] for g in group)} "
            f"| {sum(g['rto_events'] for g in group)} | {max(g.get('max_stall_ms', 0) for g in group)} "
            f"| {100.0 * tx_retrans / max(tx_frames, 1):.1f} | {rx_nobuf} "
            f"| {statistics.median(g['rssi'] for g in group):.0f} |"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sink", default="192.168.8.103:5201")
    parser.add_argument("--variants", default="150000:2:0", help="comma list of sys_khz:pio_div:ble")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--copy", type=int, default=0, help="1 = TCP_WRITE_FLAG_COPY like production")
    parser.add_argument("--rate-kbps", default="0", help="comma list; 0 = saturate, else paced payload rate")
    parser.add_argument("--inflight-cap", default="0", help="comma list of in-flight byte caps; 0 = TCP_SND_BUF")
    parser.add_argument("--retry-limit", type=int, default=0, help="set MAC short+long retry limit (0 = leave default)")
    parser.add_argument("--position", default="near")
    parser.add_argument("--out", type=Path, default=Path("bench_results/stage1.jsonl"))
    parser.add_argument("--summary", action="store_true", help="only print the table for --out")
    args = parser.parse_args()

    if args.summary:
        print(summarize(args.out))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    port = open_serial()
    for variant in args.variants.split(","):
        sys_khz, pio_div, ble = (int(v) for v in variant.split(":"))
        print(f"== variant sys={sys_khz} div={pio_div} ble={ble}", flush=True)
        port = apply_variant(port, sys_khz, pio_div, ble)
        info = query_info(port)
        if args.retry_limit:
            for wlc_set in (32, 34):  # WLC_SET_SRL, WLC_SET_LRL
                port.write(f"wl set {wlc_set} {args.retry_limit}\n".encode())
                read_until(port, f"wl set {wlc_set}", 5, echo=False)
        for rate_kbps, cap_bytes, run_index in [
            (int(r), int(c), i)
            for r in args.rate_kbps.split(",")
            for c in args.inflight_cap.split(",")
            for i in range(args.runs)
        ]:
            result = run_once(port, args.sink, args.seconds, args.copy, rate_kbps, cap_bytes)
            record = {
                "rate_kbps": rate_kbps,
                "retry_limit": args.retry_limit,
                "position": args.position,
                "arch": info.get("arch", "poll"),
                "sys_khz": sys_khz,
                "pio_div": pio_div,
                "ble": ble,
                "copy": args.copy,
                "run": run_index,
                "seconds": args.seconds,
                "started_at": time.time(),
                **result,
            }
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"== run {run_index} rate={rate_kbps or 'sat'} cap={cap_bytes or 'sndbuf'}: {result['mbps']:.2f} Mbit/s, "
                f"p10 {result['p10_sec_mbps']:.2f}, zero-s {result['zero_seconds']}, rexmit {result['rto_events']}, "
                f"wl {result['wl_delta']}, rssi {result['rssi']}",
                flush=True,
            )
            time.sleep(2)
    print(summarize(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
