#!/usr/bin/env python3
"""Summarize node audio/transport health during firmware stress runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class NodeSnapshot:
    timestamp_s: float
    node_id: str
    sample_rate_hz: int | None
    audio_status: str | None
    age_seconds: float | None
    ingest_verdict: str | None
    frames_captured: int
    frames_dropped: int
    publish_errors: int
    queue_overflows: int
    continuity_violations: int
    queue_depth: int | None
    ring_high_water: int | None
    ring_capacity: int | None
    queue_high_water: int | None
    queue_capacity: int | None
    publish_latency_ewma_ms: int | None
    wifi_rssi_dbm: int | None
    boot_id: int | None


def _get_json(url: str, *, timeout_s: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _snapshot(node: dict[str, Any], timestamp_s: float) -> NodeSnapshot:
    audio_debug = node.get("audio_debug") if isinstance(node.get("audio_debug"), dict) else {}
    runner_stats = audio_debug.get("runner_stats") if isinstance(audio_debug.get("runner_stats"), dict) else {}
    transport_health = (
        runner_stats.get("transport_health")
        if isinstance(runner_stats.get("transport_health"), dict)
        else {}
    )
    ingest_health = (
        audio_debug.get("ingest_health") if isinstance(audio_debug.get("ingest_health"), dict) else {}
    )
    return NodeSnapshot(
        timestamp_s=timestamp_s,
        node_id=str(node.get("id") or ""),
        sample_rate_hz=audio_debug.get("sample_rate_hz"),
        audio_status=audio_debug.get("status"),
        age_seconds=audio_debug.get("age_seconds"),
        ingest_verdict=ingest_health.get("verdict"),
        frames_captured=_as_int(runner_stats.get("frames_captured")),
        frames_dropped=_as_int(runner_stats.get("frames_dropped")),
        publish_errors=_as_int(runner_stats.get("publish_errors")),
        queue_overflows=_as_int(runner_stats.get("queue_overflows")),
        continuity_violations=_as_int(runner_stats.get("packet_continuity_violations")),
        queue_depth=runner_stats.get("queue_depth"),
        ring_high_water=transport_health.get("ring_frames_high_water"),
        ring_capacity=transport_health.get("ring_frames_capacity"),
        queue_high_water=transport_health.get("queue_slots_high_water"),
        queue_capacity=transport_health.get("queue_slots_capacity"),
        publish_latency_ewma_ms=transport_health.get("publish_latency_ewma_ms"),
        wifi_rssi_dbm=transport_health.get("wifi_rssi_dbm"),
        boot_id=transport_health.get("boot_id"),
    )


def _delta(current: int, previous: int) -> int:
    return max(0, current - previous)


def _format_ratio(value: int | None, capacity: int | None) -> str:
    if value is None or capacity in (None, 0):
        return "-"
    return f"{value}/{capacity} ({(100.0 * value / capacity):.0f}%)"


def _print_summary(first: NodeSnapshot, last: NodeSnapshot) -> None:
    elapsed_s = max(last.timestamp_s - first.timestamp_s, 1e-6)
    captured_delta = _delta(last.frames_captured, first.frames_captured)
    dropped_delta = _delta(last.frames_dropped, first.frames_dropped)
    continuity_delta = _delta(last.continuity_violations, first.continuity_violations)
    publish_error_delta = _delta(last.publish_errors, first.publish_errors)
    queue_overflow_delta = _delta(last.queue_overflows, first.queue_overflows)
    loss_events = dropped_delta + continuity_delta + queue_overflow_delta
    loss_ppm = (loss_events * 1_000_000.0 / captured_delta) if captured_delta > 0 else 0.0

    print(f"node_id: {last.node_id}")
    print(f"elapsed_s: {elapsed_s:.1f}")
    print(f"sample_rate_hz: {last.sample_rate_hz}")
    print(f"audio_status: {last.audio_status} age_s={last.age_seconds}")
    print(f"ingest_verdict: {last.ingest_verdict}")
    print(f"frames_captured_delta: {captured_delta}")
    print(f"loss_events_delta: {loss_events} ({loss_ppm:.1f} ppm)")
    print(f"frames_dropped_delta: {dropped_delta}")
    print(f"continuity_violations_delta: {continuity_delta}")
    print(f"queue_overflows_delta: {queue_overflow_delta}")
    print(f"publish_errors_delta: {publish_error_delta}")
    print(f"queue_depth: {last.queue_depth}")
    print(f"ring_high_water: {_format_ratio(last.ring_high_water, last.ring_capacity)}")
    print(f"queue_high_water: {_format_ratio(last.queue_high_water, last.queue_capacity)}")
    print(f"publish_latency_ewma_ms: {last.publish_latency_ewma_ms}")
    print(f"wifi_rssi_dbm: {last.wifi_rssi_dbm}")
    print(f"boot_id: {last.boot_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://192.168.8.165:8080", help="MinimapPR server base URL")
    parser.add_argument("--node-id", help="Node ID to summarize; defaults to the first node")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Polling duration")
    parser.add_argument("--interval-s", type=float, default=5.0, help="Polling interval")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="HTTP timeout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshots: list[NodeSnapshot] = []
    deadline_s = time.monotonic() + max(args.duration_s, 0.0)
    nodes_url = args.server.rstrip("/") + "/api/v1/nodes"
    while True:
        try:
            nodes = _get_json(nodes_url, timeout_s=args.timeout_s)
        except (OSError, TimeoutError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"poll failed: {exc}", file=sys.stderr)
        else:
            matching = [
                node
                for node in nodes
                if not args.node_id or str(node.get("id")) == args.node_id
            ]
            if matching:
                snapshots.append(_snapshot(matching[0], time.time()))
                latest = snapshots[-1]
                print(
                    f"{time.strftime('%H:%M:%S')} {latest.node_id} "
                    f"status={latest.audio_status} verdict={latest.ingest_verdict} "
                    f"captured={latest.frames_captured} drops={latest.frames_dropped} "
                    f"queue={latest.queue_depth} ewma_ms={latest.publish_latency_ewma_ms}"
                )
            elif args.node_id:
                print(f"node {args.node_id!r} not present", file=sys.stderr)
        if time.monotonic() >= deadline_s:
            break
        time.sleep(max(args.interval_s, 0.1))

    if len(snapshots) < 2:
        print("not enough samples to summarize", file=sys.stderr)
        return 1
    _print_summary(snapshots[0], snapshots[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
