#!/usr/bin/env python3
"""Collect comparable node, server, and sidecar ingest health snapshots.

Run one capture per firmware/configuration candidate, for example:
  python scripts/collect_ingest_regression_metrics.py --duration-seconds 1800 \
      --output /tmp/minimappr-ingest-baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen


def fetch_json(url: str, timeout_seconds: float) -> dict:
    with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 - lab endpoint supplied by operator
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def node_audio_age_seconds(nodes: list[dict], node_id: str) -> float | None:
    for node in nodes:
        if node.get("id") != node_id:
            continue
        audio_debug = node.get("audio_debug")
        if isinstance(audio_debug, dict) and isinstance(audio_debug.get("age_seconds"), (int, float)):
            return float(audio_debug["age_seconds"])
    return None


def capture_sample(args: argparse.Namespace) -> dict:
    nodes = fetch_json(f"{args.server_base_url}/api/v1/nodes", args.timeout_seconds)
    if not isinstance(nodes, list):
        raise ValueError("node endpoint did not return a list")
    server_diagnostics = fetch_json(
        f"{args.server_base_url}/api/v1/system/diagnostics", args.timeout_seconds
    )
    node_stats = fetch_json(f"{args.node_base_url}/api/v1/stats", args.timeout_seconds)
    return {
        "captured_unix_ns": time.time_ns(),
        "candidate": args.candidate,
        "node_id": args.node_id,
        "audio_age_seconds": node_audio_age_seconds(nodes, args.node_id),
        "node_stats": node_stats,
        "server_pipeline": server_diagnostics.get("pipeline"),
        "sidecar": server_diagnostics.get("sidecar"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-base-url", default="http://192.168.8.165:8080")
    parser.add_argument("--node-base-url", default="http://192.168.8.126:8082")
    parser.add_argument("--node-id", default="sirith-tetra-1a15")
    parser.add_argument("--candidate", default="unlabeled", help="firmware commit/configuration under test")
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("duration, interval, and timeout must be positive")

    deadline = time.monotonic() + args.duration_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        while time.monotonic() < deadline:
            started = time.monotonic()
            try:
                sample = capture_sample(args)
            except Exception as error:  # noqa: BLE001 - diagnostics must retain failed samples
                sample = {"captured_unix_ns": time.time_ns(), "error": str(error)}
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            stream.flush()
            remaining = args.interval_seconds - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
