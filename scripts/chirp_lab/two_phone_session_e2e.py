#!/usr/bin/env python3
"""Two-phone live session end to end (plan B.5): the Pixel hosts, the A03s
joins (NSD by default, or --host-address), and the host's SYNC_STATUS is
polled over dumpsys until enough chirp rounds have completed. Asserts the
joiner joined, rounds keep completing and the live range is within
--tolerance-m of --tape-distance-m. With --lock the screens are turned off
after the join and the run continues (the 15-minute locked run).

Writes every poll to data/field_phase1/<date>_<label>_session/polls.jsonl;
with --shadow-wav both phones also record chirp-lab bundles, so
`sync_spike analyse` can cross-check the live numbers. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

from chirp_lab_devices import REPO_ROOT, broadcast, load_lab_config, service_status, shell, start_activity

POLL_INTERVAL_S = 2.0
JOIN_TIMEOUT_S = 60.0
# Round ids start at 0 and the last round may still be in flight at the
# final poll, so the newest solved round may trail the count by two.
MAXIMUM_DIAGNOSTICS_LAG_ROUNDS = 2


def peer_rows(status: dict) -> list[dict]:
    """Rows carrying a range. Which row holds it depends on which phone the
    round solver picked as clock reference, so look at every row."""
    return [peer for peer in status.get("peers", []) if peer.get("range_m") is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wifi", action="store_true")
    parser.add_argument("--label", default="session")
    parser.add_argument("--tape-distance-m", type=float, required=True)
    parser.add_argument("--tolerance-m", type=float, default=0.05)
    parser.add_argument("--minimum-rounds", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--temperature-c", type=float, default=20.0)
    parser.add_argument("--periodic-ms", type=int, default=5_000)
    parser.add_argument("--lead-ms", type=int, default=1_000)
    parser.add_argument("--host-address", help="ip:port for the joiner (skips NSD)")
    parser.add_argument("--lock", action="store_true", help="screens off after the join")
    parser.add_argument("--shadow-wav", action="store_true")
    arguments = parser.parse_args()

    config = load_lab_config(use_wifi=arguments.wifi)
    host, joiner = config.session_device("host"), config.session_device("join")
    run_id = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{arguments.label}"
    output_directory = REPO_ROOT / "data" / "field_phase1" / f"{dt.date.today():%Y%m%d}_{arguments.label}_session"
    output_directory.mkdir(parents=True, exist_ok=True)
    common = {
        "mode": "session",
        "run_id": run_id,
        "temperature_c": arguments.temperature_c,
        "periodic_ms": arguments.periodic_ms,
        "lead_ms": arguments.lead_ms,
        "shadow_wav": arguments.shadow_wav,
    }
    start_activity(config, host, {**common, "session_role": "host"})
    time.sleep(3.0)
    start_activity(config, joiner, {**common, "session_role": "join", "host_address": arguments.host_address})

    polls = (output_directory / "polls.jsonl").open("a")
    failures: list[str] = []
    started = time.time()
    joined_at = None
    locked = False
    last_host_status: dict = {}
    while time.time() - started < arguments.duration_s:
        time.sleep(POLL_INTERVAL_S)
        host_status = service_status(config, host).get("SYNC_STATUS", {})
        joiner_status = service_status(config, joiner).get("SYNC_STATUS", {})
        polls.write(json.dumps({"t": time.time() - started, "host": host_status, "joiner": joiner_status}) + "\n")
        polls.flush()
        last_host_status = host_status or last_host_status
        if joined_at is None and joiner_status.get("joined"):
            joined_at = time.time() - started
            print(f"joiner joined after {joined_at:.1f} s (session clock offset {joiner_status.get('session_clock_offset_ms')} ms, best RTT {joiner_status.get('session_clock_best_round_trip_ms')} ms)")
            if arguments.lock and not locked:
                for device in (host, joiner):
                    shell(config, device, "input keyevent KEYCODE_SLEEP")
                locked = True
                print("screens locked")
        if joined_at is None and time.time() - started > JOIN_TIMEOUT_S:
            failures.append(f"joiner did not join within {JOIN_TIMEOUT_S:.0f} s: {joiner_status}")
            break
        rows = peer_rows(host_status)
        ranges = ", ".join(f"{row['node_id']}: {row.get('range_m')} m, θ {row.get('round_clock_offset_us')} µs, {row.get('rate_offset_ppm')} ppm" for row in rows)
        print(f"t={time.time() - started:6.1f}s rounds={host_status.get('rounds_completed')} {ranges}")
    for device in (joiner, host):
        broadcast(config, device, "SESSION_STOP")

    rounds = last_host_status.get("rounds_completed", 0)
    if rounds < arguments.minimum_rounds:
        failures.append(f"only {rounds} rounds completed (need {arguments.minimum_rounds})")
    rows = peer_rows(last_host_status)
    if not rows or rows[0].get("range_m") is None:
        failures.append("no live range to the joiner")
    else:
        error = rows[0]["range_m"] - arguments.tape_distance_m
        print(f"final range {rows[0]['range_m']:.4f} m vs tape {arguments.tape_distance_m:.3f} m: error {error:+.4f} m")
        if abs(error) > arguments.tolerance_m:
            failures.append(f"range error {error:+.3f} m exceeds ±{arguments.tolerance_m} m")
    freshest_round = max((row.get("last_round_id", -1) for row in last_host_status.get("peers", [])), default=-1)
    if rounds and freshest_round < rounds - MAXIMUM_DIAGNOSTICS_LAG_ROUNDS:
        failures.append(f"peer diagnostics are stale: last solved round {freshest_round} of {rounds} completed (rounds finish without an island solve)")
    if last_host_status.get("muted_node_ids"):
        failures.append(f"muted nodes: {last_host_status['muted_node_ids']}")
    summary = {"run_id": run_id, "tape_distance_m": arguments.tape_distance_m, "locked": locked, "joined_after_s": joined_at, "final_host_status": last_host_status, "failures": failures}
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
