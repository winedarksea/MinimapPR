#!/usr/bin/env python3
"""Bit-exact MMB3 capacity sink for the Pico 2 W multichannel bench (Stage 2).

Stands in for the ingest server at ``POST /api/v1/ingest/binary`` so a node
streaming synthetic audio can be judged on the node + link alone: it answers
202 as soon as the request body has arrived (no server processing in the
round trip) and verifies every frame afterwards, off the response path.

Per run it checks, against firmware ``SyntheticAudioSource``:
  * frame ``sequence`` continuity per boot (gaps = frames lost on the node),
  * ``start_sample_index`` continuity across frames and requests,
  * ``end_sample_index == start + samples_per_channel`` (parser enforces),
  * every sample == ((uint32(sample_index) * 257) ^ (ch * 0x1f1f) ^ 0x5a5a) as int16,
  * ``start_time_ns - sample_index / fs`` steady (no timestamp jumps),
  * node counters carried in MMB3 sections 0x1/0x4 (drops, overflows, queue).

Control endpoints (for scripts/bench/pico_capacity_run.py):
  ``POST /bench/reset``   start a new run (clears all state)
  ``GET  /bench/summary`` JSON summary with a PASS/FAIL verdict

Standalone asyncio HTTP/1.1 (keep-alive) server; the only repo import is the
real MMB3 parser, so the wire format is validated exactly as production does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from minimappr.api.binary_ingest import parse_binary_ingest_payload  # noqa: E402

INGEST_PATH = "/api/v1/ingest/binary"
MAX_HEADER_BYTES = 16384
MAX_BODY_BYTES = 4 * 1024 * 1024
# A step in (start_time_ns - sample_index/fs) larger than this between
# consecutive frames is a timestamp jump. NTP slews are sub-ms per frame.
TIMESTAMP_JUMP_TOLERANCE_NS = 1_000_000
INT16_MASK = 0xFFFF


def expected_synthetic_block(start_sample_index: int, samples_per_channel: int, channels: int) -> np.ndarray:
    """Firmware SyntheticAudioSource::sampleValue, shape (samples, channels), int16.

    The firmware truncates the absolute sample index to uint32 before the
    multiply, so the pattern wraps every 2**32 samples, and keeps the low 16 bits.
    """
    sample_index = (np.arange(samples_per_channel, dtype=np.uint64) + np.uint64(start_sample_index)) & np.uint64(
        0xFFFFFFFF
    )
    words = (sample_index * np.uint64(257)) & np.uint64(0xFFFFFFFF)
    channel_term = np.arange(channels, dtype=np.uint64) * np.uint64(0x1F1F)
    block = (words[:, None] ^ channel_term[None, :] ^ np.uint64(0x5A5A)) & np.uint64(INT16_MASK)
    return block.astype(np.uint16).view(np.int16)


@dataclass
class BootTrack:
    last_sequence: int | None = None
    last_end_sample_index: int | None = None
    timestamp_offset_ns: float | None = None
    first_counters: dict | None = None
    last_counters: dict | None = None


@dataclass
class RunState:
    started_monotonic: float = field(default_factory=time.monotonic)
    started_wall: float = field(default_factory=time.time)
    requests: int = 0
    parse_errors: int = 0
    parse_error_examples: list[str] = field(default_factory=list)
    frames: int = 0
    samples_per_channel_total: int = 0
    body_bytes: int = 0
    sequence_gaps: int = 0
    frames_missing: int = 0
    duplicate_frames: int = 0
    sample_index_gaps: int = 0
    samples_missing: int = 0
    bit_errors: int = 0
    frames_with_bit_errors: int = 0
    timestamp_jumps: int = 0
    max_timestamp_jump_ns: int = 0
    node_ids: set[str] = field(default_factory=set)
    peers: set[str] = field(default_factory=set)
    channels: set[int] = field(default_factory=set)
    sample_rates: set[int] = field(default_factory=set)
    boots: dict[tuple[str, int], BootTrack] = field(default_factory=dict)
    request_arrivals: list[float] = field(default_factory=list)
    bytes_per_second: dict[int, int] = field(default_factory=dict)
    gap_events: list[dict] = field(default_factory=list)


class CapacitySink:
    def __init__(self, gap_event_limit: int = 50) -> None:
        self.state = RunState()
        self.gap_event_limit = gap_event_limit
        self.queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()

    def reset(self) -> None:
        self.state = RunState()

    # --- verification (off the response path) -------------------------------------
    def verify_payload(self, arrival_monotonic: float, body: bytes) -> None:
        state = self.state
        state.requests += 1
        state.body_bytes += len(body)
        state.request_arrivals.append(arrival_monotonic)
        second = int(arrival_monotonic - state.started_monotonic)
        state.bytes_per_second[second] = state.bytes_per_second.get(second, 0) + len(body)
        try:
            payload = parse_binary_ingest_payload(body)
        except ValueError as exc:
            state.parse_errors += 1
            if len(state.parse_error_examples) < 5:
                state.parse_error_examples.append(str(exc))
            return
        node_id = payload.node.id
        state.node_ids.add(node_id)
        for buffered in payload.buffered_frames:
            self._verify_frame(node_id, (payload.node.metadata or {}).get("boot_count"), buffered)

    def _note_gap(self, kind: str, **details) -> None:
        if len(self.state.gap_events) < self.gap_event_limit:
            self.state.gap_events.append(
                {"kind": kind, "t_s": round(time.monotonic() - self.state.started_monotonic, 3), **details}
            )

    def _verify_frame(self, node_id: str, boot_count: int | None, buffered) -> None:
        state = self.state
        frame = buffered.frame
        diagnostics = frame.timing_diagnostics or {}
        transport = diagnostics.get("transport_health", {})
        boot_key = (node_id, int(transport.get("boot_id", boot_count or 0)))
        track = state.boots.setdefault(boot_key, BootTrack())
        state.frames += 1
        state.channels.add(frame.channels)
        state.sample_rates.add(frame.sample_rate_hz)
        samples = frame.samples_per_channel
        state.samples_per_channel_total += samples

        if track.last_sequence is not None:
            if frame.sequence <= track.last_sequence:
                # Re-sent batch the sink already accepted (node timed out after
                # the 202 left): not a loss, and its samples were verified once.
                state.duplicate_frames += 1
                return
            if frame.sequence != track.last_sequence + 1:
                missing = frame.sequence - track.last_sequence - 1
                state.sequence_gaps += 1
                state.frames_missing += missing
                self._note_gap("sequence", after=track.last_sequence, got=frame.sequence, missing=missing)
        track.last_sequence = frame.sequence

        if track.last_end_sample_index is not None and frame.start_sample_index != track.last_end_sample_index:
            state.sample_index_gaps += 1
            state.samples_missing += frame.start_sample_index - track.last_end_sample_index
            self._note_gap(
                "sample_index",
                expected=track.last_end_sample_index,
                got=frame.start_sample_index,
            )
        track.last_end_sample_index = frame.end_sample_index

        received = np.round(buffered.decoded_audio.T * 32768.0).astype(np.int32).astype(np.int16)
        expected = expected_synthetic_block(frame.start_sample_index, samples, frame.channels)
        mismatches = int(np.count_nonzero(received != expected))
        if mismatches:
            state.bit_errors += mismatches
            state.frames_with_bit_errors += 1
            self._note_gap("bit_error", sequence=frame.sequence, mismatched_samples=mismatches)

        offset_ns = frame.start_time_ns - frame.start_sample_index * 1e9 / frame.sample_rate_hz
        if track.timestamp_offset_ns is not None:
            jump_ns = int(abs(offset_ns - track.timestamp_offset_ns))
            if jump_ns > TIMESTAMP_JUMP_TOLERANCE_NS:
                state.timestamp_jumps += 1
                self._note_gap("timestamp_jump", sequence=frame.sequence, jump_ns=jump_ns)
            state.max_timestamp_jump_ns = max(state.max_timestamp_jump_ns, jump_ns)
        track.timestamp_offset_ns = offset_ns

        counters = {
            "frames_dropped": diagnostics.get("runner_frames_dropped", 0),
            "continuity_violations": diagnostics.get("runner_continuity_violations", 0),
            "publish_errors": diagnostics.get("runner_publish_errors", 0),
            "queue_overflows": diagnostics.get("runner_queue_overflows", 0),
            "queue_slots_high_water": transport.get("queue_slots_high_water", 0),
            "queue_slots_capacity": transport.get("queue_slots_capacity", 0),
            "publish_latency_ewma_ms": transport.get("publish_latency_ewma_ms", 0),
            "publish_latency_max_ms": transport.get("publish_latency_max_ms", 0),
            "wifi_rssi_dbm": transport.get("wifi_rssi_dbm", 0),
        }
        if track.first_counters is None:
            track.first_counters = counters
        track.last_counters = counters

    # --- summary -------------------------------------------------------------------
    def summary(self) -> dict:
        state = self.state
        elapsed = max(time.monotonic() - state.started_monotonic, 1e-9)
        arrivals = state.request_arrivals
        inter_arrival_ms = [1000.0 * (b - a) for a, b in zip(arrivals, arrivals[1:])]
        node_counter_deltas: dict[str, int] = {}
        for track in state.boots.values():
            if track.first_counters is None or track.last_counters is None:
                continue
            for key in ("frames_dropped", "continuity_violations", "publish_errors", "queue_overflows"):
                node_counter_deltas[key] = node_counter_deltas.get(key, 0) + int(
                    track.last_counters[key] - track.first_counters[key]
                )
        last_counters = next(
            (t.last_counters for t in reversed(list(state.boots.values())) if t.last_counters), {}
        )
        full_seconds = sorted(state.bytes_per_second.items())[1:-1]  # drop partial first/last
        per_second_mbps = [round(v * 8 / 1e6, 3) for _, v in full_seconds]
        sample_rate = next(iter(state.sample_rates), 0)
        failures = []
        if state.frames == 0:
            failures.append("no frames received")
        for name in (
            "parse_errors",
            "sequence_gaps",
            "sample_index_gaps",
            "bit_errors",
            "timestamp_jumps",
        ):
            if getattr(state, name):
                failures.append(f"{name}={getattr(state, name)}")
        for key in ("frames_dropped", "continuity_violations", "queue_overflows"):
            if node_counter_deltas.get(key, 0):
                failures.append(f"node_{key}+={node_counter_deltas[key]}")
        if len(state.boots) > 1:
            failures.append(f"node rebooted ({len(state.boots)} boots)")
        return {
            "verdict": "PASS" if not failures else "FAIL",
            "failures": failures,
            "elapsed_s": round(elapsed, 1),
            "node_ids": sorted(state.node_ids),
            "peers": sorted(state.peers),
            "boots": [f"{n}#{b}" for n, b in state.boots],
            "channels": sorted(state.channels),
            "sample_rates": sorted(state.sample_rates),
            "requests": state.requests,
            "frames": state.frames,
            "audio_seconds": round(state.samples_per_channel_total / sample_rate, 1) if sample_rate else 0,
            "goodput_mbps": round(state.body_bytes * 8 / elapsed / 1e6, 3),
            "per_second_mbps_min": min(per_second_mbps, default=0.0),
            "per_second_mbps_p50": statistics.median(per_second_mbps) if per_second_mbps else 0.0,
            "inter_arrival_ms_p50": round(statistics.median(inter_arrival_ms), 1) if inter_arrival_ms else 0,
            "inter_arrival_ms_p99": round(sorted(inter_arrival_ms)[int(0.99 * (len(inter_arrival_ms) - 1))], 1)
            if inter_arrival_ms
            else 0,
            "inter_arrival_ms_max": round(max(inter_arrival_ms), 1) if inter_arrival_ms else 0,
            "parse_errors": state.parse_errors,
            "parse_error_examples": state.parse_error_examples,
            "sequence_gaps": state.sequence_gaps,
            "frames_missing": state.frames_missing,
            "duplicate_frames": state.duplicate_frames,
            "sample_index_gaps": state.sample_index_gaps,
            "samples_missing": state.samples_missing,
            "bit_errors": state.bit_errors,
            "frames_with_bit_errors": state.frames_with_bit_errors,
            "timestamp_jumps": state.timestamp_jumps,
            "max_timestamp_step_ns": state.max_timestamp_jump_ns,
            "node_counter_deltas": node_counter_deltas,
            "node_last_counters": last_counters,
            "gap_events": state.gap_events,
            "pending_verification": self.queue.qsize(),
        }

    # --- HTTP ----------------------------------------------------------------------
    async def verifier(self) -> None:
        while True:
            arrival, body = await self.queue.get()
            try:
                self.verify_payload(arrival, body)
            except Exception as exc:  # keep the sink alive; surface as a parse error
                self.state.parse_errors += 1
                if len(self.state.parse_error_examples) < 5:
                    self.state.parse_error_examples.append(f"verifier: {exc!r}")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                if len(head) > MAX_HEADER_BYTES:
                    return
                lines = head.decode("latin-1").split("\r\n")
                method, path, _ = lines[0].split(" ", 2)
                headers = {}
                for line in lines[1:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        headers[key.strip().lower()] = value.strip()
                length = int(headers.get("content-length", "0"))
                if length > MAX_BODY_BYTES:
                    return
                body = await reader.readexactly(length) if length else b""
                arrival = time.monotonic()
                close_after = headers.get("connection", "").lower() == "close"
                if method == "POST" and path == INGEST_PATH:
                    peer = writer.get_extra_info("peername")
                    if peer:
                        self.state.peers.add(peer[0])
                    self.queue.put_nowait((arrival, body))
                    await self._respond(writer, 202, b'{"accepted":true}', close_after)
                elif method == "GET" and path == "/bench/summary":
                    while not self.queue.empty():  # settle pending verification first
                        await asyncio.sleep(0.05)
                    await self._respond(writer, 200, json.dumps(self.summary()).encode(), close_after)
                elif method == "POST" and path == "/bench/reset":
                    self.reset()
                    await self._respond(writer, 200, b'{"reset":true}', close_after)
                else:
                    await self._respond(writer, 404, b'{"error":"not found"}', close_after)
                if close_after:
                    return
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            return
        finally:
            writer.close()

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, body: bytes, close_after: bool) -> None:
        reason = {200: "OK", 202: "Accepted", 404: "Not Found"}[status]
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: {'close' if close_after else 'keep-alive'}\r\n\r\n"
            ).encode()
            + body
        )
        await writer.drain()


async def serve(host: str, port: int) -> None:
    sink = CapacitySink()
    verifier_task = asyncio.create_task(sink.verifier())
    server = await asyncio.start_server(sink.handle_client, host, port)
    print(json.dumps({"listening": f"{host}:{port}"}), flush=True)
    try:
        async with server:
            await server.serve_forever()
    finally:
        verifier_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
