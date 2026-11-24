"""Out-of-process probes for the E10 hot-path gate: DSP event stream, RSS and disk writes.

Everything here observes the server from outside (HTTP, process table, file
system) so the probes work unchanged against any checkout's server.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil

BYTES_PER_MEBIBYTE = 1024 * 1024
HTTP_TIMEOUT_SECONDS = 10.0
SSE_READ_TIMEOUT_SECONDS = 120.0
LOCALIZATION_RESULT_MARKER = '"manifest_type":"localization_result"'
RAW_AUDIO_FRAME_MARKER = '"manifest_type":"raw_audio_frame"'
CLASSIFIER_RENDER_MARKER = '"manifest_type":"classifier_render"'
# Windows tagged with a coverage window type (e.g. classification_trailing) are
# re-emissions attached to a classifier render, not trigger solves.
COVERAGE_WINDOW_TYPE_KEY = "window_type"
REPLAY_GAP_EVENT_NAME = "replay_gap"


def http_get_json(port: int, path: str) -> dict | list:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read()
    return json.loads(payload) if payload else {}


@dataclass(frozen=True)
class ObservedSolve:
    arrival_monotonic_ns: int
    stream_key: str
    trigger_sample_index_start: int
    trigger_sample_count: int
    pair_count: int


@dataclass
class DspStreamObservations:
    solves: list[ObservedSolve] = field(default_factory=list)
    # stream_key -> [(sample_index_start, sample_count)] in arrival order.
    raw_audio_spans_by_stream: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    classifier_render_count: int = 0
    replay_gap_event_count: int = 0
    malformed_event_count: int = 0
    stream_error: str | None = None


class DspStreamListener:
    """Subscribes to the sidecar's ``/api/v1/dsp/stream`` like the app's own consumer does.

    Solves are localization_result manifests carrying TDOA pairs; the
    zero-pair localization_result manifests the sidecar emits per packet are
    attempt markers, not solves. Raw-audio spans let the harness check that
    every posted sample was forwarded to consumers without a hole.
    """

    def __init__(self, sidecar_port: int) -> None:
        self.sidecar_port = sidecar_port
        self.observations = DspStreamObservations()
        self._connected = threading.Event()
        self._frozen = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hot-path-gate-dsp-stream", daemon=True)

    def start(self, connect_timeout_seconds: float) -> None:
        self._thread.start()
        if not self._connected.wait(connect_timeout_seconds):
            raise RuntimeError("timed out connecting to the sidecar DSP event stream")

    def freeze(self) -> DspStreamObservations:
        """Stop recording (before server shutdown, so teardown is not an observation) and return the result."""
        self._frozen.set()
        return self.observations

    def _run(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.sidecar_port, timeout=SSE_READ_TIMEOUT_SECONDS)
        try:
            connection.request("GET", "/api/v1/dsp/stream", headers={"Accept": "text/event-stream"})
            response = connection.getresponse()
            self._connected.set()
            event_name = ""
            while True:
                line = response.readline()
                if not line:
                    return
                text = line.decode("utf-8").rstrip("\r\n")
                if text.startswith("event:"):
                    event_name = text[len("event:"):].strip()
                elif text.startswith("data:"):
                    self._handle_event(event_name, text[len("data:"):].strip())
                elif not text:
                    event_name = ""
        except (OSError, http.client.HTTPException) as error:
            if not self._frozen.is_set():
                self.observations.stream_error = repr(error)
        finally:
            self._connected.set()
            connection.close()

    def _handle_event(self, event_name: str, data: str) -> None:
        if self._frozen.is_set():
            return
        arrival_ns = time.monotonic_ns()
        if event_name == REPLAY_GAP_EVENT_NAME:
            # The broadcast dropped events for this subscriber: the run's
            # observations are incomplete and the result must say so.
            self.observations.replay_gap_event_count += 1
            return
        if CLASSIFIER_RENDER_MARKER in data:
            self.observations.classifier_render_count += 1
            return
        is_solve_candidate = LOCALIZATION_RESULT_MARKER in data
        is_raw_audio = RAW_AUDIO_FRAME_MARKER in data
        if not (is_solve_candidate or is_raw_audio):
            return
        try:
            manifest = json.loads(data)
            first_handle = (manifest.get("source_handles") or [])[0]
            stream_key = str(first_handle["stream_key"])
            sample_index_start = int(first_handle["sample_index_start"])
            sample_count = int(first_handle["sample_count"])
        except (ValueError, KeyError, IndexError, TypeError):
            self.observations.malformed_event_count += 1
            return
        if is_raw_audio:
            self.observations.raw_audio_spans_by_stream.setdefault(stream_key, []).append(
                (sample_index_start, sample_count)
            )
            return
        pair_count = len((manifest.get("localization") or {}).get("pair_tdoas") or [])
        is_trigger_window = COVERAGE_WINDOW_TYPE_KEY not in (manifest.get("coverage_stats") or {})
        if pair_count and is_trigger_window:
            self.observations.solves.append(
                ObservedSolve(arrival_ns, stream_key, sample_index_start, sample_count, pair_count)
            )


def node_id_for_stream_key(stream_key: str, node_ids: list[str]) -> str | None:
    # Sidecar stream keys are "<node_id>__<stream>__<hash>".
    return next((node_id for node_id in node_ids if stream_key.startswith(f"{node_id}__")), None)


@dataclass(frozen=True)
class RawAudioForwardingAccounting:
    missing_sample_count: int
    duplicated_sample_count: int
    out_of_order_span_count: int
    unmapped_stream_count: int


def account_raw_audio_forwarding(
    posted_spans_by_node: dict[str, list[tuple[int, int]]],
    forwarded_spans_by_stream: dict[str, list[tuple[int, int]]],
    total_sample_count: int,
) -> RawAudioForwardingAccounting:
    """Compare accepted posted samples with the raw audio the sidecar forwarded to consumers.

    Missing samples are lost packets. Out-of-order spans (a span starting
    before the end of one already forwarded) are counted apart: the
    sidecar's ingest workers may publish concurrently, which reorders but
    does not lose audio.
    """
    node_ids = list(posted_spans_by_node)
    forwarded_counts_by_node = {node_id: np.zeros(total_sample_count, dtype=np.int32) for node_id in node_ids}
    out_of_order_span_count, unmapped_stream_count = 0, 0
    for stream_key, spans in forwarded_spans_by_stream.items():
        node_id = node_id_for_stream_key(stream_key, node_ids)
        if node_id is None:
            unmapped_stream_count += 1
            continue
        highest_end_seen = -1
        for start, count in spans:
            forwarded_counts_by_node[node_id][start : start + count] += 1
            out_of_order_span_count += int(start < highest_end_seen)
            highest_end_seen = max(highest_end_seen, start + count)
    missing_sample_count, duplicated_sample_count = 0, 0
    for node_id, spans in posted_spans_by_node.items():
        posted_mask = np.zeros(total_sample_count, dtype=bool)
        for start, count in spans:
            posted_mask[start : start + count] = True
        forwarded_counts = forwarded_counts_by_node[node_id]
        missing_sample_count += int(np.count_nonzero(posted_mask & (forwarded_counts == 0)))
        duplicated_sample_count += int(np.sum(np.maximum(forwarded_counts - 1, 0)))
    return RawAudioForwardingAccounting(
        missing_sample_count, duplicated_sample_count, out_of_order_span_count, unmapped_stream_count
    )


class PeakRssSampler:
    """Samples resident set size of named processes until stopped; keeps the peak per name."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes: dict[str, int] = {}
        self._processes: dict[str, psutil.Process] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hot-path-gate-rss", daemon=True)

    def watch(self, name: str, pid: int) -> None:
        self._processes[name] = psutil.Process(pid)
        self.peak_rss_bytes.setdefault(name, 0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=self.interval_seconds * 10)
        return {name: peak / BYTES_PER_MEBIBYTE for name, peak in self.peak_rss_bytes.items()}

    def freeze(self) -> DspStreamObservations:
        """Stop recording (before server shutdown, so teardown is not an observation) and return the result."""
        self._frozen.set()
        return self.observations

    def _run(self) -> None:
        while not self._stop.is_set():
            for name, process in list(self._processes.items()):
                try:
                    rss_bytes = process.memory_info().rss
                except psutil.Error:
                    continue
                if rss_bytes > self.peak_rss_bytes[name]:
                    self.peak_rss_bytes[name] = rss_bytes
            self._stop.wait(self.interval_seconds)


def snapshot_file_sizes(roots: list[Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for root in roots:
        if not root.exists():
            continue
        for directory, _subdirectories, file_names in os.walk(root):
            for file_name in file_names:
                path = Path(directory) / file_name
                try:
                    sizes[str(path.resolve())] = path.stat().st_size
                except FileNotFoundError:
                    continue
    return sizes


def written_files(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Files created or grown between two snapshots -> bytes added."""
    return {
        path: size - before.get(path, 0)
        for path, size in after.items()
        if path not in before or size > before[path]
    }
