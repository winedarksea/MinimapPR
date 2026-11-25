"""Drive a running ingest sidecar with a calibration bundle's uploads.

The uploads come from ``minimappr/sim/replay_frames.py`` (one firmware frame
per recorded frame, with its own packet timing). They are posted to the
sidecar's ``/api/v1/ingest/binary`` in the order they would arrive live: by
first packet timestamp, ties broken by a stated node order. Posting is as fast
as the sidecar accepts unless a pacing is given.

Launching the sidecar or the full app is the caller's business (tests use
``tests/rust_sidecar_e2e_helpers.py``); this module only needs the port.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from minimappr.calibration.bundle import CalibrationBundle
from minimappr.models import NodeType
from minimappr.sim.replay_frames import ReplayUpload, ReplayUploadFraming, bundle_uploads_by_node

BINARY_INGEST_PATH = "/api/v1/ingest/binary"
UPLOAD_HTTP_TIMEOUT_SECONDS = 5.0
NANOSECONDS_PER_SECOND = 1_000_000_000


class ReplayUploadRejected(RuntimeError):
    """The sidecar answered an upload without ``accepted: true``."""


@dataclass(frozen=True, slots=True)
class ReplayPacing:
    """Space posts by their packet-time differences divided by ``speedup`` (1.0 = real time)."""

    speedup: float

    def __post_init__(self) -> None:
        if self.speedup <= 0:
            raise ValueError("speedup must be positive")


@dataclass(slots=True)
class SidecarReplaySummary:
    uploads_posted: int = 0
    frames_by_node: dict[str, int] = field(default_factory=dict)
    first_packet_time_ns: int | None = None
    last_upload_first_packet_time_ns: int | None = None


def interleave_uploads_by_packet_time(
    uploads_by_node: Mapping[str, Sequence[ReplayUpload]],
    *,
    node_order: Sequence[str] | None = None,
) -> list[ReplayUpload]:
    """Live arrival order: first packet time, then position in ``node_order``
    (default: mapping order), then each node's own upload order."""
    ordered_node_ids = list(node_order) if node_order is not None else list(uploads_by_node)
    missing = set(uploads_by_node) - set(ordered_node_ids)
    if missing:
        raise ValueError(f"node_order is missing {sorted(missing)}")
    node_rank = {node_id: rank for rank, node_id in enumerate(ordered_node_ids)}
    keyed = [
        ((upload.first_packet_time_ns, node_rank[node_id], upload_index), upload)
        for node_id, uploads in uploads_by_node.items()
        for upload_index, upload in enumerate(uploads)
    ]
    return [upload for _key, upload in sorted(keyed, key=lambda item: item[0])]


def post_binary_upload(port: int, payload: bytes) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{BINARY_INGEST_PATH}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=UPLOAD_HTTP_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8") or "{}")
    if not isinstance(body, dict) or body.get("accepted") is not True:
        raise ReplayUploadRejected(f"sidecar did not accept the upload: {body}")
    return body


def drive_sidecar_with_uploads(
    port: int,
    schedule: Sequence[ReplayUpload],
    *,
    pacing: ReplayPacing | None = None,
    post_upload: Callable[[int, bytes], dict] = post_binary_upload,
) -> SidecarReplaySummary:
    summary = SidecarReplaySummary()
    replay_started_monotonic = time.monotonic()
    for upload in schedule:
        if summary.first_packet_time_ns is None:
            summary.first_packet_time_ns = upload.first_packet_time_ns
        if pacing is not None:
            due_after_seconds = (
                (upload.first_packet_time_ns - summary.first_packet_time_ns) / NANOSECONDS_PER_SECOND / pacing.speedup
            )
            wait_seconds = replay_started_monotonic + due_after_seconds - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        post_upload(port, upload.payload)
        summary.uploads_posted += 1
        summary.frames_by_node[upload.node_id] = summary.frames_by_node.get(upload.node_id, 0) + upload.frame_count
        summary.last_upload_first_packet_time_ns = upload.first_packet_time_ns
    return summary


def replay_bundle_through_sidecar(
    bundle: CalibrationBundle,
    port: int,
    framing: ReplayUploadFraming,
    *,
    node_order: Sequence[str] | None = None,
    node_type_by_node_id: Mapping[str, NodeType] | None = None,
    pacing: ReplayPacing | None = None,
) -> SidecarReplaySummary:
    """Encode every bundle node's uploads and post them in live arrival order."""
    uploads_by_node = bundle_uploads_by_node(
        bundle,
        framing,
        node_ids=node_order,
        node_type_by_node_id=node_type_by_node_id,
    )
    schedule = interleave_uploads_by_packet_time(uploads_by_node, node_order=node_order)
    return drive_sidecar_with_uploads(port, schedule, pacing=pacing)
