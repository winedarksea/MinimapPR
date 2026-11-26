"""Make one node of a streamed scene the solve's reporter (E5 reporter independence).

The sidecar's reporter is whichever node's frame triggers first, which the
interleaved upload order leaves to scheduling. Holding every other node's
uploads back a few rounds makes ``lead_node_id`` trigger first without
touching any packet time: TDOAs come from packet timestamps, and the coverage
wait collects the late nodes' audio as it would from a slow uplink. Held
uploads are released before the run first reads detections.

Patches ``tests.realistic_scene_full_app_run`` at runtime only, so the shared
streaming helper stays unchanged.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import pytest

import tests.realistic_scene_full_app_run as full_app_run

# 100 ms uploads: three rounds (300 ms) cover a 10 m layout's ~30 ms arrival
# spread plus the upload in which the lead's onset lands.
DEFAULT_OTHER_NODES_HOLD_ROUNDS = 3
# Receipt for an upload still held back: not accepted (yet). The range matrix
# never reads receipts; this only keeps UploadReceipt.from_response total.
HELD_UPLOAD_RESPONSE: dict = {"accepted": False}


def lead_reporter_uploads(
    monkeypatch: pytest.MonkeyPatch,
    lead_node_id: str,
    *,
    hold_rounds: int = DEFAULT_OTHER_NODES_HOLD_ROUNDS,
) -> None:
    original_build = full_app_run.build_uploads_by_node
    original_post = full_app_run.post_binary_ingest
    original_read_detections = full_app_run._rust_handoff_detections
    node_id_by_upload: dict[int, str] = {}
    held: deque[tuple[int, int, bytes]] = deque()
    lead_post_count = 0

    def recording_build(nodes: Sequence, captures_by_node, **kwargs):
        uploads = original_build(nodes, captures_by_node, **kwargs)
        for node_id, node_uploads in uploads.items():
            for upload in node_uploads:
                node_id_by_upload[id(upload)] = node_id
        return uploads

    def release_held(*, all_held: bool) -> None:
        while held and (all_held or lead_post_count - held[0][1] >= hold_rounds):
            port, _, upload = held.popleft()
            original_post(port, upload)

    def holding_post(port: int, upload: bytes):
        nonlocal lead_post_count
        if node_id_by_upload.get(id(upload)) != lead_node_id:
            held.append((port, lead_post_count, upload))
            # Not yet sent, so no ingest answer exists; the shared streamer
            # records a receipt per upload and needs a mapping.
            return HELD_UPLOAD_RESPONSE
        response = original_post(port, upload)
        lead_post_count += 1
        release_held(all_held=False)
        return response

    def releasing_read_detections(client, node_ids):
        release_held(all_held=True)
        return original_read_detections(client, node_ids)

    monkeypatch.setattr(full_app_run, "build_uploads_by_node", recording_build)
    monkeypatch.setattr(full_app_run, "post_binary_ingest", holding_post)
    monkeypatch.setattr(full_app_run, "_rust_handoff_detections", releasing_read_detections)
