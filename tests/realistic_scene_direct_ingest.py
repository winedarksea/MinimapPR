"""The Python-direct-ingest half of ``run_streaming_scene_through_full_app``.

Same firmware-shaped uploads, posted to the app's own ``/api/v1/ingest/binary``
with the Rust sidecar off, so the app's ``IngestProcessor`` trigger, Python
localization, classification and storage handle them. The classifier is a
constant bird (the sidecar runs use a constant-answer helper process too), so
the funnel measures admission and localization, not classifier recall.

Direct ingest answers every upload with per-frame ``triggered`` flags; the
receipts keep them, so trigger admission is observed without a manifest stream.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minimappr.classifiers.base import AudioClassifier
from minimappr.main import app
from minimappr.models import ClassificationResult
from tests.acoustic_scene_synthesis import AcousticScene
from tests.realistic_scene_full_app_run import (
    INGEST_BACKEND_PYTHON_DIRECT,
    PACKET_TIME_BEHIND_RECEIPT_NS,
    FullAppRunResult,
    StreamingNode,
    build_uploads_by_node,
    enter_hand_measured_positions,
    stream_uploads,
    wait_for_detections,
)
from tests.rust_sidecar_e2e_helpers import (
    FAKE_CLASSIFIER_CONFIDENCE,
    FAKE_CLASSIFIER_LABEL,
    FULL_APP_BASE_ENVIRONMENT,
)

DETECTIONS_QUERY_LIMIT = 1000
HTTP_OK = 200

# The sidecar base settings, with ingest routed to the app itself. Spool mode
# with a parked consumer: binary uploads are processed inline by the request.
DIRECT_INGEST_BASE_ENVIRONMENT = {
    **FULL_APP_BASE_ENVIRONMENT,
    "MINIMAPPR_DIRECT_INGEST_ENABLED": "true",
    "MINIMAPPR_INGEST_SIDECAR_ENABLED": "false",
    "MINIMAPPR_INGEST_STORAGE_MODE": "spool",
    "MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS": "3600",
}


class ConstantBirdClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        return ClassificationResult(
            label=FAKE_CLASSIFIER_LABEL,
            confidence=FAKE_CLASSIFIER_CONFIDENCE,
            scores={FAKE_CLASSIFIER_LABEL: FAKE_CLASSIFIER_CONFIDENCE},
            features={},
        )


def configure_full_app_with_direct_ingest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, app_environment: Mapping[str, str]
) -> None:
    environment = {
        "MINIMAPPR_DB_PATH": str(tmp_path / "http_app_direct.db"),
        "MINIMAPPR_SNIPPET_DIR": str(tmp_path / "snippets"),
        "MINIMAPPR_LARGE_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "MINIMAPPR_TRAINING_DATASET_DIR": str(tmp_path / "training"),
        "MINIMAPPR_INGEST_SPOOL_DIR": str(tmp_path / "spool"),
        **DIRECT_INGEST_BASE_ENVIRONMENT,
        **app_environment,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("minimappr.main._build_runtime_classifier", lambda settings: ConstantBirdClassifier())


def _post_direct_binary_ingest(client: TestClient, payload: bytes) -> dict:
    response = client.post("/api/v1/ingest/binary", content=payload, headers={"content-type": "application/octet-stream"})
    assert response.status_code == HTTP_OK, response.text
    return response.json()


def _node_detections(client: TestClient, node_ids: set[str]) -> list[dict]:
    response = client.get("/api/v1/detections", params={"limit": DETECTIONS_QUERY_LIMIT})
    assert response.status_code == HTTP_OK
    return [detection for detection in response.json() if detection.get("source_node_id") in node_ids]


def run_streaming_scene_through_direct_ingest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    scene: AcousticScene,
    nodes: Sequence[StreamingNode],
    captures_by_node: Mapping[str, np.ndarray],
    app_environment: Mapping[str, str],
    detection_wait_seconds: float,
    detections_complete: Callable[[FullAppRunResult], bool],
) -> FullAppRunResult:
    configure_full_app_with_direct_ingest(monkeypatch, tmp_path, app_environment)
    node_ids = {node.scene_node.node_id for node in nodes}
    with TestClient(app) as client:
        enter_hand_measured_positions(client, nodes)
        result = FullAppRunResult(
            capture_start_packet_time_ns=time.time_ns() - PACKET_TIME_BEHIND_RECEIPT_NS,
            ingest_backend=INGEST_BACKEND_PYTHON_DIRECT,
        )
        uploads = build_uploads_by_node(
            nodes,
            captures_by_node,
            sample_rate_hz=scene.sample_rate_hz,
            capture_start_packet_time_ns=result.capture_start_packet_time_ns,
        )
        stream_uploads(nodes, uploads, lambda payload: _post_direct_binary_ingest(client, payload), result)
        wait_for_detections(lambda: _node_detections(client, node_ids), result, detection_wait_seconds, detections_complete)
    return result
