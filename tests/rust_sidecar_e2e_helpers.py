"""Shared harness for end-to-end tests against the real Rust ingest sidecar.

Two launch modes:
- ``running_rust_sidecar``: the sidecar binary alone (manifests are read back
  from its ``/api/v1/dsp/results`` ring or journal);
- ``configure_full_app_with_rust_sidecar`` + ``TestClient(app)``: the full
  FastAPI app, which launches and supervises the sidecar itself, consumes its
  manifests and stores detections.

Classification is made deterministic with a fake sidecar classifier helper
process (``write_fake_sidecar_classifier_helper``) whose label is
authoritative on the Rust manifest.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minimappr.utils.audio import encode_pcm16le_b64
from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available

MINIMAPPR_ROOT = Path(__file__).resolve().parents[1]
HTTP_REQUEST_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.05
SIDECAR_READY_TIMEOUT_SECONDS = 15.0
SIDECAR_TERMINATE_TIMEOUT_SECONDS = 5.0
DSP_RESULTS_POLL_LIMIT = 200
SSE_REPLAY_IDLE_SECONDS = 1.0
FAKE_CLASSIFIER_LABEL = "winter wren"
FAKE_CLASSIFIER_CONFIDENCE = 0.91
FAKE_CLASSIFIER_RUNNER_UP_LABEL = "sparrow"
FAKE_CLASSIFIER_RUNNER_UP_CONFIDENCE = 0.04
MEBIBYTE = 1024 * 1024

# Sidecar budgets small enough for a test box; shared by both launch modes.
SIDECAR_TEST_BUDGET_ENVIRONMENT = {
    "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES": str(32 * MEBIBYTE),
    "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES": str(4 * MEBIBYTE),
    "MINIMAPPR_SIDECAR_DERIVED_CACHE_BUDGET_BYTES": str(16 * MEBIBYTE),
    "MINIMAPPR_SIDECAR_DERIVED_CACHE_ADMISSION_RESERVE_BYTES": str(2 * MEBIBYTE),
}

STANDALONE_SIDECAR_BASE_ENVIRONMENT = {
    "MINIMAPPR_SIDECAR_STORAGE_MODE": "journal",
    "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": "true",
    "MINIMAPPR_SIDECAR_HYBRID_RENDER_ENABLED": "true",
    "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE": "beamformed",
    "MINIMAPPR_LOCALIZATION_BAND_MIN_HZ": "300.0",
    "MINIMAPPR_LOCALIZATION_BAND_MAX_HZ": "3500.0",
    "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "30.0",
    "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.05",
    "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": "0.3",
    "MINIMAPPR_DEFAULT_TEMPERATURE_C": "20.0",
    "MINIMAPPR_DEFAULT_HUMIDITY": "0.5",
    **SIDECAR_TEST_BUDGET_ENVIRONMENT,
}

# Full-app settings: hybrid beamformed classification through the sidecar, a
# trigger that fires on any signal and no cooldown. Tests override per scene.
FULL_APP_BASE_ENVIRONMENT = {
    "MINIMAPPR_INGEST_STORAGE_MODE": "journal",
    "MINIMAPPR_DIRECT_INGEST_ENABLED": "false",
    "MINIMAPPR_INGEST_SIDECAR_ENABLED": "true",
    "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": "true",
    "MINIMAPPR_BIRDNET_ENABLED": "true",
    "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE": "beamformed",
    "MINIMAPPR_LOCALIZATION_ALGORITHM": "srp_phat",
    "MINIMAPPR_LOCALIZATION_STRATEGY": "fixed",
    "MINIMAPPR_LOCALIZATION_BAND_MIN_HZ": "300.0",
    "MINIMAPPR_LOCALIZATION_BAND_MAX_HZ": "3500.0",
    "MINIMAPPR_BIRDNET_CHUNKED_DISPATCH_ENABLED": "true",
    "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "30.0",
    "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS": "32.0",
    "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.05",
    "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": "0.3",
    "MINIMAPPR_DEFAULT_TEMPERATURE_C": "20.0",
    "MINIMAPPR_DEFAULT_HUMIDITY": "0.5",
    "MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS": "0.05",
    "MINIMAPPR_TRIGGER_RMS": "0.000001",
    "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": "0",
    "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS": "0.02",
    "MINIMAPPR_FUSION_WORKER_COUNT": "1",
    "MINIMAPPR_SNIPPET_RETENTION_SECONDS": "0",
    "MINIMAPPR_REPORTING_WINDOW_SECONDS": "1.0",
    "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON": "",
    **SIDECAR_TEST_BUDGET_ENVIRONMENT,
}


def pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict | list:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method=method,
        headers=headers or {},
    )
    with urllib.request.urlopen(request, timeout=HTTP_REQUEST_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def post_binary_ingest(port: int, payload: bytes) -> dict:
    response = http_json(
        port,
        "/api/v1/ingest/binary",
        method="POST",
        body=payload,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert isinstance(response, dict) and response.get("accepted") is True, response
    return response


def post_store_forward_ingest(port: int, payload: dict) -> dict:
    response = http_json(
        port,
        "/api/v1/ingest/store-forward",
        method="POST",
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert isinstance(response, dict) and response.get("accepted") is True, response
    return response


def ensure_sidecar_binary() -> Path:
    if not cargo_is_available():
        pytest.skip("cargo toolchain not available; cannot build the Rust sidecar")
    binary_path = build_rust_sidecar_binary()
    if not binary_path.exists():
        pytest.skip(f"Rust sidecar binary not found at {binary_path}")
    return binary_path


def _sidecar_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read()
    except Exception:
        return ""


def _wait_for_sidecar_ready(process: subprocess.Popen[str], *, port: int) -> None:
    deadline = time.monotonic() + SIDECAR_READY_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "Rust sidecar exited before becoming ready.\n"
                f"Captured output:\n{_sidecar_output(process)}"
            )
        try:
            payload = http_json(port, "/healthz")
            if payload.get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - transient startup race
            last_error = exc
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(
        "Timed out waiting for the Rust sidecar health endpoint.\n"
        f"Last error: {last_error}\n"
        f"Captured output:\n{_sidecar_output(process)}"
    )


@contextmanager
def running_rust_sidecar(tmp_path: Path, *, extra_env: dict[str, str] | None = None) -> Iterator[tuple[int, Path]]:
    """Run the sidecar binary alone; yields ``(port, spool_dir)``."""
    binary_path = ensure_sidecar_binary()
    port = pick_free_tcp_port()
    spool_dir = tmp_path / "sidecar_spool"
    env = {
        **os.environ,
        "MINIMAPPR_SIDECAR_PORT": str(port),
        "MINIMAPPR_INGEST_SPOOL_DIR": str(spool_dir),
        **STANDALONE_SIDECAR_BASE_ENVIRONMENT,
        **(extra_env or {}),
    }
    process = subprocess.Popen(
        [str(binary_path)],
        cwd=str(MINIMAPPR_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_sidecar_ready(process, port=port)
        yield port, spool_dir
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=SIDECAR_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=SIDECAR_TERMINATE_TIMEOUT_SECONDS)


def configure_full_app_with_rust_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sidecar_port: int,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Point the app at a fresh DB/spool and the worktree's sidecar binary.

    Enter ``TestClient(app)`` afterwards to start the app (and the sidecar).
    Returns the spool directory.
    """
    binary_path = ensure_sidecar_binary()
    spool_dir = tmp_path / "spool"
    environment = {
        "MINIMAPPR_DB_PATH": str(tmp_path / "http_app_sidecar.db"),
        "MINIMAPPR_SNIPPET_DIR": str(tmp_path / "snippets"),
        "MINIMAPPR_LARGE_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "MINIMAPPR_INGEST_SPOOL_DIR": str(spool_dir),
        "MINIMAPPR_INGEST_SIDECAR_BINARY_PATH": str(binary_path),
        "MINIMAPPR_SIDECAR_PORT": str(sidecar_port),
        **FULL_APP_BASE_ENVIRONMENT,
        **(extra_env or {}),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return spool_dir


def write_fake_sidecar_classifier_helper(tmp_path: Path) -> Path:
    """A classifier helper process that always answers ``FAKE_CLASSIFIER_LABEL``."""
    helper_path = tmp_path / "fake_sidecar_classifier_helper.py"
    response_scores = {
        FAKE_CLASSIFIER_LABEL: FAKE_CLASSIFIER_CONFIDENCE,
        FAKE_CLASSIFIER_RUNNER_UP_LABEL: FAKE_CLASSIFIER_RUNNER_UP_CONFIDENCE,
    }
    helper_path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "",
                "for line in sys.stdin:",
                "    if not line.strip():",
                "        continue",
                "    request = json.loads(line)",
                "    response = {",
                '        "request_id": request["request_id"],',
                f"        \"label\": {FAKE_CLASSIFIER_LABEL!r},",
                f"        \"label_confidence\": {FAKE_CLASSIFIER_CONFIDENCE!r},",
                f"        \"scores\": {response_scores!r},",
                "    }",
                '    sys.stdout.write(json.dumps(response) + "\\n")',
                "    sys.stdout.flush()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return helper_path


def fake_sidecar_classifier_command_json(helper_path: Path, python_executable: str) -> str:
    return json.dumps([python_executable, str(helper_path)])


def fetch_live_dsp_results(port: int) -> list[dict]:
    payloads = http_json(port, f"/api/v1/dsp/results?limit={DSP_RESULTS_POLL_LIMIT}")
    assert isinstance(payloads, list)
    return payloads


def replay_dsp_event_manifests(port: int, *, idle_seconds: float = SSE_REPLAY_IDLE_SECONDS) -> list[dict]:
    """Every manifest still in the sidecar's SSE replay ring (512 events), oldest first.

    ``/api/v1/dsp/results`` keeps only ~50 entries and raw-frame events flood
    it; the SSE stream replays everything after ``Last-Event-ID`` instead. Reads
    until the stream has been idle for ``idle_seconds``.
    """
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=idle_seconds)
    manifests: list[dict] = []
    try:
        connection.request("GET", "/api/v1/dsp/stream", headers={"Last-Event-ID": "0"})
        response = connection.getresponse()
        assert response.status == 200, response.status
        event_name = "message"
        while True:
            try:
                line = response.readline()
            except (socket.timeout, TimeoutError):
                break
            if not line:
                break
            text = line.decode("utf-8").rstrip("\r\n")
            if text.startswith("event:"):
                event_name = text.partition(":")[2].strip()
            elif text.startswith("data:") and event_name == "message":
                manifests.append(json.loads(text.partition(":")[2]))
            elif not text:
                event_name = "message"
    finally:
        connection.close()
    return manifests


def wait_for_live_dsp_results(
    port: int,
    *,
    predicate: Callable[[list[dict]], bool],
    timeout_s: float = 15.0,
    description: str,
) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    last_manifest_types: list[str] = []
    while time.monotonic() < deadline:
        payloads = fetch_live_dsp_results(port)
        last_manifest_types = [str(payload.get("manifest_type")) for payload in payloads]
        if predicate(payloads):
            return payloads
        time.sleep(POLL_INTERVAL_SECONDS)
    status = http_json(port, "/api/v1/dsp/status")
    raise AssertionError(
        f"Timed out waiting for live Rust DSP results matching {description}. "
        f"Last manifest types: {last_manifest_types}. "
        f"Status snapshot: {status}"
    )


def wait_for_detection_from_node(client: TestClient, *, node_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get("/api/v1/detections", params={"limit": 20})
        assert response.status_code == 200
        for detection in response.json():
            if detection.get("source_node_id") == node_id:
                return detection
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"Timed out waiting for detection from node {node_id}")


def store_forward_payload(
    *,
    node_id: str,
    node_type: str,
    sensor_offsets_m: list[list[float]],
    channels: np.ndarray,
    start_time_ns: int,
    sequence: int,
    cluster_id: str,
    cluster_sensor_positions: dict[str, list[float]],
    sample_rate_hz: int,
) -> dict:
    """JSON store-forward upload with an explicit cluster geometry."""
    samples_per_channel = int(channels.shape[1])
    end_time_ns = start_time_ns + int(round(samples_per_channel / sample_rate_hz * 1_000_000_000))
    return {
        "node": {
            "id": node_id,
            "node_type": node_type,
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": sensor_offsets_m,
            "capabilities": ["audio", "array_localization"],
            "metadata": {},
            "properties": {
                "cluster_sensor_positions": cluster_sensor_positions,
            },
            "cluster_id": cluster_id,
        },
        "sort_by_toa": True,
        "buffered_frames": [
            {
                "frame": {
                    "start_time_ns": start_time_ns,
                    "utc_start_ns": start_time_ns,
                    "utc_end_ns": end_time_ns,
                    "start_sample_index": 0,
                    "end_sample_index": samples_per_channel,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": int(channels.shape[0]),
                    "encoding": "pcm16le",
                    "samples_per_channel": samples_per_channel,
                    "samples_b64": encode_pcm16le_b64(channels),
                    "sequence": sequence,
                    "time_quality": "gps_locked",
                    "toa_ns": start_time_ns,
                    "tor_ns": start_time_ns + 1_000_000,
                    "source_type": "raw_sensor",
                }
            }
        ],
    }
