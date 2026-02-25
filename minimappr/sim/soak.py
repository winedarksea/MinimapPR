"""End-to-end soak harness for API + fusion runtime stability checks."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from minimappr.models import NodeSpec, NodeType
from minimappr.sim.run_demo import render_sensor_frame
from minimappr.utils.audio import encode_pcm16le_b64


@dataclass(slots=True)
class SoakConfig:
    duration_seconds: float = 300.0
    host: str = "127.0.0.1"
    port: int = 8099
    sample_rate_hz: int = 16_000
    frame_size: int = 1024
    poll_interval_seconds: float = 1.0
    db_path: Path = Path("/tmp/minimappr_soak.db")
    snippet_dir: Path = Path("/tmp/minimappr_soak_snippets")
    trigger_rms: float = 0.01
    trigger_cooldown_seconds: float = 0.2
    worker_count: int = 1


@dataclass(slots=True)
class SoakExpectations:
    min_detections: int = 25
    min_full_3d_detections: int = 5
    max_stage_failures: int = 0
    max_backpressure_drops: int = 0
    require_workers_running: bool = True
    max_frames_rejected: int = 0


@dataclass(slots=True)
class SoakResult:
    ok: bool
    errors: list[str]
    final_status: dict
    health_errors: int
    ingestion_errors: int
    poll_samples: int
    detections_total: int
    full_3d_detections: int
    localization_methods: dict[str, int]


def build_four_point_nodes() -> list[NodeSpec]:
    positions = [
        (0.0, 0.0, 1.8),
        (4.5, 0.0, 1.8),
        (0.0, 4.5, 1.8),
        (4.5, 4.5, 1.8),
    ]
    return [
        NodeSpec(
            id=f"soak-point-{idx+1}",
            node_type=NodeType.POINT,
            position_m=position,
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio", "gps_pps"],
            metadata={"hw": "soak_sim"},
        )
        for idx, position in enumerate(positions)
    ]


async def wait_for_server(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    async with httpx.AsyncClient(timeout=1.5) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError(f"Server did not become healthy within {timeout_seconds:.1f}s")


async def stream_four_point_nodes(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    duration_seconds: float,
    sample_rate_hz: int,
    frame_size: int,
) -> int:
    nodes = build_four_point_nodes()
    sensor_positions = [np.asarray(node.position_m, dtype=np.float64) for node in nodes]
    frame_duration_s = frame_size / float(sample_rate_hz)
    frame_duration_ns = int(frame_duration_s * 1_000_000_000)
    stream_start_ns = time.time_ns()
    wall_start = time.perf_counter()
    ingest_errors = 0
    frame_idx = 0
    ingest_url = f"{base_url}/api/v1/ingest/frame"

    while True:
        frame_start_s = frame_idx * frame_duration_s
        if frame_start_s >= duration_seconds:
            break
        frame_start_ns = stream_start_ns + frame_idx * frame_duration_ns

        requests = []
        for node, sensor_position in zip(nodes, sensor_positions):
            audio = render_sensor_frame(
                sensor_position_m=sensor_position,
                frame_start_s=frame_start_s,
                frame_size=frame_size,
                sample_rate_hz=sample_rate_hz,
                sound_speed_mps=343.2,
                noise_std=0.008,
            )[None, :]
            payload = {
                "node": node.model_dump(mode="json"),
                "frame": {
                    "start_time_ns": frame_start_ns,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(audio),
                    "sequence": frame_idx,
                    "source_type": "raw_sensor",
                },
            }
            requests.append(client.post(ingest_url, json=payload))

        responses = await asyncio.gather(*requests, return_exceptions=True)
        for response in responses:
            if isinstance(response, Exception):
                ingest_errors += 1
                continue
            if response.status_code != 200:
                ingest_errors += 1

        frame_idx += 1
        target = wall_start + frame_idx * frame_duration_s
        await asyncio.sleep(max(0.0, target - time.perf_counter()))

    return ingest_errors


async def poll_fusion_health(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    poll_interval_seconds: float,
    stop: asyncio.Event,
) -> tuple[list[dict], int]:
    samples: list[dict] = []
    health_errors = 0
    while not stop.is_set():
        try:
            health = await client.get(f"{base_url}/health")
            status = await client.get(f"{base_url}/api/v1/fusion/status")
            if health.status_code != 200 or status.status_code != 200:
                health_errors += 1
            else:
                samples.append(
                    {
                        "health": health.json(),
                        "status": status.json(),
                    }
                )
        except Exception:
            health_errors += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_interval_seconds)
        except asyncio.TimeoutError:
            continue
    return samples, health_errors


def detection_breakdown(db_path: Path) -> tuple[int, int, dict[str, int]]:
    if not db_path.exists():
        return 0, 0, {}
    total = 0
    full_3d = 0
    methods: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT feature_summary_json FROM detections").fetchall()
    for (raw,) in rows:
        total += 1
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        tier = str(payload.get("capability_tier", "unknown"))
        method = str(payload.get("localization_method", "unknown"))
        if tier == "full_3d":
            full_3d += 1
        methods[method] = methods.get(method, 0) + 1
    return total, full_3d, dict(sorted(methods.items(), key=lambda item: item[0]))


def evaluate_soak(
    *,
    final_status: dict,
    poll_samples: list[dict],
    health_errors: int,
    ingestion_errors: int,
    detections_total: int,
    full_3d_detections: int,
    expectations: SoakExpectations,
) -> list[str]:
    errors: list[str] = []
    metrics = final_status.get("metrics", {}) if isinstance(final_status, dict) else {}
    workers = final_status.get("workers", {}) if isinstance(final_status, dict) else {}

    if health_errors > 0:
        errors.append(f"health polling observed {health_errors} failed sample(s)")
    if ingestion_errors > 0:
        errors.append(f"ingest requests failed {ingestion_errors} time(s)")
    if detections_total < expectations.min_detections:
        errors.append(
            f"detections_total={detections_total} below minimum {expectations.min_detections}"
        )
    if full_3d_detections < expectations.min_full_3d_detections:
        errors.append(
            f"full_3d_detections={full_3d_detections} below minimum {expectations.min_full_3d_detections}"
        )

    stage_failures = int(metrics.get("localization_failures", 0)) + int(
        metrics.get("classification_failures", 0)
    ) + int(metrics.get("rules_failures", 0))
    if stage_failures > expectations.max_stage_failures:
        errors.append(
            f"stage_failures={stage_failures} above max {expectations.max_stage_failures}"
        )

    stage_drops = int(metrics.get("stage_drops_backpressure", 0))
    if stage_drops > expectations.max_backpressure_drops:
        errors.append(
            f"stage_drops_backpressure={stage_drops} above max {expectations.max_backpressure_drops}"
        )

    frames_rejected = int(metrics.get("frames_rejected", 0))
    if frames_rejected > expectations.max_frames_rejected:
        errors.append(
            f"frames_rejected={frames_rejected} above max {expectations.max_frames_rejected}"
        )

    if expectations.require_workers_running:
        if int(workers.get("localization_running", 0)) < 1:
            errors.append("localization worker is not running at end of soak")
        if int(workers.get("classification_running", 0)) < 1:
            errors.append("classification worker is not running at end of soak")
        if int(workers.get("rules_running", 0)) < 1:
            errors.append("rules worker is not running at end of soak")
        for sample in poll_samples:
            sample_workers = sample["status"].get("workers", {})
            if int(sample_workers.get("localization_running", 0)) < 1:
                errors.append("localization worker dropped during soak")
                break
            if int(sample_workers.get("classification_running", 0)) < 1:
                errors.append("classification worker dropped during soak")
                break
            if int(sample_workers.get("rules_running", 0)) < 1:
                errors.append("rules worker dropped during soak")
                break

    return errors

