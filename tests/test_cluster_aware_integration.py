from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.fusion_node import EventCandidate, FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import ClusterSpec, GeoPoint, LocalizationResult, NodeSpec, NodeType, SyncGrade, TimeQuality
from minimappr.storage.db import Storage


SAMPLE_RATE_HZ = 16_000
START_TIME_NS = 1_000_000_000
SAMPLE_COUNT = 640
WINDOW_SECONDS = SAMPLE_COUNT / SAMPLE_RATE_HZ
EVENT_TIME_NS = START_TIME_NS + int(round((WINDOW_SECONDS / 2.0) * 1_000_000_000))


class _CapturingLocalizer:
    def __init__(self) -> None:
        self.calls = 0
        self.sensor_ids: list[str] = []
        self.sensor_weights: dict[str, float] | None = None

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        del sensor_windows, sample_rate_hz, temperature_c, humidity_fraction
        self.calls += 1
        self.sensor_ids = sorted(sensor_positions.keys())
        self.sensor_weights = sensor_weights
        return LocalizationResult(
            position_m=(1.0, 1.0, 2.0),
            confidence=0.8,
            gdop=1.0,
            reference_sensor=self.sensor_ids[0],
            tdoa_s={},
        )


def _single_mic_node(node_id: str, position_m: tuple[float, float, float]) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=position_m,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )


async def _build_fusion(
    tmp_path: Path,
    *,
    cluster_aware_localization: bool,
    localizer,
    cluster_registry: ClusterRegistry,
) -> tuple[FusionNode, Storage, NodeRegistry, MultiSensorBuffer]:
    settings = Settings(
        db_path=tmp_path / "cluster_aware.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=WINDOW_SECONDS,
        classification_window_seconds=WINDOW_SECONDS,
        max_sensor_buffer_seconds=2.0,
        min_sensors_for_3d=4,
        min_sensors_for_2d=3,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        cluster_aware_localization=cluster_aware_localization,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()
    registry = NodeRegistry()
    buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)
    tracker = TrackManager(settings)
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0),
        mode="flat",
    )
    zone_matcher = ZoneMatcher(storage=storage)

    async def _live(payload: dict) -> None:
        _ = payload

    fusion = FusionNode(
        settings=settings,
        registry=registry,
        buffer=buffer,
        localizer=localizer,
        classifier=HeuristicClassifier(),
        tracker=tracker,
        storage=storage,
        live_callback=_live,
        coordinate_frame=coordinate_frame,
        zone_matcher=zone_matcher,
        cluster_registry=cluster_registry,
    )
    return fusion, storage, registry, buffer


@pytest.mark.asyncio
async def test_cluster_aware_localization_uses_cluster_sensor_scope(tmp_path: Path) -> None:
    cluster_registry = ClusterRegistry()
    localizer = _CapturingLocalizer()
    fusion, storage, registry, buffer = await _build_fusion(
        tmp_path,
        cluster_aware_localization=True,
        localizer=localizer,
        cluster_registry=cluster_registry,
    )

    nodes = {
        "n0": (0.0, 0.0, 2.0),
        "n1": (2.0, 0.0, 2.0),
        "n2": (2.0, 2.0, 2.0),
        "n3": (0.0, 2.0, 2.0),
    }
    try:
        for node_id, position in nodes.items():
            await registry.upsert(_single_mic_node(node_id, position), last_seen_ns=START_TIME_NS)
        for node_id in ("n0", "n1", "n2"):
            await registry.update_sensor_sync_grade(f"{node_id}:ch0", SyncGrade.GPS_PPS)
        await registry.update_sensor_sync_grade("n3:ch0", SyncGrade.NTP)

        await cluster_registry.upsert(ClusterSpec(
            id="square",
            member_node_ids=list(nodes.keys()),
            declared_sync_grade=SyncGrade.GPS_PPS,
        ))
        await cluster_registry.update_node_memberships(registry)

        samples = np.ones(SAMPLE_COUNT, dtype=np.float32) * 0.1
        for node_id in nodes:
            await buffer.append(
                sensor_id=f"{node_id}:ch0",
                sample_rate_hz=SAMPLE_RATE_HZ,
                start_time_ns=START_TIME_NS,
                samples=samples,
            )

        localized = await fusion._localize_candidate(
            EventCandidate(
                id="evt-0001",
                source_node_id="n0",
                event_time_ns=EVENT_TIME_NS,
                sample_rate_hz=SAMPLE_RATE_HZ,
                source_type="raw_sensor",
                time_quality=TimeQuality.GPS_LOCKED,
                source_observation_ids=[],
            )
        )

        assert localized is not None
        assert localizer.calls == 1
        assert localizer.sensor_ids == ["n0:ch0", "n1:ch0", "n2:ch0", "n3:ch0"]
        assert localized.selected_sensor_ids == ["n0:ch0", "n1:ch0", "n2:ch0", "n3:ch0"]
        assert localizer.sensor_weights == {
            "n0:ch0": pytest.approx(1.0),
            "n1:ch0": pytest.approx(1.0),
            "n2:ch0": pytest.approx(1.0),
            "n3:ch0": pytest.approx(0.25),
        }
    finally:
        await storage.close()


def test_settings_from_env_reads_cluster_aware_localization(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAPPR_CLUSTER_AWARE_LOCALIZATION", "true")

    settings = Settings.from_env()

    assert settings.cluster_aware_localization is True
    assert settings.localization_config().cluster_aware_localization is True