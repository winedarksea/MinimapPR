from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("birdnet")

from minimappr.classifiers.birdnet import BirdNETClassifier
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64
from tests.helpers import (
    SIRITH_TETRA_SENSOR_OFFSETS_M,
    load_wav_fixture_mono,
    prepend_noise_padding_to_duration,
    resample_signal,
    split_channels_into_frames,
    synthesize_delayed_array_channels,
)


HOUSE_FINCH_FIXTURE_PATH = Path(__file__).with_name("house_finch.wav")
HOUSE_FINCH_SOURCE_POSITION_M = (0.35, 0.18, 0.07)
HOUSE_FINCH_LABEL = "house finch"
DEFAULT_SITE_ORIGIN = GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0)
TIGHT_SRP_GRID_RESOLUTION_M = 0.05
TIGHT_SRP_SEARCH_PADDING_M = 0.3
TIGHT_LOCALIZATION_MAX_ERROR_M = 0.14


@pytest.fixture(scope="module")
def house_finch_fixture_48khz() -> tuple[np.ndarray, int]:
    return load_wav_fixture_mono(HOUSE_FINCH_FIXTURE_PATH)


@pytest.fixture(scope="module")
def birdnet_classifier() -> BirdNETClassifier:
    return BirdNETClassifier(min_confidence=0.05)


def _node_spec() -> NodeSpec:
    return NodeSpec(
        id="sirith-house-finch",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
        capabilities=["audio", "array_localization"],
        metadata={},
    )


def _sensor_positions() -> dict[str, np.ndarray]:
    return {
        f"sirith-house-finch:ch{index}": np.asarray(offset, dtype=np.float64)
        for index, offset in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M)
    }


def _resampled_fixture(
    fixture: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> np.ndarray:
    samples, fixture_sample_rate_hz = fixture
    return resample_signal(samples, fixture_sample_rate_hz, sample_rate_hz)


def _build_profile_settings(
    tmp_path: Path,
    *,
    sample_rate_hz: int,
    coordinate_mode: str = "flat",
    trigger_rms: float = 0.006,
    trigger_cooldown_seconds: float = 1.0,
    localization_srp_grid_resolution_m: float | None = None,
    localization_search_padding_m: float | None = None,
) -> Settings:
    settings_kwargs = dict(
        runtime_profile="birdnet_hybrid_production",
        db_path=tmp_path / f"birdnet_hybrid_{sample_rate_hz}.db",
        snippet_dir=tmp_path / f"snippets_{sample_rate_hz}",
        snippet_retention_seconds=0,
        trigger_rms=trigger_rms,
        trigger_cooldown_seconds=trigger_cooldown_seconds,
        fusion_worker_count=1,
        fusion_localization_queue_size=16,
        fusion_classification_queue_size=16,
        fusion_rules_queue_size=16,
        site_origin_lat=DEFAULT_SITE_ORIGIN.lat,
        site_origin_lon=DEFAULT_SITE_ORIGIN.lon,
        site_origin_alt_m=DEFAULT_SITE_ORIGIN.alt_m,
        coordinate_mode=coordinate_mode,
        model_chain_config_path=tmp_path / "missing_model_chain.json",
        birdnet_trigger_min_confidence=0.05,
    )
    if localization_srp_grid_resolution_m is not None:
        settings_kwargs["localization_srp_grid_resolution_m"] = localization_srp_grid_resolution_m
    if localization_search_padding_m is not None:
        settings_kwargs["localization_search_padding_m"] = localization_search_padding_m
    settings = Settings(**settings_kwargs)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    return settings


async def _start_fusion_with_profile(
    tmp_path: Path,
    *,
    sample_rate_hz: int,
    coordinate_mode: str = "flat",
    trigger_rms: float = 0.006,
    trigger_cooldown_seconds: float = 1.0,
    localization_srp_grid_resolution_m: float | None = None,
    localization_search_padding_m: float | None = None,
) -> tuple[FusionNode, Storage, Settings]:
    settings = _build_profile_settings(
        tmp_path,
        sample_rate_hz=sample_rate_hz,
        coordinate_mode=coordinate_mode,
        trigger_rms=trigger_rms,
        trigger_cooldown_seconds=trigger_cooldown_seconds,
        localization_srp_grid_resolution_m=localization_srp_grid_resolution_m,
        localization_search_padding_m=localization_search_padding_m,
    )
    storage = Storage(settings.db_path)
    await storage.initialize()
    coordinate_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode=settings.coordinate_mode)
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=build_localizer_from_settings(settings),
        classifier=create_classifier(settings),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=coordinate_frame,
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    return fusion, storage, settings


async def _stream_fixture_into_fusion(
    fusion: FusionNode,
    *,
    sample_rate_hz: int,
    frame_samples: int,
    fixture: tuple[np.ndarray, int],
    classification_window_seconds: float,
    start_time_ns: int,
) -> int:
    bird_signal = _resampled_fixture(fixture, sample_rate_hz)
    padded_signal = prepend_noise_padding_to_duration(
        bird_signal,
        sample_rate_hz,
        total_duration_seconds=(classification_window_seconds + (bird_signal.size / sample_rate_hz)),
        noise_rms=0.002,
        seed=sample_rate_hz,
    )
    return await _stream_signal_into_fusion(
        fusion,
        sample_rate_hz=sample_rate_hz,
        frame_samples=frame_samples,
        mono_signal=padded_signal,
        start_time_ns=start_time_ns,
    )


async def _stream_signal_into_fusion(
    fusion: FusionNode,
    *,
    sample_rate_hz: int,
    frame_samples: int,
    mono_signal: np.ndarray,
    start_time_ns: int,
) -> int:
    channels = synthesize_delayed_array_channels(
        mono_signal,
        sample_rate_hz,
        source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
    )
    triggered_frames = 0
    for sequence, (frame_start_ns, frame) in enumerate(
        split_channels_into_frames(
            channels,
            sample_rate_hz=sample_rate_hz,
            start_time_ns=start_time_ns,
            frame_samples=frame_samples,
        ),
        start=1,
    ):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=_node_spec(),
                frame={
                    "start_time_ns": frame_start_ns,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": int(frame.shape[0]),
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(frame),
                    "sequence": sequence,
                },
            )
        )
        assert response.accepted is True
        triggered_frames += int(response.triggered)
    return triggered_frames


async def _wait_for_localized_house_finch(storage: Storage, *, timeout_s: float = 45.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_detections: list[dict] = []
    while asyncio.get_running_loop().time() < deadline:
        last_detections = await storage.list_detections(limit=128)
        for detection in last_detections:
            if detection["label"] != HOUSE_FINCH_LABEL:
                continue
            if detection["reporting_modality"] != "localized":
                continue
            return detection
        await asyncio.sleep(0.25)
    raise AssertionError(
        "Timed out waiting for a localized house finch detection. "
        f"Last labels: {[row['label'] for row in last_detections[:10]]}"
    )


def _localization_error_m(position_m: tuple[float, float, float] | list[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(position_m, dtype=np.float64)
            - np.asarray(HOUSE_FINCH_SOURCE_POSITION_M, dtype=np.float64)
        )
    )


def _geo_error_m(
    coordinate_frame: LocalCoordinateFrame,
    position_geo: dict[str, float] | GeoPoint,
) -> float:
    geo = position_geo if isinstance(position_geo, GeoPoint) else GeoPoint(**position_geo)
    return _localization_error_m(coordinate_frame.geo_to_local(geo))


def test_house_finch_fixture_classifies_with_birdnet(
    birdnet_classifier: BirdNETClassifier,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    samples, sample_rate_hz = house_finch_fixture_48khz
    result = birdnet_classifier.classify(samples, sample_rate_hz)

    assert result.label == HOUSE_FINCH_LABEL
    assert result.confidence >= 0.05
    assert result.features["model"] == "birdnet_v2m4"


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_synthesized_house_finch_localizes_with_tight_srp(
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    signal = _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz)
    windows = {
        sensor_id: channel
        for sensor_id, channel in zip(
            _sensor_positions().keys(),
            synthesize_delayed_array_channels(
                signal,
                sample_rate_hz,
                source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
            ),
            strict=True,
        )
    }

    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
        localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
        model_chain_config_path=Path("missing_model_chain.json"),
    )
    localizer = build_localizer_from_settings(settings)
    result = localizer.localize(
        sensor_positions=_sensor_positions(),
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    error_m = float(np.linalg.norm(estimate - np.asarray(HOUSE_FINCH_SOURCE_POSITION_M, dtype=np.float64)))
    assert error_m < TIGHT_LOCALIZATION_MAX_ERROR_M
    assert result.attempted_algorithm == "srp_phat"
    assert result.resolved_algorithm == "srp_phat"


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_detects_localized_house_finch(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    fusion, storage, settings = await _start_fusion_with_profile(tmp_path, sample_rate_hz=16_000)
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=16_000,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=1_739_910_000_000_000_000,
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        assert detection["label"] == HOUSE_FINCH_LABEL
        assert detection["reporting_modality"] == "localized"
        assert detection["feature_summary"]["capability_tier"] == "full_3d"
        assert detection["feature_summary"]["localization_method"] == "srp_phat"
        assert detection["label_confidence"] >= settings.birdnet_trigger_min_confidence
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_birdnet_hybrid_production_detects_house_finch_at_native_48k(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    fusion, storage, settings = await _start_fusion_with_profile(tmp_path, sample_rate_hz=48_000)
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=48_000,
            frame_samples=8192,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=1_739_920_000_000_000_000,
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        assert detection["label"] == HOUSE_FINCH_LABEL
        assert detection["reporting_modality"] == "localized"
        assert detection["feature_summary"]["capability_tier"] == "full_3d"
        assert detection["feature_summary"]["localization_method"] == "srp_phat"
        assert detection["label_confidence"] >= settings.birdnet_trigger_min_confidence
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_synthesized_house_finch_localizes_tightly_in_geodetic_space(
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    signal = _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz)
    windows = {
        sensor_id: channel
        for sensor_id, channel in zip(
            _sensor_positions().keys(),
            synthesize_delayed_array_channels(
                signal,
                sample_rate_hz,
                source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
            ),
            strict=True,
        )
    }
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
        localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
        site_origin_lat=DEFAULT_SITE_ORIGIN.lat,
        site_origin_lon=DEFAULT_SITE_ORIGIN.lon,
        site_origin_alt_m=DEFAULT_SITE_ORIGIN.alt_m,
        coordinate_mode="geodetic",
        model_chain_config_path=Path("missing_model_chain.json"),
    )
    localizer = build_localizer_from_settings(settings)
    coordinate_frame = LocalCoordinateFrame(origin=DEFAULT_SITE_ORIGIN, mode="geodetic")
    expected_geo = coordinate_frame.local_to_geo(HOUSE_FINCH_SOURCE_POSITION_M)
    result = localizer.localize(
        sensor_positions=_sensor_positions(),
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimated_geo = coordinate_frame.local_to_geo(result.position_m)
    assert _localization_error_m(result.position_m) < TIGHT_LOCALIZATION_MAX_ERROR_M
    assert _geo_error_m(coordinate_frame, estimated_geo) < TIGHT_LOCALIZATION_MAX_ERROR_M
    assert abs(estimated_geo.lat - expected_geo.lat) < 2e-6
    assert abs(estimated_geo.lon - expected_geo.lon) < 2e-6
    assert abs(estimated_geo.alt_m - expected_geo.alt_m) < 0.05
