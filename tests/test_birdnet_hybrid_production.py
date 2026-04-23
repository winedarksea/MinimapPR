from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import correlate

pytest.importorskip("birdnet")

from minimappr.classifiers.birdnet import BirdNETClassifier
from minimappr.classifiers.factory import create_classifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.beamforming import create_beamformer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import build_localizer_from_settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64, read_wav_mono, rms
from tests.helpers import (
    SIRITH_TETRA_SENSOR_OFFSETS_M,
    geometric_array_propagation_delays_s,
    load_sensor_wav_files,
    load_wav_fixture_mono,
    prepend_noise_padding,
    resample_signal,
    split_channels_into_frames,
    synthesize_delayed_array_channels,
    write_delayed_array_wav_files,
)


HOUSE_FINCH_FIXTURE_PATH = Path(__file__).with_name("house_finch.wav")
HOUSE_FINCH_SOURCE_POSITION_M = (0.35, 0.18, 0.07)
HOUSE_FINCH_LABEL = "house finch"
DEFAULT_SITE_ORIGIN = GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0)
TIGHT_SRP_GRID_RESOLUTION_M = 0.05
TIGHT_SRP_SEARCH_PADDING_M = 0.3
TIGHT_LOCALIZATION_MAX_ERROR_M = 0.14
PRODUCTION_BIRDNET_CHUNK_OVERLAP_SECONDS = 3.0
BIRDNET_TEST_TARGET_CONTEXT_SECONDS = 3.0


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


def _synthesized_sensor_wav_paths(
    tmp_path: Path,
    fixture: tuple[np.ndarray, int],
    *,
    sample_rate_hz: int,
    classification_window_seconds: float,
    leading_padding_seconds: float,
) -> dict[str, Path]:
    bird_signal = _resampled_fixture(fixture, sample_rate_hz)
    del classification_window_seconds
    padded_signal = prepend_noise_padding(
        bird_signal,
        pad_samples=max(1, int(round(leading_padding_seconds * sample_rate_hz))),
        noise_rms=0.002,
        seed=sample_rate_hz,
    )
    return write_delayed_array_wav_files(
        tmp_path / f"house_finch_array_{sample_rate_hz}",
        mono_signal=padded_signal,
        sample_rate_hz=sample_rate_hz,
        source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        filename_prefix="sirith-house-finch_ch",
    )


def _build_profile_settings(
    tmp_path: Path,
    *,
    sample_rate_hz: int,
    snippet_retention_seconds: int = 0,
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
        snippet_retention_seconds=snippet_retention_seconds,
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
        birdnet_chunk_overlap_seconds=PRODUCTION_BIRDNET_CHUNK_OVERLAP_SECONDS,
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
    snippet_retention_seconds: int = 0,
    coordinate_mode: str = "flat",
    trigger_rms: float = 0.006,
    trigger_cooldown_seconds: float = 1.0,
    localization_srp_grid_resolution_m: float | None = None,
    localization_search_padding_m: float | None = None,
) -> tuple[FusionNode, Storage, Settings]:
    settings = _build_profile_settings(
        tmp_path,
        sample_rate_hz=sample_rate_hz,
        snippet_retention_seconds=snippet_retention_seconds,
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
    birdnet_chunk_stride_seconds = max(
        fusion.settings.localization_window_seconds,
        fusion.settings.classification_window_seconds - fusion.settings.birdnet_chunk_overlap_seconds,
    )
    leading_padding_seconds = max(
        fusion.settings.localization_window_seconds,
        birdnet_chunk_stride_seconds - BIRDNET_TEST_TARGET_CONTEXT_SECONDS,
    )
    sensor_paths = _synthesized_sensor_wav_paths(
        Path(fusion.settings.snippet_dir).parent,
        fixture,
        sample_rate_hz=sample_rate_hz,
        classification_window_seconds=classification_window_seconds,
        leading_padding_seconds=leading_padding_seconds,
    )
    windows, loaded_sample_rate_hz = load_sensor_wav_files(sensor_paths)
    assert loaded_sample_rate_hz == sample_rate_hz
    channels = np.stack(
        [windows[f"sirith-house-finch_ch{index}"] for index in range(len(SIRITH_TETRA_SENSOR_OFFSETS_M))],
        axis=0,
    )
    return await _stream_channels_into_fusion(
        fusion,
        sample_rate_hz=sample_rate_hz,
        frame_samples=frame_samples,
        channels=channels,
        start_time_ns=start_time_ns,
    )


async def _stream_channels_into_fusion(
    fusion: FusionNode,
    *,
    sample_rate_hz: int,
    frame_samples: int,
    channels: np.ndarray,
    start_time_ns: int,
) -> int:
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


def _estimate_integer_sample_lag(reference: np.ndarray, candidate: np.ndarray) -> int:
    correlation = correlate(candidate, reference, mode="full", method="fft")
    return int(np.argmax(correlation) - (reference.size - 1))


def _align_start_time_to_birdnet_chunk(start_time_ns: int, settings: Settings) -> int:
    stride_ns = max(
        1,
        int(
            max(
                settings.localization_window_seconds,
                settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds,
            )
            * 1_000_000_000
        ),
    )
    return int(start_time_ns) - (int(start_time_ns) % stride_ns)


def _best_contiguous_subsegment_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, int, bool]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.size == 0 or candidate.size == 0:
        raise AssertionError("Signals must be non-empty")

    if reference.size <= candidate.size:
        shorter = reference
        longer = candidate
        reference_is_shorter = True
    else:
        shorter = candidate
        longer = reference
        reference_is_shorter = False

    if shorter.size == longer.size:
        offset = 0
    else:
        correlation = correlate(longer, shorter, mode="valid", method="fft")
        offset = int(np.argmax(correlation))

    aligned = longer[offset : offset + shorter.size]
    centered_shorter = shorter - np.mean(shorter, dtype=np.float32)
    centered_aligned = aligned - np.mean(aligned, dtype=np.float32)
    denominator = float(
        np.linalg.norm(centered_shorter.astype(np.float64))
        * np.linalg.norm(centered_aligned.astype(np.float64))
    )
    normalized_correlation = 0.0
    if denominator > 0.0:
        normalized_correlation = float(
            np.dot(centered_shorter.astype(np.float64), centered_aligned.astype(np.float64))
            / denominator
        )
    normalized_rms_error = float(
        rms(aligned - shorter) / max(rms(shorter), 1e-9)
    )
    return normalized_correlation, normalized_rms_error, offset, reference_is_shorter


def _dominant_active_region(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    analysis_window_ms: float = 25.0,
    edge_padding_ms: float = 75.0,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AssertionError("Expected non-empty samples")

    window_samples = max(8, int(round((analysis_window_ms / 1000.0) * sample_rate_hz)))
    power = np.square(samples.astype(np.float64))
    window = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    rolling_rms = np.sqrt(np.convolve(power, window, mode="same"))

    peak_rms = float(np.max(rolling_rms))
    noise_floor_rms = float(np.percentile(rolling_rms, 60))
    threshold_rms = max(noise_floor_rms * 3.0, peak_rms * 0.2, 1e-5)
    active_indices = np.flatnonzero(rolling_rms >= threshold_rms)
    if active_indices.size == 0:
        return samples.copy()

    gap_tolerance_samples = max(1, int(round(0.15 * sample_rate_hz)))
    segment_start = int(active_indices[0])
    segment_end = int(active_indices[0])
    best_start = segment_start
    best_end = segment_end
    best_energy = float(np.sum(power[segment_start : segment_end + 1]))

    for index in active_indices[1:]:
        current_index = int(index)
        if current_index - segment_end <= gap_tolerance_samples:
            segment_end = current_index
            continue

        current_energy = float(np.sum(power[segment_start : segment_end + 1]))
        if current_energy > best_energy:
            best_start = segment_start
            best_end = segment_end
            best_energy = current_energy
        segment_start = current_index
        segment_end = current_index

    current_energy = float(np.sum(power[segment_start : segment_end + 1]))
    if current_energy > best_energy:
        best_start = segment_start
        best_end = segment_end

    edge_padding_samples = max(1, int(round((edge_padding_ms / 1000.0) * sample_rate_hz)))
    trimmed_start = max(0, best_start - edge_padding_samples)
    trimmed_end = min(samples.size, best_end + edge_padding_samples)
    return samples[trimmed_start:trimmed_end].copy()


def _longest_internal_quiet_gap_seconds(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    analysis_window_ms: float = 25.0,
) -> float:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return 0.0

    window_samples = max(8, int(round((analysis_window_ms / 1000.0) * sample_rate_hz)))
    power = np.square(samples.astype(np.float64))
    window = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    rolling_rms = np.sqrt(np.convolve(power, window, mode="same"))

    peak_rms = float(np.max(rolling_rms))
    if peak_rms <= 1e-8:
        return float(samples.size) / float(sample_rate_hz)

    quiet_threshold_rms = max(peak_rms * 0.08, 1e-6)
    quiet_mask = rolling_rms < quiet_threshold_rms
    if quiet_mask.size <= 2:
        return 0.0

    # Ignore leading/trailing quiet padding and only measure holes inside the
    # already-isolated active region.
    active_indices = np.flatnonzero(~quiet_mask)
    if active_indices.size == 0:
        return float(samples.size) / float(sample_rate_hz)
    interior_quiet = quiet_mask[active_indices[0] : active_indices[-1] + 1]

    longest_quiet_run = 0
    current_quiet_run = 0
    for is_quiet in interior_quiet:
        if bool(is_quiet):
            current_quiet_run += 1
            longest_quiet_run = max(longest_quiet_run, current_quiet_run)
        else:
            current_quiet_run = 0
    return float(longest_quiet_run) / float(sample_rate_hz)


async def _reconstruct_expected_classification_signal(
    fusion: FusionNode,
    detection: dict,
    *,
    sample_rate_hz: int,
    classification_window_seconds: float,
) -> np.ndarray:
    selected_sensor_ids = [str(sensor_id) for sensor_id in detection["source_sensors"]]
    classification_windows = await fusion.buffer.get_synchronized_window_ending_at(
        sensor_ids=selected_sensor_ids,
        end_time_ns=int(detection["timestamp_ns"]),
        window_seconds=classification_window_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    assert classification_windows

    reference_sensor = str(detection["reference_sensor"])
    classification_path = str(detection["feature_summary"]["classification_path"])
    if classification_path == "omni":
        expected_signal = classification_windows[reference_sensor]
    elif classification_path.startswith("beamformed:"):
        selected_positions = {
            sensor_id: _sensor_positions()[sensor_id]
            for sensor_id in selected_sensor_ids
        }
        localization_position_m = tuple(float(value) for value in detection["position_m"])
        expected_signal = fusion.beamformer.beamform(
            selected_positions,
            classification_windows,
            sample_rate_hz,
            localization_position_m,
            fusion.environment_provider.get_speed_of_sound(localization_position_m),
        )
    else:
        raise AssertionError(f"Unsupported classification path: {classification_path}")

    if fusion.classification_preprocessor is not None:
        expected_signal = fusion.classification_preprocessor.process(
            expected_signal,
            sample_rate_hz,
        )
    return np.asarray(expected_signal, dtype=np.float32)


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


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_synthesized_house_finch_beamforming_recovers_source_waveform(
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    source_signal = _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz)
    sensor_positions = _sensor_positions()
    sensor_windows = {
        sensor_id: channel
        for sensor_id, channel in zip(
            sensor_positions.keys(),
            synthesize_delayed_array_channels(
                source_signal,
                sample_rate_hz,
                source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
            ),
            strict=True,
        )
    }

    # Use FFT delay-and-sum to avoid interpolation artifacts in this fidelity test.
    beamformer = create_beamformer("freq_domain_das")
    beamformed_signal = beamformer.beamform(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=sample_rate_hz,
        steer_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        sound_speed_mps=343.2,
    )

    source_active_region = _dominant_active_region(source_signal, sample_rate_hz)
    beamformed_active_region = _dominant_active_region(beamformed_signal, sample_rate_hz)
    correlation, normalized_error, _, _ = _best_contiguous_subsegment_metrics(
        source_active_region,
        beamformed_active_region,
    )

    assert _longest_internal_quiet_gap_seconds(beamformed_active_region, sample_rate_hz) <= 0.15
    assert correlation >= 0.80
    assert normalized_error <= 0.55


@pytest.mark.parametrize("sample_rate_hz", [16_000, 48_000])
def test_house_finch_fixture_expands_to_four_sensor_wavs_with_expected_relative_delays(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
    sample_rate_hz: int,
) -> None:
    sensor_paths = _synthesized_sensor_wav_paths(
        tmp_path,
        house_finch_fixture_48khz,
        sample_rate_hz=sample_rate_hz,
        classification_window_seconds=30.0,
        leading_padding_seconds=24.0,
    )

    assert sorted(sensor_paths.keys()) == [
        "sirith-house-finch_ch0",
        "sirith-house-finch_ch1",
        "sirith-house-finch_ch2",
        "sirith-house-finch_ch3",
    ]
    assert all(path.exists() for path in sensor_paths.values())

    windows, loaded_sample_rate_hz = load_sensor_wav_files(sensor_paths)
    assert loaded_sample_rate_hz == sample_rate_hz

    stacked_windows = np.stack(
        [windows[f"sirith-house-finch_ch{index}"] for index in range(len(SIRITH_TETRA_SENSOR_OFFSETS_M))],
        axis=0,
    )
    assert stacked_windows.shape[0] == 4
    assert not np.allclose(stacked_windows[0], stacked_windows[1])

    expected_delays_s = geometric_array_propagation_delays_s(
        source_position_m=HOUSE_FINCH_SOURCE_POSITION_M,
        sensor_offsets_m=SIRITH_TETRA_SENSOR_OFFSETS_M,
    )
    expected_relative_samples = np.round(
        (expected_delays_s - expected_delays_s.min()) * sample_rate_hz
    ).astype(np.int64)
    earliest_sensor_index = int(np.argmin(expected_delays_s))
    observed_relative_samples = np.asarray(
        [
            _estimate_integer_sample_lag(
                stacked_windows[earliest_sensor_index],
                stacked_windows[index],
            )
            for index in range(stacked_windows.shape[0])
        ],
        dtype=np.int64,
    )
    assert np.max(np.abs(observed_relative_samples - expected_relative_samples)) <= 1

    result = build_localizer_from_settings(
        Settings(
            runtime_profile="birdnet_hybrid_production",
            localization_srp_grid_resolution_m=TIGHT_SRP_GRID_RESOLUTION_M,
            localization_search_padding_m=TIGHT_SRP_SEARCH_PADDING_M,
            model_chain_config_path=Path("missing_model_chain.json"),
        )
    ).localize(
        sensor_positions=_sensor_positions(),
        sensor_windows={
            sensor_id.replace("_", ":"): windows[sensor_id]
            for sensor_id in sensor_paths
        },
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )
    assert _localization_error_m(result.position_m) < TIGHT_LOCALIZATION_MAX_ERROR_M
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
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_910_000_000_000_000, settings),
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
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_920_000_000_000_000, settings),
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
async def test_birdnet_hybrid_production_persists_contiguous_classification_audio_snippet(
    tmp_path: Path,
    house_finch_fixture_48khz: tuple[np.ndarray, int],
) -> None:
    sample_rate_hz = 16_000
    fusion, storage, settings = await _start_fusion_with_profile(
        tmp_path,
        sample_rate_hz=sample_rate_hz,
        snippet_retention_seconds=3600,
    )
    try:
        triggered_frames = await _stream_fixture_into_fusion(
            fusion,
            sample_rate_hz=sample_rate_hz,
            frame_samples=1024,
            fixture=house_finch_fixture_48khz,
            classification_window_seconds=settings.classification_window_seconds,
            start_time_ns=_align_start_time_to_birdnet_chunk(1_739_930_000_000_000_000, settings),
        )
        assert triggered_frames >= 1

        detection = await _wait_for_localized_house_finch(storage)
        assert detection["snippet_path"] is not None

        snippet_path = Path(detection["snippet_path"])
        assert snippet_path.exists()

        snippet_signal, snippet_sample_rate_hz = read_wav_mono(snippet_path)
        assert snippet_sample_rate_hz == sample_rate_hz
        assert snippet_signal.size > 0

        expected_signal = await _reconstruct_expected_classification_signal(
            fusion,
            detection,
            sample_rate_hz=sample_rate_hz,
            classification_window_seconds=settings.classification_window_seconds,
        )
        expected_correlation, expected_error, _, _ = _best_contiguous_subsegment_metrics(
            expected_signal,
            snippet_signal,
        )
        assert expected_correlation >= 0.995
        assert expected_error <= 0.02

        # The attached snippet may include front padding and classifier-oriented
        # conditioning, so compare the dominant active region rather than the
        # entire stored clip. This still catches obvious gaps or severe damage.
        fixture_signal = _dominant_active_region(
            _resampled_fixture(house_finch_fixture_48khz, sample_rate_hz),
            sample_rate_hz,
        )
        snippet_active_region = _dominant_active_region(
            snippet_signal,
            sample_rate_hz,
        )
        fixture_correlation, _, _, _ = _best_contiguous_subsegment_metrics(
            fixture_signal,
            snippet_active_region,
        )
        assert _longest_internal_quiet_gap_seconds(snippet_active_region, sample_rate_hz) <= 0.15
        # This integration test validates contiguous snippet persistence and signal identity against
        # the exact classification input reconstruction. Beamformer fidelity against the true source
        # waveform is covered by test_synthesized_house_finch_beamforming_recovers_source_waveform.
        assert fixture_correlation >= 0.20
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
