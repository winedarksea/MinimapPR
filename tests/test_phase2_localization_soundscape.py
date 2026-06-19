from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.advanced_localization import EspritLocalizer, MusicLocalizer, SRPPhatLocalizer
from minimappr.core.ambisonics import AmbisonicSpatialEncoder, SoundscapeRenderer, SpatialSourceFrame
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.beamforming import DelayAndSumBeamformer, MVDRBeamformer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_dispatch import LocalizationDispatcher, build_localizer_from_settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import ClassificationResult, GeoPoint, IngestFrameRequest, LocalizationResult, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64, rms
from tests.helpers import StubLocalizer, shift_signal


@pytest.mark.parametrize(
    ("localizer", "max_error_m"),
    [
        (SRPPhatLocalizer(max_tau_s=0.03, grid_resolution_m=0.4, search_padding_m=1.5), 1.7),
        (MusicLocalizer(max_tau_s=0.03, azimuth_step_deg=7.0, elevation_step_deg=10.0), 2.5),
        (EspritLocalizer(max_tau_s=0.03), 2.8),
    ],
)
def test_phase2_localizers_return_reasonable_solution(localizer, max_error_m: float) -> None:
    sample_rate_hz = 16_000
    n = int(sample_rate_hz * 0.2)
    rng = np.random.default_rng(77)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2

    sensor_positions = {
        "s0": np.array([0.0, 0.0, 1.8]),
        "s1": np.array([5.5, 0.0, 1.9]),
        "s2": np.array([0.0, 4.5, 2.1]),
        "s3": np.array([5.0, 4.5, 2.0]),
        "s4": np.array([2.2, 2.0, 2.4]),
    }
    source = np.asarray([2.5, 2.2, 1.4], dtype=np.float64)

    windows = {}
    for sensor_id, position in sensor_positions.items():
        distance = float(np.linalg.norm(source - position))
        windows[sensor_id] = shift_signal(excitation, sample_rate_hz, distance / sound_speed)

    result = localizer.localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    error_m = float(np.linalg.norm(estimate - source))
    assert error_m < max_error_m
    assert 0.0 <= result.confidence <= 1.0
    assert np.isfinite(result.gdop) or np.isinf(result.gdop)
    assert result.position_covariance_m2 is not None
    assert result.residual_rms_seconds is not None


def test_tight_array_srp_far_field_is_not_replaced_with_near_array_position() -> None:
    sample_rate_hz = 48_000
    n = int(sample_rate_hz * 0.18)
    rng = np.random.default_rng(91)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2

    sensor_positions = {
        f"s{idx}": np.asarray(offset, dtype=np.float64)
        for idx, offset in enumerate(
            [
                (-0.016238, 0.025000, -0.010205),
                (0.027063, 0.000000, -0.010205),
                (-0.016238, -0.025000, -0.010205),
                (0.005413, 0.000000, 0.030615),
            ]
        )
    }
    source = np.asarray([82.0, 34.0, 11.0], dtype=np.float64)
    centroid = np.mean(np.vstack(list(sensor_positions.values())), axis=0)
    expected_direction = source - centroid
    expected_direction /= np.linalg.norm(expected_direction)

    distances = {
        sensor_id: float(np.linalg.norm(source - position))
        for sensor_id, position in sensor_positions.items()
    }
    min_delay_s = min(distances.values()) / sound_speed
    windows = {}
    for sensor_id, distance in distances.items():
        windows[sensor_id] = shift_signal(excitation, sample_rate_hz, (distance / sound_speed) - min_delay_s)

    result = SRPPhatLocalizer(
        max_tau_s=0.003,
        grid_resolution_m=0.5,
        search_padding_m=1.0,
        far_field_default_range_m=60.0,
        far_field_max_range_m=250.0,
    ).localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )
    result_with_different_legacy_ranges = SRPPhatLocalizer(
        max_tau_s=0.003,
        grid_resolution_m=0.5,
        search_padding_m=8.0,
        far_field_default_range_m=60.0,
        far_field_max_range_m=500.0,
    ).localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    estimated_offset = estimate - centroid
    estimated_range_m = float(np.linalg.norm(estimated_offset))
    estimated_direction = estimated_offset / estimated_range_m

    assert estimated_range_m > 20.0
    assert float(np.dot(estimated_direction, expected_direction)) > 0.97
    assert result.position_covariance_m2 is not None
    assert result.range_observability is not None
    assert result.range_observability < 0.10
    assert result.residual_rms_seconds is not None
    # Bearing is well-determined on a tight array → bearing_projected, not plain asymptotic
    assert result.range_projection_mode == "range_bearing_projected"
    # Confidence must exceed old UNOBSERVABLE_CONFIDENCE_CAP (0.20) since bearing is good
    assert result.confidence > 0.20
    np.testing.assert_allclose(
        result.position_m,
        result_with_different_legacy_ranges.position_m,
        rtol=1.0e-9,
        atol=1.0e-6,
    )


@pytest.mark.parametrize("source_range_m", [20.0, 50.0, 100.0, 150.0, 500.0])
def test_tight_array_srp_far_field_reports_bearing_with_honest_range_uncertainty(
    source_range_m: float,
) -> None:
    sample_rate_hz = 48_000
    n = int(sample_rate_hz * 0.18)
    rng = np.random.default_rng(92)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2

    sensor_positions = {
        f"s{idx}": np.asarray(offset, dtype=np.float64)
        for idx, offset in enumerate(
            [
                (-0.016238, 0.025000, -0.010205),
                (0.027063, 0.000000, -0.010205),
                (-0.016238, -0.025000, -0.010205),
                (0.005413, 0.000000, 0.030615),
            ]
        )
    }
    expected_direction = np.asarray([0.89, 0.41, 0.18], dtype=np.float64)
    expected_direction /= np.linalg.norm(expected_direction)
    source = expected_direction * source_range_m
    centroid = np.mean(np.vstack(list(sensor_positions.values())), axis=0)

    distances = {
        sensor_id: float(np.linalg.norm(source - position))
        for sensor_id, position in sensor_positions.items()
    }
    min_delay_s = min(distances.values()) / sound_speed
    windows = {
        sensor_id: shift_signal(excitation, sample_rate_hz, (distance / sound_speed) - min_delay_s)
        for sensor_id, distance in distances.items()
    }

    result = SRPPhatLocalizer(
        max_tau_s=0.003,
        grid_resolution_m=0.5,
        search_padding_m=1.0,
        far_field_default_range_m=60.0,
        far_field_max_range_m=250.0,
    ).localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    estimated_offset = estimate - centroid
    estimated_direction = estimated_offset / np.linalg.norm(estimated_offset)
    covariance = np.asarray(result.position_covariance_m2, dtype=np.float64)
    radial_variance_m2 = float(estimated_direction @ covariance @ estimated_direction)
    lateral_variance_m2 = float((np.trace(covariance) - radial_variance_m2) / 2.0)

    assert float(np.dot(estimated_direction, expected_direction)) > 0.99
    assert estimated_offset.dot(estimated_offset) > 250.0**2
    # Bearing well-determined → bearing_projected (bearing observed, range uncertain)
    assert result.range_projection_mode == "range_bearing_projected"
    assert result.range_observability is not None
    assert result.range_observability < 0.10
    # Confidence must exceed old UNOBSERVABLE_CONFIDENCE_CAP (0.20); bearing is valid
    assert result.confidence > 0.20
    # Covariance is explicitly elongated: radial uncertainty >> lateral uncertainty
    assert radial_variance_m2 > lateral_variance_m2 * 100.0


@pytest.mark.parametrize(
    ("source", "max_position_error_m"),
    [
        (np.asarray([0.20, 0.10, 0.15], dtype=np.float64), 0.10),
        (np.asarray([1.5, 0.6, 0.4], dtype=np.float64), 1.50),
        (np.asarray([4.0, -2.0, 1.0], dtype=np.float64), 4.50),
    ],
)
def test_tight_array_srp_near_field_regression_keeps_bearing_and_covariance(
    source: np.ndarray,
    max_position_error_m: float,
) -> None:
    sample_rate_hz = 48_000
    n = int(sample_rate_hz * 0.18)
    rng = np.random.default_rng(93)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sensor_positions = {
        "s0": np.array([0.0, 0.0, 0.0]),
        "s1": np.array([0.05, 0.0, 0.0]),
        "s2": np.array([0.0, 0.05, 0.0]),
        "s3": np.array([0.0, 0.0, 0.05]),
    }
    centroid = np.mean(np.vstack(list(sensor_positions.values())), axis=0)
    expected_direction = source - centroid
    expected_direction /= np.linalg.norm(expected_direction)
    distances = {
        sensor_id: float(np.linalg.norm(source - position))
        for sensor_id, position in sensor_positions.items()
    }
    min_delay_s = min(distances.values()) / 343.2
    windows = {
        sensor_id: shift_signal(excitation, sample_rate_hz, (distance / 343.2) - min_delay_s)
        for sensor_id, distance in distances.items()
    }

    result = SRPPhatLocalizer(
        max_tau_s=0.003,
        grid_resolution_m=0.05,
        search_padding_m=0.3,
        far_field_default_range_m=60.0,
        far_field_max_range_m=250.0,
    ).localize(sensor_positions, windows, sample_rate_hz, 20.0, 0.5)

    estimate = np.asarray(result.position_m, dtype=np.float64)
    position_error_m = float(np.linalg.norm(estimate - source))
    estimated_direction = estimate - centroid
    estimated_direction /= np.linalg.norm(estimated_direction)
    assert float(np.dot(estimated_direction, expected_direction)) > 0.95
    assert position_error_m <= max_position_error_m
    assert result.range_projection_mode == "range_refined"
    assert result.position_covariance_m2 is not None


def test_birdnet_hybrid_profile_uses_fixed_srp_for_tight_array_near_field() -> None:
    sample_rate_hz = 48_000
    n = int(sample_rate_hz * 0.18)
    rng = np.random.default_rng(95)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sensor_positions = {
        "s0": np.array([0.0, 0.0, 0.0]),
        "s1": np.array([0.05, 0.0, 0.0]),
        "s2": np.array([0.0, 0.05, 0.0]),
        "s3": np.array([0.0, 0.0, 0.05]),
    }
    source = np.asarray([0.20, 0.10, 0.15], dtype=np.float64)
    distances = {
        sensor_id: float(np.linalg.norm(source - position))
        for sensor_id, position in sensor_positions.items()
    }
    min_delay_s = min(distances.values()) / 343.2
    windows = {
        sensor_id: shift_signal(excitation, sample_rate_hz, (distance / 343.2) - min_delay_s)
        for sensor_id, distance in distances.items()
    }
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        localization_srp_grid_resolution_m=0.05,
        localization_search_padding_m=0.3,
    )
    result = build_localizer_from_settings(settings).localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    assert float(np.linalg.norm(estimate - source)) <= 0.10
    assert result.attempted_algorithm == "srp_phat"
    assert result.resolved_algorithm == "srp_phat"
    assert result.range_projection_mode == "range_refined"
    assert result.position_covariance_m2 is not None


def test_birdnet_hybrid_profile_uses_fixed_srp_for_tight_array_far_field() -> None:
    sample_rate_hz = 48_000
    n = int(sample_rate_hz * 0.18)
    rng = np.random.default_rng(96)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2
    sensor_positions = {
        f"s{idx}": np.asarray(offset, dtype=np.float64)
        for idx, offset in enumerate(
            [
                (-0.016238, 0.025000, -0.010205),
                (0.027063, 0.000000, -0.010205),
                (-0.016238, -0.025000, -0.010205),
                (0.005413, 0.000000, 0.030615),
            ]
        )
    }
    source = np.asarray([82.0, 34.0, 11.0], dtype=np.float64)
    centroid = np.mean(np.vstack(list(sensor_positions.values())), axis=0)
    expected_direction = source - centroid
    expected_direction /= np.linalg.norm(expected_direction)
    distances = {
        sensor_id: float(np.linalg.norm(source - position))
        for sensor_id, position in sensor_positions.items()
    }
    min_delay_s = min(distances.values()) / sound_speed
    windows = {
        sensor_id: shift_signal(excitation, sample_rate_hz, (distance / sound_speed) - min_delay_s)
        for sensor_id, distance in distances.items()
    }
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        localization_srp_grid_resolution_m=0.5,
        localization_search_padding_m=1.0,
        localization_far_field_default_range_m=60.0,
        localization_far_field_max_range_m=250.0,
    )
    result = build_localizer_from_settings(settings).localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.asarray(result.position_m, dtype=np.float64)
    estimated_offset = estimate - centroid
    estimated_range_m = float(np.linalg.norm(estimated_offset))
    estimated_direction = estimated_offset / estimated_range_m
    assert estimated_range_m > 20.0
    assert float(np.dot(estimated_direction, expected_direction)) > 0.97
    assert result.attempted_algorithm == "srp_phat"
    assert result.resolved_algorithm == "srp_phat"
    # Bearing well-determined → bearing_projected (bearing observed, range uncertain)
    assert result.range_projection_mode == "range_bearing_projected"
    assert result.range_observability is not None
    assert result.range_observability < 0.10
    # Confidence is bounded by wavelength_factor (signal is aliased for a 5cm array),
    # so it may be low; just verify it's a valid float in [0, 1].
    assert 0.0 <= result.confidence <= 1.0


def test_localization_dispatch_geometry_aware_and_cascade() -> None:
    stubs = {
        "gcc_phat": StubLocalizer("gcc", 0.2),
        "srp_phat": StubLocalizer("srp", 0.5),
        "music": StubLocalizer("music", 0.8),
        "esprit": StubLocalizer("esprit", 0.7),
    }
    dispatcher = LocalizationDispatcher(
        strategy="geometry_aware",
        default_algorithm="gcc_phat",
        algorithms=stubs,
    )

    tight_positions = {
        "a": np.array([0.0, 0.0, 0.0]),
        "b": np.array([0.05, 0.0, 0.0]),
        "c": np.array([0.0, 0.05, 0.0]),
        "d": np.array([0.0, 0.0, 0.05]),
    }
    windows = {sensor_id: np.ones(256, dtype=np.float32) for sensor_id in tight_positions}
    _ = dispatcher.localize(tight_positions, windows, 16_000, 20.0, 0.5)
    assert stubs["esprit"].calls == 1
    assert dispatcher.last_algorithm_name() == "esprit"

    cascade = LocalizationDispatcher(
        strategy="cascade",
        default_algorithm="gcc_phat",
        refine_confidence_threshold=0.4,
        algorithms=stubs,
    )
    _ = cascade.localize(tight_positions, windows, 16_000, 20.0, 0.5)
    assert cascade.last_algorithm_name() == "music"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def test_beamformers_improve_target_correlation() -> None:
    sample_rate_hz = 16_000
    n = 4096
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    clean = (0.5 * np.sin(2 * np.pi * 650.0 * t) + 0.4 * np.sin(2 * np.pi * 1100.0 * t)).astype(np.float32)
    interferer = (0.7 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    rng = np.random.default_rng(303)

    sensor_positions = {
        "s0": np.array([0.0, 0.0, 0.0]),
        "s1": np.array([0.2, 0.0, 0.0]),
        "s2": np.array([0.0, 0.2, 0.0]),
        "s3": np.array([0.2, 0.2, 0.0]),
    }
    target_pos = np.asarray([3.0, 0.6, 0.0], dtype=np.float64)
    interferer_pos = np.asarray([0.8, 3.5, 0.0], dtype=np.float64)
    sound_speed = 343.2

    windows: dict[str, np.ndarray] = {}
    for sensor_id, position in sensor_positions.items():
        target_delay = float(np.linalg.norm(target_pos - position) / sound_speed)
        int_delay = float(np.linalg.norm(interferer_pos - position) / sound_speed)
        mix = shift_signal(clean, sample_rate_hz, target_delay)
        mix += shift_signal(interferer, sample_rate_hz, int_delay)
        mix += rng.normal(0.0, 0.03, size=n).astype(np.float32)
        windows[sensor_id] = mix

    target_ref = shift_signal(
        clean,
        sample_rate_hz,
        min(float(np.linalg.norm(target_pos - p) / sound_speed) for p in sensor_positions.values()),
    )
    ref = windows["s0"]
    das = DelayAndSumBeamformer().beamform(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        steer_position_m=(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
        sound_speed_mps=sound_speed,
    )
    mvdr = MVDRBeamformer(diagonal_loading=5e-3).beamform(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        steer_position_m=(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
        sound_speed_mps=sound_speed,
    )

    corr_ref = _corr(ref, target_ref)
    corr_das = _corr(das, target_ref)
    corr_mvdr = _corr(mvdr, target_ref)

    assert corr_das > corr_ref
    assert corr_mvdr > corr_ref


def test_ambisonic_encoder_and_renderer_export(tmp_path: Path) -> None:
    sr = 16_000
    n = 1024
    t = np.arange(n, dtype=np.float64) / sr
    front = (0.4 * np.sin(2 * np.pi * 600.0 * t)).astype(np.float32)
    right = (0.4 * np.sin(2 * np.pi * 900.0 * t)).astype(np.float32)

    encoder = AmbisonicSpatialEncoder(distance_reference_m=1.0)
    renderer = SoundscapeRenderer(encoder=encoder, suppress_labels={"speech_like"})
    result = renderer.render(
        [
            SpatialSourceFrame(samples=front, position_m=(3.0, 0.0, 0.0), label="bird_like"),
            SpatialSourceFrame(samples=right, position_m=(0.0, 3.0, 0.0), label="speech_like"),
        ]
    )

    assert result.bformat.shape == (4, n)
    assert result.rendered_sources == 1
    assert result.suppressed_sources == 1
    assert float(np.mean(result.bformat[1])) != 0.0

    bformat_path = tmp_path / "soundscape_bformat.wav"
    surround_path = tmp_path / "soundscape_surround.wav"
    renderer.export_bformat_wav(path=bformat_path, bformat=result.bformat, sample_rate_hz=sr)
    downmix = renderer.export_5_1_wav(path=surround_path, bformat=result.bformat, sample_rate_hz=sr)

    assert downmix.shape[0] == 6
    with wave.open(str(bformat_path), "rb") as wav:
        assert wav.getnchannels() == 4
        assert wav.getframerate() == sr
    with wave.open(str(surround_path), "rb") as wav:
        assert wav.getnchannels() == 6
        assert wav.getframerate() == sr


class _RmsDrivenClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del sample_rate_hz
        level = rms(samples)
        if level >= 0.2:
            return ClassificationResult(
                label="bird_like",
                confidence=0.94,
                scores={"bird_like": 0.94, "ambient": 0.06},
                features={"rms": level},
            )
        return ClassificationResult(
            label="ambient",
            confidence=0.35,
            scores={"bird_like": 0.35, "ambient": 0.65},
            features={"rms": level},
        )


class _FixedLocalizer:
    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        del sensor_windows, sample_rate_hz, temperature_c, humidity_fraction
        reference = sorted(sensor_positions.keys())[0]
        return LocalizationResult(
            position_m=(1.0, 2.0, 1.5),
            confidence=0.9,
            gdop=1.0,
            reference_sensor=reference,
            tdoa_s={},
        )


class _BoostingBeamformer:
    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        del sensor_positions, sample_rate_hz, steer_position_m, sound_speed_mps, frequency_weights
        first = next(iter(sensor_windows.values()))
        return np.full_like(first, 0.35, dtype=np.float32)


@pytest.mark.asyncio
async def test_fusion_node_beamformed_classification_selects_higher_confidence(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "phase2_fusion.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        beamformed_classification_enabled=True,
        beamformer_type="delay_and_sum",
        beamformed_classification_min_sensor_count=2,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=_FixedLocalizer(),
        classifier=_RmsDrivenClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
        beamformer=_BoostingBeamformer(),
    )
    await fusion.start()

    channels = np.random.default_rng(404).normal(0.0, 0.03, size=(4, 1024)).astype(np.float32)
    request = IngestFrameRequest(
        node=NodeSpec(
            id="phase2-array",
            node_type=NodeType.SIRITH_TETRA,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[
                (0.0, 0.0, 0.0),
                (0.05, 0.0, 0.0),
                (0.0, 0.05, 0.0),
                (0.0, 0.0, 0.05),
            ],
            capabilities=["audio"],
        ),
        frame={
            "start_time_ns": 1_739_810_600_000_000_000,
            "sample_rate_hz": 16_000,
            "channels": 4,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(channels),
            "sequence": 1,
        },
    )
    response = await fusion.ingest(request)
    assert response.accepted is True
    assert response.triggered is True

    await asyncio.sleep(0.25)
    detections = await storage.list_detections(limit=5)
    assert detections
    latest = detections[0]
    assert latest["label"] == "bird_like"
    assert latest["feature_summary"]["classification_path"].startswith("beamformed:")
    assert float(latest["feature_summary"]["beamformed_confidence"]) > float(
        latest["feature_summary"]["omni_confidence"]
    )

    await fusion.stop()
    await storage.close()
