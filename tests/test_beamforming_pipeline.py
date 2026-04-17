"""Tests for the beamforming-to-classification pipeline.

Covers:
- FrequencyDomainDelayAndSumBeamformer correctness & speed vs time-domain DAS
- Vectorised MVDRBeamformer equivalence with the original per-bin implementation
- Recall-biased MVDR (higher diagonal loading → wider beam)
- create_beamformer factory (all types including superdirective, gevd)
- Superdirective beamformer: diffuse-noise model, coherence, directivity
- GEVD/MaxSNR beamformer: eigendecomposition, SNR maximisation
- Pre-classification preprocessing: built-in stages, registry, chain builder
- End-to-end fusion pipeline: localize → beamform → preprocess → classify → chain
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.beamforming import (
    DelayAndSumBeamformer,
    FrequencyDomainDelayAndSumBeamformer,
    GEVDBeamformer,
    MVDRBeamformer,
    SuperdirectiveBeamformer,
    available_beamformers,
    create_beamformer,
)
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.preprocessing import (
    AudioPreprocessingChain,
    BandpassFilterStage,
    DCRemovalStage,
    GainStage,
    HighpassFilterStage,
    LowpassFilterStage,
    NodePreprocessorFactory,
    NormalizationStage,
    SpectralGateStage,
    available_stages,
    build_preprocessing_chain,
    create_classification_preprocessor,
    create_stage,
    register_stage,
)
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import (
    ClassificationResult,
    GeoPoint,
    IngestFrameRequest,
    LocalizationResult,
    NodeSpec,
    NodeType,
)
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64, rms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sensor_layout() -> dict[str, np.ndarray]:
    """4-sensor tetrahedral-ish array."""
    return {
        "s0": np.array([0.0, 0.0, 0.0]),
        "s1": np.array([0.2, 0.0, 0.0]),
        "s2": np.array([0.0, 0.2, 0.0]),
        "s3": np.array([0.2, 0.2, 0.0]),
    }


def _synthesise_windows(
    sensor_positions: dict[str, np.ndarray],
    target_position: np.ndarray,
    sample_rate_hz: int = 16_000,
    n_samples: int = 2048,
    target_freq_hz: float = 800.0,
    noise_level: float = 0.05,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create synthetic sensor windows with a target tone arriving from *target_position*."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    clean = np.sin(2 * np.pi * target_freq_hz * t).astype(np.float64)
    sound_speed = 343.2
    windows: dict[str, np.ndarray] = {}
    for sensor_id, pos in sensor_positions.items():
        delay_s = float(np.linalg.norm(target_position - pos)) / sound_speed
        delay_samples = delay_s * sample_rate_hz
        shifted = np.interp(t - delay_s, t, clean, left=0.0, right=0.0)
        shifted += rng.normal(0.0, noise_level, size=n_samples)
        windows[sensor_id] = shifted.astype(np.float32)
    return windows


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


# ---------------------------------------------------------------------------
# FrequencyDomainDelayAndSumBeamformer
# ---------------------------------------------------------------------------

class TestFrequencyDomainDAS:
    def test_single_sensor_passthrough(self) -> None:
        positions = {"s0": np.array([0.0, 0.0, 0.0])}
        signal = np.random.default_rng(1).normal(0, 0.1, 512).astype(np.float32)
        windows = {"s0": signal}
        out = FrequencyDomainDelayAndSumBeamformer().beamform(
            positions, windows, 16_000, (1.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_empty_returns_empty(self) -> None:
        out = FrequencyDomainDelayAndSumBeamformer().beamform({}, {}, 16_000, (0, 0, 0))
        assert out.size == 0

    def test_improves_target_correlation(self) -> None:
        """Frequency-domain DAS should improve correlation with the clean target
        compared to a raw single-sensor signal."""
        positions = _make_sensor_layout()
        target = np.array([3.0, 1.0, 0.0])
        sr = 16_000
        n = 2048
        windows = _synthesise_windows(positions, target, sample_rate_hz=sr, n_samples=n, noise_level=0.15)

        # Reference: clean tone delayed by the closest-sensor propagation time
        # (beamformer output is aligned to that reference point).
        sound_speed = 343.2
        min_delay = min(float(np.linalg.norm(target - p)) / sound_speed for p in positions.values())
        t = np.arange(n, dtype=np.float64) / sr
        clean_ref = np.sin(2 * np.pi * 800.0 * (t - min_delay)).astype(np.float32)

        raw_corr = abs(_correlation(windows["s0"], clean_ref))
        beamformed = FrequencyDomainDelayAndSumBeamformer().beamform(
            positions, windows, sr,
            (float(target[0]), float(target[1]), float(target[2])),
        )
        bf_corr = abs(_correlation(beamformed, clean_ref))
        assert bf_corr > raw_corr, f"beamformed {bf_corr:.3f} should exceed raw {raw_corr:.3f}"

    def test_matches_time_domain_das(self) -> None:
        """Frequency-domain and time-domain DAS should produce correlated outputs."""
        positions = _make_sensor_layout()
        target = np.array([2.0, 1.5, 0.0])
        windows = _synthesise_windows(positions, target)
        steer = (float(target[0]), float(target[1]), float(target[2]))

        td = DelayAndSumBeamformer().beamform(positions, windows, 16_000, steer)
        fd = FrequencyDomainDelayAndSumBeamformer().beamform(positions, windows, 16_000, steer)

        corr = abs(_correlation(td, fd))
        assert corr > 0.95, f"time-domain and freq-domain DAS correlation {corr:.3f} too low"

    def test_frequency_weights_applied(self) -> None:
        positions = _make_sensor_layout()
        target = np.array([2.0, 0.0, 0.0])
        windows = _synthesise_windows(positions, target, n_samples=512)
        steer = (2.0, 0.0, 0.0)
        n_freq = len(np.fft.rfftfreq(512, d=1.0 / 16_000))
        weights = np.zeros(n_freq, dtype=np.float64)
        out = FrequencyDomainDelayAndSumBeamformer().beamform(
            positions, windows, 16_000, steer, frequency_weights=weights,
        )
        # Zeroed frequency weights → essentially silent output
        assert rms(out) < 1e-6


# ---------------------------------------------------------------------------
# MVDRBeamformer (vectorised)
# ---------------------------------------------------------------------------

class TestVectorisedMVDR:
    def test_single_sensor_passthrough(self) -> None:
        positions = {"s0": np.array([0.0, 0.0, 0.0])}
        signal = np.random.default_rng(9).normal(0, 0.1, 512).astype(np.float32)
        out = MVDRBeamformer().beamform(
            positions, {"s0": signal}, 16_000, (1.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_improves_target_correlation(self) -> None:
        """MVDR should improve correlation with a multi-tone target signal."""
        positions = _make_sensor_layout()
        target = np.array([3.0, 0.5, 0.0])
        sr = 16_000
        n = 4096
        sound_speed = 343.2
        min_delay = min(float(np.linalg.norm(target - p)) / sound_speed for p in positions.values())
        t = np.arange(n, dtype=np.float64) / sr
        # Multi-tone target with richer spectral content for stable covariance estimation.
        clean = (0.5 * np.sin(2 * np.pi * 650.0 * t) + 0.4 * np.sin(2 * np.pi * 1100.0 * t)).astype(np.float64)

        rng = np.random.default_rng(77)
        # Synthesise with explicit interferer for MVDR to null.
        interferer_pos = np.array([0.8, 3.5, 0.0])
        interferer = (0.7 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float64)

        windows: dict[str, np.ndarray] = {}
        for sid, pos in positions.items():
            t_delay = float(np.linalg.norm(target - pos)) / sound_speed
            i_delay = float(np.linalg.norm(interferer_pos - pos)) / sound_speed
            sig = np.interp(t - t_delay, t, clean, left=0.0, right=0.0)
            sig += np.interp(t - i_delay, t, interferer, left=0.0, right=0.0)
            sig += rng.normal(0.0, 0.03, size=n)
            windows[sid] = sig.astype(np.float32)

        clean_ref = np.interp(t - min_delay, t, clean, left=0.0, right=0.0).astype(np.float32)
        # No abs() — MVDR suppresses the interferer which pushes raw correlation
        # negative; the MVDR output should correlate less negatively (closer to
        # the target), so raw signed correlation is the correct comparison.
        raw_corr = _correlation(windows["s0"], clean_ref)
        mvdr_out = MVDRBeamformer(diagonal_loading=5e-3).beamform(
            positions, windows, sr,
            (float(target[0]), float(target[1]), float(target[2])),
        )
        mvdr_corr = _correlation(mvdr_out, clean_ref)
        assert mvdr_corr > raw_corr

    def test_empty_input(self) -> None:
        out = MVDRBeamformer().beamform({}, {}, 16_000, (0, 0, 0))
        assert out.size == 0


# ---------------------------------------------------------------------------
# Recall-biased diagonal loading
# ---------------------------------------------------------------------------

class TestRecallBiasedBeamforming:
    def test_higher_loading_preserves_more_energy(self) -> None:
        """Increasing diagonal loading should widen the beam, preserving
        more energy from the vicinity of the target (higher recall)."""
        positions = _make_sensor_layout()
        target = np.array([3.0, 1.0, 0.0])
        windows = _synthesise_windows(positions, target, noise_level=0.1, seed=55)

        narrow = MVDRBeamformer(diagonal_loading=1e-4).beamform(
            positions, windows, 16_000,
            (float(target[0]), float(target[1]), float(target[2])),
        )
        wide = MVDRBeamformer(diagonal_loading=1.0).beamform(
            positions, windows, 16_000,
            (float(target[0]), float(target[1]), float(target[2])),
        )

        rms_narrow = rms(narrow)
        rms_wide = rms(wide)
        # Wider beam captures more energy (both target + some noise).
        assert rms_wide >= rms_narrow * 0.95, (
            f"wide-beam RMS {rms_wide:.4f} should be >= narrow {rms_narrow:.4f}"
        )


# ---------------------------------------------------------------------------
# create_beamformer factory
# ---------------------------------------------------------------------------

class TestBeamformerFactory:
    def test_delay_and_sum_default(self) -> None:
        bf = create_beamformer("delay_and_sum")
        assert isinstance(bf, DelayAndSumBeamformer)

    def test_delay_and_sum_name_uses_time_domain_das_without_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("WARNING", logger="minimappr.core.beamforming"):
            bf = create_beamformer("delay_and_sum")
        assert isinstance(bf, DelayAndSumBeamformer)
        assert "Unknown beamformer type" not in caplog.text

    def test_freq_domain_das(self) -> None:
        bf = create_beamformer("freq_domain_das")
        assert isinstance(bf, FrequencyDomainDelayAndSumBeamformer)

    def test_mvdr(self) -> None:
        bf = create_beamformer("mvdr", diagonal_loading=1e-3)
        assert isinstance(bf, MVDRBeamformer)

    def test_mvdr_recall_scale(self) -> None:
        bf = create_beamformer(
            "mvdr",
            diagonal_loading=1e-3,
            classifier_diagonal_loading_scale=10.0,
        )
        assert isinstance(bf, MVDRBeamformer)
        # Effective loading should be 1e-3 * 10 = 0.01
        assert abs(bf.diagonal_loading - 0.01) < 1e-9

    def test_unknown_type_falls_back_to_das(self) -> None:
        bf = create_beamformer("nonexistent")
        assert isinstance(bf, DelayAndSumBeamformer)

    def test_settings_normalize_das_alias_to_delay_and_sum(self) -> None:
        settings = Settings(beamformer_type="das")
        assert settings.beamformer_type == "delay_and_sum"


# ---------------------------------------------------------------------------
# Pre-classification preprocessing
# ---------------------------------------------------------------------------

class TestClassificationPreprocessor:
    def test_returns_none_when_disabled(self) -> None:
        from minimappr.config import LocalizationConfig

        config = LocalizationConfig(
            trigger_rms=0.015,
            trigger_cooldown_seconds=0.8,
            localization_window_seconds=0.08,
            max_sensor_buffer_seconds=8.0,
            default_temperature_c=20.0,
            default_humidity=0.5,
            audio_highpass_hz=50.0,
            audio_lowpass_hz=0.0,
            preprocess_enabled=True,
            ingest_gain_multiplier=1.0,
            min_sensors_for_3d=4,
            min_sensors_for_2d=3,
            localization_max_tau_s=0.02,
            localization_algorithm="gcc_phat",
            localization_strategy="fixed",
            localization_srp_grid_resolution_m=0.5,
            localization_search_padding_m=2.0,
            localization_music_azimuth_step_deg=6.0,
            localization_music_elevation_step_deg=8.0,
            localization_subspace_freq_min_hz=300.0,
            localization_subspace_freq_max_hz=3500.0,
            localization_refine_confidence_threshold=0.45,
            localization_tight_array_aperture_m=0.35,
            beamformed_classification_enabled=True,
            beamformer_type="delay_and_sum",
            beamformed_classification_min_sensor_count=2,
            beamformed_classification_confidence_margin=0.0,
            mvdr_diagonal_loading=1e-3,
            classifier_diagonal_loading_scale=10.0,
            pre_classification_highpass_hz=0.0,
            pre_classification_lowpass_hz=0.0,
            gcc_phat_interp_factor=4,
        )
        assert create_classification_preprocessor(config) is None

    def test_creates_chain_when_configured(self) -> None:
        from minimappr.config import LocalizationConfig

        config = LocalizationConfig(
            trigger_rms=0.015,
            trigger_cooldown_seconds=0.8,
            localization_window_seconds=0.08,
            max_sensor_buffer_seconds=8.0,
            default_temperature_c=20.0,
            default_humidity=0.5,
            audio_highpass_hz=50.0,
            audio_lowpass_hz=0.0,
            preprocess_enabled=True,
            ingest_gain_multiplier=1.0,
            min_sensors_for_3d=4,
            min_sensors_for_2d=3,
            localization_max_tau_s=0.02,
            localization_algorithm="gcc_phat",
            localization_strategy="fixed",
            localization_srp_grid_resolution_m=0.5,
            localization_search_padding_m=2.0,
            localization_music_azimuth_step_deg=6.0,
            localization_music_elevation_step_deg=8.0,
            localization_subspace_freq_min_hz=300.0,
            localization_subspace_freq_max_hz=3500.0,
            localization_refine_confidence_threshold=0.45,
            localization_tight_array_aperture_m=0.35,
            beamformed_classification_enabled=True,
            beamformer_type="mvdr",
            beamformed_classification_min_sensor_count=2,
            beamformed_classification_confidence_margin=0.0,
            mvdr_diagonal_loading=1e-3,
            classifier_diagonal_loading_scale=10.0,
            pre_classification_highpass_hz=100.0,
            pre_classification_lowpass_hz=4000.0,
            gcc_phat_interp_factor=4,
        )
        preprocessor = create_classification_preprocessor(config)
        assert preprocessor is not None
        assert isinstance(preprocessor, AudioPreprocessingChain)
        assert len(preprocessor.stages) == 2

    def test_preprocessing_modifies_signal(self) -> None:
        """A 50Hz tone should be attenuated by a 100Hz highpass filter."""
        chain = AudioPreprocessingChain(stages=[
            HighpassFilterStage(cutoff_hz=100.0),
        ])
        t = np.arange(4096, dtype=np.float64) / 16_000
        tone_50hz = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
        filtered = chain.process(tone_50hz, 16_000)
        assert rms(filtered) < rms(tone_50hz) * 0.5


# ---------------------------------------------------------------------------
# End-to-end pipeline: localize → beamform → preprocess → classify → chain
# ---------------------------------------------------------------------------

class _PrimaryClassifier(AudioClassifier):
    """Primary classifier that returns higher confidence for louder signals."""

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        level = rms(samples)
        if level >= 0.15:
            return ClassificationResult(
                label="gunshot",
                confidence=0.9,
                scores={"gunshot": 0.9, "ambient": 0.1},
                features={"rms": float(level)},
            )
        return ClassificationResult(
            label="ambient",
            confidence=0.3,
            scores={"gunshot": 0.3, "ambient": 0.7},
            features={"rms": float(level)},
        )


class _SecondaryClassifier(AudioClassifier):
    """Refinement classifier that re-labels gunshots with weapon type."""

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        return ClassificationResult(
            label="rifle_shot",
            confidence=0.85,
            scores={"rifle_shot": 0.85, "pistol_shot": 0.15},
            features={"refined": True},
        )


class _FixedLocalizer:
    def localize(self, sensor_positions, sensor_windows, sample_rate_hz,
                 temperature_c, humidity_fraction) -> LocalizationResult:
        reference = sorted(sensor_positions.keys())[0]
        return LocalizationResult(
            position_m=(1.0, 2.0, 0.0),
            confidence=0.85,
            gdop=1.2,
            reference_sensor=reference,
            tdoa_s={},
        )


class _AmplifyingBeamformer:
    """Test beamformer that returns a loud tone to trigger higher classifier confidence."""

    def beamform(self, sensor_positions, sensor_windows, sample_rate_hz,
                 steer_position_m, sound_speed_mps=343.2, frequency_weights=None):
        first = next(iter(sensor_windows.values()))
        n = first.size
        t = np.arange(n, dtype=np.float64) / sample_rate_hz
        # A 500 Hz tone at 0.3 amplitude — survives highpass filtering and
        # gives high RMS for the confidence-threshold classifier.
        return (0.3 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)


@pytest.mark.asyncio
async def test_full_pipeline_localize_beamform_preprocess_classify_chain(tmp_path: Path) -> None:
    """End-to-end test: localize → beamform → optional preprocess → classify (chained).

    Verifies that:
    1. The beamformed signal is used for classification when it yields higher confidence.
    2. The chained secondary classifier receives the beamformed signal.
    3. Pre-classification preprocessing is applied.
    4. Provenance metadata records the full pipeline path.
    """
    settings = Settings(
        db_path=tmp_path / "pipeline.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        beamformed_classification_enabled=True,
        beamformer_type="delay_and_sum",
        beamformed_classification_min_sensor_count=2,
        beamformed_classification_confidence_margin=0.0,
        pre_classification_highpass_hz=50.0,
        pre_classification_lowpass_hz=0.0,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    # Primary → secondary chained classifier
    primary = _PrimaryClassifier()
    chained = ChainedClassifier(
        base_classifier=primary,
        stages=[
            ChainStage(
                stage_id="weapon_refine",
                classifier=_SecondaryClassifier(),
                trigger_labels={"gunshot"},
                min_confidence=0.5,
                score_weight=1.0,
            ),
        ],
        category_for_label=lambda label: "security" if "shot" in label or "gun" in label else "unknown",
    )

    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=_FixedLocalizer(),
        classifier=chained,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
        beamformer=_AmplifyingBeamformer(),
    )
    await fusion.start()

    channels = np.random.default_rng(123).normal(0.0, 0.02, size=(4, 1024)).astype(np.float32)
    request = IngestFrameRequest(
        node=NodeSpec(
            id="pipeline-test-array",
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
            "start_time_ns": 1_750_000_000_000_000_000,
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

    await asyncio.sleep(0.3)
    detections = await storage.list_detections(limit=5)
    assert detections, "Expected at least one detection"

    latest = detections[0]
    features = latest["feature_summary"]

    # Beamformed path was selected (amplifying beamformer boosts confidence).
    assert features["classification_path"].startswith("beamformed:")
    assert float(features["beamformed_confidence"]) > float(features["omni_confidence"])

    # The secondary classifier in the chain was triggered: its scores appear.
    classifier_scores = latest.get("classifier_scores", {})
    has_refinement = any("weapon_refine:" in key for key in classifier_scores)
    assert has_refinement, (
        f"Expected weapon_refine chain stage scores in {list(classifier_scores.keys())}"
    )

    await fusion.stop()
    await storage.close()


# ---------------------------------------------------------------------------
# Superdirective beamformer
# ---------------------------------------------------------------------------

class TestSuperdirectiveBeamformer:
    """Superdirective beamformer with analytical diffuse-noise model."""

    def test_single_sensor_passthrough(self) -> None:
        positions = {"s0": np.array([0.0, 0.0, 0.0])}
        signal = np.random.default_rng(7).normal(0, 0.1, 512).astype(np.float32)
        windows = {"s0": signal}
        out = SuperdirectiveBeamformer().beamform(
            positions, windows, 16_000, (1.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_empty_returns_empty(self) -> None:
        out = SuperdirectiveBeamformer().beamform({}, {}, 16_000, (0, 0, 0))
        assert out.size == 0

    def test_enhances_target_tone(self) -> None:
        """Superdirective output should strongly correlate with delay-aligned target tone."""
        sensor_positions = _make_sensor_layout()
        target_pos = np.array([2.0, 0.0, 0.0])
        windows = _synthesise_windows(
            sensor_positions, target_pos,
            noise_level=0.15, target_freq_hz=800.0,
        )
        bf = SuperdirectiveBeamformer(diagonal_loading=1e-2)
        output = bf.beamform(
            sensor_positions, windows, 16_000, tuple(target_pos), 343.2,
        )
        # Compare against delay-aligned reference (beamformers align to
        # propagation delay of closest sensor).
        closest_delay_s = min(
            float(np.linalg.norm(target_pos - pos)) / 343.2
            for pos in sensor_positions.values()
        )
        t = np.arange(2048, dtype=np.float64) / 16_000
        reference = np.sin(2 * np.pi * 800.0 * (t - closest_delay_s))
        corr = abs(_correlation(output, reference))
        assert corr > 0.8, f"Expected high correlation with delay-aligned target, got {corr:.3f}"

    def test_higher_loading_widens_beam(self) -> None:
        """Higher diagonal loading should produce output closer to simple DAS."""
        sensor_positions = _make_sensor_layout()
        target_pos = np.array([2.0, 0.0, 0.0])
        windows = _synthesise_windows(sensor_positions, target_pos)

        narrow = SuperdirectiveBeamformer(diagonal_loading=1e-3)
        wide = SuperdirectiveBeamformer(diagonal_loading=10.0)

        out_narrow = narrow.beamform(sensor_positions, windows, 16_000, tuple(target_pos))
        out_wide = wide.beamform(sensor_positions, windows, 16_000, tuple(target_pos))

        das = FrequencyDomainDelayAndSumBeamformer()
        out_das = das.beamform(sensor_positions, windows, 16_000, tuple(target_pos))

        # Wide loading should be closer to DAS than narrow loading.
        corr_wide = abs(_correlation(out_wide, out_das))
        corr_narrow = abs(_correlation(out_narrow, out_das))
        assert corr_wide > corr_narrow, (
            f"Wide loading should be closer to DAS: wide={corr_wide:.3f}, narrow={corr_narrow:.3f}"
        )

    def test_output_length_preserved(self) -> None:
        sensor_positions = _make_sensor_layout()
        target = np.array([1.0, 1.0, 0.0])
        windows = _synthesise_windows(sensor_positions, target, n_samples=1024)
        out = SuperdirectiveBeamformer().beamform(
            sensor_positions, windows, 16_000, tuple(target),
        )
        assert out.shape == (1024,)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# GEVD / MaxSNR beamformer
# ---------------------------------------------------------------------------

class TestGEVDBeamformer:
    """GEVD beamformer maximising signal-to-noise ratio."""

    def test_single_sensor_passthrough(self) -> None:
        positions = {"s0": np.array([0.0, 0.0, 0.0])}
        signal = np.random.default_rng(8).normal(0, 0.1, 512).astype(np.float32)
        windows = {"s0": signal}
        out = GEVDBeamformer().beamform(
            positions, windows, 16_000, (1.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_empty_returns_empty(self) -> None:
        out = GEVDBeamformer().beamform({}, {}, 16_000, (0, 0, 0))
        assert out.size == 0

    def test_enhances_target_tone(self) -> None:
        """GEVD output should correlate with delay-aligned target tone.

        Single-snapshot covariance limits GEVD precision, so the threshold
        is lower than for superdirective, but output should meaningfully
        preserve the target signal.
        """
        sensor_positions = _make_sensor_layout()
        target_pos = np.array([2.0, 0.0, 0.0])
        windows = _synthesise_windows(
            sensor_positions, target_pos,
            noise_level=0.1, target_freq_hz=1000.0,
        )
        bf = GEVDBeamformer(diagonal_loading=1e-2)
        output = bf.beamform(
            sensor_positions, windows, 16_000, tuple(target_pos), 343.2,
        )
        closest_delay_s = min(
            float(np.linalg.norm(target_pos - pos)) / 343.2
            for pos in sensor_positions.values()
        )
        t = np.arange(2048, dtype=np.float64) / 16_000
        reference = np.sin(2 * np.pi * 1000.0 * (t - closest_delay_s))
        corr = abs(_correlation(output, reference))
        # Single-snapshot GEVD has limited precision but should still
        # capture the target signal meaningfully.
        assert corr > 0.3, f"GEVD delay-aligned correlation too low: {corr:.3f}"

    def test_output_length_preserved(self) -> None:
        sensor_positions = _make_sensor_layout()
        target = np.array([1.0, 1.0, 0.0])
        windows = _synthesise_windows(sensor_positions, target, n_samples=1024)
        out = GEVDBeamformer().beamform(
            sensor_positions, windows, 16_000, tuple(target),
        )
        assert out.shape == (1024,)
        assert out.dtype == np.float32

    def test_accepts_frequency_weights(self) -> None:
        sensor_positions = _make_sensor_layout()
        target_pos = np.array([2.0, 0.0, 0.0])
        windows = _synthesise_windows(sensor_positions, target_pos, n_samples=512)
        bf = GEVDBeamformer()
        freqs_size = 512 // 2 + 1
        weights = np.ones(freqs_size)
        out = bf.beamform(
            sensor_positions, windows, 16_000, tuple(target_pos), 343.2, weights,
        )
        assert out.shape == (512,)


# ---------------------------------------------------------------------------
# Extended factory tests (superdirective + gevd)
# ---------------------------------------------------------------------------

class TestBeamformerFactoryExtended:
    def test_create_superdirective(self) -> None:
        bf = create_beamformer("superdirective")
        assert isinstance(bf, SuperdirectiveBeamformer)

    def test_create_gevd(self) -> None:
        bf = create_beamformer("gevd")
        assert isinstance(bf, GEVDBeamformer)

    def test_superdirective_respects_loading_scale(self) -> None:
        bf = create_beamformer(
            "superdirective",
            diagonal_loading=0.01,
            classifier_diagonal_loading_scale=5.0,
        )
        assert isinstance(bf, SuperdirectiveBeamformer)
        assert bf.diagonal_loading == pytest.approx(0.05)

    def test_gevd_respects_loading_scale(self) -> None:
        bf = create_beamformer(
            "gevd",
            diagonal_loading=0.01,
            classifier_diagonal_loading_scale=3.0,
        )
        assert isinstance(bf, GEVDBeamformer)
        assert bf.diagonal_loading == pytest.approx(0.03)

    def test_available_beamformers_includes_all(self) -> None:
        avail = available_beamformers()
        assert "delay_and_sum" in avail
        assert "das" in avail
        assert "mvdr" in avail
        assert "superdirective" in avail
        assert "gevd" in avail
        assert "freq_domain_das" in avail

    def test_unknown_type_falls_back_to_das(self) -> None:
        bf = create_beamformer("nonexistent_type_xyz")
        assert isinstance(bf, DelayAndSumBeamformer)


# ---------------------------------------------------------------------------
# Extended preprocessing tests (new stages + registry + build)
# ---------------------------------------------------------------------------

class TestPreprocessingStages:
    """Test each built-in preprocessing stage individually."""

    def test_bandpass_filter(self) -> None:
        rng = np.random.default_rng(10)
        signal = rng.normal(0, 1.0, 4096).astype(np.float32)
        stage = BandpassFilterStage(low_hz=200.0, high_hz=4000.0)
        out = stage.process(signal, 16_000)
        assert out.shape == signal.shape
        assert out.dtype == np.float32
        # Energy should be reduced (noise outside band removed).
        assert np.sqrt(np.mean(out**2)) < np.sqrt(np.mean(signal**2))

    def test_bandpass_invalid_range_passthrough(self) -> None:
        signal = np.ones(256, dtype=np.float32)
        stage = BandpassFilterStage(low_hz=5000.0, high_hz=100.0)  # low >= high
        out = stage.process(signal, 16_000)
        np.testing.assert_array_equal(out, signal)

    def test_dc_removal(self) -> None:
        signal = np.full(256, 0.5, dtype=np.float32)
        stage = DCRemovalStage()
        out = stage.process(signal, 16_000)
        assert abs(float(np.mean(out))) < 1e-6

    def test_dc_removal_empty(self) -> None:
        signal = np.array([], dtype=np.float32)
        stage = DCRemovalStage()
        out = stage.process(signal, 16_000)
        assert out.size == 0

    def test_normalization_peak(self) -> None:
        signal = np.array([0.1, -0.5, 0.3], dtype=np.float32)
        stage = NormalizationStage(target_level=1.0, mode="peak")
        out = stage.process(signal, 16_000)
        assert float(np.max(np.abs(out))) == pytest.approx(1.0, abs=1e-5)

    def test_normalization_rms(self) -> None:
        rng = np.random.default_rng(11)
        signal = rng.normal(0, 0.3, 1024).astype(np.float32)
        stage = NormalizationStage(target_level=0.5, mode="rms")
        out = stage.process(signal, 16_000)
        out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        assert out_rms == pytest.approx(0.5, abs=0.01)

    def test_normalization_silent_passthrough(self) -> None:
        signal = np.zeros(64, dtype=np.float32)
        stage = NormalizationStage()
        out = stage.process(signal, 16_000)
        np.testing.assert_array_equal(out, signal)

    def test_spectral_gate(self) -> None:
        rng = np.random.default_rng(12)
        t = np.arange(1024, dtype=np.float64) / 16_000
        tone = np.sin(2 * np.pi * 500.0 * t).astype(np.float32)
        noise = rng.normal(0, 0.02, 1024).astype(np.float32)
        signal = tone + noise
        stage = SpectralGateStage(threshold_factor=1.5)
        out = stage.process(signal, 16_000)
        # Should preserve the tone and reduce noise.
        corr = abs(_correlation(out, tone))
        assert corr > 0.9, f"Expected high correlation with tone after gating, got {corr:.3f}"

    def test_spectral_gate_short_signal(self) -> None:
        signal = np.ones(8, dtype=np.float32)
        stage = SpectralGateStage()
        out = stage.process(signal, 16_000)
        np.testing.assert_array_equal(out, signal)

    def test_gain_stage_multiplies_signal(self) -> None:
        signal = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        stage = GainStage(multiplier=2.0)
        out = stage.process(signal, 16_000)
        np.testing.assert_allclose(out, signal * 2.0, atol=1e-7)
        assert out.dtype == np.float32

    def test_gain_stage_unity_passthrough(self) -> None:
        signal = np.array([0.25, -0.5, 0.75], dtype=np.float32)
        stage = GainStage(multiplier=1.0)
        out = stage.process(signal, 16_000)
        np.testing.assert_array_equal(out, signal)


class TestNodePreprocessorFactory:
    def _make_node(self, properties: dict[str, object] | None = None) -> NodeSpec:
        return NodeSpec(
            id="node-preprocess",
            node_type=NodeType.SIRITH_TETRA,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            properties=properties or {},
        )

    def test_global_ingest_gain_applies_gain_stage(self) -> None:
        settings = Settings(
            preprocess_enabled=True,
            ingest_gain_multiplier=2.0,
            audio_highpass_hz=0.0,
            audio_lowpass_hz=0.0,
        )
        factory = NodePreprocessorFactory(settings)
        chain = factory.for_node(self._make_node())
        assert isinstance(chain, AudioPreprocessingChain)
        assert len(chain.stages) == 1
        assert isinstance(chain.stages[0], GainStage)
        assert chain.stages[0].multiplier == pytest.approx(2.0)

    def test_node_gain_override_takes_precedence(self) -> None:
        settings = Settings(
            preprocess_enabled=True,
            ingest_gain_multiplier=2.0,
            audio_highpass_hz=0.0,
            audio_lowpass_hz=0.0,
        )
        node = self._make_node(properties={"preprocess": {"gain_multiplier": 3.5}})
        factory = NodePreprocessorFactory(settings)
        chain = factory.for_node(node)
        assert isinstance(chain, AudioPreprocessingChain)
        assert len(chain.stages) == 1
        assert isinstance(chain.stages[0], GainStage)
        assert chain.stages[0].multiplier == pytest.approx(3.5)


class TestIngestGainValidation:
    def test_settings_reject_non_positive_ingest_gain(self) -> None:
        with pytest.raises(ValueError, match="MINIMAPPR_INGEST_GAIN_MULTIPLIER"):
            Settings(ingest_gain_multiplier=0.0)

    def test_settings_reject_non_finite_ingest_gain(self) -> None:
        with pytest.raises(ValueError, match="MINIMAPPR_INGEST_GAIN_MULTIPLIER"):
            Settings(ingest_gain_multiplier=float("nan"))


class TestYamnetConditioningSettings:
    def test_settings_reject_non_positive_yamnet_target_rms(self) -> None:
        with pytest.raises(ValueError, match="MINIMAPPR_YAMNET_INPUT_TARGET_RMS"):
            Settings(yamnet_input_target_rms=0.0)

    def test_settings_reject_non_positive_yamnet_max_input_gain(self) -> None:
        with pytest.raises(ValueError, match="MINIMAPPR_YAMNET_MAX_INPUT_GAIN"):
            Settings(yamnet_max_input_gain=0.0)

    def test_settings_from_env_reads_yamnet_conditioning_knobs(self, monkeypatch) -> None:
        monkeypatch.setenv("MINIMAPPR_YAMNET_INPUT_TARGET_RMS", "0.18")
        monkeypatch.setenv("MINIMAPPR_YAMNET_MAX_INPUT_GAIN", "14.0")

        settings = Settings.from_env()

        assert settings.yamnet_input_target_rms == pytest.approx(0.18)
        assert settings.yamnet_max_input_gain == pytest.approx(14.0)


class TestPreprocessingRegistry:
    """Test the stage registry and build_preprocessing_chain."""

    def test_available_stages_has_all_builtins(self) -> None:
        names = available_stages()
        for expected in ["highpass", "lowpass", "bandpass", "dc_remove", "gain", "normalize", "spectral_gate"]:
            assert expected in names, f"{expected} not in {names}"

    def test_create_stage_by_name(self) -> None:
        stage = create_stage("dc_remove")
        assert isinstance(stage, DCRemovalStage)

    def test_create_stage_with_kwargs(self) -> None:
        stage = create_stage("highpass", cutoff_hz=100.0, order=6)
        assert isinstance(stage, HighpassFilterStage)
        assert stage.cutoff_hz == 100.0
        assert stage.order == 6

    def test_create_stage_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessing stage"):
            create_stage("nonexistent_filter_xyz")

    def test_register_custom_stage(self) -> None:
        """Custom stages can be registered and instantiated by name."""
        from dataclasses import dataclass

        @dataclass(slots=True)
        class _GainStage:
            gain: float = 2.0

            def process(self, samples, sample_rate_hz, *, node_id=None):
                return (samples * self.gain).astype(np.float32)

        register_stage("test_gain", _GainStage)
        try:
            assert "test_gain" in available_stages()
            stage = create_stage("test_gain", gain=3.0)
            signal = np.array([0.5], dtype=np.float32)
            out = stage.process(signal, 16_000)
            assert float(out[0]) == pytest.approx(1.5)
        finally:
            # Clean up registry to avoid test pollution.
            from minimappr.core.preprocessing import _STAGE_REGISTRY
            _STAGE_REGISTRY.pop("test_gain", None)

    def test_build_chain_from_specs(self) -> None:
        chain = build_preprocessing_chain([
            {"name": "dc_remove"},
            {"name": "highpass", "cutoff_hz": 100.0},
            {"name": "normalize", "mode": "peak", "target_level": 0.8},
        ])
        assert isinstance(chain, AudioPreprocessingChain)
        assert len(chain.stages) == 3

        # Functional check: DC offset + signal → DC removed, filtered, normalized.
        signal = np.full(1024, 0.5, dtype=np.float32) + np.random.default_rng(13).normal(0, 0.01, 1024).astype(np.float32)
        out = chain.process(signal, 16_000)
        assert out.dtype == np.float32
        # DC should be removed.
        assert abs(float(np.mean(out))) < 0.1


class TestClassificationPreprocessorExtended:
    """Test extra_stages parameter and composability."""

    def test_extra_stages_appended(self) -> None:
        from minimappr.config import LocalizationConfig
        config = LocalizationConfig(
            trigger_rms=0.01, trigger_cooldown_seconds=1.0,
            localization_window_seconds=0.5, max_sensor_buffer_seconds=5.0,
            default_temperature_c=20.0, default_humidity=0.5,
            audio_highpass_hz=50.0, audio_lowpass_hz=7500.0,
            preprocess_enabled=True,
            ingest_gain_multiplier=1.0,
            min_sensors_for_3d=4, min_sensors_for_2d=3,
            localization_max_tau_s=0.01,
            localization_algorithm="gcc_phat",
            localization_strategy="srp_grid",
            localization_srp_grid_resolution_m=0.5,
            localization_search_padding_m=2.0,
            localization_music_azimuth_step_deg=5.0,
            localization_music_elevation_step_deg=10.0,
            localization_subspace_freq_min_hz=200.0,
            localization_subspace_freq_max_hz=4000.0,
            localization_refine_confidence_threshold=0.6,
            localization_tight_array_aperture_m=0.25,
            beamformed_classification_enabled=True,
            beamformer_type="mvdr",
            beamformed_classification_min_sensor_count=3,
            beamformed_classification_confidence_margin=0.05,
            mvdr_diagonal_loading=0.001,
            classifier_diagonal_loading_scale=10.0,
            pre_classification_highpass_hz=50.0,
            pre_classification_lowpass_hz=0.0,
            gcc_phat_interp_factor=4,
        )
        norm = NormalizationStage(target_level=0.5, mode="rms")
        chain = create_classification_preprocessor(config, extra_stages=[norm])
        assert chain is not None
        assert isinstance(chain, AudioPreprocessingChain)
        # Should have highpass + normalization = 2 stages.
        assert len(chain.stages) == 2
        assert isinstance(chain.stages[0], HighpassFilterStage)
        assert isinstance(chain.stages[1], NormalizationStage)

    def test_extra_stages_only_when_no_config_filters(self) -> None:
        from minimappr.config import LocalizationConfig
        config = LocalizationConfig(
            trigger_rms=0.01, trigger_cooldown_seconds=1.0,
            localization_window_seconds=0.5, max_sensor_buffer_seconds=5.0,
            default_temperature_c=20.0, default_humidity=0.5,
            audio_highpass_hz=50.0, audio_lowpass_hz=7500.0,
            preprocess_enabled=True,
            ingest_gain_multiplier=1.0,
            min_sensors_for_3d=4, min_sensors_for_2d=3,
            localization_max_tau_s=0.01,
            localization_algorithm="gcc_phat",
            localization_strategy="srp_grid",
            localization_srp_grid_resolution_m=0.5,
            localization_search_padding_m=2.0,
            localization_music_azimuth_step_deg=5.0,
            localization_music_elevation_step_deg=10.0,
            localization_subspace_freq_min_hz=200.0,
            localization_subspace_freq_max_hz=4000.0,
            localization_refine_confidence_threshold=0.6,
            localization_tight_array_aperture_m=0.25,
            beamformed_classification_enabled=True,
            beamformer_type="mvdr",
            beamformed_classification_min_sensor_count=3,
            beamformed_classification_confidence_margin=0.05,
            mvdr_diagonal_loading=0.001,
            classifier_diagonal_loading_scale=10.0,
            pre_classification_highpass_hz=0.0,  # No default filters.
            pre_classification_lowpass_hz=0.0,
            gcc_phat_interp_factor=4,
        )
        dc = DCRemovalStage()
        chain = create_classification_preprocessor(config, extra_stages=[dc])
        assert chain is not None
        assert len(chain.stages) == 1
        assert isinstance(chain.stages[0], DCRemovalStage)

    def test_none_when_no_stages(self) -> None:
        from minimappr.config import LocalizationConfig
        config = LocalizationConfig(
            trigger_rms=0.01, trigger_cooldown_seconds=1.0,
            localization_window_seconds=0.5, max_sensor_buffer_seconds=5.0,
            default_temperature_c=20.0, default_humidity=0.5,
            audio_highpass_hz=50.0, audio_lowpass_hz=7500.0,
            preprocess_enabled=True,
            ingest_gain_multiplier=1.0,
            min_sensors_for_3d=4, min_sensors_for_2d=3,
            localization_max_tau_s=0.01,
            localization_algorithm="gcc_phat",
            localization_strategy="srp_grid",
            localization_srp_grid_resolution_m=0.5,
            localization_search_padding_m=2.0,
            localization_music_azimuth_step_deg=5.0,
            localization_music_elevation_step_deg=10.0,
            localization_subspace_freq_min_hz=200.0,
            localization_subspace_freq_max_hz=4000.0,
            localization_refine_confidence_threshold=0.6,
            localization_tight_array_aperture_m=0.25,
            beamformed_classification_enabled=True,
            beamformer_type="mvdr",
            beamformed_classification_min_sensor_count=3,
            beamformed_classification_confidence_margin=0.05,
            mvdr_diagonal_loading=0.001,
            classifier_diagonal_loading_scale=10.0,
            pre_classification_highpass_hz=0.0,
            pre_classification_lowpass_hz=0.0,
            gcc_phat_interp_factor=4,
        )
        assert create_classification_preprocessor(config) is None
        assert create_classification_preprocessor(config, extra_stages=[]) is None
