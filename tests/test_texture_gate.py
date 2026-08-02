"""Noise-floor texture gate: metric units + orchestrator annotation/demotion.

The gate exists because classification windows containing only a node's noise
floor are boosted ~30 dB by the bounded-RMS AGC and confidently mislabeled
("Zipper (clothing)", "Fireworks", ...). It flags such windows using two
level-invariant discriminators and — once ``confidence_factor`` drops below 1.0
— demotes their confidence without ever rewriting the label or dropping the
detection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.classification import ClassificationOrchestrator
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.taxonomy import DEFAULT_CATEGORY_TO_IFF
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import ClassificationResult, GeoPoint
from minimappr.storage.db import Storage
from minimappr.utils.audio import (
    energy_contrast_db,
    framed_rms_db,
    framed_spectral_flatness_median,
)


SAMPLE_RATE_HZ = 16_000
CONTRAST_THRESHOLD_DB = 8.0
FLATNESS_MIN = 0.2


def _gated(samples: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> bool | None:
    """Apply the orchestrator's AND condition to a raw signal."""
    contrast = energy_contrast_db(samples, sample_rate_hz)
    flatness = framed_spectral_flatness_median(samples, sample_rate_hz)
    if contrast is None or flatness is None:
        return None
    return contrast < CONTRAST_THRESHOLD_DB and flatness > FLATNESS_MIN


def _white_noise(seconds: float = 2.0, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(20260801)
    return rng.normal(0.0, scale, size=int(SAMPLE_RATE_HZ * seconds)).astype(np.float32)


def _click_train(seconds: float = 2.0) -> np.ndarray:
    """Impulsive event over a very quiet floor — a real acoustic event."""
    rng = np.random.default_rng(4242)
    samples = rng.normal(0.0, 0.0005, size=int(SAMPLE_RATE_HZ * seconds)).astype(np.float32)
    burst = (0.3 * np.hanning(200)).astype(np.float32)
    for start in range(0, samples.size - burst.size, 4_000):
        samples[start : start + burst.size] += burst
    return samples


def _steady_tone(seconds: float = 2.0, amplitude: float = 0.01) -> np.ndarray:
    """Drone-hum proxy: steady ~-40 dBFS tone with no energy contrast at all."""
    t = np.arange(int(SAMPLE_RATE_HZ * seconds), dtype=np.float32) / SAMPLE_RATE_HZ
    return (amplitude * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


# ── Metric units ────────────────────────────────────────────────────────────


class TestTextureMetrics:
    def test_framed_rms_db_drops_partial_trailing_frame(self) -> None:
        samples = np.ones(SAMPLE_RATE_HZ + 700, dtype=np.float32)
        frames_db = framed_rms_db(samples, SAMPLE_RATE_HZ, frame_ms=100.0)
        assert frames_db.size == 10
        assert np.allclose(frames_db, 0.0, atol=1e-4)

    def test_white_noise_is_flagged(self) -> None:
        noise = _white_noise()
        assert energy_contrast_db(noise, SAMPLE_RATE_HZ) < CONTRAST_THRESHOLD_DB
        assert framed_spectral_flatness_median(noise, SAMPLE_RATE_HZ) > FLATNESS_MIN
        assert _gated(noise) is True

    def test_verdict_is_level_invariant(self) -> None:
        """A 30 dB quieter copy of the same texture gets the same verdict.

        Long-range detections are faint by nature; the gate must key on shape,
        not on absolute level, or it would silently truncate detection range.
        """
        noise = _white_noise()
        faint = (noise * 0.03).astype(np.float32)
        assert _gated(faint) is True
        assert energy_contrast_db(faint, SAMPLE_RATE_HZ) == pytest.approx(
            energy_contrast_db(noise, SAMPLE_RATE_HZ), abs=0.05
        )
        assert framed_spectral_flatness_median(faint, SAMPLE_RATE_HZ) == pytest.approx(
            framed_spectral_flatness_median(noise, SAMPLE_RATE_HZ), abs=0.01
        )

    def test_impulsive_event_passes_on_contrast(self) -> None:
        clicks = _click_train()
        assert energy_contrast_db(clicks, SAMPLE_RATE_HZ) > CONTRAST_THRESHOLD_DB
        assert _gated(clicks) is False

    def test_steady_tone_passes_on_flatness(self) -> None:
        """Design constraint: a hovering drone has no contrast but is tonal."""
        tone = _steady_tone()
        assert energy_contrast_db(tone, SAMPLE_RATE_HZ) < CONTRAST_THRESHOLD_DB
        assert framed_spectral_flatness_median(tone, SAMPLE_RATE_HZ) < FLATNESS_MIN
        assert _gated(tone) is False

    @pytest.mark.parametrize("size", [0, int(SAMPLE_RATE_HZ * 0.15)])
    def test_short_and_empty_windows_fail_open(self, size: int) -> None:
        samples = _white_noise()[:size]
        assert energy_contrast_db(samples, SAMPLE_RATE_HZ) is None
        assert framed_spectral_flatness_median(samples, SAMPLE_RATE_HZ) is None

    def test_invalid_sample_rate_fails_open(self) -> None:
        noise = _white_noise()
        assert energy_contrast_db(noise, 0) is None
        assert framed_spectral_flatness_median(noise, 0) is None


# ── Orchestrator wiring ─────────────────────────────────────────────────────


class _StubClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        return ClassificationResult(
            label="Zipper (clothing)",
            confidence=0.8,
            scores={"Zipper (clothing)": 0.8, "Fireworks": 0.1},
            features={},
        )


class _StorageStub:
    async def upsert_label(self, *, name: str, category: str, source: str, created_ns: int) -> str:
        del name, category, source, created_ns
        return "label-texture"


class _TaxonomyStub:
    def category_for_label(self, label: str) -> str:
        del label
        return "unknown"

    def iff_for_category(self, category: str) -> str:
        return DEFAULT_CATEGORY_TO_IFF.get(category.strip().lower(), "unknown")


class _EnvironmentStub:
    def get_speed_of_sound(self, position_m: tuple[float, float, float]) -> float:
        del position_m
        return 343.0


class _PassthroughBeamformer:
    """Returns the first sensor window unchanged — enough to exercise the path."""

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
        return next(iter(sensor_windows.values()))


def _orchestrator(
    *,
    classifier: AudioClassifier | None = None,
    flags: list[bool] | None = None,
    **gate_kwargs: object,
) -> ClassificationOrchestrator:
    return ClassificationOrchestrator(
        classifier=classifier or _StubClassifier(),
        beamformer=_PassthroughBeamformer(),
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
        on_texture_gate_demotion=(None if flags is None else lambda: flags.append(True)),
        **gate_kwargs,  # type: ignore[arg-type]
    )


class TestOrchestratorTextureGate:
    @pytest.mark.asyncio
    async def test_annotate_only_default_leaves_confidence_untouched(self) -> None:
        flags: list[bool] = []
        orchestrator = _orchestrator(flags=flags)

        result = await orchestrator.classify_omni_only(
            reference_signal=_white_noise(),
            sample_rate_hz=SAMPLE_RATE_HZ,
            event_time_ns=1,
        )

        annotation = result.classification.features["texture_gate"]
        assert annotation["gated"] is True
        assert "original_confidence" not in annotation
        assert result.classification.confidence == pytest.approx(0.8)
        # Fired even in annotate-only mode: the counter measures what demotion
        # would suppress.
        assert flags == [True]

    @pytest.mark.asyncio
    async def test_demotion_scales_confidence_and_preserves_scores(self) -> None:
        orchestrator = _orchestrator(texture_gate_confidence_factor=0.25)

        result = await orchestrator.classify_omni_only(
            reference_signal=_white_noise(),
            sample_rate_hz=SAMPLE_RATE_HZ,
            event_time_ns=1,
        )

        annotation = result.classification.features["texture_gate"]
        assert annotation["gated"] is True
        assert annotation["original_confidence"] == pytest.approx(0.8)
        assert result.classification.confidence == pytest.approx(0.2)
        # Label and per-class scores are never rewritten.
        assert result.classification.label == "Zipper (clothing)"
        assert result.classification.scores["Zipper (clothing)"] == pytest.approx(0.8)
        # The raw omni result keeps its unmodified confidence for reporting.
        assert result.omni_classification.confidence == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_real_event_is_annotated_but_not_gated(self) -> None:
        flags: list[bool] = []
        orchestrator = _orchestrator(flags=flags, texture_gate_confidence_factor=0.25)

        result = await orchestrator.classify_omni_only(
            reference_signal=_click_train(),
            sample_rate_hz=SAMPLE_RATE_HZ,
            event_time_ns=1,
        )

        annotation = result.classification.features["texture_gate"]
        assert annotation["gated"] is False
        assert annotation["contrast_db"] > CONTRAST_THRESHOLD_DB
        assert result.classification.confidence == pytest.approx(0.8)
        assert flags == []

    @pytest.mark.asyncio
    async def test_disabled_gate_adds_no_annotation(self) -> None:
        orchestrator = _orchestrator(
            texture_gate_enabled=False,
            texture_gate_confidence_factor=0.25,
        )

        result = await orchestrator.classify_omni_only(
            reference_signal=_white_noise(),
            sample_rate_hz=SAMPLE_RATE_HZ,
            event_time_ns=1,
        )

        assert "texture_gate" not in result.classification.features
        assert result.classification.confidence == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_short_window_is_skipped(self) -> None:
        orchestrator = _orchestrator(texture_gate_confidence_factor=0.25)

        result = await orchestrator.classify_omni_only(
            reference_signal=_white_noise(seconds=0.15),
            sample_rate_hz=SAMPLE_RATE_HZ,
            event_time_ns=1,
        )

        assert result.classification.features["texture_gate"] == {"skipped": "short_window"}
        assert result.classification.confidence == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_degraded_results_are_left_alone(self) -> None:
        class _FailingClassifier(AudioClassifier):
            def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
                del samples, sample_rate_hz
                raise RuntimeError("backend crashed")

        flags: list[bool] = []
        orchestrator = _orchestrator(
            classifier=_FailingClassifier(),
            flags=flags,
            texture_gate_confidence_factor=0.25,
        )

        result = await orchestrator.classify_omni_only(
            reference_signal=_white_noise(),
            sample_rate_hz=SAMPLE_RATE_HZ,
            event_time_ns=1,
        )

        assert result.classification.features.get("reason") == "classification_error"
        assert "texture_gate" not in result.classification.features
        assert flags == []

    @pytest.mark.asyncio
    async def test_adopted_sidecar_result_is_gated(self) -> None:
        """Sidecar-authoritative results funnel through the same server-side gate."""
        flags: list[bool] = []
        orchestrator = _orchestrator(flags=flags, texture_gate_confidence_factor=0.25)

        result = await orchestrator.adopt_authoritative_classification(
            classification=ClassificationResult(
                label="Fireworks",
                confidence=0.6,
                scores={"Fireworks": 0.6},
                features={},
            ),
            event_time_ns=1,
            classification_signal=_white_noise(),
            sample_rate_hz=SAMPLE_RATE_HZ,
        )

        assert result.classification.features["texture_gate"]["gated"] is True
        assert result.classification.confidence == pytest.approx(0.15)
        assert flags == [True]

    @pytest.mark.asyncio
    async def test_beamformed_only_path_is_gated(self) -> None:
        """The beamformed-only trigger path funnels through the same gate."""
        flags: list[bool] = []
        orchestrator = _orchestrator(flags=flags, texture_gate_confidence_factor=0.25)
        noise = _white_noise()
        sensor_ids = ["s0", "s1", "s2", "s3"]

        result = await orchestrator.classify_beamformed_only(
            sample_rate_hz=SAMPLE_RATE_HZ,
            capability_tier="localization",
            selected_sensor_ids=sensor_ids,
            selected_positions={
                sid: np.array(pos, dtype=np.float64)
                for sid, pos in zip(
                    sensor_ids,
                    [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.05)],
                )
            },
            selected_windows={sid: noise for sid in sensor_ids},
            localization_position_m=(5.0, 5.0, 0.0),
            event_time_ns=1,
        )

        assert result is not None
        assert result.classification.features["texture_gate"]["gated"] is True
        assert result.classification.confidence == pytest.approx(0.2)
        assert flags == [True]

    @pytest.mark.asyncio
    async def test_adopted_result_with_empty_signal_is_a_no_op(self) -> None:
        flags: list[bool] = []
        orchestrator = _orchestrator(flags=flags, texture_gate_confidence_factor=0.25)

        result = await orchestrator.adopt_authoritative_classification(
            classification=ClassificationResult(
                label="Fireworks",
                confidence=0.6,
                scores={"Fireworks": 0.6},
                features={},
            ),
            event_time_ns=1,
            classification_signal=np.zeros(0, dtype=np.float32),
            sample_rate_hz=SAMPLE_RATE_HZ,
        )

        assert "texture_gate" not in result.classification.features
        assert result.classification.confidence == pytest.approx(0.6)
        assert flags == []


# ── Settings plumbing ───────────────────────────────────────────────────────


class TestTextureGateSettings:
    def test_defaults_ship_in_annotate_only_mode(self) -> None:
        settings = Settings()
        assert settings.classification_texture_gate_enabled is True
        assert settings.classification_texture_gate_contrast_db == pytest.approx(8.0)
        assert settings.classification_texture_gate_flatness_min == pytest.approx(0.2)
        # 1.0 = annotate only. Demotion is turned on by a live config PATCH once
        # the flagged population has been reviewed.
        assert settings.classification_texture_gate_confidence_factor == pytest.approx(1.0)

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_TEXTURE_GATE_ENABLED", "false")
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_TEXTURE_GATE_CONTRAST_DB", "6.5")
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_TEXTURE_GATE_FLATNESS_MIN", "0.35")
        monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_TEXTURE_GATE_CONFIDENCE_FACTOR", "0.25")

        settings = Settings.from_env()

        assert settings.classification_texture_gate_enabled is False
        assert settings.classification_texture_gate_contrast_db == pytest.approx(6.5)
        assert settings.classification_texture_gate_flatness_min == pytest.approx(0.35)
        assert settings.classification_texture_gate_confidence_factor == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("classification_texture_gate_contrast_db", -1.0, "CONTRAST_DB"),
            ("classification_texture_gate_contrast_db", float("nan"), "CONTRAST_DB"),
            ("classification_texture_gate_flatness_min", 1.5, "FLATNESS_MIN"),
            ("classification_texture_gate_confidence_factor", -0.1, "CONFIDENCE_FACTOR"),
            ("classification_texture_gate_confidence_factor", 1.5, "CONFIDENCE_FACTOR"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: float, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Settings(**{field: value})

    @pytest.mark.asyncio
    async def test_orchestrator_receives_settings(self, monkeypatch, tmp_path: Path) -> None:
        """The four knobs reach the orchestrator as direct constructor kwargs."""
        from minimappr.core import fusion_node as fusion_node_module

        captured: dict[str, object] = {}
        real_orchestrator = fusion_node_module.ClassificationOrchestrator

        def _capture(**kwargs: object) -> ClassificationOrchestrator:
            captured.update(kwargs)
            return real_orchestrator(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(fusion_node_module, "ClassificationOrchestrator", _capture)

        settings = Settings(
            db_path=tmp_path / "texture_gate.db",
            snippet_dir=tmp_path / "snippets",
            classification_texture_gate_enabled=False,
            classification_texture_gate_contrast_db=5.0,
            classification_texture_gate_flatness_min=0.4,
            classification_texture_gate_confidence_factor=0.5,
        )
        settings.snippet_dir.mkdir(parents=True, exist_ok=True)
        storage = Storage(settings.db_path)
        await storage.initialize()
        try:
            fusion_node_module.FusionNode(
                settings=settings,
                registry=NodeRegistry(),
                buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
                localizer=LocalizationEngine(max_tau_s=0.03),
                classifier=_StubClassifier(),
                tracker=TrackManager(settings),
                storage=storage,
                live_callback=lambda payload: asyncio.sleep(0, result=None),
                coordinate_frame=LocalCoordinateFrame(
                    origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0),
                    mode="flat",
                ),
                zone_matcher=ZoneMatcher(storage=storage),
            )
        finally:
            await storage.close()

        assert captured["texture_gate_enabled"] is False
        assert captured["texture_gate_contrast_db"] == pytest.approx(5.0)
        assert captured["texture_gate_flatness_min"] == pytest.approx(0.4)
        assert captured["texture_gate_confidence_factor"] == pytest.approx(0.5)
        assert callable(captured["on_texture_gate_demotion"])
