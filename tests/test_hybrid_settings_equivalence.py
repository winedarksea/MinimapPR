"""The explicit-settings combo reproduces the old birdnet_hybrid_production profile.

Lightweight (no birdnet/tensorflow required): asserts backend resolution, the
chunking policy gate, and the sidecar spawn env (incl. the hybrid-render flag)
match what the removed runtime profile produced.
"""

from __future__ import annotations

import pytest

from minimappr.classifiers import availability
from minimappr.ingest_sidecar_runtime import (
    _sidecar_hybrid_render_enabled,
    build_ingest_sidecar_environment,
)
from tests.hybrid_settings import hybrid_production_settings


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()
    yield
    availability.probe_backends.cache_clear()
    availability.resolve_backend.cache_clear()


def test_hybrid_settings_reproduce_profile_fields() -> None:
    s = hybrid_production_settings()
    assert s.classifier_backend == "birdnet"
    assert s.resolved_classifier_backend() == "birdnet"  # explicit passes through
    assert s.localization_algorithm == "srp_phat"
    assert s.localization_strategy == "fixed"
    assert s.classification_audio_source == "omni"
    assert s.birdnet_chunked_dispatch_enabled is True
    assert s.classification_window_seconds >= 30.0
    assert s.max_sensor_buffer_seconds >= 32.0
    assert s.localization_band_min_hz == 300.0
    assert s.localization_band_max_hz == 3500.0
    assert s.reporting_window_seconds >= 30.0


def test_hybrid_render_flag_requires_beamformed_source() -> None:
    # Old profile classified from the Rust hybrid render. Under the new formula the
    # hybrid render flag needs backend=birdnet AND audio_source=beamformed.
    omni = hybrid_production_settings()  # audio_source="omni"
    assert _sidecar_hybrid_render_enabled(omni) is False

    beamformed = hybrid_production_settings(classification_audio_source="beamformed")
    assert _sidecar_hybrid_render_enabled(beamformed) is True


def test_sidecar_env_carries_new_vars_not_runtime_profile() -> None:
    s = hybrid_production_settings(classification_audio_source="beamformed")
    env = build_ingest_sidecar_environment(s, default_classifier_command_json_builder=lambda _s: None)
    assert "MINIMAPPR_RUNTIME_PROFILE" not in env
    assert env["MINIMAPPR_SIDECAR_HYBRID_RENDER_ENABLED"] == "true"
    assert env["MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE"] == "beamformed"
    assert env["MINIMAPPR_MIN_LOCALIZATION_CONFIDENCE"] == str(s.min_localization_confidence)
    assert env["MINIMAPPR_LOCALIZATION_BAND_MIN_HZ"] == "300.0"
    assert env["MINIMAPPR_LOCALIZATION_BAND_MAX_HZ"] == "3500.0"
