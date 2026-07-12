"""Shared test helper reproducing the removed ``birdnet_hybrid_production``
runtime profile as explicit settings.

The ``runtime_profile`` "mode" system was removed; its behaviors are now plain
settings. This helper applies the exact settings the old profile forced, so
tests that previously passed ``runtime_profile="birdnet_hybrid_production"``
keep the same effective configuration. Profile-controlled keys win over caller
overrides (mirroring the old ``__post_init__`` ordering where the profile ran
after field assignment).
"""

from __future__ import annotations

from minimappr.config import DEFAULT_BIRDNET_HYBRID_RULES_CONFIG_PATH, Settings


def hybrid_production_kwargs(**overrides) -> dict:
    """Return Settings kwargs equivalent to the old birdnet_hybrid_production profile."""
    kwargs = dict(overrides)
    classification_window_seconds = max(
        float(overrides.get("classification_window_seconds", 30.0)), 30.0
    )
    kwargs.update(
        birdnet_enabled=True,
        localization_algorithm="srp_phat",
        localization_strategy="fixed",
        # The old profile forced omni (Python beamformer off); allow tests to vary
        # this to exercise the Rust hybrid-render path.
        classification_audio_source=overrides.get("classification_audio_source", "omni"),
        skip_localization_for_classification=False,
        birdnet_chunked_dispatch_enabled=True,
        birdnet_chunk_overlap_seconds=min(
            float(overrides.get("birdnet_chunk_overlap_seconds", 2.0)), 2.0
        ),
        classification_window_seconds=classification_window_seconds,
        localization_band_min_hz=300.0,
        localization_band_max_hz=3500.0,
        reporting_window_seconds=max(
            float(overrides.get("reporting_window_seconds", 30.0)), 30.0
        ),
    )
    kwargs["max_sensor_buffer_seconds"] = max(
        float(overrides.get("max_sensor_buffer_seconds", 32.0)),
        classification_window_seconds + 2.0,
    )
    if "rules_config_path" not in overrides:
        kwargs["rules_config_path"] = DEFAULT_BIRDNET_HYBRID_RULES_CONFIG_PATH
    return kwargs


def hybrid_production_settings(**overrides) -> Settings:
    return Settings(**hybrid_production_kwargs(**overrides))
