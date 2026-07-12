"""Tests for classification_audio_source, backend auto, and runtime_profile removal."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.cross_node_beamform import (
    CrossNodeBeamConfig,
    CrossNodeBeamformer,
    NodeAudioCandidate,
    omni_mix,
)
from minimappr.models import SyncGrade


def test_runtime_profile_env_raises_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAPPR_RUNTIME_PROFILE", "birdnet_hybrid_production")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    msg = str(exc.value)
    assert "MINIMAPPR_BIRDNET_ENABLED=true" in msg
    assert "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE=omni" in msg


def test_runtime_profile_default_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAPPR_RUNTIME_PROFILE", "default")
    # Must not raise.
    Settings.from_env()


def test_legacy_beamformed_bool_maps_to_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE", raising=False)
    monkeypatch.setenv("MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED", "false")
    s = Settings.from_env()
    assert s.classification_audio_source == "omni"
    assert s.beamformed_classification_enabled is False

    monkeypatch.setenv("MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED", "true")
    s = Settings.from_env()
    assert s.classification_audio_source == "beamformed"
    assert s.beamformed_classification_enabled is True


def test_new_var_wins_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE", "nearest_node_omni")
    monkeypatch.setenv("MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED", "true")
    s = Settings.from_env()
    assert s.classification_audio_source == "nearest_node_omni"


def test_invalid_audio_source_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(classification_audio_source="bogus")


def test_localization_config_carries_audio_source() -> None:
    s = Settings(classification_audio_source="nearest_node_omni", min_localization_confidence=0.5)
    lc = s.localization_config()
    assert lc.classification_audio_source == "nearest_node_omni"
    assert lc.min_localization_confidence == 0.5
    fc = s.fusion_config()
    assert fc.classification_audio_source == "nearest_node_omni"


def _candidate(node_id: str, position, channels) -> NodeAudioCandidate:
    return NodeAudioCandidate(
        node_id=node_id,
        node_position_m=np.asarray(position, dtype=np.float64),
        orientation=None,
        mic_positions_local_m=np.zeros((channels.shape[0], 3), dtype=np.float64),
        channels=channels,
        sample_rate_hz=16_000,
        sync_grade=SyncGrade.GPS_PPS,
        position_trusted=True,
        covers_event=True,
    )


def test_omni_mix_is_channel_mean() -> None:
    channels = np.array([[1.0, 3.0], [3.0, 5.0]])
    np.testing.assert_allclose(omni_mix(_candidate("n", (0, 0, 0), channels)), [2.0, 4.0])


@pytest.mark.asyncio
async def test_cross_node_nearest_omni_uses_nearest_node() -> None:
    near = _candidate("near", (1.0, 0.0, 0.0), np.ones((2, 8)))
    far = _candidate("far", (50.0, 0.0, 0.0), np.full((2, 8), 0.5))
    config = CrossNodeBeamConfig(
        enabled=True,
        max_range_m=100.0,
        classification_audio_source="nearest_node_omni",
    )
    beamformer = CrossNodeBeamformer(config)

    classified_nodes: list[str] = []

    async def classify_fn(samples, sample_rate_hz):
        # nearest node's omni mix is all-ones.
        classified_nodes.append("near" if float(samples.mean()) > 0.75 else "far")
        return {"bird": 0.9}

    result = await beamformer.classify_across_nodes(
        candidates=[far, near],
        source_position_m=(0.0, 0.0, 0.0),
        classify_fn=classify_fn,
    )
    assert result is not None
    assert result["fusion_method"] == "nearest_node_omni"
    assert result["best_node_id"] == "near"
    assert result["node_count"] == 1
    assert classified_nodes == ["near"]
