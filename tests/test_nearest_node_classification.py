"""Unit tests for nearest-node omni classification selection in FusionNode."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from minimappr.config import Settings
from minimappr.core.fusion_node import FusionNode


def test_nearest_reference_sensor_argmin() -> None:
    positions = {
        "a": np.array([0.0, 0.0, 0.0]),
        "b": np.array([10.0, 0.0, 0.0]),
        "c": np.array([3.0, 4.0, 0.0]),
    }
    assert FusionNode._nearest_reference_sensor(positions, (2.0, 0.0, 0.0)) == "a"
    assert FusionNode._nearest_reference_sensor(positions, (9.0, 1.0, 0.0)) == "b"
    assert FusionNode._nearest_reference_sensor({}, (0.0, 0.0, 0.0)) is None


def _product(confidence: float, position_m):
    branch = SimpleNamespace(
        localization_confidence=confidence,
        localization_position_m=position_m,
    )
    return SimpleNamespace(
        localization_branch=branch,
        selected_positions={
            "near": np.array([1.0, 0.0, 0.0]),
            "far": np.array([50.0, 0.0, 0.0]),
        },
    )


def test_nearest_node_selected_when_confident_and_enabled() -> None:
    settings = Settings(classification_audio_source="nearest_node_omni", min_localization_confidence=0.2)
    fake = SimpleNamespace(
        settings=settings,
        _nearest_reference_sensor=FusionNode._nearest_reference_sensor,
    )
    product = _product(confidence=0.9, position_m=(0.0, 0.0, 0.0))
    assert FusionNode._nearest_node_omni_sensor(fake, product) == "near"


def test_low_confidence_falls_back_to_none() -> None:
    settings = Settings(classification_audio_source="nearest_node_omni", min_localization_confidence=0.5)
    fake = SimpleNamespace(
        settings=settings,
        _nearest_reference_sensor=FusionNode._nearest_reference_sensor,
    )
    product = _product(confidence=0.1, position_m=(0.0, 0.0, 0.0))
    assert FusionNode._nearest_node_omni_sensor(fake, product) is None


def test_disabled_when_source_is_beamformed() -> None:
    settings = Settings(classification_audio_source="beamformed")
    fake = SimpleNamespace(
        settings=settings,
        _nearest_reference_sensor=FusionNode._nearest_reference_sensor,
    )
    product = _product(confidence=0.9, position_m=(0.0, 0.0, 0.0))
    assert FusionNode._nearest_node_omni_sensor(fake, product) is None
