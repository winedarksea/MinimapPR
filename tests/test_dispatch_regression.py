"""Regression tests for LocalizationDispatcher fallback/failover paths.

Covers:
- _run_algorithm fallback from non-gcc algorithm to gcc_phat on LocalizationError
- fallback_count increments on error-triggered fallback
- cascade strategy exception handling mid-cascade
- localize_2d provenance fields (attempted_algorithm, resolved_algorithm)
- fixed strategy provenance
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.localization import LocalizationError
from minimappr.core.localization_dispatch import LocalizationDispatcher
from minimappr.models import LocalizationResult
from tests.helpers import FailingLocalizer, StubLocalizer


# Shared test fixtures.
_POSITIONS = {
    "a": np.array([0.0, 0.0, 0.0]),
    "b": np.array([5.0, 0.0, 0.0]),
    "c": np.array([0.0, 5.0, 0.0]),
    "d": np.array([0.0, 0.0, 5.0]),
}
_WINDOWS = {sid: np.ones(256, dtype=np.float32) for sid in _POSITIONS}
_SR = 16_000
_TEMP = 20.0
_HUM = 0.5


def test_run_algorithm_fallback_on_error() -> None:
    """When a non-gcc algorithm raises LocalizationError, _run_algorithm
    falls back to gcc_phat and increments fallback_count."""
    gcc = StubLocalizer("gcc", 0.3)
    failing_music = FailingLocalizer("music")
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="music",
        algorithms={"gcc_phat": gcc, "music": failing_music},
    )
    result = dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert failing_music.calls == 1
    assert gcc.calls == 1
    assert dispatcher.fallback_count() == 1
    assert result.attempted_algorithm == "music"
    assert result.resolved_algorithm == "gcc_phat"
    assert dispatcher.last_algorithm_name() == "gcc_phat"
    assert dispatcher.last_attempted_algorithm_name() == "music"


def test_run_algorithm_gcc_phat_error_propagates() -> None:
    """When gcc_phat itself fails, the error propagates (no further fallback)."""
    failing_gcc = FailingLocalizer("gcc_phat")
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="gcc_phat",
        algorithms={"gcc_phat": failing_gcc},
    )
    with pytest.raises(LocalizationError):
        dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert failing_gcc.calls == 1
    assert dispatcher.fallback_count() == 0


def test_sensor_weights_trigger_gcc_fallback_for_unsupported_algorithm() -> None:
    """Weighted localization should not silently ignore weights on non-gcc algorithms."""

    class WeightedGccStub:
        def __init__(self) -> None:
            self.calls = 0

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
            assert sensor_weights is not None
            reference_sensor = sorted(sensor_positions.keys())[0]
            return LocalizationResult(
                position_m=(0.0, 0.0, 0.0),
                confidence=0.3,
                gdop=1.0,
                reference_sensor=reference_sensor,
                tdoa_s={},
            )

    gcc = WeightedGccStub()
    music = StubLocalizer("music", 0.8)
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="music",
        algorithms={"gcc_phat": gcc, "music": music},
    )

    result = dispatcher.localize(
        _POSITIONS,
        _WINDOWS,
        _SR,
        _TEMP,
        _HUM,
        sensor_weights={sid: 1.0 for sid in _POSITIONS},
    )

    assert music.calls == 0
    assert gcc.calls == 1
    assert dispatcher.fallback_count() == 1
    assert result.attempted_algorithm == "music"
    assert result.resolved_algorithm == "gcc_phat"


def test_fallback_count_accumulates() -> None:
    """Multiple fallback events accumulate in fallback_count."""
    gcc = StubLocalizer("gcc", 0.3)
    failing = FailingLocalizer("esprit")
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="esprit",
        algorithms={"gcc_phat": gcc, "esprit": failing},
    )
    for _ in range(3):
        dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert dispatcher.fallback_count() == 3
    assert gcc.calls == 3
    assert failing.calls == 3


def test_cascade_exception_during_refinement() -> None:
    """Cascade continues past algorithms that raise errors during refinement."""
    gcc = StubLocalizer("gcc", 0.2)  # Below threshold → triggers cascade
    failing_music = FailingLocalizer("music")
    good_srp = StubLocalizer("srp", 0.6)
    dispatcher = LocalizationDispatcher(
        strategy="cascade",
        default_algorithm="gcc_phat",
        refine_confidence_threshold=0.4,
        algorithms={
            "gcc_phat": gcc,
            "music": failing_music,
            "srp_phat": good_srp,
        },
    )
    result = dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    # music failed but srp succeeded with higher confidence than gcc.
    assert result.confidence == 0.6
    assert result.resolved_algorithm == "srp_phat"
    assert result.attempted_algorithm == "gcc_phat"
    assert failing_music.calls == 1
    assert good_srp.calls == 1


def test_cascade_all_refinements_fail() -> None:
    """When all cascade refinements fail, the default result is returned."""
    gcc = StubLocalizer("gcc", 0.2)
    failing_music = FailingLocalizer("music")
    failing_esprit = FailingLocalizer("esprit")
    failing_srp = FailingLocalizer("srp_phat")
    dispatcher = LocalizationDispatcher(
        strategy="cascade",
        default_algorithm="gcc_phat",
        refine_confidence_threshold=0.4,
        algorithms={
            "gcc_phat": gcc,
            "music": failing_music,
            "esprit": failing_esprit,
            "srp_phat": failing_srp,
        },
    )
    result = dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert result.confidence == 0.2
    assert result.resolved_algorithm == "gcc_phat"
    assert result.attempted_algorithm == "gcc_phat"


def test_cascade_above_threshold_skips_refinement() -> None:
    """When the default algorithm meets the confidence threshold, no refinement runs."""
    gcc = StubLocalizer("gcc", 0.9)
    music = StubLocalizer("music", 0.95)
    dispatcher = LocalizationDispatcher(
        strategy="cascade",
        default_algorithm="gcc_phat",
        refine_confidence_threshold=0.4,
        algorithms={"gcc_phat": gcc, "music": music},
    )
    result = dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert result.confidence == 0.9
    assert music.calls == 0  # Never tried


def test_localize_2d_provenance_fields() -> None:
    """localize_2d stamps attempted_algorithm and resolved_algorithm as gcc_phat."""
    gcc = StubLocalizer("gcc", 0.5)
    dispatcher = LocalizationDispatcher(
        strategy="geometry_aware",
        default_algorithm="gcc_phat",
        algorithms={"gcc_phat": gcc, "music": StubLocalizer("music", 0.8)},
    )
    result = dispatcher.localize_2d(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert result.attempted_algorithm == "gcc_phat"
    assert result.resolved_algorithm == "gcc_phat"
    assert gcc.localize_2d_calls == 1
    assert dispatcher.last_algorithm_name() == "gcc_phat"
    assert dispatcher.last_attempted_algorithm_name() == "gcc_phat"


def test_fixed_strategy_provenance() -> None:
    """Fixed strategy stamps provenance with the selected algorithm name."""
    srp = StubLocalizer("srp", 0.7)
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="srp_phat",
        algorithms={"gcc_phat": StubLocalizer("gcc", 0.3), "srp_phat": srp},
    )
    result = dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)

    assert result.attempted_algorithm == "srp_phat"
    assert result.resolved_algorithm == "srp_phat"
    assert srp.calls == 1


def test_unknown_algorithm_falls_back_to_gcc() -> None:
    """If default_algorithm is unknown, __post_init__ resets it to gcc_phat."""
    gcc = StubLocalizer("gcc", 0.4)
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="nonexistent",
        algorithms={"gcc_phat": gcc},
    )
    assert dispatcher.default_algorithm == "gcc_phat"
    result = dispatcher.localize(_POSITIONS, _WINDOWS, _SR, _TEMP, _HUM)
    assert result.attempted_algorithm == "gcc_phat"
    assert result.resolved_algorithm == "gcc_phat"
