"""Per-pair GCC-PHAT band and taper (``core/localization_band_policy.py``).

The ceiling and taper values match ``minimappr-core/src/localization_band_policy_tests.rs``
so both engines split pairs alike.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.localization import gcc_phat
from minimappr.core.localization_band_policy import (
    LocalizationBandPolicy,
    PairFrontEnd,
    SensorPairFrontEnds,
    array_alias_ceiling_hz,
    band_policy_from_config,
    normalize_band_hz,
    tukey_taper,
)
from minimappr.config import Settings
from minimappr.spatial_audio.geometry import SIRITH_MIC_POSITIONS_M

SOUND_SPEED_MPS = 343.2
# Firmware `kPlanarSensorOffsetsM`: four corners on a 25 mm circle plus the centre.
_PLANAR_CORNER_M = 0.025 / np.sqrt(2.0)
SIRITH_PLANAR_SENSOR_OFFSETS_M = [
    [_PLANAR_CORNER_M, _PLANAR_CORNER_M, 0.0],
    [-_PLANAR_CORNER_M, _PLANAR_CORNER_M, 0.0],
    [-_PLANAR_CORNER_M, -_PLANAR_CORNER_M, 0.0],
    [_PLANAR_CORNER_M, -_PLANAR_CORNER_M, 0.0],
    [0.0, 0.0, 0.0],
]


def test_sirith_tetra_ceiling_comes_from_its_widest_pair() -> None:
    assert array_alias_ceiling_hz(SIRITH_MIC_POSITIONS_M, SOUND_SPEED_MPS) == pytest.approx(3266.0, abs=2.0)


def test_sirith_planar_ceiling_comes_from_its_diagonal() -> None:
    assert array_alias_ceiling_hz(np.asarray(SIRITH_PLANAR_SENSOR_OFFSETS_M), SOUND_SPEED_MPS) == pytest.approx(
        3432.0, abs=1.0
    )


def test_single_or_coincident_mics_have_no_alias_ceiling() -> None:
    assert array_alias_ceiling_hz([np.array([1.0, 2.0, 3.0])], SOUND_SPEED_MPS) == 0.0
    assert array_alias_ceiling_hz([np.array([1.0, 2.0, 3.0])] * 2, SOUND_SPEED_MPS) == 0.0


def _tetra_plus_points() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    positions = {f"tetra:ch{index}": np.asarray(offset) for index, offset in enumerate(SIRITH_MIC_POSITIONS_M)}
    positions["point-a:ch0"] = np.array([20.0, 0.0, 1.5])
    positions["point-b:ch0"] = np.array([0.0, 30.0, 1.5])
    return positions, {sensor_id: sensor_id.split(":")[0] for sensor_id in positions}


def test_intra_array_pairs_use_their_array_ceiling_and_cross_node_pairs_the_wide_band() -> None:
    positions, node_ids = _tetra_plus_points()
    fronts = SensorPairFrontEnds(LocalizationBandPolicy(), positions, node_ids, SOUND_SPEED_MPS)
    intra = fronts.pair_front_end("tetra:ch0", "tetra:ch3")
    assert intra.taper_fraction is None
    assert intra.band_hz[0] == 50.0
    assert intra.band_hz[1] == pytest.approx(3266.0, abs=2.0)
    assert fronts.pair_front_end("tetra:ch2", "point-a:ch0") == PairFrontEnd(band_hz=(50.0, 11_000.0), taper_fraction=0.2)
    assert fronts.is_cross_node_pair("point-a:ch0", "point-b:ch0")


def test_only_solving_mics_set_an_arrays_ceiling() -> None:
    positions, node_ids = _tetra_plus_points()
    del positions["tetra:ch3"]
    fronts = SensorPairFrontEnds(LocalizationBandPolicy(), positions, node_ids, SOUND_SPEED_MPS)
    assert fronts.pair_front_end("tetra:ch0", "tetra:ch1").band_hz[1] == pytest.approx(3432.0, abs=2.0)


def test_an_explicit_intra_array_ceiling_overrides_the_geometry() -> None:
    positions, node_ids = _tetra_plus_points()
    policy = LocalizationBandPolicy(band_min_hz=300.0, intra_array_band_max_hz=3500.0)
    fronts = SensorPairFrontEnds(policy, positions, node_ids, SOUND_SPEED_MPS)
    assert fronts.pair_front_end("tetra:ch1", "tetra:ch2") == PairFrontEnd(band_hz=(300.0, 3500.0), taper_fraction=None)


def test_no_node_ids_put_every_sensor_on_one_array() -> None:
    positions = {f"ch{index}": np.asarray(offset) for index, offset in enumerate(SIRITH_MIC_POSITIONS_M)}
    fronts = SensorPairFrontEnds(LocalizationBandPolicy(), positions, None, SOUND_SPEED_MPS)
    assert not fronts.is_cross_node_pair("ch0", "ch3")
    assert fronts.pair_front_end("ch0", "ch3").taper_fraction is None


def test_zero_taper_keeps_the_rectangular_window_on_cross_node_pairs() -> None:
    positions, node_ids = _tetra_plus_points()
    fronts = SensorPairFrontEnds(
        LocalizationBandPolicy(cross_node_window_taper_fraction=0.0), positions, node_ids, SOUND_SPEED_MPS
    )
    assert fronts.pair_front_end("tetra:ch0", "point-a:ch0").taper_fraction is None


def test_tukey_taper_ramps_both_ends_and_leaves_the_middle() -> None:
    tapered = tukey_taper(np.ones(20), 0.4)
    expected_ramp = [0.0, 0.1464466, 0.5, 0.8535534]
    np.testing.assert_allclose(tapered[:4], expected_ramp, atol=1e-6)
    np.testing.assert_allclose(tapered[::-1][:4], expected_ramp, atol=1e-6)
    assert np.all(tapered[4:16] == 1.0)


def test_tukey_taper_of_zero_fraction_or_tiny_window_is_a_no_op() -> None:
    np.testing.assert_array_equal(tukey_taper(np.ones(3), 0.0), np.ones(3))
    np.testing.assert_array_equal(tukey_taper(np.ones(1), 1.0), np.ones(1))


def test_normalize_band_mirrors_the_rust_clamp() -> None:
    assert normalize_band_hz((50.0, 0.0), 16_000) == (50.0, 8_000.0)
    assert normalize_band_hz((50.0, 11_000.0), 16_000) == (50.0, 8_000.0)
    assert normalize_band_hz((300.0, 3_500.0), 16_000) == (300.0, 3_500.0)
    assert normalize_band_hz((5_000.0, 3_000.0), 16_000) is None
    assert normalize_band_hz(None, 16_000) is None


def test_shipped_settings_derive_the_array_ceiling_and_widen_cross_node_pairs() -> None:
    policy = band_policy_from_config(Settings().localization_config())
    assert policy == LocalizationBandPolicy()


def _two_node_call(sample_rate_hz: int, true_lag_s: float, cut_through_call: bool) -> tuple[np.ndarray, np.ndarray]:
    """A 3.5-6.5 kHz FM call heard at two far-apart nodes, cut by one shared window."""
    duration_s = 0.4
    window_n = int(0.3 * sample_rate_hz)
    rng = np.random.default_rng(7)

    def call(t: np.ndarray) -> np.ndarray:
        inside = (t >= 0.0) & (t <= duration_s)
        phase = 2.0 * np.pi * (3_500.0 * t + 3_000.0 * t * t / (2.0 * duration_s))
        return np.where(inside, np.sin(phase), 0.0)

    start_s = 0.15 if cut_through_call else -0.05
    t = start_s + np.arange(window_n) / sample_rate_hz
    near = call(t) + 0.01 * rng.standard_normal(window_n)
    far = 0.5 * call(t - true_lag_s) + 0.01 * rng.standard_normal(window_n)
    return near, far


@pytest.mark.parametrize("true_lag_s", [0.004, 0.021])
def test_cross_node_front_end_recovers_a_call_lag_the_array_band_misses(true_lag_s: float) -> None:
    # The E8 failure: a 3.5-6.5 kHz call cut mid-call by the event window.
    # Under the old 3430 Hz rectangular front end PHAT locks to the shared
    # window step (zero lag); the cross-node band plus taper finds the lag.
    sample_rate_hz = 24_000
    near, far = _two_node_call(sample_rate_hz, true_lag_s, cut_through_call=True)
    max_tau_s = 0.05
    legacy_tau_s, _ = gcc_phat(far, near, sample_rate_hz, max_tau_s, band_hz=(50.0, 3430.0))
    front = SensorPairFrontEnds(
        LocalizationBandPolicy(),
        {"a": np.zeros(3), "b": np.array([30.0, 0.0, 0.0])},
        {"a": "a", "b": "b"},
        SOUND_SPEED_MPS,
    ).pair_front_end("a", "b")
    split_tau_s, _ = gcc_phat(
        tukey_taper(far, front.taper_fraction),
        tukey_taper(near, front.taper_fraction),
        sample_rate_hz,
        max_tau_s,
        band_hz=front.band_hz,
    )
    assert abs(split_tau_s - true_lag_s) < 2.0 / sample_rate_hz
    assert abs(legacy_tau_s - true_lag_s) > 10.0 / sample_rate_hz
