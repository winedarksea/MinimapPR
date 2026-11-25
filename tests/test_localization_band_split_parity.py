"""Rust <-> Python parity of the per-pair GCC-PHAT front end (band split by pair type).

A tetra plus three point nodes hear one bird call through a shared event
window cut mid-call. Both engines must give intra-array pairs the tetra's
own c / (2 d_max) band and cross-node pairs the wide, tapered band, and must
agree on every cross-node lag (and recover the true one). The Rust side runs
the sidecar's hidden ``srp-oracle`` with a node partition.
"""

from __future__ import annotations

import itertools
import json
import subprocess

import numpy as np
import pytest

from minimappr.core.localization import gcc_phat
from minimappr.core.localization_band_policy import LocalizationBandPolicy, SensorPairFrontEnds, tukey_taper
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M
from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available

pytestmark = pytest.mark.skipif(not cargo_is_available(), reason="cargo toolchain required for the srp-oracle")

SAMPLE_RATE_HZ = 24_000
SOUND_SPEED_MPS = 343.2
WINDOW_SECONDS = 0.3
CALL_SECONDS = 0.45
SOURCE_M = np.array([8.0, 12.0, 3.0])
POINT_NODES_M = ([18.0, 2.0, 1.5], [3.0, 22.0, 1.5], [-12.0, 9.0, 1.5])
# Python's FFT length (len(a) + len(b)) differs from Rust's power of two, so
# the lags are not bit-identical. Most pairs agree to 0.03 samples; the sweep's
# whitened correlation has a near-tied lobe ~2.4 samples (~3 cm) off the truth
# that either engine may pick. The old 50-3430 Hz rectangular front end
# misses these lags by milliseconds.
LAG_AGREEMENT_SAMPLES = 3.0
TRUE_LAG_TOLERANCE_SAMPLES = 3.0


def _call(t: np.ndarray) -> np.ndarray:
    """A 3.5-6.5 kHz FM sweep over a 1.2 kHz harmonic stack (energy on both sides of 3.3 kHz)."""
    inside = (t >= 0.0) & (t <= CALL_SECONDS)
    sweep = np.sin(2.0 * np.pi * (3_500.0 * t + 3_000.0 * t * t / (2.0 * CALL_SECONDS)))
    harmonics = sum(np.sin(2.0 * np.pi * 1_200.0 * k * t) / k for k in (1, 2))
    return np.where(inside, sweep + 0.5 * harmonics, 0.0)


def _scene() -> tuple[np.ndarray, np.ndarray, list[int]]:
    tetra_m = np.asarray(SIRITH_TETRA_SENSOR_OFFSETS_M) + np.array([0.0, 0.0, 1.5])
    positions = np.vstack([tetra_m, np.asarray(POINT_NODES_M)])
    array_index_by_channel = [0, 0, 0, 0, 1, 2, 3]
    rng = np.random.default_rng(11)
    window_t = 0.12 + np.arange(int(WINDOW_SECONDS * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    channels = []
    for position in positions:
        distance_m = float(np.linalg.norm(SOURCE_M - position))
        signal = _call(window_t - distance_m / SOUND_SPEED_MPS) / max(distance_m, 1.0)
        channels.append(signal + 1e-3 * rng.standard_normal(window_t.size))
    return np.asarray(channels), positions, array_index_by_channel


def _rust_pairs(channels: np.ndarray, positions: np.ndarray, array_index_by_channel: list[int]) -> dict:
    payload = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sound_speed_mps": SOUND_SPEED_MPS,
        "mic_positions_m": positions.tolist(),
        "channels": [channel.tolist() for channel in channels],
        "array_index_by_channel": array_index_by_channel,
    }
    completed = subprocess.run(
        [str(build_rust_sidecar_binary()), "srp-oracle"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return {(pair["ch_a"], pair["ch_b"]): pair for pair in json.loads(completed.stdout)["pair_tdoas"]}


def test_both_engines_split_the_band_by_pair_type_and_agree_on_cross_node_lags() -> None:
    channels, positions, array_index_by_channel = _scene()
    rust_pairs = _rust_pairs(channels, positions, array_index_by_channel)
    sensor_ids = [f"ch{index}" for index in range(len(positions))]
    python_fronts = SensorPairFrontEnds(
        LocalizationBandPolicy(),
        dict(zip(sensor_ids, positions)),
        {sensor: f"node{array}" for sensor, array in zip(sensor_ids, array_index_by_channel)},
        SOUND_SPEED_MPS,
    )
    arrival_s = np.linalg.norm(SOURCE_M - positions, axis=1) / SOUND_SPEED_MPS
    sample_s = 1.0 / SAMPLE_RATE_HZ
    for ch_a, ch_b in itertools.combinations(range(len(positions)), 2):
        rust = rust_pairs[(ch_a, ch_b)]
        front = python_fronts.pair_front_end(sensor_ids[ch_a], sensor_ids[ch_b])
        assert rust["band_hz"] == pytest.approx(list(front.band_hz), abs=0.5)
        assert rust["window_taper_fraction"] == pytest.approx(front.taper_fraction)
        if array_index_by_channel[ch_a] == array_index_by_channel[ch_b]:
            assert front.taper_fraction is None
            assert front.band_hz[1] == pytest.approx(3266.0, abs=2.0)
            continue
        assert front.band_hz == (50.0, 11_000.0)
        assert front.phat_beta == pytest.approx(0.7)
        python_lag_s, _ = gcc_phat(
            tukey_taper(channels[ch_a], front.taper_fraction),
            tukey_taper(channels[ch_b], front.taper_fraction),
            SAMPLE_RATE_HZ,
            0.1,
            band_hz=front.band_hz,
            phat_beta=front.phat_beta,
        )
        true_lag_s = arrival_s[ch_a] - arrival_s[ch_b]
        assert abs(rust["lag_seconds"] - python_lag_s) < LAG_AGREEMENT_SAMPLES * sample_s, (ch_a, ch_b)
        assert abs(rust["lag_seconds"] - true_lag_s) < TRUE_LAG_TOLERANCE_SAMPLES * sample_s, (ch_a, ch_b)
