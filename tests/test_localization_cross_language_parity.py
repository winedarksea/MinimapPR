"""Cross-language agreement: Rust SRP-PHAT sidecar vs Python Cartesian solver.

Feeds identical synthetic tetrahedral windows to both estimators and asserts they
agree on bearing, and that both flag the range as unobservable for a single tight
array. This defends the parity contract: the two engines need not produce identical
*positions*, but they must agree on bearing and on range observability.

The Rust estimate is obtained via the sidecar's hidden ``srp-oracle`` subcommand
(see ``minimappr-ingest-sidecar/src/main.rs``), which runs
``estimate_tetrahedral_steering`` on channels supplied as JSON on stdin.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from minimappr.core.localization import LocalizationEngine, speed_of_sound_mps
from minimappr.core.range_projection import normalize_range_mode, range_mode_is_unobservable
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, synthesize_delayed_array_channels


MINIMAPPR_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_DIR = MINIMAPPR_ROOT / "minimappr-ingest-sidecar"
SIDECAR_BINARY_PATH = SIDECAR_DIR / "target" / "debug" / "minimappr-ingest-sidecar"

SAMPLE_RATE_HZ = 48_000


pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo toolchain required to build the Rust sidecar oracle",
)


@functools.lru_cache(maxsize=1)
def _sidecar_binary() -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "minimappr-ingest-sidecar"],
        cwd=str(SIDECAR_DIR),
        check=True,
    )
    return SIDECAR_BINARY_PATH


def _run_rust_oracle(
    *,
    channels: np.ndarray,
    mic_positions_m: np.ndarray,
    sound_speed_mps: float,
) -> dict:
    payload = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sound_speed_mps": sound_speed_mps,
        "mic_positions_m": mic_positions_m.tolist(),
        "channels": [channel.astype(float).tolist() for channel in channels],
    }
    completed = subprocess.run(
        [str(_sidecar_binary()), "srp-oracle"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _python_localization(
    *,
    channels: np.ndarray,
    mic_positions_m: np.ndarray,
    temperature_c: float,
    humidity_fraction: float,
):
    sensor_positions = {
        f"ch{index}": mic_positions_m[index] for index in range(len(mic_positions_m))
    }
    sensor_windows = {
        f"ch{index}": channels[index].astype(np.float32) for index in range(len(channels))
    }
    engine = LocalizationEngine(max_tau_s=0.02)
    return engine.localize(
        sensor_positions,
        sensor_windows,
        SAMPLE_RATE_HZ,
        temperature_c,
        humidity_fraction,
    )


@pytest.mark.parametrize("distance_m", [6.0, 22.0])
def test_rust_and_python_agree_on_bearing_and_range_observability(distance_m: float) -> None:
    temperature_c = 20.0
    humidity_fraction = 0.0
    sound_speed_mps = speed_of_sound_mps(
        temperature_c=temperature_c, humidity_fraction=humidity_fraction
    )

    mic_positions_m = np.asarray(SIRITH_TETRA_SENSOR_OFFSETS_M, dtype=np.float64)
    centroid = mic_positions_m.mean(axis=0)
    # Range is unobservable for a ~5 cm aperture at any distance along this
    # bearing; both engines must agree regardless of how far the source is.
    direction = _unit(np.array([20.0, 10.0, 3.0]))
    source_m = centroid + direction * distance_m
    true_bearing = direction

    rng = np.random.default_rng(20260613)
    mono = rng.normal(0.0, 0.3, size=8192).astype(np.float32)
    channels = synthesize_delayed_array_channels(
        mono,
        SAMPLE_RATE_HZ,
        source_position_m=tuple(source_m),
        sensor_offsets_m=SIRITH_TETRA_SENSOR_OFFSETS_M,
        sound_speed_mps=sound_speed_mps,
    )

    rust = _run_rust_oracle(
        channels=channels,
        mic_positions_m=mic_positions_m,
        sound_speed_mps=sound_speed_mps,
    )
    python_result = _python_localization(
        channels=channels,
        mic_positions_m=mic_positions_m,
        temperature_c=temperature_c,
        humidity_fraction=humidity_fraction,
    )

    rust_bearing = _unit(np.asarray(rust["steering_direction"], dtype=np.float64))
    python_bearing = _unit(np.asarray(python_result.position_m, dtype=np.float64) - centroid)

    # Both engines point at the true source bearing...
    assert float(np.dot(rust_bearing, true_bearing)) > 0.9
    assert float(np.dot(python_bearing, true_bearing)) > 0.9
    # ...and therefore at each other.
    assert float(np.dot(rust_bearing, python_bearing)) > 0.9

    # Both flag range as unobservable for a single tight array (canonical vocab).
    assert range_mode_is_unobservable(normalize_range_mode(rust["range_projection_mode"]))
    assert range_mode_is_unobservable(python_result.range_projection_mode)
