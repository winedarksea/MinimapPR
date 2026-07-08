from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from minimappr.models import NodeOrientation, NodeSpec, NodeType
from minimappr.spatial_audio import PROFILES, encode_ambisonics
from minimappr.spatial_audio.geometry import (
    SIRITH_MIC_POSITIONS_M,
    centroid_corrected_positions,
    rotate_positions,
)
from minimappr.spatial_audio.parametric import enhance_foa_parametric
from minimappr.spatial_audio.profiles import PARAMETRIC_V2
from minimappr.spatial_audio.objects import (
    subtract_rendered_object_bed_stft,
    slerp_unit_vectors,
)
from minimappr.spatial_audio.stft import cola_error, istft_channels, stft_channels
from minimappr.core.ambi_atob import encode_mono_to_bformat
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M


_V2_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "atob_foa_tetra_golden_v2.npz"


def test_ambisonics_profiles_load_from_checked_in_json() -> None:
    assert set(PROFILES) >= {"linear_v1", "parametric_v2"}
    assert PROFILES["linear_v1"].max_parametric_blend == 0.0
    assert PROFILES["parametric_v2"].max_parametric_blend == 0.85


def test_sirith_geometry_matches_firmware_offsets() -> None:
    corrected, _ = centroid_corrected_positions(SIRITH_MIC_POSITIONS_M)
    np.testing.assert_allclose(
        corrected,
        np.asarray(SIRITH_TETRA_SENSOR_OFFSETS_M, dtype=np.float64),
        atol=5e-7,
    )


def test_node_orientation_defaults_identity_and_rotates_offsets() -> None:
    node = NodeSpec(
        id="tetra",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
    )
    assert node.orientation == NodeOrientation()

    rotated = rotate_positions(
        np.asarray([(1.0, 0.0, 0.0)], dtype=np.float64),
        NodeOrientation(yaw_deg=90.0),
    )
    np.testing.assert_allclose(rotated[0], (0.0, 1.0, 0.0), atol=1e-12)


def test_sqrt_hann_stft_round_trips_with_quarter_hop() -> None:
    rng = np.random.default_rng(123)
    samples = rng.standard_normal((4, 4096)).astype(np.float32) * 0.1
    frame_size = 512
    hop_size = 128
    spectra = stft_channels(samples, frame_size=frame_size, hop_size=hop_size)
    reconstructed = istft_channels(
        spectra,
        frame_size=frame_size,
        hop_size=hop_size,
        n_samples=samples.shape[1],
    )
    assert cola_error(frame_size, hop_size) < 1e-3
    np.testing.assert_allclose(
        reconstructed[:, frame_size:-frame_size],
        samples[:, frame_size:-frame_size],
        atol=1e-6,
    )


def test_parametric_v2_improves_directional_energy_without_high_rate_scope() -> None:
    sample_rate_hz = 16_000
    duration_s = 0.5
    t = np.arange(int(sample_rate_hz * duration_s), dtype=np.float64) / sample_rate_hz
    mono = (
        0.35 * np.sin(2.0 * math.pi * 1200.0 * t)
        + 0.15 * np.sin(2.0 * math.pi * 2200.0 * t)
    ).astype(np.float32)
    direction = np.asarray((0.86, 0.43, 0.27), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    linear = np.zeros((4, mono.size), dtype=np.float32)
    linear[0] = mono / math.sqrt(2.0)
    linear[1:4] = (0.08 * direction[:, np.newaxis] * mono[np.newaxis, :]).astype(np.float32)

    parametric = enhance_foa_parametric(linear, sample_rate_hz, profile=PARAMETRIC_V2)

    def xyz_to_w_db(foa: np.ndarray) -> float:
        w_energy = float(np.mean(foa[0] ** 2)) + 1e-12
        xyz_energy = float(np.mean(foa[1:4] ** 2)) + 1e-12
        return 10.0 * math.log10(xyz_energy / w_energy)

    assert xyz_to_w_db(parametric) - xyz_to_w_db(linear) >= 6.0
    assert float(np.max(np.abs(parametric))) <= 0.9801


def test_slerp_unit_vectors_uses_great_circle_midpoint() -> None:
    midpoint = slerp_unit_vectors((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5)
    expected = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(midpoint, (expected, expected, 0.0), atol=1e-12)


def test_stft_wiener_object_subtraction_reduces_object_and_keeps_other_source() -> None:
    sample_rate_hz = 16_000
    n_samples = sample_rate_hz
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    selected = (0.35 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
    unselected = (0.25 * np.sin(2.0 * math.pi * 1700.0 * t)).astype(np.float32)
    selected_bed = encode_mono_to_bformat(selected, (1.0, 0.0, 0.0))
    unselected_bed = encode_mono_to_bformat(unselected, (0.0, 1.0, 0.0))
    bed_full = (selected_bed + unselected_bed).astype(np.float32)

    cleaned = subtract_rendered_object_bed_stft(
        bed_full,
        selected_bed,
        n_samples,
        sample_rate_hz=sample_rate_hz,
    )

    stable = slice(2048, -2048)
    selected_rms = float(np.sqrt(np.mean(selected_bed[:, stable] ** 2)))
    residual_rms = float(np.sqrt(np.mean((cleaned[:, stable] - unselected_bed[:, stable]) ** 2)))
    unselected_rms = float(np.sqrt(np.mean(unselected_bed[:, stable] ** 2)))

    assert residual_rms <= selected_rms * 0.10
    assert residual_rms <= unselected_rms * 0.10
    assert np.all(np.isfinite(cleaned))


def test_parametric_v2_golden_fixture_stays_within_tolerance() -> None:
    with np.load(_V2_FIXTURE_PATH) as fixture:
        channels = fixture["channels"]
        golden = fixture["bed"]
        sample_rate_hz = int(fixture["sample_rate_hz"])

    current = encode_ambisonics(channels, sample_rate_hz, profile="parametric_v2")
    np.testing.assert_allclose(current, golden, rtol=1e-5, atol=2e-5)
