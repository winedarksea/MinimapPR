"""Parity: Rust ``minimappr_core::onset_clustering`` vs the Python onset-SRP pipeline.

Goldens (``tests/fixtures/onset_clustering_parity_goldens.npz``) come from Python
``localize_onset_clusters`` via ``python -m tests.onset_clustering_parity_cases``;
the Rust side runs through the sidecar's hidden ``onset-clustering-oracle``.

Tolerances, per stage (why):

* Integer / decision outputs are exact: onset sample indices, frame reference
  sensors, envelope flags, grid sizes, which frame each peak came from, cluster
  membership, statuses and reasons, range-projection modes, height verdicts,
  classification nodes. Every step is deterministic, so any mismatch is a bug.
* Spectral quantities (onset strengths, curve weights, bandwidths, peak values,
  SRP peak weights) at 1e-7 relative: both sides are f64 FFT pipelines that
  differ only in summation order and FFT rounding (~1e-13), with margin for
  the max/median operations that amplify it slightly.
* Refined geometry (positions, covariances, TDOAs, residuals, observability,
  range intervals) at 1e-5 relative with an absolute floor of 1e-5 x the
  array's largest magnitude: these pass through soft-L1 TRF iterations and
  pseudo-inverses (the solver gate's tolerance).
* Covariances and GDOP at 1e-4 relative to the largest element: they come
  from ``pinv(JᵀJ)`` of a forward-difference (√eps step) Jacobian, whose
  ~1e-8 relative rounding noise is amplified by the covariance condition
  number (up to ~1e5 on range-elongated or height-ambiguous fixes). Measured
  worst case 2e-5.
* Confidence compared as ``-ln(confidence)`` at 1e-5 relative: it is
  ``exp(-residual / (4 sample periods))``, so a 1e-6 relative residual
  difference becomes a ~1e-5 relative confidence difference for weak fits.

A second test re-runs Python against the same goldens so a server-side change
shows up as a golden refresh, not a mysterious Rust failure.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from tests.onset_clustering_parity_cases import GOLDEN_PATH, case_inputs, oracle_case, run_python_window
from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available

SPECTRAL_RELATIVE_TOLERANCE = 1.0e-7
GEOMETRY_RELATIVE_TOLERANCE = 1.0e-5
COVARIANCE_RELATIVE_TOLERANCE = 1.0e-4
LAG_ABSOLUTE_TOLERANCE_S = 1.0e-12
# A single-bin band has an RMS bandwidth of exactly 0 in one summation order and
# ~1e-13 Hz in another; only the 300 Hz narrowband test reads it.
FREQUENCY_ABSOLUTE_TOLERANCE_HZ = 1.0e-6

EXACT_CLUSTER_FIELDS = (
    "status", "status_reasons", "onset_samples", "template_node_ids", "contributing_node_ids", "loudest_node_id",
    "reference_sensor", "height_resolution", "range_projection_mode", "refined_by_least_squares", "frame_count",
    "independent_onset_count",
)
SPECTRAL_CLUSTER_FIELDS = ("band_min_hz", "band_max_hz", "weight", "used_envelope_fraction")
GEOMETRY_CLUSTER_FIELDS = ("position_m", "range_interval_m", "range_observability", "residual_rms_seconds")
COVARIANCE_CLUSTER_FIELDS = ("covariance_m2", "gdop")


def _load_goldens():
    with np.load(GOLDEN_PATH) as archive:
        arrays = {key: archive[key] for key in archive.files}
    descriptors = [json.loads(str(raw)) for raw in arrays["case_descriptors_json"]]
    audio = {key[len("audio_"):]: value for key, value in arrays.items() if key.startswith("audio_") and not key.endswith("_sensors")}
    sensors = {key[len("audio_"):-len("_sensors")]: json.loads(str(value)) for key, value in arrays.items() if key.endswith("_sensors")}
    goldens = [json.loads(str(arrays[f"case{index}_result_json"])) for index in range(len(descriptors))]
    return list(case_inputs(descriptors, audio, sensors, goldens)), goldens


def _close(actual, expected, relative: float, label: str, absolute_floor: float = 0.0) -> None:
    if expected is None:
        assert actual is None, f"{label}: expected None, got {actual}"
        return
    assert actual is not None, f"{label}: expected {expected}, got None"
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected_array))) if expected_array.size else 0.0
    np.testing.assert_allclose(actual_array, expected_array, rtol=relative, atol=max(relative * scale, absolute_floor), err_msg=label)


def _compare_trace(actual: dict, expected: dict, label: str) -> None:
    assert [n["node_id"] for n in actual["onsets_by_node"]] == [n["node_id"] for n in expected["onsets_by_node"]], label
    for node_actual, node_expected in zip(actual["onsets_by_node"], expected["onsets_by_node"]):
        node_label = f"{label} onsets {node_expected['node_id']}"
        assert [o[0] for o in node_actual["onsets"]] == [o[0] for o in node_expected["onsets"]], node_label
        _close([o[1:] for o in node_actual["onsets"]], [o[1:] for o in node_expected["onsets"]], SPECTRAL_RELATIVE_TOLERANCE, node_label)
    assert len(actual["frames"]) == len(expected["frames"]), f"{label} frame count"
    for frame_actual, frame_expected in zip(actual["frames"], expected["frames"]):
        frame_label = f"{label} frame {frame_expected['frame_index']}"
        for key in ("frame_index", "template_node_id", "reference_sensor", "onset_sample"):
            assert frame_actual[key] == frame_expected[key], f"{frame_label} {key}"
        _close(frame_actual["weight"], frame_expected["weight"], SPECTRAL_RELATIVE_TOLERANCE, f"{frame_label} weight")
        assert [c["sensor_b"] for c in frame_actual["curves"]] == [c["sensor_b"] for c in frame_expected["curves"]], frame_label
        for curve_actual, curve_expected in zip(frame_actual["curves"], frame_expected["curves"]):
            curve_label = f"{frame_label} curve {curve_expected['sensor_b']}"
            assert curve_actual["used_envelope"] == curve_expected["used_envelope"], curve_label
            assert abs(curve_actual["peak_lag_s"] - curve_expected["peak_lag_s"]) <= LAG_ABSOLUTE_TOLERANCE_S, curve_label
            for key in ("weight_sum", "peak_value", "value_sum"):
                _close(curve_actual[key], curve_expected[key], SPECTRAL_RELATIVE_TOLERANCE, f"{curve_label} {key}")
            for key in ("effective_bandwidth_hz", "centre_frequency_hz"):
                _close(curve_actual[key], curve_expected[key], SPECTRAL_RELATIVE_TOLERANCE, f"{curve_label} {key}", FREQUENCY_ABSOLUTE_TOLERANCE_HZ)
    assert actual["is_direction_grid"] == expected["is_direction_grid"], label
    assert actual["grid_point_count"] == expected["grid_point_count"], f"{label} grid size"
    assert len(actual["peaks"]) == len(expected["peaks"]), f"{label} peak count"
    for index, (peak_actual, peak_expected) in enumerate(zip(actual["peaks"], expected["peaks"])):
        peak_label = f"{label} peak {index}"
        assert peak_actual["frame_index"] == peak_expected["frame_index"], peak_label
        assert peak_actual["is_frame_primary"] == peak_expected["is_frame_primary"], peak_label
        for key in ("position_m", "range_m", "value", "weight"):
            _close(peak_actual[key], peak_expected[key], SPECTRAL_RELATIVE_TOLERANCE, f"{peak_label} {key}")
    assert len(actual["clusters"]) == len(expected["clusters"]), f"{label} cluster count"
    for index, (cluster_actual, cluster_expected) in enumerate(zip(actual["clusters"], expected["clusters"])):
        cluster_label = f"{label} pre-refine cluster {index}"
        for key in ("frame_indices", "status", "status_reasons", "independent_onset_count"):
            assert cluster_actual[key] == cluster_expected[key], f"{cluster_label} {key}"
        _close(cluster_actual["weight"], cluster_expected["weight"], SPECTRAL_RELATIVE_TOLERANCE, cluster_label)


def _compare_evidence(actual: dict | None, expected: dict | None, label: str) -> None:
    """Attribution level evidence: arrivals exact, SNRs spectral, geometry-derived values at 1e-5."""
    if expected is None:
        assert actual is None, f"{label}: unexpected attribution evidence"
        return
    label = f"{label} attribution evidence"
    assert actual is not None, label
    for key in ("node_band_snr_db", "node_distances_m", "node_onset_arrivals"):
        assert sorted(actual[key]) == sorted(expected[key]), f"{label} {key} nodes"
    for node, arrivals in expected["node_onset_arrivals"].items():
        assert [a[0] for a in actual["node_onset_arrivals"][node]] == [a[0] for a in arrivals], f"{label} arrivals {node}"
    snr_nodes = sorted(expected["node_band_snr_db"])
    _close([actual["node_band_snr_db"][n] for n in snr_nodes], [expected["node_band_snr_db"][n] for n in snr_nodes],
           SPECTRAL_RELATIVE_TOLERANCE, f"{label} band SNR")
    distance_nodes = sorted(expected["node_distances_m"])
    _close([actual["node_distances_m"][n] for n in distance_nodes], [expected["node_distances_m"][n] for n in distance_nodes],
           GEOMETRY_RELATIVE_TOLERANCE, f"{label} distances")
    for key in ("fitted_source_level_db", "spreading_residual_db"):
        _close(actual[key], expected[key], GEOMETRY_RELATIVE_TOLERANCE, f"{label} {key}")


def _negative_log(value):
    return None if value is None else -np.log(max(float(value), np.finfo(float).tiny))


def _compare_result(index: int, actual: dict, expected: dict) -> None:
    label = f"case {index}"
    assert "error" not in actual, f"{label}: {actual}"
    _compare_trace(actual["trace"], expected["trace"], label)
    for key in ("onset_count", "frame_count", "consumed_onset_samples_by_node"):
        assert actual[key] == expected[key], f"{label} {key}"
    assert len(actual["clusters"]) == len(expected["clusters"]), f"{label} cluster count"
    for rank, (cluster_actual, cluster_expected) in enumerate(zip(actual["clusters"], expected["clusters"])):
        cluster_label = f"{label} cluster {rank}"
        for key in EXACT_CLUSTER_FIELDS:
            assert cluster_actual[key] == cluster_expected[key], f"{cluster_label} {key}"
        for key in SPECTRAL_CLUSTER_FIELDS:
            _close(cluster_actual[key], cluster_expected[key], SPECTRAL_RELATIVE_TOLERANCE, f"{cluster_label} {key}")
        for key in GEOMETRY_CLUSTER_FIELDS:
            _close(cluster_actual[key], cluster_expected[key], GEOMETRY_RELATIVE_TOLERANCE, f"{cluster_label} {key}")
        for key in COVARIANCE_CLUSTER_FIELDS:
            _close(cluster_actual[key], cluster_expected[key], COVARIANCE_RELATIVE_TOLERANCE, f"{cluster_label} {key}")
        assert sorted(cluster_actual["tdoa_s"]) == sorted(cluster_expected["tdoa_s"]), cluster_label
        _close(
            [cluster_actual["tdoa_s"][key] for key in sorted(cluster_expected["tdoa_s"])],
            [cluster_expected["tdoa_s"][key] for key in sorted(cluster_expected["tdoa_s"])],
            GEOMETRY_RELATIVE_TOLERANCE,
            f"{cluster_label} tdoa_s",
        )
        _compare_evidence(cluster_actual["attribution_evidence"], cluster_expected["attribution_evidence"], cluster_label)
        _close(
            _negative_log(cluster_actual["confidence"]),
            _negative_log(cluster_expected["confidence"]),
            GEOMETRY_RELATIVE_TOLERANCE,
            f"{cluster_label} -ln(confidence)",
        )


@pytest.mark.skipif(not cargo_is_available(), reason="cargo toolchain required to build the Rust sidecar oracle")
def test_rust_onset_clustering_matches_python_goldens() -> None:
    inputs, goldens = _load_goldens()
    cases = [oracle_case(*case[1:]) for case in inputs]
    completed = subprocess.run(
        [str(build_rust_sidecar_binary()), "onset-clustering-oracle"],
        input=json.dumps({"cases": cases}),
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(completed.stdout)["results"]
    assert len(results) == len(goldens)
    for index, (result, golden) in enumerate(zip(results, goldens)):
        _compare_result(index, result, golden)


def test_python_onset_clustering_still_matches_goldens() -> None:
    inputs, goldens = _load_goldens()
    for index, (case, golden) in enumerate(zip(inputs, goldens)):
        _compare_result(index, run_python_window(*case[1:]), golden)
