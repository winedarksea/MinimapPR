"""Golden cases for the Rust ``minimappr_core::tdoa_solver`` parity gate.

Defines deterministic synthetic inputs for Python
``cartesian_tdoa.solve_cartesian_tdoa`` and
``multi_node_bearing_fusion.fuse_bearings``, runs Python on them, and (when
executed as ``python -m tests.tdoa_solver_parity_cases``) writes
``tests/fixtures/tdoa_solver_parity_goldens.npz``. The parity test feeds the
same JSON inputs to the sidecar's ``tdoa-solver-oracle`` subcommand.

Golden layout: ``case_inputs_json`` (one JSON object per case) plus, per case
``i``: ``case{i}_position_m``, ``case{i}_covariance_m2``, ``case{i}_scalars``
and ``case{i}_mode`` (cartesian) or ``case{i}_rejection`` (fusion reject).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from minimappr.core.cartesian_tdoa import solve_cartesian_tdoa
from minimappr.core.multi_node_bearing_fusion import BearingObservation, fuse_bearings
from minimappr.core.spatial_constraints import AmplitudeRatioConstraint, SpatialBearingConstraint
from minimappr.core.subspace_bearing import BearingPrior
from minimappr.core.tdoa_measurements import PairTdoaMeasurement, solve_reference_tdoas

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "tdoa_solver_parity_goldens.npz"
SOUND_SPEED_MPS = 343.0
SAMPLE_RATE_HZ = 48_000
INTERPOLATION_FACTOR = 4
CARTESIAN_SCALAR_NAMES = ("confidence", "gdop", "residual_rms_seconds", "radial_observability", "condition_number")
FUSION_SCALAR_NAMES = ("range_m", "confidence")
SIRITH_TETRA_OFFSETS_M = (
    (-0.016238, 0.025000, -0.010205),
    (0.027063, 0.000000, -0.010205),
    (-0.016238, -0.025000, -0.010205),
    (0.005413, 0.000000, 0.030615),
)


def _tetra(node: str, origin: tuple[float, float, float]) -> dict[str, list[float]]:
    return {
        f"{node}:ch{index}": [origin[axis] + offset[axis] for axis in range(3)]
        for index, offset in enumerate(SIRITH_TETRA_OFFSETS_M)
    }


def _cartesian_case(
    *,
    sensor_positions: dict[str, list[float]],
    source_m: tuple[float, float, float],
    noise_std_s: float,
    seed: int,
    reference_sensor: str | None = None,
    **extra: object,
) -> dict:
    rng = np.random.default_rng(seed)
    source = np.asarray(source_m, dtype=np.float64)
    sensor_ids = sorted(sensor_positions)
    measurements = []
    for sensor_a, sensor_b in itertools.combinations(sensor_ids, 2):
        distance_a = float(np.linalg.norm(source - np.asarray(sensor_positions[sensor_a])))
        distance_b = float(np.linalg.norm(source - np.asarray(sensor_positions[sensor_b])))
        measurements.append(
            {
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "tdoa_seconds": (distance_a - distance_b) / SOUND_SPEED_MPS + float(rng.normal(0.0, noise_std_s)),
                "correlation_peak": float(rng.uniform(0.3, 0.95)),
                "weight": float(rng.uniform(0.5, 1.0)),
            }
        )
    reference = reference_sensor or sensor_ids[0]
    reference_tdoa_s = solve_reference_tdoas(
        measurements=[PairTdoaMeasurement(**measurement) for measurement in measurements],
        sensor_ids=sensor_ids,
        reference_sensor=reference,
    )
    case = {
        "kind": "cartesian",
        "sensor_positions": sensor_positions,
        "measurements": measurements,
        "sound_speed_mps": SOUND_SPEED_MPS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "interpolation_factor": INTERPOLATION_FACTOR,
        "reference_sensor": reference,
        "reference_tdoa_s": reference_tdoa_s,
    }
    case.update(extra)
    return case


def _unit(vector: tuple[float, float, float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float64)
    return (array / np.linalg.norm(array)).tolist()


def _fusion_observation(node_id, origin, toward, *, lateral_std_m=0.5, confidence=0.8, event_time_ns=0):
    direction = np.asarray(toward, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return {
        "node_id": node_id,
        "origin_m": list(origin),
        "direction": (direction / np.linalg.norm(direction)).tolist(),
        "lateral_std_m": lateral_std_m,
        "confidence": confidence,
        "event_time_ns": event_time_ns,
    }


def build_cases() -> list[dict]:
    one_sample_s = 1.0 / (SAMPLE_RATE_HZ * INTERPOLATION_FACTOR)
    distributed = {
        "n0": [0.0, 0.0, 1.5], "n1": [18.0, 1.0, 1.2], "n2": [17.0, 19.0, 1.8], "n3": [-1.0, 17.5, 1.0],
    }
    six_mics = dict(distributed, n4=[8.0, -6.0, 2.5], n5=[9.0, 26.0, 0.8])
    two_tetras = dict(_tetra("a", (0.0, 0.0, 1.5)), **_tetra("b", (12.0, 3.0, 1.5)))
    collinear = {"l0": [0.0, 0.0, 1.5], "l1": [6.0, 0.0, 1.5], "l2": [12.0, 0.0, 1.5], "l3": [18.0, 0.0, 1.5]}
    cases = [
        _cartesian_case(sensor_positions=distributed, source_m=(7.0, 9.0, 6.0), noise_std_s=3 * one_sample_s, seed=1),
        _cartesian_case(sensor_positions=six_mics, source_m=(-30.0, 25.0, 12.0), noise_std_s=2 * one_sample_s, seed=2),
        _cartesian_case(sensor_positions=distributed, source_m=(9.0, 8.0, 15.0), noise_std_s=one_sample_s, seed=3),
        _cartesian_case(sensor_positions=_tetra("t", (0.0, 0.0, 1.5)), source_m=(0.6, 0.5, 1.8), noise_std_s=0.2 * one_sample_s, seed=4),
        _cartesian_case(sensor_positions=_tetra("t", (0.0, 0.0, 1.5)), source_m=(25.0, 18.0, 6.0), noise_std_s=0.2 * one_sample_s, seed=5),
        _cartesian_case(
            sensor_positions=_tetra("t", (0.0, 0.0, 1.5)), source_m=(25.0, 18.0, 6.0), noise_std_s=0.2 * one_sample_s, seed=6,
            bearing_prior={"direction": _unit((25.0, 18.0, 4.5)), "strength": 2.0, "quality": 0.7},
        ),
        _cartesian_case(
            sensor_positions=two_tetras, source_m=(6.0, 14.0, 4.0), noise_std_s=0.5 * one_sample_s, seed=7,
            bearing_constraints=[
                {"node_id": "a", "origin_m": [0.0, 0.0, 1.5], "direction": _unit((6.0, 14.0, 2.5)), "strength": 1.5, "quality": 0.8},
                {"node_id": "b", "origin_m": [12.0, 3.0, 1.5], "direction": _unit((-6.0, 11.0, 2.5)), "strength": 1.0, "quality": 0.6},
            ],
            amplitude_constraints=[
                {"node_a": "a", "node_b": "b", "origin_a_m": [0.0, 0.0, 1.5], "origin_b_m": [12.0, 3.0, 1.5],
                 "measured_log_amplitude_ratio": float(np.log(np.hypot(12.5, 2.5) / np.hypot(15.2, 2.5))), "strength": 0.5},
            ],
        ),
        # Collinear along x: the fix is only defined up to rotation about the
        # array axis (TRF drifts along that null space on rounding noise), so
        # the parity test compares rotation invariants for this case.
        _cartesian_case(
            sensor_positions=collinear, source_m=(9.0, 14.0, 3.0), noise_std_s=one_sample_s, seed=8,
            parity_invariant={"kind": "axisymmetric", "axis_point_m": [0.0, 0.0, 1.5], "axis_direction": [1.0, 0.0, 0.0]},
        ),
        _cartesian_case(
            sensor_positions=distributed, source_m=(7.0, 9.0, 6.0), noise_std_s=2 * one_sample_s, seed=9,
            radial_refinement_enabled=False, far_field_initial_range_m=80.0, far_field_prior_radial_std_m=120.0,
        ),
        {"kind": "bearing_fusion", "observations": [
            _fusion_observation("a", (0.0, 0.0, 1.5), (20.0, 30.0, 6.0)),
            _fusion_observation("b", (40.0, 0.0, 1.5), (20.5, 29.0, 5.0), lateral_std_m=1.2, confidence=0.6, event_time_ns=200_000_000),
        ]},
        {"kind": "bearing_fusion", "observations": [
            _fusion_observation("a", (0.0, 0.0, 1.5), (60.0, 80.0, 10.0)),
            _fusion_observation("b", (30.0, -5.0, 1.0), (61.0, 79.0, 9.0), lateral_std_m=2.0, confidence=0.4),
            _fusion_observation("c", (90.0, 10.0, 2.0), (59.0, 81.0, 11.0), lateral_std_m=1.0, confidence=0.9),
            _fusion_observation("c", (90.0, 10.0, 2.0), (0.0, 0.0, 0.0), confidence=0.3),
        ], "config": {"min_separation_deg": 5.0, "max_condition": 1.0e4, "max_range_m": 1200.0,
                      "sound_speed_mps": 340.0, "window_seconds": 1.5}},
        {"kind": "bearing_fusion", "observations": [
            _fusion_observation("a", (0.0, 0.0, 0.0), (0.0, 1000.0, 0.0)),
            _fusion_observation("b", (1.0, 0.0, 0.0), (0.0, 1000.0, 0.0)),
        ]},
        {"kind": "bearing_fusion", "observations": [
            _fusion_observation("a", (0.0, 0.0, 1.5), (20.0, 30.0, 6.0)),
            _fusion_observation("b", (40.0, 0.0, 1.5), (20.0, 30.0, 6.0), event_time_ns=5_000_000_000),
        ]},
        {"kind": "bearing_fusion", "observations": [
            _fusion_observation("a", (0.0, 0.0, 0.0), (1500.0, 900.0, 0.0)),
            _fusion_observation("b", (300.0, 0.0, 0.0), (1500.0, 900.0, 0.0)),
        ]},
    ]
    return cases


def run_python_case(case: dict) -> dict:
    """Run the Python reference on one case; same output schema as the Rust oracle."""
    if case["kind"] == "bearing_fusion":
        config = case.get("config") or {}
        observations = [
            BearingObservation(
                node_id=raw["node_id"],
                origin_m=np.asarray(raw["origin_m"], dtype=np.float64),
                direction=np.asarray(raw["direction"], dtype=np.float64),
                lateral_std_m=raw["lateral_std_m"],
                confidence=raw["confidence"],
                event_time_ns=raw["event_time_ns"],
                range_prior_m=0.0,
                expiry_ns=0,
            )
            for raw in case["observations"]
        ]
        fused = fuse_bearings(observations, **config)
        if isinstance(fused, str):
            return {"rejection": fused}
        return {
            "position_m": np.asarray(fused.position_m).tolist(),
            "covariance_m2": np.asarray(fused.covariance_m2).tolist(),
            "range_m": fused.range_m,
            "confidence": fused.confidence,
            "contributor_node_ids": fused.contributor_node_ids,
        }
    prior = case.get("bearing_prior")
    solve = solve_cartesian_tdoa(
        sensor_positions={key: np.asarray(value, dtype=np.float64) for key, value in case["sensor_positions"].items()},
        measurements=[PairTdoaMeasurement(**measurement) for measurement in case["measurements"]],
        sound_speed_mps=case["sound_speed_mps"],
        sample_rate_hz=case["sample_rate_hz"],
        interpolation_factor=case["interpolation_factor"],
        reference_sensor=case["reference_sensor"],
        reference_tdoa_s=case["reference_tdoa_s"],
        bearing_prior=None if prior is None else BearingPrior(
            direction=np.asarray(prior["direction"], dtype=np.float64), strength=prior["strength"], quality=prior["quality"],
        ),
        bearing_constraints=[
            SpatialBearingConstraint(
                node_id=raw["node_id"], origin_m=np.asarray(raw["origin_m"], dtype=np.float64),
                direction=np.asarray(raw["direction"], dtype=np.float64), strength=raw["strength"], quality=raw["quality"],
            )
            for raw in case.get("bearing_constraints", [])
        ],
        amplitude_constraints=[
            AmplitudeRatioConstraint(
                node_a=raw["node_a"], node_b=raw["node_b"],
                origin_a_m=np.asarray(raw["origin_a_m"], dtype=np.float64),
                origin_b_m=np.asarray(raw["origin_b_m"], dtype=np.float64),
                measured_log_amplitude_ratio=raw["measured_log_amplitude_ratio"], strength=raw["strength"],
            )
            for raw in case.get("amplitude_constraints", [])
        ],
        far_field_initial_range_m=case.get("far_field_initial_range_m", 50.0),
        far_field_prior_radial_std_m=case.get("far_field_prior_radial_std_m"),
        radial_refinement_enabled=case.get("radial_refinement_enabled", True),
    )
    return {
        "position_m": np.asarray(solve.position_m).tolist(),
        "covariance_m2": np.asarray(solve.covariance_m2).tolist(),
        "confidence": solve.confidence,
        "gdop": solve.gdop,
        "residual_rms_seconds": solve.residual_rms_seconds,
        "radial_observability": solve.radial_observability,
        "range_projection_mode": solve.range_projection_mode,
        "condition_number": solve.condition_number,
    }


def golden_arrays(cases: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"case_inputs_json": np.asarray([json.dumps(case) for case in cases])}
    for index, case in enumerate(cases):
        result = run_python_case(case)
        if "rejection" in result:
            arrays[f"case{index}_rejection"] = np.asarray(result["rejection"])
            continue
        arrays[f"case{index}_position_m"] = np.asarray(result["position_m"], dtype=np.float64)
        arrays[f"case{index}_covariance_m2"] = np.asarray(result["covariance_m2"], dtype=np.float64)
        names = FUSION_SCALAR_NAMES if case["kind"] == "bearing_fusion" else CARTESIAN_SCALAR_NAMES
        arrays[f"case{index}_scalars"] = np.asarray([result[name] for name in names], dtype=np.float64)
        if case["kind"] == "cartesian":
            arrays[f"case{index}_mode"] = np.asarray(result["range_projection_mode"])
    return arrays


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, **golden_arrays(build_cases()))
    print(f"wrote {GOLDEN_PATH}")
