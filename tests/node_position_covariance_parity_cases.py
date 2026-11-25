"""Cross-language parity cases for the node-position covariance term.

Python ``minimappr/core/node_position_covariance.py`` and Rust
``minimappr-core/src/tdoa_node_position_covariance.rs`` implement the same
``G R G^T`` propagation of stated node position errors. The cases below are
written with their Python result to ``tests/fixtures/node_position_covariance_parity_cases.json``;
the Rust test ``node_position_term_matches_the_python_parity_fixture`` and
``tests/test_node_position_covariance.py`` both check against that file.

Regenerate after an intended change: ``python -m tests.node_position_covariance_parity_cases``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from minimappr.core.node_position_covariance import WeightedTdoaRow, node_position_error_covariance_m2

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "node_position_covariance_parity_cases.json"
SOUND_SPEED_MPS = 343.0
TETRA_OFFSETS_M = [
    [-0.016238, 0.025000, -0.010205],
    [0.027063, 0.000000, -0.010205],
    [-0.016238, -0.025000, -0.010205],
    [0.005413, 0.000000, 0.030615],
]


def _reference_rows(sensor_count: int, reference: int) -> list[dict]:
    """The sidecar SRP's rows: unweighted ``arrival(ch) - arrival(reference)``."""
    return [{"sensor_a": index, "sensor_b": reference, "weight": 1.0} for index in range(sensor_count) if index != reference]


def _all_pair_rows(weights_by_pair) -> list[dict]:
    return [{"sensor_a": a, "sensor_b": b, "weight": weight} for (a, b), weight in weights_by_pair.items()]


def _cases() -> list[dict]:
    point_nodes = [[0.0, 0.0, 1.5], [60.0, 0.0, 2.0], [0.0, 60.0, 1.0], [60.0, 60.0, 3.0], [30.0, 30.0, 6.0]]
    tetra_plus_points = [list(np.add([0.0, 0.0, 1.5], offset)) for offset in TETRA_OFFSETS_M] + [
        [40.0, 5.0, 2.0],
        [-5.0, 35.0, 1.0],
        [30.0, 30.0, 4.0],
    ]
    tetra_plus_points_nodes = [0, 0, 0, 0, 1, 2, 3]
    rng = np.random.default_rng(20261125)
    tetra_pair_weights = {
        (a, b): (0.81 if tetra_plus_points_nodes[a] == tetra_plus_points_nodes[b] else float(rng.uniform(1e-4, 0.5)))
        for a, b in itertools.combinations(range(len(tetra_plus_points)), 2)
    }
    near_field_spread = [[0.0, 0.0, 1.0], [2.0, 0.2, 1.3], [0.3, 2.1, 0.8], [1.8, 1.9, 2.4]]
    return [
        {
            "name": "point_network_one_hand_measured_srp_reference_rows",
            "position_m": [35.0, 20.0, 4.0],
            "sensor_positions_m": point_nodes,
            "tdoa_rows": _reference_rows(len(point_nodes), 0),
            "additional_normal_matrix": [[0.0] * 3 for _ in range(3)],
            "node_index_by_sensor": [0, 1, 2, 3, 4],
            "node_position_std_m": [0.0, 0.3, 0.1, 0.1, 0.0],
        },
        {
            "name": "tetra_plus_points_weighted_pairs_mixed_accuracy",
            "position_m": [22.0, 41.0, 7.0],
            "sensor_positions_m": tetra_plus_points,
            "tdoa_rows": _all_pair_rows(tetra_pair_weights),
            "additional_normal_matrix": [[0.0] * 3 for _ in range(3)],
            "node_index_by_sensor": tetra_plus_points_nodes,
            "node_position_std_m": [0.05, 0.3, 0.0, 1.5],
        },
        {
            "name": "tetra_plus_points_with_bearing_constraint_rows",
            "position_m": [-80.0, 150.0, 12.0],
            "sensor_positions_m": tetra_plus_points,
            "tdoa_rows": _all_pair_rows(tetra_pair_weights),
            "additional_normal_matrix": [[4.0e-9, 1.0e-9, 0.0], [1.0e-9, 2.0e-9, 0.0], [0.0, 0.0, 1.0e-10]],
            "node_index_by_sensor": tetra_plus_points_nodes,
            "node_position_std_m": [0.3, 0.3, 0.3, 0.3],
        },
        {
            "name": "near_field_spread_all_hand_measured",
            "position_m": [1.1, 0.9, 1.6],
            "sensor_positions_m": near_field_spread,
            "tdoa_rows": _reference_rows(len(near_field_spread), 2),
            "additional_normal_matrix": [[0.0] * 3 for _ in range(3)],
            "node_index_by_sensor": [0, 1, 2, 3],
            "node_position_std_m": [0.3, 0.3, 0.3, 0.3],
        },
        {
            "name": "no_stated_accuracy_keeps_todays_covariance",
            "position_m": [35.0, 20.0, 4.0],
            "sensor_positions_m": point_nodes,
            "tdoa_rows": _reference_rows(len(point_nodes), 0),
            "additional_normal_matrix": [[0.0] * 3 for _ in range(3)],
            "node_index_by_sensor": [0, 1, 2, 3, 4],
            "node_position_std_m": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]


def python_covariance_for_case(case: dict) -> np.ndarray | None:
    return node_position_error_covariance_m2(
        position_m=case["position_m"],
        sensor_positions_m=case["sensor_positions_m"],
        tdoa_rows=[WeightedTdoaRow(**row) for row in case["tdoa_rows"]],
        additional_normal_matrix=np.asarray(case["additional_normal_matrix"], dtype=np.float64),
        sound_speed_mps=SOUND_SPEED_MPS,
        node_index_by_sensor=case["node_index_by_sensor"],
        node_position_std_m=case["node_position_std_m"],
    )


def build_fixture() -> dict:
    cases = []
    for case in _cases():
        covariance = python_covariance_for_case(case)
        cases.append({**case, "expected_covariance_m2": None if covariance is None else covariance.tolist()})
    return {"sound_speed_mps": SOUND_SPEED_MPS, "cases": cases}


if __name__ == "__main__":
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=1) + "\n")
    print(f"wrote {FIXTURE_PATH}")
