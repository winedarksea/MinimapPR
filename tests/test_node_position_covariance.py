"""Stated node position accuracy widens the published covariance (owner decision).

Hand-measured nodes are trusted: their entered positions are used, and their
stated accuracy widens the error. The term is the nodes' rigid position errors
propagated through the solve's own sensitivity (``core/node_position_covariance.py``,
mirrored by Rust ``tdoa_node_position_covariance.rs``).
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from minimappr.core.node_position_covariance import (
    NodePositionUncertainty,
    WeightedTdoaRow,
    node_position_error_covariance_m2,
)
from minimappr.core.rust_tdoa_solve import solve_localization_from_rust_tdoas
from tests.node_position_covariance_parity_cases import FIXTURE_PATH, build_fixture
from tests.range_observability_oracle import OracleNode, position_covariance_m2

SOUND_SPEED_MPS = 343.0
SAMPLE_RATE_HZ = 48_000
INTERPOLATION_FACTOR = 4
# chi-square(2 dof) at 95 %: the E8 horizontal coverage threshold.
HORIZONTAL_COVERAGE_CHI_SQUARE_95 = 5.991_464_547_107_979
HAND_MEASURED_STD_M = 0.3
SURVEYED_STD_M = 0.1
# E8-like point network: two hand-measured nodes among surveyed ones, ~75 m spread.
POINT_NODE_POSITIONS_M = {
    "n0": (0.0, 0.0, 1.5),
    "n1": (70.0, 5.0, 2.5),
    "n2": (10.0, 65.0, 1.0),
    "n3": (75.0, 70.0, 4.0),
    "n4": (40.0, 30.0, 8.0),
    "n5": (-5.0, 40.0, 3.0),
}
STATED_NODE_STD_M = {
    "n0": HAND_MEASURED_STD_M,
    "n1": SURVEYED_STD_M,
    "n2": SURVEYED_STD_M,
    "n3": HAND_MEASURED_STD_M,
    "n4": SURVEYED_STD_M,
    "n5": SURVEYED_STD_M,
}


def test_python_matches_the_cross_language_parity_fixture() -> None:
    """The Rust test ``node_position_term_matches_the_python_parity_fixture``
    checks the same file, so the two implementations agree case by case."""
    stored = json.loads(FIXTURE_PATH.read_text())
    regenerated = build_fixture()
    assert [case["name"] for case in stored["cases"]] == [case["name"] for case in regenerated["cases"]]
    for stored_case, fresh_case in zip(stored["cases"], regenerated["cases"]):
        if stored_case["expected_covariance_m2"] is None:
            assert fresh_case["expected_covariance_m2"] is None, stored_case["name"]
            continue
        np.testing.assert_allclose(
            fresh_case["expected_covariance_m2"], stored_case["expected_covariance_m2"], rtol=1e-9, atol=0.0
        )


def _four_point_nodes():
    sensors = [[0.0, 0.0, 1.5], [60.0, 0.0, 2.0], [0.0, 60.0, 1.0], [55.0, 62.0, 6.0]]
    return sensors, [0, 1, 2, 3], [0.0, HAND_MEASURED_STD_M, SURVEYED_STD_M, 0.0]


def test_sidecar_and_re_solve_formulations_agree_on_an_exactly_determined_layout() -> None:
    """The sidecar SRP propagates through unweighted reference-channel rows, the
    Python re-solve through weighted all-pair rows. With four nodes both are
    exactly determined, so any weighting gives the same sensitivity and the two
    published terms must match; with more nodes each follows its own estimator."""
    sensors, node_index, node_std_m = _four_point_nodes()
    source_m = [25.0, 31.0, 5.0]
    reference_rows = [WeightedTdoaRow(sensor_a=index, sensor_b=0, weight=1.0) for index in range(1, 4)]
    rng = np.random.default_rng(4)
    weighted_pair_rows = [
        WeightedTdoaRow(sensor_a=a, sensor_b=b, weight=float(rng.uniform(1e-4, 0.9)))
        for a, b in itertools.combinations(range(4), 2)
    ]

    def term(rows):
        return node_position_error_covariance_m2(
            position_m=source_m,
            sensor_positions_m=sensors,
            tdoa_rows=rows,
            additional_normal_matrix=None,
            sound_speed_mps=SOUND_SPEED_MPS,
            node_index_by_sensor=node_index,
            node_position_std_m=node_std_m,
        )

    np.testing.assert_allclose(term(weighted_pair_rows), term(reference_rows), rtol=1e-6, atol=1e-12)


def test_term_equals_the_observability_oracles_node_position_covariance() -> None:
    """Noise-free timing, so the oracle's inverse Fisher information is pure
    node-position error; on an exactly determined layout it must equal the
    propagated term (same ``sigma^2 u_i . u_j / c^2`` arrival model)."""
    sensors, node_index, node_std_m = _four_point_nodes()
    source_m = np.asarray([25.0, 31.0, 5.0])
    reference_rows = [WeightedTdoaRow(sensor_a=index, sensor_b=0, weight=1.0) for index in range(1, 4)]
    propagated = node_position_error_covariance_m2(
        position_m=source_m,
        sensor_positions_m=sensors,
        tdoa_rows=reference_rows,
        additional_normal_matrix=None,
        sound_speed_mps=SOUND_SPEED_MPS,
        node_index_by_sensor=node_index,
        node_position_std_m=node_std_m,
    )
    from tests.range_matrix_cells import ORACLE_CONFIG

    # A vanishing timing floor keeps the oracle's arrival covariance invertible.
    oracle_config = type(ORACLE_CONFIG)(**{**ORACLE_CONFIG.__dict__, "lag_quantum_seconds": 1e-12})
    oracle_nodes = [
        OracleNode(
            node_id=f"n{index}",
            sensor_positions_m=(tuple(sensor),),
            sensor_in_band_snr_db=(200.0,),
            clock_offset_std_seconds=0.0,
            position_std_m=max(std_m, 1e-9),
        )
        for index, (sensor, std_m) in enumerate(zip(sensors, node_std_m))
    ]
    oracle = position_covariance_m2(oracle_nodes, source_m, oracle_config)
    np.testing.assert_allclose(propagated, oracle, rtol=1e-3, atol=1e-8)


def _horizontal_coverage(*, with_stated_accuracy: bool, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    sensor_ids = [f"{node_id}:ch0" for node_id in POINT_NODE_POSITIONS_M]
    uncertainty = NodePositionUncertainty.from_sensor_position_std(
        sensor_node_ids={sensor_id: sensor_id.split(":")[0] for sensor_id in sensor_ids},
        sensor_position_std_m={sensor_id: STATED_NODE_STD_M[sensor_id.split(":")[0]] for sensor_id in sensor_ids},
    )
    covered = solved = 0
    for _ in range(trials):
        source_m = np.asarray([rng.uniform(5.0, 65.0), rng.uniform(5.0, 60.0), rng.uniform(1.0, 10.0)])
        # The entered positions carry real errors of the stated size; the sound
        # travels from the true ones.
        entered_positions = {
            f"{node_id}:ch0": np.asarray(position) + STATED_NODE_STD_M[node_id] * rng.normal(size=3)
            for node_id, position in POINT_NODE_POSITIONS_M.items()
        }
        pair_tdoas = []
        for (index_a, sensor_a), (index_b, sensor_b) in itertools.combinations(enumerate(sensor_ids), 2):
            true_a = np.asarray(POINT_NODE_POSITIONS_M[sensor_a.split(":")[0]])
            true_b = np.asarray(POINT_NODE_POSITIONS_M[sensor_b.split(":")[0]])
            lag_s = (np.linalg.norm(source_m - true_a) - np.linalg.norm(source_m - true_b)) / SOUND_SPEED_MPS
            pair_tdoas.append(
                {
                    "ch_a": index_a,
                    "ch_b": index_b,
                    "lag_seconds": float(lag_s + rng.normal(0.0, 2.0e-5)),
                    "confidence": 0.9,
                    "timing_uncertainty_seconds": 1.4e-6,
                }
            )
        result = solve_localization_from_rust_tdoas(
            sensor_positions=entered_positions,
            ordered_sensor_ids=sensor_ids,
            pair_tdoas=pair_tdoas,
            steering_direction=None,
            sound_speed_mps=SOUND_SPEED_MPS,
            sample_rate_hz=SAMPLE_RATE_HZ,
            interpolation_factor=INTERPOLATION_FACTOR,
            far_field_default_range_m=50.0,
            node_position_uncertainty=uncertainty if with_stated_accuracy else None,
        )
        if result is None:
            continue
        solved += 1
        horizontal_error_m = (np.asarray(result.position_m) - source_m)[:2]
        horizontal_covariance_m2 = np.asarray(result.position_covariance_m2)[:2, :2]
        mahalanobis_squared = float(horizontal_error_m @ np.linalg.solve(horizontal_covariance_m2, horizontal_error_m))
        covered += mahalanobis_squared <= HORIZONTAL_COVERAGE_CHI_SQUARE_95
    assert solved >= 0.9 * trials
    return covered / solved


def test_hand_measured_nodes_restore_e8_coverage_on_the_re_solve() -> None:
    """E8 found 0.37 horizontal 95 % coverage at sites with a hand-measured node
    on the default python_cartesian handoff. With the stated accuracies in the
    covariance the same synthetic site covers close to the stated 95 %."""
    timing_only = _horizontal_coverage(with_stated_accuracy=False, trials=120, seed=7)
    widened = _horizontal_coverage(with_stated_accuracy=True, trials=120, seed=7)
    assert timing_only < 0.6, timing_only
    assert widened >= 0.88, widened


def test_handoff_re_solve_reads_position_std_from_the_solve_provenance() -> None:
    """The python_cartesian handoff takes each sensor's ``position_std_m`` from
    the manifest: an operator-placed planar node at 0.3 m widens the published
    covariance without moving an unbiased fix; ``null`` keeps today's."""
    from minimappr.core.localized_handoff_solution import SnapshotGeometrySolutionResolver
    from tests.localization_solve_manifest_fixtures import (
        PLANAR_NODE_ID,
        POINT_NODE_ID,
        mixed_cluster_solve_localization_payload,
        planar_node_payload,
    )
    from tests.test_localized_handoff_geometry_and_solve import (
        SITE_FRAME_ID,
        SOURCE_POSITION_M,
        _request,
        _resolve,
        _solution_host,
    )

    def resolve(planar_position_std_m: float | None):
        localization = mixed_cluster_solve_localization_payload(
            arrival_order=(PLANAR_NODE_ID, POINT_NODE_ID),
            source_position_m=SOURCE_POSITION_M,
            frame_id=SITE_FRAME_ID,
            reporting_node_id=PLANAR_NODE_ID,
            sidecar_position_m=(0.0, 0.0, 0.0),
        )
        for sensor in localization["solve_provenance"]["sensors"]:
            if sensor["node_id"] == PLANAR_NODE_ID:
                sensor["position_source"] = "operator_override"
                sensor["position_std_m"] = planar_position_std_m
        request = _request(localization, reporter_payload=planar_node_payload())
        return SnapshotGeometrySolutionResolver(_solution_host("python_cartesian")).resolve_handoff_solution(
            payload=request, geometry=_resolve(request), reporting_modality="localized"
        )

    unstated = resolve(None)
    hand_measured = resolve(HAND_MEASURED_STD_M)
    assert hand_measured.method == "python_cartesian_rust_tdoa"
    assert math.dist(hand_measured.position_m, unstated.position_m) < 1e-6
    assert np.trace(np.asarray(hand_measured.position_covariance_m2)) > np.trace(
        np.asarray(unstated.position_covariance_m2)
    ) + HAND_MEASURED_STD_M**2


@pytest.mark.asyncio
async def test_registry_states_the_same_accuracy_the_geometry_push_sends() -> None:
    """Python-ingest path: an operator-pinned position states its entered
    accuracy (or the push's default), a stationary estimate its spread, and a
    node with neither states nothing (so its solves keep today's covariance)."""
    from minimappr.core.node_registry import DEFAULT_OPERATOR_PLACEMENT_POSITION_STD_M, NodeRegistry
    from minimappr.models import NodeOverrides, NodeSpec, NodeType

    registry = NodeRegistry()
    for node_id in ("hand", "hand-default", "estimated", "plain"):
        await registry.upsert(
            NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=(1.0, 2.0, 3.0), capabilities=["audio"]),
            last_seen_ns=1,
        )
    await registry.set_overrides("hand", NodeOverrides(position_m=(1.0, 2.0, 3.0), position_std_m=HAND_MEASURED_STD_M))
    await registry.set_overrides("hand-default", NodeOverrides(position_m=(1.0, 2.0, 3.0)))
    await registry.set_node_position_std_m("estimated", 0.7)
    assert await registry.sensor_position_std_m() == {
        "hand:ch0": HAND_MEASURED_STD_M,
        "hand-default:ch0": DEFAULT_OPERATOR_PLACEMENT_POSITION_STD_M,
        "estimated:ch0": 0.7,
    }
