"""Node-position term of a TDOA fix's covariance (owner decision: a stated
node position accuracy widens the error; it never excludes the node).

A node whose stated position is off by a rigid translation ``d`` (all of its
channels move together) shifts each arrival at its sensors by ``-u_s . d / c``,
``u_s`` the unit vector from sensor ``s`` toward the source. Like a clock
offset, the error is shared by every TDOA of that node (correlated per node,
not independent per pair) and grows with geometric dilution, so it is
propagated through the solve's own least-squares sensitivity:
``G R G^T`` with ``G = (J^T W J)^-1 J^T W`` at the solved position and ``R``
built from each node's ``d ~ N(0, sigma^2 I)`` (``position_std_m`` read as a
per-axis, isotropic 1-sigma). This is the direction-dependent projection, the
arrival covariance the E5 observability oracle uses; for a single-sensor node
it equals a clock offset of ``sigma / c``.

Line-for-line mirror of Rust ``minimappr-core/src/tdoa_node_position_covariance.rs``;
``tests/fixtures/node_position_covariance_parity_cases.json`` pins the two.
A node with no stated accuracy contributes nothing (the behaviour before
positions carried one).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Same ridge as the Rust clock and node-position terms: keeps a direction no
# measurement sees (a planar layout's normal) from blowing up the inverse.
RELATIVE_NORMAL_MATRIX_RIDGE = 1.0e-12
DISTANCE_EPSILON_M = 1.0e-12


@dataclass(frozen=True, slots=True)
class WeightedTdoaRow:
    """``arrival(sensor_a) - arrival(sensor_b)``, residual scaled by ``sqrt(weight)``."""

    sensor_a: int
    sensor_b: int
    weight: float


@dataclass(frozen=True, slots=True)
class NodePositionUncertainty:
    """Sensor -> node, and each node's per-axis 1-sigma position error (m).

    A sensor or node missing from the maps states no error.
    """

    sensor_node_ids: Mapping[str, str]
    node_position_std_m: Mapping[str, float]

    @classmethod
    def from_sensor_position_std(
        cls,
        sensor_node_ids: Mapping[str, str],
        sensor_position_std_m: Mapping[str, float | None],
    ) -> "NodePositionUncertainty | None":
        """Per-sensor accuracies (each inherited from its node) folded to nodes.

        ``None`` when no sensor states one, so callers keep today's covariance.
        The largest accuracy among a node's sensors is kept so a mixed snapshot
        never understates it (the sidecar's rule).
        """
        node_position_std_m: dict[str, float] = {}
        for sensor_id, position_std_m in sensor_position_std_m.items():
            node_id = sensor_node_ids.get(sensor_id)
            if node_id is None or position_std_m is None or not position_std_m > 0.0:
                continue
            node_position_std_m[node_id] = max(node_position_std_m.get(node_id, 0.0), float(position_std_m))
        if not node_position_std_m:
            return None
        return cls(sensor_node_ids=dict(sensor_node_ids), node_position_std_m=node_position_std_m)


def node_position_error_covariance_m2(
    *,
    position_m: Sequence[float],
    sensor_positions_m: Sequence[Sequence[float]],
    tdoa_rows: Sequence[WeightedTdoaRow],
    additional_normal_matrix: np.ndarray | None,
    sound_speed_mps: float,
    node_index_by_sensor: Sequence[int],
    node_position_std_m: Sequence[float],
) -> np.ndarray | None:
    """Covariance (m^2) from the nodes' stated position errors, or ``None``.

    ``None`` when no node states an error, no row involves such a node, or the
    geometry cannot be inverted. ``additional_normal_matrix`` is ``J^T J`` of
    the non-TDOA rows the solve's covariance includes (bearing / amplitude
    constraints): they shape the sensitivity but carry no node-position error.
    """

    def node_states_an_error(node: int) -> bool:
        return 0 <= node < len(node_position_std_m) and node_position_std_m[node] > 0.0

    if not any(
        node_states_an_error(node_index_by_sensor[row.sensor_a]) or node_states_an_error(node_index_by_sensor[row.sensor_b])
        for row in tdoa_rows
    ) or not sound_speed_mps > 0.0:
        return None

    position = np.asarray(position_m, dtype=np.float64)
    sensors = np.asarray(sensor_positions_m, dtype=np.float64)
    offsets_m = position[None, :] - sensors
    unit_toward_source = offsets_m / (np.linalg.norm(offsets_m, axis=1, keepdims=True) + DISTANCE_EPSILON_M)

    sensor_a = np.asarray([row.sensor_a for row in tdoa_rows], dtype=np.int64)
    sensor_b = np.asarray([row.sensor_b for row in tdoa_rows], dtype=np.int64)
    row_scale = np.sqrt(np.maximum([row.weight for row in tdoa_rows], 0.0)) / sound_speed_mps
    jacobian = (unit_toward_source[sensor_a] - unit_toward_source[sensor_b]) * row_scale[:, None]
    normal_matrix = jacobian.T @ jacobian
    if additional_normal_matrix is not None:
        normal_matrix = normal_matrix + np.asarray(additional_normal_matrix, dtype=np.float64)
    trace = float(np.trace(normal_matrix))
    if not (np.isfinite(trace) and trace > 0.0):
        return None
    normal_matrix = normal_matrix + np.eye(3) * (trace * RELATIVE_NORMAL_MATRIX_RIDGE)
    try:
        normal_inverse = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError:
        return None

    node_of_a = np.asarray([node_index_by_sensor[index] for index in sensor_a], dtype=np.int64)
    node_of_b = np.asarray([node_index_by_sensor[index] for index in sensor_b], dtype=np.int64)
    covariance_m2 = np.zeros((3, 3), dtype=np.float64)
    for node, position_std_m in enumerate(node_position_std_m):
        if not position_std_m > 0.0:
            continue
        a_on_node = (node_of_a == node).astype(np.float64)
        b_on_node = (node_of_b == node).astype(np.float64)
        if not np.any(a_on_node + b_on_node):
            continue
        # Row r shifts by sqrt(w_r) (-[node(a)=k] u_a + [node(b)=k] u_b) . d / c;
        # column j of `shifts` is the shift per metre of d along axis j.
        shifts = (
            b_on_node[:, None] * unit_toward_source[sensor_b] - a_on_node[:, None] * unit_toward_source[sensor_a]
        ) * row_scale[:, None]
        sensitivity = normal_inverse @ (jacobian.T @ shifts)
        covariance_m2 += float(position_std_m) ** 2 * (sensitivity @ sensitivity.T)
    return covariance_m2 if np.all(np.isfinite(covariance_m2)) else None


def pair_measurement_node_position_covariance_m2(
    *,
    position_m: np.ndarray,
    sensor_positions: Mapping[str, np.ndarray],
    measurements: Sequence,
    additional_normal_matrix: np.ndarray | None,
    sound_speed_mps: float,
    uncertainty: NodePositionUncertainty,
) -> np.ndarray | None:
    """The node-position term for a solve over ``PairTdoaMeasurement`` rows
    (``core/cartesian_tdoa.py``), keyed by sensor and node id."""
    sensor_ids = sorted(sensor_positions)
    sensor_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
    node_ids = sorted(set(uncertainty.node_position_std_m))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    # A sensor of a node with no stated error gets its own index past the
    # stated nodes: it moves no row.
    unstated_node_index = len(node_ids)
    node_index_by_sensor = [
        node_index.get(uncertainty.sensor_node_ids.get(sensor_id, ""), unstated_node_index) for sensor_id in sensor_ids
    ]
    return node_position_error_covariance_m2(
        position_m=position_m,
        sensor_positions_m=[sensor_positions[sensor_id] for sensor_id in sensor_ids],
        tdoa_rows=[
            WeightedTdoaRow(
                sensor_a=sensor_index[measurement.sensor_a],
                sensor_b=sensor_index[measurement.sensor_b],
                weight=float(measurement.weight),
            )
            for measurement in measurements
        ],
        additional_normal_matrix=additional_normal_matrix,
        sound_speed_mps=sound_speed_mps,
        node_index_by_sensor=node_index_by_sensor,
        node_position_std_m=[float(uncertainty.node_position_std_m[node_id]) for node_id in node_ids],
    )
