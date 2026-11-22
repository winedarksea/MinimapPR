"""Spreading-law level check for onset fixes in a node's near field (Field Mode S-fixes).

Why: the node near-field shells let SRP place a source 1-8 m from a node, which
a bird next to a node needs (S gate: a wren 2.3 m from a tetra was placed 52 m
out). They also let a weak secondary cluster led by one node's onsets land in
that node's near field (overlap scene: a fix 2.4 m from tetra-a with no bird
there). The W2 level check only rejects a nearest node *quieter* than the
farthest; a fix d_near from one node and d_far from another must also be about
20 log10(d_far / d_near) dB louder at the near node (free-field spreading). A
near-field fix whose level gradient falls short of that by more than
``tolerance_db`` is inconsistent. Levels are per-node noise-floor-normalised
band SNR (``node_band_snr_db``), as in the W2 check; no absolute level is used.
"""

from __future__ import annotations

import numpy as np

from minimappr.core.onset_cluster_echo_checks import node_band_snr_db
from minimappr.core.onset_detection import SubbandHistory
from minimappr.core.onset_srp_types import OnsetFrame


def near_node_level_gradient_holds(
    *,
    position_m: np.ndarray,
    led_frames: list[OnsetFrame],
    node_histories: dict[str, SubbandHistory],
    node_positions_m: dict[str, np.ndarray],
    near_field_radius_m: float,
    tolerance_db: float,
    sound_speed_mps: float,
    sample_rate_hz: int,
    frame_samples: int,
) -> bool:
    """True unless the fix lies within ``near_field_radius_m`` of a node and the measured
    nearest-vs-farthest level difference trails the spreading prediction by > tolerance."""
    if len(node_positions_m) < 2 or not led_frames:
        return True
    distances = {node: float(np.linalg.norm(position_m - pos)) for node, pos in node_positions_m.items()}
    nearest = min(distances, key=distances.get)
    farthest = max(distances, key=distances.get)
    if distances[nearest] > near_field_radius_m or nearest == farthest or distances[nearest] <= 0.0:
        return True
    expected_db = 20.0 * np.log10(distances[farthest] / distances[nearest])
    differences = []
    for frame in led_frames:
        template_distance = distances.get(frame.template_node_id, distances[nearest])
        readings = []
        for node in (nearest, farthest):
            arrival = frame.onset_sample + int(round((distances[node] - template_distance) / sound_speed_mps * sample_rate_hz))
            readings.append(
                node_band_snr_db(
                    node_histories[node], start_sample=arrival, frame_samples=frame_samples,
                    band_min_hz=frame.band_min_hz, band_max_hz=frame.band_max_hz,
                )
            )
        if readings[0] is not None and readings[1] is not None:
            differences.append(readings[0] - readings[1])
    if not differences:
        return True
    return float(np.median(differences)) >= expected_db - tolerance_db


def near_node_fix_led_only_by_that_node(
    *,
    position_m: np.ndarray,
    led_frames: list[OnsetFrame],
    node_positions_m: dict[str, np.ndarray],
    near_field_radius_m: float,
) -> bool:
    """True when the fix lies in one node's near field and every onset that led the
    cluster was templated on that node. Such frames are dominated by the node's own
    range-blind compact pairs (a second bird's mixed frames, measured on the overlap and
    collinear scenes), so the shells, not the data, chose the range; a real bird next to a
    node is also heard leading onsets at the other nodes."""
    if len(node_positions_m) < 2 or not led_frames:
        return False
    distances = {node: float(np.linalg.norm(position_m - pos)) for node, pos in node_positions_m.items()}
    nearest = min(distances, key=distances.get)
    if distances[nearest] > near_field_radius_m:
        return False
    return all(frame.template_node_id == nearest for frame in led_frames)
