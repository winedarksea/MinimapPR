"""Per-cluster level evidence for species attribution (Field Mode W3-attribution, S5).

For every onset cluster (accepted, tentative or echo) we record, per node:

* the noise-floor-relative band SNR at the cluster's led onsets, read at each
  onset's predicted arrival on that node and in that onset's own band;
* the predicted arrival (span sample) of each led onset on that node, with its
  band, so a later masking step knows which time-frequency regions belong to
  the cluster in any node's classification window;
* the consistency of the SNR profile with 1/r spreading from the cluster's
  position (RMS residual after fitting the unknown source level).

Levels are dB above each node's own floor, so mic-sensitivity differences cancel.
Nodes whose reading is below ``audible_snr_db`` are censored (the true level
could be anywhere under the floor): they are excluded from the source-level
fit and penalised only when the fit predicts them clearly audible.

Pure and synchronous (runs inside the clustering worker thread).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from minimappr.core.onset_cluster_echo_checks import node_band_snr_db
from minimappr.core.onset_detection import SubbandHistory
from minimappr.core.onset_srp_types import OnsetFrame

DECIBELS_PER_AMPLITUDE_DECADE = 20.0
# Distances below this are clamped before taking log10 (a node at the source).
MINIMUM_SPREADING_DISTANCE_M = 0.1


@dataclass(frozen=True, slots=True)
class ClusterOnsetArrival:
    """One led onset of a cluster as it reaches one node."""

    span_sample: int
    band_min_hz: float
    band_max_hz: float


@dataclass(frozen=True, slots=True)
class ClusterAttributionEvidence:
    node_band_snr_db: dict[str, float]
    node_distances_m: dict[str, float]
    node_onset_arrivals: dict[str, tuple[ClusterOnsetArrival, ...]]
    # Source level (dB above floor at 1 m) fitted from audible nodes; None with < 1 audible node.
    fitted_source_level_db: float | None
    # RMS of measured-minus-predicted over audible nodes (+ censored violations); None with < 2 nodes.
    spreading_residual_db: float | None

    def predicted_snr_db(self, node_id: str) -> float | None:
        """Spreading-model SNR of this cluster at a node (None without a fit)."""
        if self.fitted_source_level_db is None or node_id not in self.node_distances_m:
            return None
        distance = max(self.node_distances_m[node_id], MINIMUM_SPREADING_DISTANCE_M)
        return self.fitted_source_level_db - DECIBELS_PER_AMPLITUDE_DECADE * math.log10(distance)


def _spreading_fit(
    snr_by_node: dict[str, float], distances_m: dict[str, float], audible_snr_db: float
) -> tuple[float | None, float | None]:
    audible = [node for node, snr in snr_by_node.items() if snr >= audible_snr_db]
    if not audible:
        return None, None
    spreading = {
        node: DECIBELS_PER_AMPLITUDE_DECADE * math.log10(max(distances_m[node], MINIMUM_SPREADING_DISTANCE_M))
        for node in snr_by_node
    }
    source_level = float(np.mean([snr_by_node[node] + spreading[node] for node in audible]))
    if len(snr_by_node) < 2:
        return source_level, None
    residuals = []
    for node, measured in snr_by_node.items():
        predicted = source_level - spreading[node]
        if node in audible:
            residuals.append(measured - predicted)
        elif predicted > audible_snr_db:
            # Censored: only the part of the prediction above audibility is a violation.
            residuals.append(predicted - max(measured, audible_snr_db))
    if not residuals:
        return source_level, None
    return source_level, float(np.sqrt(np.mean(np.square(residuals))))


def build_cluster_attribution_evidence(
    *,
    position_m: np.ndarray,
    led_frames: list[OnsetFrame],
    node_histories: dict[str, SubbandHistory],
    node_positions_m: dict[str, np.ndarray],
    sound_speed_mps: float,
    sample_rate_hz: int,
    frame_samples: int,
    audible_snr_db: float,
) -> ClusterAttributionEvidence:
    position = np.asarray(position_m, dtype=np.float64)
    distances = {node: float(np.linalg.norm(position - pos)) for node, pos in node_positions_m.items()}
    arrivals: dict[str, list[ClusterOnsetArrival]] = {node: [] for node in node_positions_m}
    readings: dict[str, list[float]] = {node: [] for node in node_positions_m}
    for frame in led_frames:
        template_distance = distances.get(frame.template_node_id, min(distances.values()))
        for node in node_positions_m:
            arrival = frame.onset_sample + int(
                round((distances[node] - template_distance) / sound_speed_mps * sample_rate_hz)
            )
            arrivals[node].append(ClusterOnsetArrival(arrival, frame.band_min_hz, frame.band_max_hz))
            history = node_histories.get(node)
            if history is None:
                continue
            snr = node_band_snr_db(
                history,
                start_sample=arrival,
                frame_samples=frame_samples,
                band_min_hz=frame.band_min_hz,
                band_max_hz=frame.band_max_hz,
            )
            if snr is not None:
                readings[node].append(snr)
    snr_by_node = {node: float(np.median(values)) for node, values in readings.items() if values}
    source_level, residual = _spreading_fit(
        snr_by_node, {node: distances[node] for node in snr_by_node}, audible_snr_db
    )
    return ClusterAttributionEvidence(
        node_band_snr_db=snr_by_node,
        node_distances_m=distances,
        node_onset_arrivals={node: tuple(items) for node, items in arrivals.items()},
        fitted_source_level_db=source_level,
        spreading_residual_db=residual,
    )
