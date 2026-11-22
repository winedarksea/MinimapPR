"""Telling a second bird from an echo or sidelobe (Field Mode S2 echo checks).

Three invariants, cheapest first (plan: "Telling a second bird from an echo"):

1. Independent onsets. An echo is time-locked 3-30 ms behind its source's onset
   in the same band, and mostly appears as a secondary peak in the source's own
   frames. A secondary cluster whose frames are not led by onsets of its own is
   an echo.
2. Level consistency (multi-node only). Nodes nearer a candidate must hold more
   noise-floor-normalised in-band energy than farther ones, within a loose
   margin; a failure only holds the cluster as tentative (never deletes it),
   so a faint distant bird is never rejected on level.
3. Persistence (single compact array only). A secondary cluster needs several
   independent onsets; the strongest cluster in a window is always accepted,
   so a single faint note still surfaces.
"""

from __future__ import annotations

import numpy as np

from minimappr.core.onset_detection import SubbandHistory
from minimappr.core.onset_srp_types import (
    CLUSTER_STATUS_ECHO,
    CLUSTER_STATUS_TENTATIVE,
    OnsetCluster,
    OnsetFrame,
)

# Bands must overlap by at least this fraction of the narrower one to be "same band".
SAME_BAND_MIN_OVERLAP_FRACTION = 0.5


def band_overlap_fraction(frame_a: OnsetFrame, frame_b: OnsetFrame) -> float:
    low = max(frame_a.band_min_hz, frame_b.band_min_hz)
    high = min(frame_a.band_max_hz, frame_b.band_max_hz)
    narrower = min(frame_a.band_max_hz - frame_a.band_min_hz, frame_b.band_max_hz - frame_b.band_min_hz)
    if narrower <= 0.0:
        return 0.0
    return max(0.0, high - low) / narrower


def _frame_trails_stronger_onset(
    frame: OnsetFrame,
    stronger_led_frames: list[OnsetFrame],
    *,
    echo_min_lag_samples: int,
    echo_max_lag_samples: int,
) -> bool:
    for leader in stronger_led_frames:
        if leader.template_node_id != frame.template_node_id:
            continue
        lag = frame.onset_sample - leader.onset_sample
        if (
            echo_min_lag_samples <= lag <= echo_max_lag_samples
            and band_overlap_fraction(frame, leader) >= SAME_BAND_MIN_OVERLAP_FRACTION
        ):
            return True
    return False


def apply_onset_echo_and_persistence_checks(
    clusters: list[OnsetCluster],
    frames: dict[int, OnsetFrame],
    *,
    sample_rate_hz: int,
    echo_min_lag_seconds: float,
    echo_max_lag_seconds: float,
    echo_min_independent_fraction: float,
    persistence_min_independent_onsets: int,
    apply_persistence: bool = True,
) -> None:
    """Mutates cluster status in place; `clusters` must be strongest first.

    Persistence is for single compact arrays (plan: "independent onsets and
    persistence carry the decision there"); multi-node windows use the level
    check instead, so a second bird's single note can still be accepted.
    """
    echo_min_lag_samples = int(round(echo_min_lag_seconds * sample_rate_hz))
    echo_max_lag_samples = int(round(echo_max_lag_seconds * sample_rate_hz))
    for rank, cluster in enumerate(clusters):
        stronger_led_frames = [
            frames[index] for other in clusters[:rank] for index in other.led_frame_indices
        ]
        independent = [
            index
            for index in cluster.led_frame_indices
            if not _frame_trails_stronger_onset(
                frames[index],
                stronger_led_frames,
                echo_min_lag_samples=echo_min_lag_samples,
                echo_max_lag_samples=echo_max_lag_samples,
            )
        ]
        cluster.independent_onset_count = len(independent)
        if rank == 0:
            continue
        independent_fraction = len(independent) / max(1, len(cluster.frame_indices))
        if independent_fraction < echo_min_independent_fraction:
            cluster.status = CLUSTER_STATUS_ECHO
            cluster.status_reasons.append("no_independent_onsets")
        elif apply_persistence and len(independent) < persistence_min_independent_onsets:
            cluster.status = CLUSTER_STATUS_TENTATIVE
            cluster.status_reasons.append("not_persistent")


def node_band_snr_db(
    history: SubbandHistory,
    *,
    start_sample: int,
    frame_samples: int,
    band_min_hz: float,
    band_max_hz: float,
) -> float | None:
    """Mean in-band dB above the node's own noise floor over one frame."""
    first_hop = max(0, start_sample // history.hop_samples)
    last_hop = min(history.energy_db.shape[0], (start_sample + frame_samples) // history.hop_samples + 1)
    if last_hop <= first_hop:
        return None
    centers = history.band_centers_hz
    bands = (centers >= band_min_hz) & (centers <= band_max_hz)
    if not np.any(bands):
        return None
    return float(np.mean(history.snr_db()[first_hop:last_hop][:, bands]))


def level_consistency_holds(
    *,
    position_m: np.ndarray,
    led_frames: list[OnsetFrame],
    node_histories: dict[str, SubbandHistory],
    node_positions_m: dict[str, np.ndarray],
    sound_speed_mps: float,
    sample_rate_hz: int,
    frame_samples: int,
    margin_db: float,
) -> bool:
    """Nearest node's band SNR must not trail the farthest's by more than margin_db.

    Levels are per-node noise-floor-normalised (dB above that node's own floor),
    which cancels mic-sensitivity differences between nodes.
    """
    if len(node_histories) < 2 or not led_frames:
        return True
    distances = {node: float(np.linalg.norm(position_m - pos)) for node, pos in node_positions_m.items()}
    nearest = min(distances, key=distances.get)
    farthest = max(distances, key=distances.get)
    if nearest == farthest:
        return True
    differences = []
    for frame in led_frames:
        template_distance = distances.get(frame.template_node_id, distances[nearest])
        readings = {}
        for node in (nearest, farthest):
            arrival = frame.onset_sample + int(
                round((distances[node] - template_distance) / sound_speed_mps * sample_rate_hz)
            )
            readings[node] = node_band_snr_db(
                node_histories[node],
                start_sample=arrival,
                frame_samples=frame_samples,
                band_min_hz=frame.band_min_hz,
                band_max_hz=frame.band_max_hz,
            )
        if readings[nearest] is not None and readings[farthest] is not None:
            differences.append(readings[nearest] - readings[farthest])
    if not differences:
        return True
    return float(np.median(differences)) >= -margin_db
