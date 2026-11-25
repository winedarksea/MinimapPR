"""Compact-array near-field collapse guard for Cartesian TDOA solves.

Why this exists: with a compact array (one 5 cm tetrahedral node) the TDOA
residual surface has a spurious near-field basin within a few apertures of the
array. Four mics give only three independent TDOAs, which a 3-D point always
fits exactly, so a point a few centimetres out "explains" measurement noise as
wavefront curvature. `solve_cartesian_tdoa` reported that point as
RANGE_REFINED with range observability ~1.0: every single-node fix collapsed
onto its node, and the beamformer then steered at a point 3 cm away (Field Mode
W2 root cause, see docs/field_mode_progress/onset_srp.md). Even a 400 m source
with ideal TDOAs collapsed to 6 cm.

The guard acts on every RANGE_REFINED solve of a compact array, wherever it
landed: the near-field point is replaced by the best plane-wave projection
unless it beats that projection by more than the timing noise floor
(`CURVATURE_EVIDENCE_NOISE_MULTIPLIER` interpolated sample periods, RMS). That
test is the range-observability question itself, so no distance cutoff is
needed: a genuine near-field source (e.g. 0.27 m from a 5 cm array) leaves a
plane-wave misfit far above the noise and keeps its fix, while a point that
only beats the plane wave by noise is a collapse at any distance. Multi-node
layouts keep the legacy answer.

History: the guard used to act only within 4 apertures of the centroid (~20 cm
for a 5 cm tetra). Collapses also settle just outside that (4-6 apertures);
one landed at 0.216 m from a 0.0525 m tetra (4.1 apertures) and was published
as a RANGE_REFINED fix with a ~0.2 m covariance for a source 100 m away (E5
range matrix, `100m-single_tetra-*`, after R4's onset trigger left most of the
80 ms solve window before the burst). Without a cutoff, a lone tetra 1-3 m from
a source reports a bearing, which is what its aperture can honestly observe
(the E5 oracle's verdict for that cell too).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPSILON = 1.0e-12
# A near-field point must beat the plane wave by this many timing-noise std
# (RMS) to be believed.
CURVATURE_EVIDENCE_NOISE_MULTIPLIER = 1.0
# Only arrays at most this wide are guarded. Wider (multi-node) layouts can
# observe range, and when their near and far fits tie it is because the TDOAs
# are inconsistent (e.g. cycle-skipped cross-node pairs on a pure tone); those
# keep the legacy answer and its low confidence. 0.5 m covers tetra nodes and
# stereo phones with margin.
COMPACT_ARRAY_MAX_APERTURE_M = 0.5


@dataclass(frozen=True, slots=True)
class FarFieldAlternative:
    position_m: np.ndarray
    residual_rms_seconds: float


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float64) / norm if norm > EPSILON else None


def far_field_alternative_if_curvature_unsupported(
    *,
    centroid_m: np.ndarray,
    near_field_position_m: np.ndarray,
    plane_wave_direction: np.ndarray | None,
    residual_function: Callable[[np.ndarray], np.ndarray],
    near_field_residual_rms_seconds: float,
    timing_noise_std_seconds: float,
    far_field_radius_m: float,
    array_aperture_m: float,
) -> FarFieldAlternative | None:
    """Return a plane-wave projection when a compact-array solve has collapsed.

    None means keep the near-field solution: the array is not compact, or its
    curvature evidence beats the noise.
    """
    if array_aperture_m > COMPACT_ARRAY_MAX_APERTURE_M:
        return None
    near_offset = np.asarray(near_field_position_m, dtype=np.float64) - centroid_m
    candidate_directions = [
        direction
        for direction in (_unit(near_offset), _unit(plane_wave_direction))
        if direction is not None
    ]
    best: FarFieldAlternative | None = None
    for direction in candidate_directions:
        # A fixed far-field radius is a plane wave for a compact array; an
        # unbounded radial search can run off to 1e14 m on an asymptotic cost.
        projected_position_m = centroid_m + direction * far_field_radius_m
        residual_rms_s = _rms(residual_function(projected_position_m))
        if not math.isfinite(residual_rms_s):
            continue
        if best is None or residual_rms_s < best.residual_rms_seconds:
            best = FarFieldAlternative(
                position_m=projected_position_m,
                residual_rms_seconds=residual_rms_s,
            )
    if best is None:
        return None
    curvature_evidence_s = best.residual_rms_seconds - near_field_residual_rms_seconds
    if curvature_evidence_s > CURVATURE_EVIDENCE_NOISE_MULTIPLIER * timing_noise_std_seconds:
        return None
    return best
