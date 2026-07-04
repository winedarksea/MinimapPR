"""Amplitude/SNR-informed range prior (Phase 1c).

A tetrahedral array a few centimetres across cannot resolve range beyond the near
field — wavefront curvature across the aperture is sub-sample. For those
unobservable-range solves the estimator projects the well-observed bearing out to a
*prior* distance. Historically that distance was a fixed 50 m guess. This module
derives a physically-motivated prior from the received sound-pressure level using
inverse-square spreading::

    r = 10 ** ((L_ref - L_recv) / 20)

where ``L_ref`` is the (assumed) source level at 1 m and ``L_recv`` is the received
level. A source 10 dB quieter than reference is ~3.16× farther; 20 dB → 10× farther.

The prior only substitutes the *projection distance* for unobservable-range modes
(RANGE_ASYMPTOTIC / RANGE_BEARING_PROJECTED / RANGE_BOUNDARY). It never overrides a
``range_refined`` solve, where the data resolved range directly. The Rust sidecar
mirrors this formula in ``src/srp_phat.rs`` and the shared contract lives in
``minimappr-ingest-sidecar/RANGE_PROJECTION_CONTRACT.md``.
"""

from __future__ import annotations

import math

# Reference RMS for the dBFS-style SPL proxy. Kept as a module constant so the
# Python SPL proxy and the amplitude prior share one definition. ``received_level``
# here is 20·log10(rms) + gain_offset_db, i.e. dB relative to full scale offset by
# the per-node calibration gain — the same quantity assembly.py records as ``spl_db``.
_MIN_RMS = 1e-9


def received_level_db_from_rms(rms: float, gain_offset_db: float = 0.0) -> float:
    """Received level (dB) from a window RMS and per-node calibration offset.

    Shared by the detection SPL proxy (``assembly.py``) and the amplitude range
    prior so the two never drift. ``rms`` is floored to avoid ``log10(0)``.
    """

    return float(20.0 * math.log10(max(float(rms), _MIN_RMS))) + float(gain_offset_db)


def amplitude_range_prior_m(
    received_level_db: float,
    *,
    reference_source_level_db: float,
    min_range_m: float,
    max_range_m: float,
) -> tuple[float, bool]:
    """Inverse-square range prior from a received level, clamped to a physical band.

    Returns ``(range_m, clamped)`` where ``clamped`` is True when the raw
    inverse-square estimate fell outside ``[min_range_m, max_range_m]`` and was
    clipped — used to drive the ``localization_amplitude_prior_clamped`` metric.
    """

    if not math.isfinite(received_level_db):
        return float(max(min_range_m, 0.0)), True
    raw_range_m = 10.0 ** ((float(reference_source_level_db) - float(received_level_db)) / 20.0)
    lo = max(float(min_range_m), 0.0)
    hi = max(float(max_range_m), lo)
    clamped_range_m = min(max(raw_range_m, lo), hi)
    was_clamped = not (lo <= raw_range_m <= hi)
    return float(clamped_range_m), was_clamped
