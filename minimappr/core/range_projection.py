"""Canonical ``range_projection_mode`` vocabulary shared across estimators.

Both the Python Cartesian solver and the Rust SRP-PHAT sidecar tag every
localization result with a ``range_projection_mode`` describing how the radial
(range) axis was resolved. Historically the two estimators emitted divergent
strings for the same semantic state, so downstream confidence/observability
capping only fired for one path. These constants are the single source of truth
for the Python side; the Rust side mirrors them in ``src/range_projection.rs``
and the shared contract is documented in
``minimappr-ingest-sidecar/RANGE_PROJECTION_CONTRACT.md``.
"""

from __future__ import annotations

# Range observed from the data (well-conditioned TDOA / grid fit).
RANGE_REFINED = "range_refined"

# Range unobservable: the estimate is dominated by the far-field prior rather
# than the measurements. Covers the Python "asymptotic" radial fit and the Rust
# far-field "prior_projected" state.
RANGE_ASYMPTOTIC = "range_asymptotic"

# Estimate clamped at the search-grid boundary (Rust near-field grid). Range is
# likewise not observable from the data and must receive the same haircut.
RANGE_BOUNDARY = "range_boundary"

# Legacy strings emitted by un-migrated Rust sidecars. Kept so a sidecar that has
# not yet been rebuilt with the canonical vocabulary still receives the correct
# haircut. ``normalize_range_mode`` folds these into the canonical values.
LEGACY_PRIOR_PROJECTED = "prior_projected"
LEGACY_BOUNDED_GRID_BOUNDARY = "bounded_grid_boundary"

_LEGACY_ALIASES: dict[str, str] = {
    LEGACY_PRIOR_PROJECTED: RANGE_ASYMPTOTIC,
    LEGACY_BOUNDED_GRID_BOUNDARY: RANGE_BOUNDARY,
}

# Modes whose range axis is not observable from the measurements. Downstream
# consumers cap confidence/observability uniformly for these regardless of which
# estimator produced the result.
UNOBSERVABLE_RANGE_MODES: frozenset[str] = frozenset(
    {RANGE_ASYMPTOTIC, RANGE_BOUNDARY}
)

# Caps applied to any localization whose range axis is unobservable. These are
# the single source of truth for both the Python solver (applied inline during
# the Cartesian solve) and the path-agnostic downstream haircut in the fusion
# node (which catches Rust-origin estimates).
UNOBSERVABLE_CONFIDENCE_CAP = 0.20
UNOBSERVABLE_RANGE_OBSERVABILITY_CAP = 0.05


def normalize_range_mode(mode: str | None) -> str | None:
    """Fold legacy sidecar values into the canonical vocabulary.

    Unknown or canonical values pass through unchanged; ``None`` stays ``None``.
    """

    if mode is None:
        return None
    return _LEGACY_ALIASES.get(mode, mode)


def range_mode_is_unobservable(mode: str | None) -> bool:
    """Return True when the range axis is not observable for ``mode``.

    Accepts both canonical and legacy strings so callers can classify a raw
    inbound value without normalizing it first.
    """

    return normalize_range_mode(mode) in UNOBSERVABLE_RANGE_MODES


def apply_unobservable_range_haircut(
    *,
    mode: str | None,
    confidence: float,
    range_observability: float | None,
) -> tuple[float, float | None]:
    """Cap confidence/observability when the range axis is unobservable.

    Path-agnostic: applied both inside the Python solver and at the Rust ingest
    seam so an estimate receives the same haircut regardless of which engine
    produced it. Idempotent.
    """

    if range_mode_is_unobservable(mode):
        confidence = min(confidence, UNOBSERVABLE_CONFIDENCE_CAP)
        if range_observability is not None:
            range_observability = min(
                range_observability, UNOBSERVABLE_RANGE_OBSERVABILITY_CAP
            )
    return confidence, range_observability
