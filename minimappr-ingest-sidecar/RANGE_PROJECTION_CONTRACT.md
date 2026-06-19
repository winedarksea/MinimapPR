# `range_projection_mode` contract (Rust ↔ Python)

Every localization estimate is tagged with a `range_projection_mode` describing how
the radial (range) axis was resolved. The Rust SRP-PHAT sidecar and the Python
Cartesian solver both emit these tags so downstream consumers can apply a uniform
confidence / range-observability haircut regardless of which engine produced the
estimate.

Canonical definitions live in two mirrored files; **keep them in sync**:

- Rust: [`src/range_projection.rs`](src/range_projection.rs)
- Python: [`minimappr/core/range_projection.py`](../minimappr/core/range_projection.py)

## Canonical modes

| Mode | Meaning | Range observable? |
|------|---------|-------------------|
| `range_refined` | Range resolved from a well-conditioned TDOA / grid fit. | yes |
| `range_asymptotic` | Estimate dominated by the far-field prior; range unobservable **and** the bearing fit is weak. | no |
| `range_boundary` | Estimate clamped at the search-grid boundary (Rust near-field grid); range unobservable. | no |
| `range_bearing_projected` | Bearing is well-observed; position is projected along the bearing ray at the far-field default range with a covariance explicitly elongated along the ray. Range unobservable but **direction is trustworthy**. | no (range), yes (bearing) |

Design intent: **prefer `range_bearing_projected` over `range_asymptotic`.** A known
direction with honest range uncertainty is far more useful on the COP than a
discarded or fully-haircut estimate. `range_asymptotic` should remain reachable only
when even the bearing cannot be trusted (degenerate geometry).

## Legacy aliases (folded by `normalize_range_mode`)

| Legacy string | Canonical |
|---------------|-----------|
| `prior_projected` | `range_asymptotic` |
| `bounded_grid_boundary` | `range_boundary` |

These exist so an un-rebuilt sidecar still receives the correct haircut. New code
must emit only the canonical strings.

## Caps (single source of truth)

Applied by the Python solver inline, by the path-agnostic haircut at the Rust→Python
ingest seam (`apply_unobservable_range_haircut`), and mirrored as Rust constants /
`confidence_cap_for_mode` / `range_observability_cap_for_mode` in `range_projection.rs`.

| Mode | Confidence cap | Range-observability cap |
|------|----------------|--------------------------|
| `range_refined` | none | none |
| `range_asymptotic`, `range_boundary` | `0.20` | `0.05` |
| `range_bearing_projected` | `0.85` | `0.05` |

`range_bearing_projected` keeps a high confidence cap because the **bearing** is
valid; only range observability is driven to the floor to signal range uncertainty.

> Note: Rust `is_unobservable()` returns `true` for `range_bearing_projected` (its
> *range* is unobservable), but the confidence cap differs — always go through
> `confidence_cap_for_mode` rather than assuming the harsh cap.

## Upgrade criteria (intentionally path-specific)

The canonical single-node solver is `python_cartesian`: the sidecar's pairwise TDOAs
+ bearing are re-homed into the Python Cartesian estimator, which owns the final
position and `range_projection_mode`. The Rust SRP-PHAT estimate still feeds the
bearing prior and the render steering, so both paths classify range — by different
but compatible criteria:

- **Python** (`core/cartesian_tdoa.py::solve_cartesian_tdoa`):
  - Forces `range_refined → range_asymptotic` when the final Jacobian condition
    number exceeds `1e6` (`_RANGE_PROJECTION_CONDITION_THRESHOLD`).
  - Upgrades `range_asymptotic → range_bearing_projected` when the preliminary
    (bearing) Jacobian is well-conditioned (`cond < 1e8`).
- **Rust** (`src/srp_phat.rs`):
  - `far_field_candidate` emits `range_bearing_projected` (instead of
    `range_asymptotic`) when the far-field direction-fit residual is low
    (`direction_fit_residual < 0.5`).
  - `select_candidate` prefers the far-field bearing-projected candidate over a
    near-field grid solution when the near-field range is unobservable
    (`range_observability < 0.15`), the far candidate has clearly better contrast
    (`+0.2`), is genuinely distant (`> 100 m`), and fits at least ~1.7× better
    (`far_residual ≤ 0.6 × near_residual`). The `0.6` ratio replaced an earlier
    `0.25` that phase-slope-refined TDOAs no longer reach (cleaner delays let both
    near and far models fit the near-planar wavefront, collapsing the residual gap).

These two criteria operate on different representations (Jacobian conditioning vs.
direction-fit residual). They are not expected to produce bit-identical
classifications on the same input; `python_cartesian` is authoritative for the
emitted position and mode.
