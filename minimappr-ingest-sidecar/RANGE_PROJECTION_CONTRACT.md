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

### Rust confidence parity (Phase 1a)

The Rust SRP-PHAT confidence must match the identical Python solve:

- The base score `0.6·fit + 0.25·peak + 0.15·contrast` is attenuated by
  `range_observability.clamp(0.35, 1.0)` **only** for non–bearing-projected modes.
  For `range_bearing_projected` the multiplier is skipped (mirrors Python's
  `obs_factor = 1.0` branch), because the cone's huge radial eigenvalue makes
  observability artificially tiny and it must not penalise a well-observed bearing.
- The per-mode `confidence_cap_for_mode` / `range_observability_cap_for_mode` are
  then applied Rust-side (idempotent with the Python ingest-seam haircut).

## Amplitude/SNR range prior (Phase 1c)

For unobservable-range modes only, the *projection distance* may be derived from the
received level instead of a fixed 50 m guess, via inverse-square spreading:

```
r = 10 ^ ((L_ref − L_recv) / 20)     clamped to [min_range_m, max_range_m]
```

- `L_recv` = reference-channel received level (dBFS): `20·log10(rms) + gain_offset`.
- `L_ref` = assumed source level at 1 m (`localization_amplitude_reference_level_db`,
  default 100 dB). Clamp band default `[5 m, 1000 m]`.
- Implemented identically in `minimappr/core/amplitude_range.py::amplitude_range_prior_m`
  and `src/range_projection.rs::amplitude_range_prior_m`.
- **Rule:** the prior only substitutes the projection distance for the
  unobservable-range modes (`range_asymptotic` / `range_bearing_projected` /
  `range_boundary`). It never overrides a `range_refined` solve, where the data
  resolved range directly (`far_field_initial_range_m` is unused in that branch).
- The prior's radial std is `std_factor × prior_range` (default `2.0`, ≈ ±6 dB
  source-level uncertainty). The cone radial std is `max(4×range, prior_std, 200 m)`
  in both languages (`cartesian_tdoa.py` and `srp_phat.rs`).
- Ships disabled (`localization_amplitude_range_prior_enabled = false`); enable after
  a per-node `gain_offset_db` calibration check. The Rust sidecar reports raw dBFS in
  the manifest (`received_level_dbfs`); the Python path applies the prior on re-solve.

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

## Coplanar half-space constraint (D7)

A coplanar array (e.g. the 5-mic Sirith Planar node) cannot resolve which side of
its own plane a source sits on: the TDOA measurements are identical for a source and
its mirror image reflected across the array plane. `NodeSpec.half_space`
(`"upper"` | `"lower"` | `"none"`, default `"upper"` for planar nodes, `"none"`
otherwise) disambiguates which side is physically valid, and both engines report
whether the constraint was actually applied to this estimate via
`half_space_applied`:

- **Rust** (`src/srp_phat.rs`): `SrpPhatConfig::half_space` clamps the near-field
  grid's z-bound to the array's own mean-z plane in `grid_from_bounds` before the
  search runs (`HalfSpace::Upper`/`Lower` restrict; `HalfSpace::None`, the tetra
  default, is a no-op). `half_space_applied` on `LocalizationManifestPayload` is set
  whenever `half_space != HalfSpace::None` for that manifest — it reflects
  configuration, not whether the estimate actually landed off-plane.
- **Python** (`core/localization.py::LocalizationEngine.localize`): the
  unconstrained Cartesian solve is free to converge on either mirror image;
  `spatial_audio/geometry.py::reflect_position_into_half_space` mirrors the
  solved position (and `reflect_covariance_into_half_space` the covariance's
  xz/yz cross terms) back across the array's mean-z plane when it lands on the
  wrong side. `LocalizationResult.half_space_applied` is `True` whenever
  `half_space` was `"upper"`/`"lower"` for the call (regardless of whether a
  reflection was actually needed).
- Only the z-axis case (a horizontal array plane) is implemented in both engines.
  A pitched or rolled planar node needs the array's rotated normal, not the raw
  z-axis — tracked as a follow-up (plan risk #5).
- Multi-node bearing fusion (`core/multi_node_bearing_fusion.py`) needs no
  separate half-space logic: `BearingObservation.direction` is derived from the
  already-corrected single-node position (`fusion_node.py`), so the elevation
  sign is inherited automatically once the single-node solve applies D7.
