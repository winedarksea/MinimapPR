# Tracking

`TrackManager` (`minimappr/core/tracking.py`) turns localized detections into tracks.
These notes cover the knobs that keep an intermittent source, such as a bird calling every
few minutes or a vehicle passing behind cover, on one track id rather than a new one for each
burst. All three mechanisms are on by default.

Every key below is in the `PATCH /api/v1/config` allowlist and persists to `data/config.yml`.
`TrackManager` copies its config once at startup, so a patched value takes effect on the next
restart. The PATCH response does not list these keys under `restart_required`.

## Per-category lifecycle

| Key | Default | Meaning |
|-----|---------|---------|
| `track_stale_seconds` / `kalman_process_noise` | `20.0` / `2.0` | Base values |
| `track_stale_seconds_wildlife` / `_vehicle` / `_human` / `_security` | `120` / `60` / `60` / `20` | Stale window per label category |
| `kalman_process_noise_wildlife` / `_vehicle` / `_human` / `_security` | `0.5` / `4.0` / `1.0` / `2.0` | Kalman Q per category |

A value of `0.0` inherits the base. The `unknown` category always uses the base, so BLE
tracks and unclassified detections behave as before. Drop and reap windows derive from
the category's stale window through `track_drop_multiplier` (`3.0`) and
`track_reap_multiplier` (`5.0`).

## Kalman coast guard

Without a guard, a long coast grows process noise without bound and extrapolates velocity
linearly forever.

- `kalman_max_coast_process_seconds` (`10.0`) caps the dt used for Q growth. Position and
  velocity still propagate with the real dt.
- `kalman_coast_velocity_half_life_seconds` (`10.0`): once `dt > half_life / 2`, the
  transition switches to Ornstein-Uhlenbeck-style velocity decay. This bounds coast
  displacement to about `(half_life / ln 2) * |v|`. Shorter gaps keep the exact
  constant-velocity transition. `0` disables the guard.

## Class-aware association

- `association_category_gate_enabled` (`true`) is a hard gate: a detection only associates
  with tracks of the same label category. `unknown` on either side is a wildcard.
- `association_fingerprint_weight` (`3.0`) adds a cosine similarity between the detection's
  classifier scores and the track's fingerprint. It only *re-ranks* candidates that are
  already inside the positional gate and never widens it.
- Fingerprints are an EWMA of classifier scores. `track_fingerprint_alpha` (`0.3`) sets the
  EWMA weight, and `track_fingerprint_top_k` (`8`) sets how many labels are kept.

Design note: fingerprints live in a side dict on `TrackManager` (`_fingerprints`), not as a
`TrackState` field. `TrackState` is serialized to the DB, the live hub and federation peers,
and the fingerprint is internal association state.

## Dormant reacquisition

When a confirmed track (one that reached the confirm threshold) is dropped, its identity is
parked in a dormant registry. A later detection that would mint a new track first checks the
registry. It revives the old id when the record:

- is within `dormant_reacquire_radius_m` (`20.0`) and younger than `dormant_ttl_seconds`
  (`1800`),
- has a compatible label category, and
- has an exact label match, or a fingerprint cosine of at least
  `dormant_fingerprint_min_similarity` (`0.4`).

Among the records that qualify, the one with the best mix of distance and similarity wins.
The revived track's confidence is the larger of two values: the new detection's confidence,
or the parked confidence decayed by the time spent dormant with half-life
`dormant_confidence_half_life_seconds` (`300`). The registry
holds at most `dormant_max_records` (`64`) records and evicts the oldest first.
`dormant_reacquire_enabled` is the kill switch.

Revivals are counted in `track_dormant_reacquired_count` (`FusionMetrics`), which appears
on the tracking card of the `/settings/pipeline` graph.
