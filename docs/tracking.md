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

## Tracking modes

`tracking_mode_by_category` (`minimappr/core/tracking_mode_config.py`) picks the dynamics a track
runs once it has a label:

| Mode | Default for | Behaviour |
|------|-------------|-----------|
| `stationary_hold` | `wildlife` | Perched filter: zero velocity, fixes averaged in, relocation gate, ghost revert and concurrent-perch guard (the `tracking_perched_*` keys tune it) |
| `continuous_motion` | every other category | Constant-velocity Kalman filter |
| `none` | — | The category is not tracked; `TrackManager.update` returns `None` |

An unlabelled track runs `continuous_motion`. When a label arrives or changes,
`apply_label_category_tracking_mode` switches the track's dynamics under the tracker lock. A
label of `unknown` keeps the track's current dynamics. `unknown` can't be configured as a key.
The mapping replaces the old `tracking_perched_filter_enabled` and
`tracking_perched_label_categories` flags. It is PATCHable, and a change takes effect on restart.
Rust mirrors the enum and defaults in `minimappr-core/src/wildlife_tracking/tracking_mode.rs`,
checked by `tests/test_wildlife_tracking_cross_language_parity.py`. The Rust perched tracker does
not read the mode yet; it always runs the `stationary_hold` path.

`entity_reporting_enabled` defaults to on.

## Continuous sources (drone update-gap gate)

`tests/test_continuous_source_tracking.py` (E6) feeds sidecar manifests for a drone fly-by, a
hover, two crossing drones and a drone with birds calling nearby through a real `FusionNode`. The
gate requires every drone solve to update exactly one track, once, stamped with the solve's
window-centre time. The largest gap between a drone's updates must stay within 1.5 × the solve
cadence (1.2 s at the default 0.8 s trigger cooldown). Each drone stays on one track and no other
source updates it. Before this work a drone track moved once per 28 s render (1 of 24 solves).
The 120 s scenes run with `MINIMAPPR_ENABLE_CONTINUOUS_TRACKING_LONG_SCENES=1`.

## One observation per solve

Each localized solve updates the tracker once, before classification, so nothing on the
classification side can starve a track: chunk suppression, the saturated-window pre-gate, an
empty classification, a classification-queue backlog drop, a cross-reporter duplicate merge, or a
zone or reporting-policy suppression. Both ingest paths hand every solve to
`PositionObservationPipeline` (`minimappr/core/position_observation_pipeline.py`) through the
`PositionObservationSink` seam.

- **Two steps per solve.** First the position: the fix associates on geometry alone (its label
  is still `unknown`, a wildcard for every gate) and leaves the track's label and dynamics
  unchanged. Later the label: the solve's classification attaches its label to the track that
  solve already moved, without moving it again (`core/tracking_position_observation_methods.py`).
- **Idempotent by `solve_id`.** This is the provenance `solve_id` on the Rust path, where a
  covering render repeats its event's id, and the candidate id on the Python-ingest path. A second
  delivery of a solve is counted (`position_observations_duplicate_solve`) and ignored.
- **Hidden until classified.** With `position_observation_unclassified_track_policy` set to
  `start_hidden_tentative` (the default), an unmatched fix from an event-triggered solve starts a
  tentative track. That track stays out of snapshots, storage, the live feed, federation and rules
  until a classification reveals it, including after it is dropped and until it is reaped. It is
  never parked in the dormant registry, since it has no identity to reacquire. A scheduled render, or a render without provenance, may only
  update an existing track. `require_classification` means no track starts before a
  classification.
- **Ambiguous fixes wait for their label.** When more than one track gates an unclassified fix,
  geometry cannot tell the sources apart (two birds a few metres apart, crossing drones). The fix
  is then left to its classification's label-aware `update`, which has the species gates and
  fingerprint ranking (`position_observations_deferred_ambiguous`).
- **The label never moves the track.** A solve's classification attaches its label to the track
  that solve's position already moved; `update_count` and `last_seen_ns` stay as they are, and a
  hidden track becomes visible. The label-aware `update` runs instead only when the track is gone,
  or when the classification contradicts the track. A contradiction is a different category, or
  a rejection by the label-dependent association gates (`association_candidates`: species
  log-odds, concurrent-perch guard), which run again once the label exists. A label whose category
  has tracking mode `none` ends an unclassified provisional track but never a labelled one. This
  matters only with `association_category_gate_enabled` off, where it used to delete an
  established track.
- **Abstentions say nothing.** A classification with an `unknown` label or zero confidence (a
  classifier abstention, an `unsettled` onset attribution) never labels, reveals or re-labels a
  track. It is judged on the label, not the category: a real species label that the taxonomy
  files under `unknown` still names a source. It
  retracts a hidden track that only its own fix started; otherwise it refreshes recency only.
- **Classifications that reporting drops still reach the tracker.** A `suppress` decision, a
  non-entity `enrich_existing`, or a cross-reporter duplicate merge still labels the solve's
  track, or resolves its held fix (`label_observed_solve_track_without_report`).
- **Bounded FIFO, drop-oldest.** `position_observation_queue_max_items` (`1024`) bounds the queue.
  Overflow drops the oldest observation and counts it (`position_observations_dropped_queue_full`).
  Observations are never coalesced: two callers of the same label in one cluster at the same time
  are two observations. A dropped solve's classification, if one arrives, updates the tracker
  instead. `position_observation_solve_ledger_max_entries` (`8192`, never below the queue size)
  bounds how many solves are remembered in full while their classification is pending. An evicted
  entry leaves its solve id and track id in
  `position_observation_solve_id_memory_max_entries` (`32768`, never below the ledger size). That
  is about 34 minutes at 16 event solves per second, well past any render interval or classifier
  backlog. A covering render or a classification that arrives after eviction is still a duplicate,
  or attaches only its label, and never applies the position a second time.
- **Event solves off the event loop.** The SSE consumer awaits every event-only solve. With the
  default `localization_single_node_solver: python_cartesian`, the Python re-solve of a
  six-sensor cluster solve took a median of 11 ms (max 15 ms) on the event loop. It now runs in a
  worker thread, one at a time, so ordering and idempotency are unchanged. A covering render
  repeats its event's solve, but it does not count the solve's rejections or silent drops again,
  and the bearing-fusion store keeps one cone per node and event.
- **Entity feed.** An absorbed entity bout whose solve was observed attaches its label to the
  observed track. It no longer nudges that track a second time through partial binding.

### Hold rule for perched tracks (a plan change)

Plan point 4 asked for one tracker update per solve "ahead of every suppression path". The
milestone 3 review found a case where applying the position first is wrong. A perched bird's
`stationary_hold` filter averages every fix in and never forgets one. When a neighbour's
overlapping call is solved from a poor geometry, the fix lands on the perched track before any
label exists, and an abstaining classification cannot undo it. In `gate_collinear_trail` this
moved the thrush's median error from 0.12 m to 0.45 m. It also skipped the species gate and
concurrent-perch guard: a Hermit Thrush fix moved a perched robin's track instead of starting
its own.

So a fix whose only gating track is a **labelled `stationary_hold` track** is held for its
classification (`position_observations_held_for_label`):

- An informative label applies the fix to the track that gated it, unless the label-dependent
  gates (species log-odds, concurrent-perch guard, both run against every live track) reject
  that track or it is gone. In that case the label-aware `update` places it, for example on a
  new track for a second bird (`position_observations_held_applied_with_label`). The fix is not
  re-associated from scratch: in `gate_flock_echo` a poorly conditioned fix then jumped to a
  nearer provisional track whose filter was still loose.
- An abstention, an empty classification, or a classification that finds nothing trackable
  drops the fix (`position_observations_held_dropped`).
- If no classification ever arrives, the fix is applied geometry-only
  (`position_observations_held_applied_geometry_only`). That covers a classification-queue drop,
  chunk suppression, a saturated report window, a timeout, a backend failure and ledger eviction,
  and `position_observation_held_fix_max_wait_seconds` (`30`) is a deadline for anything else. A
  released fix never moves `last_seen_ns` backwards.

Continuous-motion and hidden tracks still update at once, so the drone update-gap gate is
unaffected. Event-only Rust solves are never held, because they carry no classification and only
one per render interval is ever covered by a render. Holding them would only delay them by the
deadline, and the gates that run again at attach still catch a covering render's contradicting
label.

This keeps the reason for point 4: no capacity or dedupe drop starves the tracker, and only the
classifier's own verdict can stop a fix reaching a perched bird. It is the same kind of
exception as the ambiguity rule above.

`gate_flock_echo` recalls less than it did before milestone 3 (`event_recall_sum` 0.56 to 0.33).
This is expected: the extra recall came from a misassociation that P5 removed.

`onset_overlap_two_species` (all-on) has a median error of 0.44 m, against 0.33 m before
milestone 3. Before milestone 3, an abstaining classification (`unknown`, confidence 0) founded
a track, and the tanager's first call joined it. Abstentions no longer found or move tracks, so
the tanager's track now starts from its own first fix. That fix is poorly conditioned in height
(variance 211 m²), but the stationary_hold filter restarts at it with its default isotropic
4 m² variance, not the fix's.

Starting the filter from the founding fix's covariance was tried (branch
`loc/m3-hold-filter-init`, not merged). It brings this scene to 0.34 m, but makes
`gate_countersinging_pair` worse (0.32 → 0.51 m) and `gate_collinear_trail` worse
(0.08 → 0.21 m). In those coplanar and collinear layouts the solver's fix covariances have long
axes (up to 900 m²) that are tilted, so a later fix's height correction drags the horizontal
position along the tilt. The real fix is calibrating the correlation structure of the solver
covariance. The same branch also carries a separate, smaller fix: a relocation-gate hop fused
the same fix twice.
