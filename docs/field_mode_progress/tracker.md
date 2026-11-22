# W2-tracker: perched tracker, species belief, partial binding, two thresholds (S3 + S4)

Branch `fm/tracker` (base `field-mode` @ bca0407). Every behaviour change is behind a
flag that defaults to off; with all flags off the tracker, assembly and BirdNET path
behave exactly as before (full suite green, legacy numbers pinned in
`tests/test_tracker_field_mode_scenarios.py`).

## What was built

| Piece | File | Flag |
| --- | --- | --- |
| Perched random-walk filter (position-only Kalman, variance grows linearly with silence so sigma grows with sqrt(dt); a fix > `relocation_snap_distance_m` from the perch restarts the filter there instead of smearing) | `core/track_filters_perched.py` (`PerchedRandomWalkTrackFilter`) | `tracking_perched_filter_enabled` |
| Per-track motion-model routing (perched for `tracking_perched_label_categories`, default `wildlife`; drones/vehicles keep the 6D CV Kalman) | `core/track_filters_perched.py` (`MotionModelRoutingTrackFilter`) | same |
| Relocation gate: a 8-30 m same-species hop after >= 2 s silence, with no nearer same-species track, keeps the identity; plus **ghost-perch revert** for flocks (below) | `core/track_perched_relocation.py` | same |
| Per-track species log-odds (capped per update and in total), replacing the single-label overwrite for `tracking_species_log_odds_label_categories` (default `wildlife`); exposes `species_belief` (top-N label -> p), `species_top_label`, `species_top_probability` on `TrackState`; `track.label`/`confidence` become the top belief | `core/track_species_belief.py`, `core/track_field_mode_extensions.py` | `tracking_species_log_odds_enabled` |
| Hidden tentative tracks: a track built *only* from sub-display evidence whose belief is below `tracking_species_display_min_belief` is left out of `TrackManager.snapshot()` (API, WS, federation, housekeeping), never persisted, never sent to rules | `core/track_field_mode_extensions.py`, `core/assembly_track_resolution.py` | log-odds (+ two thresholds to produce sub-display evidence) |
| Partial-constraint binding of detections with no full fix (omni / bearing-only / non-localizable tier) to a track seen in the last 25 s: species overlap (top-3 each side) + every constraint consistent + unambiguous. Constraints: tetra DOA bearing (bearing-only solves), cross-node TDOA hyperbola (`tdoa_s` entries on another node), node-level ranking (reporting node ~nearest, 7 dB margin, hard bound). Soft position nudge (gain 0.3) toward bearing/hyperbola | `core/track_partial_binding.py`, `core/track_partial_constraint_evidence.py`, `core/tracking_field_mode_methods.py` | `tracking_partial_constraint_binding_enabled` |
| Two thresholds: the BirdNET member is built with `birdnet_localize_to_track_min_confidence` (0.15) instead of its display floor and wrapped so results under the display floor carry `below_display_threshold=True`. Such a detection feeds track belief, never writes audio, and is silently dropped from rules/WS unless its track is visible (`reason=below_display_threshold`) | `classifiers/birdnet_two_threshold.py`, `classifiers/factory.py`, `core/assembly.py`, `core/fusion_node.py` | `birdnet_two_threshold_enabled` |

Assembly now delegates the track decision to `core/assembly_track_resolution.py`
(`resolve_detection_track`), which is also what the scenario harness drives.

### Config keys (all restart-required; one `# --- Field Mode W2-tracker ---` block per file)

`tracking_perched_filter_enabled` (False), `tracking_perched_label_categories` ("wildlife"),
`tracking_perched_position_diffusion_m2_per_second` (0.05), `tracking_perched_relocation_snap_distance_m` (6),
`tracking_perched_relocation_gate_max_m` (30), `tracking_perched_relocation_min_silence_seconds` (2),
`tracking_perched_relocation_ghost_seconds` (30), `tracking_species_log_odds_enabled` (False),
`tracking_species_log_odds_label_categories` ("wildlife"), `tracking_species_log_odds_prior_probability` (0.1),
`tracking_species_log_odds_per_update_cap` (3.0), `tracking_species_log_odds_total_cap` (6.0),
`tracking_species_log_odds_evidence_top_k` (3), `tracking_species_display_min_belief` (0.5),
`tracking_species_belief_exposed_top_n` (5), `tracking_partial_constraint_binding_enabled` (False),
`tracking_partial_binding_window_seconds` (25), `..._bearing_sigma_deg` (10), `..._hyperbola_sigma_m` (2),
`..._snr_level_margin_db` (7), `..._max_normalized_residual` (3), `..._species_top_k` (3),
`..._soft_update_gain` (0.3), `..._ambiguity_score_ratio` (0.5), `birdnet_two_threshold_enabled` (False),
`birdnet_localize_to_track_min_confidence` (0.15).

Registered in `config.py` (TrackingConfig + Settings fields, env loader `MINIMAPPR_<KEY>`,
`tracking_config()` pass-through, `field_mode_tracker_setting_errors` validation),
`settings_store.py`, `main.py` (GET config, PATCH merged-view validation, restart-required
for both python and rust ingest), `core/config_groups.py` (Tracking group),
`core/pipeline_graph.py` (Tracking node + BirdNET member params).

## Design decisions worth knowing

- **Log-odds update.** `L += clip(logit(p) - logit(prior), +-per_update_cap)`, `L` clipped to
  `logit(prior) +- total_cap`. One detection scoring p gives belief p (when under the cap),
  so the display threshold on belief means the same as on a detection. Unscored labels are
  untouched (classifiers only report top-k). Three bouts 0.30/0.45/0.40 -> 0.97.
  Per-update cap 3.0 (not 2.0): with 2.0 and prior 0.1 a single detection could never pass
  0.45, so a bird heard once at 0.9 would have read 0.45.
- **Only the winning member's scores are evidence.** Composite score maps mix vocabularies
  (YAMNet "Animal" next to BirdNET species); a belief over both lets a generic class win.
- **Hidden only if never revealed.** A track is hidden only when *all* its evidence was
  sub-display. Any display-level detection reveals it permanently. So turning on log-odds
  never hides anything that surfaces today (an earlier draft hid every new track whose
  first detection sat under 0.5 belief, which created duplicate "detection" entities).
- **Ghost-perch revert (flock guard).** The relocation gate alone merged a 3-robin flock
  (perches 22-24 m apart, alternating bouts) into one track: each new bird "relocated" the
  only robin track. Now the old perch is kept as a ghost for 30 s; a same-species fix back
  on the ghost reverts the hop (track returns, the new perch becomes a new track) and bars
  both from relocating for the window; a track may hop once per window. Cost: one wrong
  detection per pair, and a real second hop inside 30 s starts a new track.
- **Node-ranking constraint is a hard bound at the margin**, not a sigma (otherwise
  3 x 7 dB = 21 dB of slack bound an omni detection from the far node).
- **Partial binding never binds on species alone** (at least one spatial constraint), and
  refuses when two same-species tracks fit comparably (`ambiguity_score_ratio`).

## Measurements

### Tracker scenarios (synthetic detection sequences, realistic jittered fixes)

`python scripts/measure_tracker_scenarios.py --seeds 0 1 2 3 4` (< 2 s). Harness:
`minimappr/sim/tracker_detection_scenarios.py` drives `resolve_detection_track` (real
assembly decision) on the 3-node layout of `synthetic_entity_scenarios`; 1-1.2 m fix
jitter; BirdNET emulated (flag off: top score < `birdnet_trigger_min_confidence` = 0.5 ->
unknown; flag on: kept down to 0.15, flagged below display). Scored with
`replay_entity_metrics` at 5 m radius, 1 s slop; a surfaced detection carries its visible
track's label (what the UI shows). Means over 5 seeds:

| Scenario | Config | Surfaced dets | Visible tracks | L&C recall | Shown-label acc. | Swap rate | Duplicates | Multi-individual entities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intermittent_perched_robin | off | 3.00 | 1.00 | 0.38 | 1.00 | n/a | 0.00 | 0.00 |
| intermittent_perched_robin | perched | 3.00 | 1.00 | 0.38 | 1.00 | n/a | 0.00 | 0.00 |
| intermittent_perched_robin | log-odds+2thr | 8.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| intermittent_perched_robin | all-on | 8.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | off | 8.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | perched | 8.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | log-odds+2thr | 8.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| robin_hops_15m | all-on | 8.00 | 1.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_alternating | off | 10.00 | 4.00 | 0.65 | 1.00 | n/a | 4.20 | 2.20 |
| two_species_alternating | perched | 10.00 | 2.00 | 0.83 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_alternating | log-odds+2thr | 11.60 | 3.60 | 0.67 | 0.69 | n/a | 4.40 | 2.80 |
| two_species_alternating | all-on | 12.00 | 2.00 | 1.00 | 1.00 | n/a | 0.00 | 0.00 |
| two_species_overlapping | off | 9.00 | 2.40 | 0.63 | 1.00 | 0.00 | 1.40 | 1.00 |
| two_species_overlapping | perched | 9.00 | 2.00 | 0.75 | 1.00 | 0.00 | 0.00 | 0.00 |
| two_species_overlapping | log-odds+2thr | 12.00 | 2.00 | 0.73 | 0.73 | 0.27 | 2.00 | 2.00 |
| two_species_overlapping | all-on | 12.00 | 2.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| weak_fix_weak_id | off | 2.00 | 0.00 | 0.00 | 1.00 | n/a | 0.00 | 0.00 |
| weak_fix_weak_id | perched | 2.00 | 0.00 | 0.00 | 1.00 | n/a | 0.00 | 0.00 |
| weak_fix_weak_id | log-odds+2thr | 4.00 | 1.00 | 0.40 | 1.00 | n/a | 1.00 | 0.00 |
| weak_fix_weak_id | all-on | 4.00 | 1.00 | 0.80 | 1.00 | n/a | 0.00 | 0.00 |
| flock_same_species | off | 9.00 | 5.40 | 0.75 | 1.00 | n/a | 6.00 | 3.60 |
| flock_same_species | perched | 9.00 | 3.00 | 0.75 | 1.00 | n/a | 0.00 | 0.00 |
| flock_same_species | log-odds+2thr | 10.60 | 6.00 | 0.88 | 1.00 | n/a | 7.60 | 4.60 |
| flock_same_species | all-on | 11.00 | 3.00 | 0.92 | 1.00 | n/a | 1.00 | 1.00 |

Readings:

- **All flags on** beats legacy on every scenario: recall 0.38 -> 1.00 (intermittent bird),
  0.65 -> 1.00 and 4.2 -> 0 duplicates (two species), 0.00 -> 0.80 (weak fix + weak ID),
  6.0 -> 1.0 duplicates (flock; the 1 is the single ghost-revert mistake), swaps 0.
- **Legacy CV Kalman mixes two birds 12 m apart**: its covariance grows as q*dt^4 over 4-8 s
  gaps, so the Mahalanobis gate (max 32 m) lets the tanager's fix update the robin's track
  (and vice versa). The perched filter alone removes this (duplicates 0).
- **Two thresholds without the perched filter is worse than legacy** on the two-species
  scenarios (swap rate 0.27, shown-label accuracy 0.69-0.73): the extra weak, confused
  detections land on the wrong CV track. **Enable `birdnet_two_threshold_enabled` only with
  `tracking_perched_filter_enabled`** (and log-odds). Recorded as a W3 cutover constraint.
- `weak_fix_weak_id` all-on recall is 0.80, not 1.0: the omni bouts bind (partial binding,
  node-ranking constraint) and the track is classified, but the first (0.30, hidden) bout
  never surfaces by design.
- Swap rate is only scorable in `two_species_overlapping` (others are not concurrent).

### Replay bundles (synthetic_entity_scenarios, full FusionNode)

`scripts/measure_replay_entity_baseline.py` path with the four flags on vs off, seed 0:
**identical** to the baseline in `server_baseline.md` for all three scenarios (same
detections, tracks, labels, recall 0.0/0.25/0.0 at 15 m, unassociated 1/2/1 at 5 m). As
expected: every fix collapses onto node-a (W2-onset-srp's problem), the stub classifier's
labels are not `wildlife` (so log-odds does not apply), and one or two detections per
bundle give the tracker nothing to fix. The Rust simulator was not used for the same
reason (it replays through the same collapsing path); rerun both once W2-onset-srp lands.

## Tests

- `tests/test_track_species_belief.py` (11): update rule, caps, recovery, winner-member
  evidence, flag-off overwrite, belief exposure, category scoping, hidden vs revealed.
- `tests/test_track_perched_filter.py` (11): sqrt(dt) growth, jitter averaging, snap,
  routing keeps CV for drones, 12 s gaps one track no drift, 15 m hop keeps identity and
  zero velocity (legacy reads it as velocity), refusals (too soon, nearer same-species
  track, different species), flock ghost revert.
- `tests/test_track_partial_binding.py` (8): bearing/hyperbola/ranking residuals and
  corrections, species/stale/ambiguity/no-constraint refusals, omni bind via the real
  assembly decision (belief lifts a hidden track), far-node refusal, flag-off untouched.
- `tests/test_birdnet_two_threshold.py` (5): wrapper flags, composite-winner lookup,
  factory builds with the low floor only when flagged, sub-display never starts a
  belief-less track, hidden track surfaces once belief passes.
- `tests/test_tracker_field_mode_scenarios.py` (11): the scenario gates above.

## Open issues / follow-ups

- **Rust sidecar BirdNET path** (`ingest_sidecar_runtime.py`, rust ingest) still uses the
  single display floor; two thresholds only apply to the Python classifier factory.
- **Node-ranking constraint uses "reporting node ~ nearest" only.** No per-node levels
  reach assembly today; when W2-onset-srp/W3-attribution produce per-node in-band energy,
  feed real levels (normalised by each node's noise floor) into `SnrRankingConstraint`.
- Partial binding needs >= 2 nodes for an omni detection (a single node gives no
  constraint) and a bearing only for `bearing_only` display mode; omni detections from a
  tetra node do not currently carry their DOA bearing to assembly.
- `track.label_id` still follows the last detection when log-odds sets `track.label` to
  the top belief label.
- `species_belief` and hidden/revealed state are memory-only (not DB columns), like the
  fingerprint; a restart forgets them.
- A sub-display BirdNET result now competes in the composite with YAMNet (it used to be
  "unknown", an abstention), so with two thresholds a weak BirdNET species can win over a
  YAMNet generic label; the detection is then hidden unless a track surfaces it.
- Thresholds (0.15 localize-to-track, 0.5 display belief, caps) are v2.4 starting points;
  W3-birdnet-cutover re-tunes them on V3.0 and golden bundles. No per-island clip budget
  here (that is mesh-side).
- `tracking.py` was already 776 lines; it is now 856 (new methods moved to
  `tracking_field_mode_methods.py` / `track_field_mode_extensions.py`). `assembly.py` 534 -> 548.
- Same-species flock: one ghost-revert mistake per pair; a bird hopping twice within 30 s
  starts a new track. A two-mode IMM stays the Phase 5 option.
