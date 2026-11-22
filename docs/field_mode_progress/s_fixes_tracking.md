# S-fixes tracking: tracker and entity fixes from the S gate

Branch `fm/s-fixes-tracking` (from `field-mode` @ 753f5ea, "Merge fm/s-gate into field-mode").
Measured 2026-11-22 on a heavily loaded machine. This fixes the tracker and entity issues the S gate
reported (`server_track_gate.md`, "Reported, not fixed" 1, 2 and 3, plus the plan-gate near-miss).
The S2 localization issues (4, 5, 6) belong to a parallel branch and are not touched here.

Every change sits behind the existing flags. With all flags off, behaviour is identical: the baseline
rows match the pre-fix run on all 10 sim scenes, metric for metric. New knobs live in
`# --- Field Mode S-fixes tracking ---` blocks. They are registered in `config.py` (fields, env,
validation, `tracking_config()`), `settings_store.py`, `main.py`, `core/config_groups.py` and
`core/pipeline_graph.py`, and every one requires a restart.

## Fixes

| # | Problem (gate report) | Fix | Files | Gating |
| --- | --- | --- | --- | --- |
| 1a | Cross-species track merge: association checked only the category | **Species association gate.** Tracks are filtered before the associator and the relocation gate. A detection that claims a species (top score ≥ `tracking_species_association_detection_claim_probability`, 0.15) never joins a track whose belief is confident in another species (top ≥ `tracking_species_association_confident_probability`, 0.5). There are two exceptions: either side supporting the other's species, and a *weak* claim (below 0.5) that sits on the track's own perch (not a fix the perched filter would snap to). Unknown labels and abstentions stay wildcards | `core/track_species_association_gate.py`, hook `FieldModeTrackExtensions.association_candidates`, one call in `tracking.py` | `tracking_species_log_odds_enabled` + `tracking_species_association_gate_enabled` (True) |
| 1b | `believes_label` accepted any species anywhere in the belief | A label is believed if it is the top species or has a belief ≥ `entity_absorption_min_species_belief` (0.3; the prior is 0.1). A bound track is followed only when the entity believes the detection's label or the track's top species | `bird_entity_model.py`, `bird_entity_registry.py` | `entity_reporting_enabled` |
| 2a | Flock: the relocation / ghost logic hopped a track onto a third bird's perch | **Concurrent-perch guard.** A fix that the perched filter would snap to (beyond the snap distance *and* χ²-inconsistent) may not move a same-species perched track in two cases: the track sang less than `min_silence` ago, or another same-species track sang *since* its last bout (within `tracking_perched_concurrent_activity_seconds`, 30 s). The most recent singer may still hop, so the single-bird hop is unchanged | `core/track_perched_concurrency.py` | `tracking_perched_filter_enabled` + `tracking_perched_concurrent_perch_guard_enabled` (True) |
| 2b | Flock: 12 m fixed absorption radius > perch spacing | **Covariance-gated absorption.** For a fix against a pinpointed entity, the horizontal Mahalanobis distance (fix + entity covariance, 2 dof) must be ≤ `entity_absorption_chi2_gate` (9.21), with a `entity_absorption_min_fix_distance_m` (3 m) jitter floor. There is no fixed cap. A fix without a covariance keeps the old 12 m rule | `core/bird_entity_absorption_geometry.py`, `bird_entity_matching.py` | `entity_absorption_covariance_gate_enabled` (True) |
| 2c | Flock: a shared track drags several birds into one entity | **Concurrency.** Each entity keeps its recent raw fixes. A fix within `entity_concurrent_fix_window_seconds` (2 s) that is inconsistent beyond both covariances is another bird. Such a fix is not absorbed, and it does not follow the track binding: it becomes a new entity from its own fix and does not take the binding | same + `bird_entity_registry.py` | entity flag |
| 3 | Canonical row overwritten with each raw fix | Rows in entity mode persist the **bound track's fused position and covariance**, which is what the entity shows. The raw fix is kept in `feature_summary.raw_fix_position_m` / `raw_fix_position_covariance_m2` and still feeds the tracker | `core/entity_canonical_row_position.py`, `assembly.py` (`persist_fused_track_position`), `fusion_node.py` | `entity_canonical_row_fused_position_enabled` (True) |
| 4 | A far fix at 391 m on the right bearing (true ~80 m) started a duplicate entity | **Bearing evidence.** `decide()` now receives the fix covariance and range-projection mode. A fix whose own horizontal 1σ exceeds the pinpoint limit (15 m), or whose range is `range_bearing_projected`, is matched by covariance gate 2b: its covariance runs hundreds of metres along the ray, so a fix on the bird's bearing is consistent. If the entity is located at least as well, the fix is absorbed like an unpositioned bout (`enrich_existing`), so it creates no far-off track, does not move the entity and does not rewrite the row | `bird_entity_absorption_geometry.py`, `entity_reporting_policy.py`, `entity_reporting_coordinator.py`, `fusion_node.py` | `entity_bearing_evidence_absorption_enabled` (True) |

Why the detection claim floor is below the display floor (found on the real-audio scene):

- After 1a with a single 0.5 floor, the tanager was labelled, but its bouts still landed on the
  grosbeak's track 7 m away.
- Sub-display tanager bouts (two-threshold) are wildcards under a 0.5 floor. They joined the grosbeak
  track and built a tanager belief inside it, and that belief then "supported" the confident tanager
  bouts.
- With a 0.15 claim floor, the weak bouts start the tanager's own (hidden) track.
- A weak misclassified bout *on* a known bird's perch still joins that bird, so W2's "one wrong bout
  does not split" behaviour holds (`test_track_species_belief.py` stays unchanged).

## Gate results (`scripts/server_track_gate.py`, seed 0, horizontal scoring, entity positions)

Commands (binary built in the worktree):

```
MINIMAPPR_SIM_BINARY=target/release/minimappr-sim python scripts/server_track_gate.py run --json a.json \
  --scenes <5 scenes> --configs all_on baseline all_on-minus:<stream>... --work-dir wd
python scripts/server_track_gate.py summarize a1.json b1.json a2.json b2.json
```

**Plan gate, sim matrix: FAIL → PASS.** On the pre-fix code, this run fails on the faint bird's
duplicate and on "position no worse" in two scenes; this run's baseline happens to place
overlap / flock rows closer than the doc's first measurement did.

| Config | Indiv. recall (sum of 10) | Event recall (sum) | Swaps | Dup | Echo | Merged | Unassoc. | Median of scene medians (m) | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (before = after, identical) | 0.00 | 0.00 | 0 | 0 | 1 | 0 | 4 | 6.78 | — |
| all_on, before | 8.83 | 4.47 | 0 | 1 | 0 | 0 | 1 | 0.62 | FAIL: faint-bird duplicate; overlap median +9.1 m, flock +2.1 m vs baseline |
| **all_on, after** | **9.50** | 4.25 | **0** | **0** | **0** | **0** | **0** | **0.43** | **PASS** |

Event recall falls 4.47 → 4.25 for two reasons:

- The far fix in `onset_faint_far_bird` is now absorbed instead of surfacing as its own row
  (1.00 → 0.67 there).
- `onset_overlap_two_species` loses a row that was 37 m off.

One row per bird is the design, so individual recall is the headline.

ALL-ON per scene, before → after:

| Scene | Dets | Indiv. recall | Swaps | Dup | Echo | Merged | Unassoc. | Median err (m) | P90 err (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate_collinear_trail | 3 → 3 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0.23 → 0.23 | 16.44 → 16.44 |
| gate_countersinging_pair | 2 → 2 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0.53 → 0.53 | 9.96 → 9.96 |
| **gate_flock_echo** | 2 → 3 | **0.33 → 1.00** | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | **8.90 → 0.43** | 9.88 → 10.11 |
| gate_omni_then_localized | 1 → 1 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 1.38 → 1.38 | 1.38 → 1.38 |
| **onset_faint_far_bird** | 4 → 3 | 1.00 → 1.00 | 0 → 0 | **1 → 0** | 0 → 0 | 0 → 0 | 0 → 0 | 0.62 → 0.62 | 0.62 → 0.62 |
| **onset_overlap_two_species** | 3 → 2 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | **1 → 0** | **15.48 → 0.34** | **37.49 → 0.36** |
| onset_same_band_pair | 2 → 2 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0.78 → 0.78 | 1.32 → 1.32 |
| onset_same_band_two_species | 2 → 2 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0.25 → 0.25 | 0.38 → 0.38 |
| onset_single_tetra_overlap | 1 → 1 | 0.50 → 0.50 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | — | — |
| onset_trunk_echo | 1 → 1 | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0.08 → 0.08 | 0.08 → 0.08 |

Per stream, all on code after the fixes (sums over the 10 scenes):

| Config | Indiv. recall | Event recall | Swaps | Dup | Echo | Merged | Unassoc. | Median of medians (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_on | 9.50 | 4.25 | 0 | 0 | 0 | 0 | 0 | 0.43 |
| − S2 onset | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 6 | 12.82 |
| − S3 perched | 9.50 | 4.60 | 0 | 2 | 0 | 0 | 0 | 0.25 |
| − S3 log-odds | 9.50 | 4.08 | 0 | 0 | 0 | 0 | 0 | 0.53 |
| − S4 partial binding | 9.50 | 4.25 | 0 | 0 | 0 | 0 | 0 | 0.43 |
| − S4 two-threshold | 9.50 | 4.08 | 0 | 0 | 0 | 0 | 0 | 0.53 |
| − S5 attribution | 8.50 | 3.89 | 2 | 1 | 0 | 0 | 0 | 0.50 |
| − S5 relative trigger | 9.00 | 3.92 | 0 | 0 | 0 | 0 | 0 | 0.43 |
| − S6 entity | 9.50 | 4.42 | 0 | 1 | 0 | 1 | 0 | 0.70 |

Each stream's contribution (all_on − all_on-minus):

- S6 went from **−1.67 recall** (gate) to **0 recall, −1 duplicate, −1 merged**: entities now only help.
- S3 perched gives −2 duplicates.
- S5 attribution gives +1.00 recall, −2 swaps and −1 duplicate.
- S5 relative trigger gives +0.50 recall.
- S3 log-odds, S4 partial binding and S4 two-threshold change nothing on these scenes.

**Real audio (`fixture_birdnet_four_birds`, BirdNET v2.4, ALL-ON): the Scarlet Tanager is
recovered.** Rows are placed at entity positions.

| Run | Dets | Indiv. recall | Event recall | Label acc. | Swaps | Dup | Merged | Unassoc. | Median / P90 err (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_on, before (this machine) | 4 | 0.75 | 0.50 | 0.80 | 0 | 0 | 0 | 0 | 1.88 / 3.96 |
| **all_on, after** (3 replays: 1.00 / 1.00 / 1.00) | 4-5 | **1.00** | 0.50 | **1.00** | 0 | 0 | 0 | 0 | 0.17-0.49 / 0.62-2.05 |
| all_on − S6 entity, after | 12 | 0.25 | 0.25 | 1.00 | 0 | 2 | 3 | 0 | 0.10 / 0.76 |
| all_on − S3 log-odds, after | 7 | 0.75 | 0.38 | 0.80 | 0 | 0 | 0 | 1 | 0.48 / 11.57 |

The fixture replay is not bit-for-bit repeatable under load: BirdNET / LOCO queue timing changes which
bouts get labelled. One pre-fix run labelled the tanager's first bout `unknown`. A first post-fix run
(before the claim-floor change) labelled it and put it on the grosbeak track. That run is what found
the weak-claim leak described above.

**W2/W3 tracker scenarios** (`scripts/measure_tracker_scenarios.py --seeds 0 1 2 3 4`):

- Every all-on, perched and entity row is unchanged, including `robin_hops_15m` (hop identity kept)
  and the flock rows.
- Only the log-odds+2thr rows (perched off) moved:

| Scenario (log-odds+2thr) | L&C recall | Shown-label acc. | Dup | Multi-individual |
| --- | --- | --- | --- | --- |
| two_species_alternating | 0.67 → 0.92 | 0.69 → 0.92 | 4.4 → 2.0 | 2.8 → 1.0 |
| two_species_overlapping | 0.73 → 0.73 | 0.73 → 0.73 | 2.0 → 2.4 | 2.0 → 1.4 |
| two_species_one_omni_each | 0.33 → 0.67 | 0.67 → 1.00 | 4.0 → 2.0 | 2.0 → 0.0 |

## Tests

- `tests/test_track_s_fixes_association.py` (10): claim / contradiction rules.
  - Species gate: a confident other species is refused; unknown joins; a sub-display other species
    off-perch builds its own track; a weak misclassified bout on the perch joins the bird; gate off,
    log-odds off or flags off keep the legacy merge.
  - Concurrent-perch guard: a rotating flock keeps one track per perch, and without the guard the third
    bird relocates the first track (the W2 mistake); a single-bird hop keeps its identity; a noisy far
    perch is never flagged.
- `tests/test_entity_s_fixes_absorption.py` (7):
  - Absorption: flock perches 10 m apart become two entities while on-perch jitter is absorbed; with
    the gate off the fixed radius merges them; a far fix on the bearing absorbs and an off-bearing one
    does not.
  - Concurrency: concurrent inconsistent fixes are refused even on a loose track; a shared track does
    not drag a concurrent bird, and its binding is not taken.
  - Policy: bearing evidence becomes `enrich_existing` and leaves the entity where it is, a
    bearing-projected range also absorbs, and a good fix still upgrades.
  - Rows: the fused-row position helper.
- `tests/test_bird_entity_registry.py`: the belief-floor case is updated. A prior-level 0.05 belief no
  longer absorbs; floor 0 restores the W3 rule.
- `tests/test_server_track_gate.py`: `gate_flock_echo` and `onset_faint_far_bird` are added to the
  ALL-ON pytest gate (both now pass: recall 1.0, 0 duplicates / swaps / echoes / merges). 7 passed in 52 s.
- Full suite:
  `python -m pytest tests/ --ignore=tests/test_soak_harness.py -q -p no:cacheprovider` →
  **2019 passed, 17 skipped** (6 min 24 s).

## Open issues

- **Perched tracks start overconfident.** `PerchedRandomWalkTrackFilter.initialize_track` uses
  `kalman_initial_position_variance` (4 m², 1σ 2 m) and ignores the fix covariance. A far bird whose
  first fix has 1σ ≈ 60 m becomes a "pinpointed" entity with σ ≈ 2 m. Bearing-evidence absorption
  copes with this, but the entity's claimed uncertainty is wrong. Seeding the filter from the
  measurement covariance would widen the association gates too, so it needs its own gate run.
- **Range observability is not used** to classify a fix as bearing evidence: values on good and bad
  fixes overlap (0.16-0.3 on both). The covariance and `range_bearing_projected` carry the signal.
  When the S2 branch makes the far-range covariance or projection mode honest, this path picks it up.
- The concurrent-perch guard refuses a real hop when a same-species neighbour sang since the hopper's
  last bout. The cost is a new track for that bird (and a new entity unless absorption catches it).
- `bird_entity_registry.py` is at 499 lines. Further growth should move into a new module.
- Only one sim seed and one real-audio scene. The fixture result is from 3 replays under load.
