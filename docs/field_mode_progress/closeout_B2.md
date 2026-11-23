# Closeout B2/B3: window label reconciliation, collinear mirror guard

Branch `field-mode-closeout`. Plan steps B2 and B3.

| Step | Commit | What |
| --- | --- | --- |
| B2 | e898d02 | Window label reconciliation: Python module, FusionNode seam, flag and timeout keys, Rust filter, parity case, regression seeds |
| B2 | 52129c7 | Only gradient-settled siblings are held (same outcome, less held work) |
| B3 | 9741888 | Collinear-node mirror guard for onset fixes from point nodes only, with the single-mic trail scene |
| B3 | 2de13a5 | The rigid-array replay test judges the guard's per-row tag rather than positions across two replays |

## B2: same-band two-species merge, post-window label reconcile

**The rule.** In one analysis window, a gradient-settled cluster drops every label that another
settled sibling (gradient or LOCO) claims as its top label. This is the reviewer's first option.

- **Clarification: a cluster keeps its own top label.** A cluster never drops its own top label,
  even when a sibling shares it. Without this, two same-species birds in one window (both
  gradient-settled, same top label) would each strip the other's label and both end up unlabelled.
- The reviewer's fallback ("keep only the settled top label") was not needed; see the gate.

**Python** (`minimappr/core/onset_window_label_reconciliation.py`, 248 lines).

- **Sibling count.** `labels_to_drop_by_cluster` is the pure rule.
  `OnsetSrpLocalizationStage._attach_sibling_count` sets
  `ClusterAttributionContext.sibling_product_count` on every product that carries an attribution
  context. `window_id` was already the trigger candidate id.
- **Where it runs.** In `FusionNode._classify_and_assemble`, after classification and attribution
  (split into `_classify_and_attribute_localized` and `_assemble_localized_classification`), each
  attributed product is submitted to `OnsetWindowLabelReconciler`.
- **The hot path never blocks.**
  - A sibling that is not gradient-settled (single_cluster, LOCO, unsettled) cannot lose labels.
    It goes on to assembly at once and leaves only its `(tier, top label)` claim in the window.
  - A gradient-settled sibling that arrives early is parked. The product is flagged
    `held_for_window_label_reconciliation`, so the worker skips the empty-outcome and chunk
    bookkeeping, and the worker moves on.
  - The last sibling to arrive reconciles the held siblings and assembles them on its own worker.
- **Windows that never complete.**
  - A per-window `loop.call_later` timeout (`onset_window_label_reconciliation_timeout_seconds`,
    2.0 s) releases the window with whatever arrived. The release goes through a background task
    that assembles the siblings and enqueues them to the rules queue.
  - Siblings that will never arrive are withdrawn at once, so their window does not wait for the
    timeout. This covers the chunk-suppressed and report-window-saturated skips, a failed render,
    and a classification timeout or exception. Withdrawal is idempotent.
  - A late sibling of a window that has already been released passes through unreconciled and is
    counted.
- **Memory bounds.** At most 256 windows are held open. Beyond that, the oldest is released early.
  The 1024 most recently finished window ids are remembered, which is a memory bound only, like
  `MAX_TRACKED_RERUN_WINDOWS`.
- **Single-product windows** pass straight through, with no added latency.
- **Drain and shutdown.** `FusionNode.stop()` and `drain_fusion_queues` call
  `release_pending_window_reconciliation()` after the classification queue joins and before the
  rules queue joins.
- **Visibility.**
  - A product whose labels were dropped is annotated with
    `onset_attribution.window_reconciled_dropped_labels`.
  - New FusionMetrics counters: `onset_window_reconciled_count`, `_timeout_count`,
    `_dropped_label_count` and `_late_sibling_count`. Two of them are shown on the pipeline graph's
    attribution node.
- **Flag** `onset_window_label_reconciliation_enabled` (default **False**) plus the timeout key,
  both registered in config.py (with env and validation), settings_store.py, main.py,
  config_groups.py and pipeline_graph.py. It only takes effect when attribution is on.

**Rust.**

- The pure rule, plus `top_scoring_label` (first maximum), is in
  `minimappr-core/src/onset_clustering/window_label_reconciliation.rs`, with 4 unit tests.
- `mesh/call_attribution.rs::reconcile_window_labels` (public, so the parity oracle calls the
  exact mesh code) applies the rule between the decision loop and the apply loop. It is gated by
  the new `MeshAttributionConfig::window_label_reconciliation_enabled`.
  - It defaults to **false**, to match the server.
  - The Layer 2 entity scenarios turn it on in `run_scene`, so all 7 entity scenarios run the
    reconciled path.
- **Divergence (by construction):** the mesh takes a cluster's top label as the argmax of its
  attributed scores, while the server uses the classifier's `label`. These are the same for every
  classifier the server uses.

**Parity.** New case kind `window_label_reconciliation`: 2 × 120 random sibling windows with 2–4
clusters, all four tiers and shared labels. It drops 31 labels. Python runs the real
`reconcile_window_classifications` on `ClassifiedResult`s, and Rust runs the mesh's
`reconcile_window_labels` through `attribution-oracle`. Goldens were regenerated and
`test_attribution_trigger_cross_language_parity.py` passes (2/2). As a sabotage check, removing
the own-top-label exemption on the Rust side makes the parity test fail.

**Regression seeds** 5007, 10015 and 10187 were added to `PULL_REQUEST_SEEDS`, which now holds 13
seeds.

| Check | Before | After |
| --- | --- | --- |
| `same_band_two_species` seeds 5007 / 10015 / 10187 | each FAIL, "1 entities merged several birds" | PASS |
| All 23 scenarios × the 3 new seeds | 23/24 tests pass (same_band fails) | 24/24 |
| `same_band_two_species`, seeds 10000–10299 | 3 fail (10015, 10187 merged; 10287 duplicate) | 0 fail |
| `same_band_two_species`, seeds 5000–5059 | 1 fail (5007) | 0 fail |
| `asymmetric_bouts`, seeds 10000–10299 and 5000–5059 | 0 fail | 0 fail |
| countersinging / collinear / couples / omni_to_localized, seeds 10000–10299 | — | 0 fail |
| `flock_echo`, seeds 10000–10299 | — | 1 fail (10200, "3 echo calls reached the entity layer"; pre-existing, documented in the port review) |

The failure: the gradient-settled robin cluster carried the tanager at 0.83 next to the robin at
0.85, while the LOCO tanager cluster's top label was the tanager. The robin's fix then absorbed
into the tanager's entity. After the fix, the robin cluster carries only the robin.

### Server gate (all_on, with and without the flag)

Commands:

```
MINIMAPPR_SIM_BINARY=target/release/minimappr-sim python scripts/server_track_gate.py run \
  --configs all_on --scenes <10 scenes> fixture_birdnet_four_birds [--settings '{"onset_window_label_reconciliation_enabled": true}']
python scripts/server_track_gate.py summarize off.json on.json
```

The rows of the second run were relabelled `all_on+reconcile` so that `summarize` keeps the two
apart.

The table below has two sources:

- **Sim scenes:** the final code. The two configurations ran in parallel. A first parallel run, on
  the pre-52129c7 code, gave the same sim rows apart from load noise on one median.
- **Fixture row:** two unloaded sequential reruns of each configuration, all four identical.
  - A run made with `cargo test` going at the same time (108 s replay wall instead of 46 s) gave
    the flag-on fixture 7 detections and 1 duplicate. Its own flag-off run gave 5 detections. The
    unloaded reruns show both are load noise: the fixture replay depends on load, as
    `server_track_gate.md` notes.
- **No server scene ever triggers a label drop.** No stored row in any flag-on run carries
  `window_reconciled_dropped_labels`. On the server, the oracle and BirdNET never put a settled
  sibling's top label into a gradient-settled cluster's scores beside a different top label.

| Scene | Config | Dets | Indiv. recall | Event recall | Swaps | Dup | Merged | Median err (m) | P90 err (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onset_overlap_two_species | all_on | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0.33 | 0.57 |
| onset_overlap_two_species | +reconcile | 3 | 1.00 | 0.75 | 0 | 0 | 0 | 0.63 | 8.07 |
| onset_trunk_echo | both | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0.65 | 0.65 |
| onset_faint_far_bird | both | 3 | 1.00 | 1.00 | 0 | 0 | 0 | 0.15 | 0.15 |
| onset_same_band_pair | both | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0.43 | 0.52 |
| onset_same_band_two_species | both | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0.47 | 0.68 |
| onset_single_tetra_overlap | both | 1 | 0.50 | 0.25 | 0 | 0 | 0 | — | — |
| gate_countersinging_pair | both | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0.32 | 9.99 |
| gate_flock_echo | both | 5 | 1.00 | 0.56 | 0 | 0 | 0 | 0.56 | 10.23 |
| gate_omni_then_localized | both | 1 | 1.00 | 0.25 | 0 | 0 | 0 | 0.18 | 0.18 |
| gate_collinear_trail | both | 2 | 1.00 | 0.33 | 0 | 0 | 0 | 0.12 | 0.12 |
| fixture_birdnet_four_birds | both (×2 each, unloaded) | 6 | 1.00 | 0.50 | 0 | 0 | 0 | 0.39 | 12.59 |

"both" means the rows are identical with and without the flag.

| Summary (10 sim scenes) | Indiv. recall sum | Event recall sum | Swaps | Dup | Echo | Merged | Median of medians |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_on | 9.50 | 4.64 | 0 | 0 | 0 | 0 | 0.33 |
| all_on + reconcile | 9.50 | 4.89 | 0 | 0 | 0 | 0 | 0.43 |

**Verdict: PASS** against the plan's requirement. Individual recall is unchanged in every scene,
including the asymmetric-bouts-style single-species scenes. There are 0 swaps, 0 duplicates and
0 merges, and the fixture is unchanged. The Layer 2 merge it targets is fixed (0/300 + 0/60 after,
against 3/300 + 1/60 before).

**One side effect, which is deterministic** (two reruns of each configuration agree):
`onset_overlap_two_species` surfaces one more robin row.

- **Cause.** In window 1 the robin is split into two gradient-settled clusters 20 ms apart. Both
  are held and then assembled back to back by the last sibling.
  - The second cluster's reporting decision is made before the first one's detection has passed
    through the rules stage (tracker and entity binding).
  - So it is inserted as its own row instead of being folded into the first.
  - It shares the same track and entity, so no duplicate entity results.
  - No labels were dropped. This is a pure ordering effect of holding.
- **Scoring effect.**
  - Event recall rises from 0.50 to 0.75.
  - The scorer's event matcher pairs the extra robin row with the tanager's 1.3 s call (11 m away).
    That moves event-level label accuracy from 1.00 to 0.75 and the p90 error from 0.57 to
    8.07 m. It is not a swap: the swap metric is 0.
- **Why it was left.** Fixing it would need the held siblings' rules-stage outcomes to feed each
  other's reporting decisions, which serialises them. That conflicts with "never block the hot
  path", so I left it.

**Should it join the live-trial flag list?** It passes the plan's gate criteria. On the server,
though, it showed no gain: no server scene reproduces the Layer 2 merge, and nothing was dropped.
Its one cost there is the ordering side effect above. My recommendation is to keep it off for the
first Bluefin trial and add it once real data shows same-window cross-species absorption. This is
the user's decision (see below).

## B3: collinear mirror guard (non-rigid nodes only)

- **The port.** `minimappr/core/collinear_node_mirror_guard.py` is a line-for-line port of
  `node_set_is_collinear`: the minor/major horizontal standard-deviation ratio, true for fewer
  than 2 nodes. It adds `fix_is_collinear_mirror_ambiguous`, which returns false as soon as any
  contributing node has at least 2 sensor positions (a rigid array) or the ratio is 0. Otherwise it
  runs the collinearity test on the contributing nodes' sensor centroids.
- **Where it applies.** In `_product_for_cluster`, a mirror-ambiguous fix is given
  `range_projection_mode = range_asymptotic`. This is the existing "range unobservable"
  representation, not a new one:
  - `spatial_display_mode_for_detection` renders it `bearing_only`;
  - `apply_unobservable_range_haircut` caps its confidence and observability;
  - by `_localization_quality_is_reportable`'s own contract, these modes create no track;
  - the entity registry treats `bearing_only` as direction-known, never pinpointed.

  The product is also tagged `onset_cluster_collinear_mirror_ambiguous`.
- **Config** `localization_collinear_mirror_max_aspect_ratio` (0.2, validated to [0, 1], 0
  disables), registered in all five places. It sits in the onset localization group and appears
  as an onset-node param in the pipeline graph.
- **No separate flag.** The plan specifies only the ratio key. The guard is therefore live by
  default for point-node-only collinear sites. Set the ratio to 0 to turn it off. Sites with any
  rigid array in a fix are provably untouched.
- **New scene** `minimappr-sim/scenarios/gate_collinear_trail_single_mic.toml`:
  `gate_collinear_trail` with single-mic phones in place of the tetras.
- **Tests** (`tests/test_collinear_node_mirror_guard.py`, 6):
  - **Unit tests:** the Rust test's trail and ring layouts; a point-node trail is ambiguous and a
    triangle is not; a contributing rigid array disables the guard, but a non-contributing one
    does not; a ratio of 0 disables it.
  - **Replay of the single-mic trail** through a full FusionNode, all_on, with
    `beamformed_classification_min_sensor_count=1` so single-mic sensors can classify:
    - unguarded, both birds are placed on the wrong side of the trail (vireo truth y = −8 placed
      at y = +16.8; thrush truth y = +9 placed at y = −8.0);
    - guarded, every row is `bearing_only` and none is placed.

    With the guard wiring removed from the stage, this test fails.
  - **Rigid-array check:** a guarded replay of the tetra `gate_collinear_trail`. Every onset row
    carries `onset_cluster_collinear_mirror_ambiguous = false`, none is `range_asymptotic`, and
    each labelled bird is placed on its true side.
    - Removing the rigid exemption makes this test and the unit test fail.
    - A first version compared the positions of two replays (guard on and off). It passed when
      run alone but failed once in the full suite, because replay timing depends on load. The
      guard's own tag does not.

## Test numbers

| | Rust (`cargo test --workspace`) | Python (`pytest tests/ --ignore=tests/test_soak_harness.py`) |
| --- | --- | --- |
| Baseline (after B5) | 555 passed / 4 ignored | 2066 passed / 9 skipped |
| After B2 + B3 | 559 passed / 0 failed / 4 ignored (+4 core unit tests) | 2083 passed / 0 failed / 9 skipped (+17 new tests) |

New Python tests:

- `test_onset_window_label_reconciliation.py` (11);
- `test_collinear_node_mirror_guard.py` (6).

Parity regained a case kind, still 2 tests. The Layer 2 PR seed count grew from 10 to 13 for every
scenario.

## Left open / for the user
- **B2 flag for the live trial.** I recommend keeping it off for the first trial (reasons above).
  The user decides.
- **B2 ordering side effect.** Holding can surface an extra same-entity row when one bird is split
  into two gradient-settled clusters in one window.
- **The server gate cannot exercise B2's drop.** No server scene reproduces the merge, so the
  merge fix is shown in Layer 2 and the parity test only. A server scene with a loud and a quiet
  same-band bird, where the quiet one is LOCO-settled, would close this gap.
- **B3 has no master flag** (per plan). The ratio set to 0 is the off switch.
- **Point-node classification still blocks real use.** Single-mic-only sites still cannot classify
  at the default `beamformed_classification_min_sensor_count=2` (S gate Reported 9). The replay
  test lowers it to see the rows.
