# Review: W4-port-server-track (S2/S3/S5/S6 in `minimappr-core`)

Independent promotion review of `field-mode` @ 7a8f093 (merges `fm/port-server-track` 132170f and
`fm/port-onset` 14ebf24). The diff under review is `430659f..7a8f093`, about 13 k lines. Fixes are on
`review/port-server-track`. No Python source changed. Where Python has the bug, it is documented here
and left alone.

## Verdict

**Promote, with the fixes below.** The port is careful. The goldens come from the real server
classes. The decision-level parity is exact, and the tolerances are justified. The core builds for
wasm32, Android and iOS. Every file is under 500 lines.

The real problems were elsewhere:

- the mesh host's attribution seam did not build the window context the way the server does;
- one numeric guard was missing;
- the bookkeeping grew without bound;
- the Layer 2 entity metrics were weaker than they looked: broken echo rejection passed half the PR
  seeds.

## How I checked

- **Read** every new core module against its Python reference. I covered S5 and S6 in full and S3's
  association, absorption, relocation and log-odds. For S2 I spot-checked the attribution evidence
  and its panic paths only: it already has stage-by-stage parity.
- **Measured branch coverage** of the goldens by decoding the `.npz` files and counting outcomes for
  every decision branch.
- **Broke the code on purpose** to see whether Layer 2 notices:
  - disabled revert re-pointing;
  - disabled the echo rule;
  - ignored attributed species;
  - turned entity tracking off.
- **Swept** 60 seeds (5000+) and 300 seeds (10000+) per entity scenario. I compared the 300-seed
  sweep against a build of the untouched 7a8f093 tree, so I could tell my regressions from existing
  failures.

## Findings

### High

1. **Layer 2's echo metric could not detect broken echo rejection.**
   - I made `is_echo_of` always return false. `flock_echo` still passed 5 of 10 PR seeds with
     "echo 0". The other 5 failed only through an unrelated "pinpointed birds 3" check.
   - Cause: a trunk echo 4 m behind the perch has the same species. It is absorbed into its own
     bird's entity, so it never forms an echo-only entity, and the "echo promotion" metric never
     fires.
   - **Fixed:** new `EntityMetrics::echo_calls_surfaced` (any call traced to an echo that reached the
     entity layer) with an invariant. With the echo rule off, all 10 seeds now fail with "N echo
     calls reached the entity layer".
2. **The entity invariants passed vacuously when no call reached the entity layer.**
   - With `entity_tracking.enabled = false`, the metrics read zero entities and report "dup 0,
     merged 0, swaps 0".
   - **Fixed:** the invariants now require every scripted bird to reach at least one entity.
3. **The mesh host built the S5 window context differently from the server.** This is
   `mesh/call_attribution.rs`, which replaces `onset_srp_localization_stage._attach_attribution_context`.
   It differed in two ways:
   - `shared_onset_max_lag_samples` was `None`. The server uses `round(echo_max_lag_seconds × fs)`.
     So the target's own note, or its echo, seen in another cluster was counted as an interferer and
     masked, which erases the target.
   - Clusters colocated with the target were interferers. These are clusters within
     `cluster_radius_m + cluster_range_fraction × range` of the target: one source split in two. The
     server excludes them.
   - **Fixed:** new `MeshAttributionConfig { cascade, shared_onset_max_lag_seconds,
     colocated_cluster_radius_m, colocated_cluster_range_fraction }`. Its defaults come from
     `OnsetClusteringConfig`, which holds the server's values. `MeshConfig::attribution` now has this
     type.
4. **The LOCO rerun budget did not bound classifier CPU in the host.**
   - `LocalizerRerunHost::mask_interferers` called `classify_cluster_window`, which masks *and*
     reruns the classifier. The cascade consults the budget only after masking.
   - So every capped rerun still ran the classifier. That is exactly the cost the per-window cap
     exists to stop on phones.
   - **Fixed:**
     - new `ClipSetLocalizer::masked_cell_fraction`, whose default delegates and is documented to be
       overridden by costly classifiers;
     - the rerun now happens lazily in `rerun_masked`, so only after the budget admits it.

### Medium

5. **Negative Mahalanobis distances were accepted in absorption.** This is
   `bird_entity/absorption_geometry.rs::horizontal_mahalanobis_squared`.
   - Python rejects `value < 0`, which an indefinite covariance from a bad Jacobian can produce. Rust
     only checked `is_finite`.
   - So a non-PSD fix would pass every χ² gate and win absorption with a negative score.
   - No parity case reached it, because every golden covariance was `B Bᵀ + εI`.
   - **Fixed**, with:
     - a Rust unit test;
     - a new parity case, `entity_geometry_degenerate_case`, which uses indefinite, zero and
       rank-one covariances and needs the fix to match Python.
6. **Per-call outcome history grew without bound** (`HostEntityTracking::call_outcomes`).
   - **Fixed:** it is now a `VecDeque`, capped by `EntityTrackingConfig::max_retained_call_outcomes`
     (4096; oldest first).
   - Re-pointing only needs the calls of one ghost window.
7. **`wire_ids` leaked.** Entities evicted by the registry's capacity LRU never emit `Expired`, so
   their `(island, entity) → wire id` entries stayed forever. **Fixed:** they are pruned at each
   maintenance tick.
8. **The attribution parity gate missed some branches and outputs.** Two branches never ran:
   `bands_overlap_not_geometric` (the collinear-trail S-fix) and `loco_rerun_cap`.
   The comparison also left out two outputs:
   - the annotation's `reason` key, so `loco_disabled`, `loco_rerun_cap` and
     `interferers_silent_in_window` were not told apart;
   - the masking seeds handed to the host.

   **Fixed:**
   - both outputs are now compared;
   - new `attribution_band_overlap_case` (6 windows sharing one rerun budget) reaches both branches.

   Every tier, gradient reason, LOCO reason and stop reason is now exercised.

### Low

9. **A NaN classifier score poisons a label's log-odds forever.** This affects both Python
   (`track_species_belief.py`, where `min(max(nan, ε), 1−ε)` is NaN) and Rust.
   - **Rust fixed:** it skips NaN evidence. Inf still clamps, as in Python.
   - **Python left as is,** per the ground rules. Recommend the same one-line guard server-side.
   - Parity cannot carry NaN (JSON), so this divergence is invisible to the gate.
10. **Non-finite fixes could reach the tracker.** **Fixed:** the host treats a non-finite fix as
    heard-nearby (not placed), with a unit test.
11. **Multi-cluster windows reported every cluster from the call's classifying node.**
    - **Fixed** for windows with at least 2 clusters: each cluster reports from its own loudest node,
      whose classification it carries.
    - A lone cluster keeps the call's node. Switching it caused a `collinear_trail` duplicate (seed
      5001). Heard-nearby matching compares against the entity's reporting-node position, within
      `absorption_max_node_distance_m`, and on an 8 m-spaced trail the loudest node changes from
      bout to bout.
    - That anchor is fragile, but it is server behaviour (`spatial_mismatch_score`). Left as is.

### Checked and fine

- **The goldens are real.**
  - `TrackManager` is driven with the S3 flags on, together with `BirdEntityRegistry` and
    `EntityReportingPolicy` in FusionNode order, with the real revert listener.
  - The cascade goldens come from the real `OnsetClusterAttributor` (only `_mask_interferers` and
    the classifier are scripted).
  - The trigger goldens come from `NoiseFloorRelativeTrigger`.
  - The second test of each gate re-derives the goldens from Python.
- **The tolerances are honest.**
  - Decisions are exact.
  - Values are 1e-9 relative on closed-form f64.
  - The comparator scales tolerance by the enclosing list's largest magnitude. That is loose only
    for mixed-magnitude nested lists, which no current output has.
- **Coverage before the fixes.** The pipeline scenarios reach every reporting action, change kind,
  state, revert and expiry. The association, concurrency, species-gate and two-birds cases hit both
  outcomes.
- **Divergences checked and found to match the reference:**
  - association: plain-gate χ² calibration, 3σ physical gate, eigenvalue floor;
  - absorption: 2-dof χ² gate (9.21) and distance floor;
  - entity LRU eviction: first minimum in creation order, matching dict order;
  - rerun-budget LRU and trigger LRU: last use;
  - log-odds caps: per-update ±3, total ±6 around the prior;
  - revert listener order: it fires inside the tracker update, before association, and the registry
    is untouched in between;
  - maintenance uses the tracker snapshot without aging, as the server coordinator does.
- **Panics.** The remaining `expect`s in new core code are internal invariants (an entry inserted or
  looked up a line earlier, or keys taken from the same map). None depends on the data.
- **Entity ids stay unique across terms.** `next_bird_entity_id` is floored at `term << 32` on
  takeover. Per-island pipelines key wire ids by `(island, entity)`.
- **Revert re-pointing works and the metrics depend on it.** With re-pointing disabled, `flock_echo`
  fails all 10 PR seeds with duplicate entities. Note what this means: `flock_echo` passes because
  of host bookkeeping beyond the server. The server itself still shows the hop.
- **`entity_tracking.enabled = false` restores W3.**
  - All 16 pre-existing scenarios pass with it off, and the 7 entity scenarios fail as they should.
  - The sim's `localize()` equals `localize_clusters()[0]`, the call the old single-source localizer
    picked.
  - The pipeline still calls `localize_clusters` and attribution, but the W3 update reads only
    `record.fix` and the call's belief.
- **The collinear guard's threshold lives in config** (`collinear_mirror_max_aspect_ratio`, 0.2, a
  ratio of standard deviations). Its doc comment now gives what it flags. It is a heuristic: a thin
  triangle is near-degenerate rather than truly mirror-ambiguous. It is sound as a "don't pinpoint"
  guard.
- **Label swaps are detected.** Feeding the call's merged belief to every cluster makes
  `same_band_two_species` fail all 10 seeds.
- **Ground-truth leakage in the Layer 2 stand-in.**
  - **Leaks, documented here, not fixed:** arrivals are grouped across nodes by their scripted
    `call_index`. That is perfect association, standing in for onset clustering, so Layer 2 cannot
    fail on cluster mis-association. The single-source W3 localizer already did this. Real
    association is covered only by the S2 parity windows.
  - **Does not leak:** echo rejection uses arrival timing and band only. The classifier stand-in
    grades the arrival's scripted species by SNR, which is the intended oracle, and neither sees
    positions. The localizer sees estimated positions and clock mappings only.

## Left as is, and why

- **`same_band_two_species` merges (seed 5007; 10015 and 10187 in a 300-seed sweep) and one
  duplicate (10287).** All four fail identically on the untouched 7a8f093 build, so they are faithful
  to the server, not regressions. The stream's "59/60" understates the rate a little (3/300 at
  10000+).
  - **My pick: fix it in attribution, not in absorption.** When the gradient tier settles, it proves
    the classifier mostly heard the target. So the target's top label is justified; the rest of the
    window's score list is not.
    - Drop from a gradient-settled cluster's evidence any label that is the top label of another
      settled cluster in the same window. That is the stream's first candidate.
    - If a server gate run shows that is not enough, keep only the settled top label.
  - **Not the absorption-side fix** (the absorbing label must be the entity's top label beyond kσ).
    It tightens absorption for every bird, and it would split exactly the case log-odds exists for:
    one bird whose top label flips between weak bouts (`asymmetric_bouts`).
  - Either change needs the server S gate run before it is ported.
- **`flock_echo` seed 10200 now fails** ("3 echo calls reached the entity layer"). Only the new,
  stricter metric catches it; the echo leaked on 7a8f093 too, silently absorbed.
  - Cause: the sim's echo is an omnidirectional image source. Nodes beyond the trunk hear it
    *earlier* than the direct path, so it is not time-locked there and passes the S2 independent
    onset rule.
  - A real trunk reflection would not reach those nodes. This is a stand-in limitation, about 1 in
    300 seeds.
  - It stays as an honest nightly signal rather than being hidden.
- **The successor host starts entity tracking from scratch, and S2 on real clips is not behind the
  host seam.** These are open items in the stream notes. I agree with them.
- **Clippy style lints** in `onset_clustering` are style only, as the stream says.

## Files changed in review

- `minimappr-core/src/bird_entity/absorption_geometry.rs`: negative-distance guard and test.
- `minimappr-core/src/wildlife_tracking/species_log_odds.rs`: NaN evidence guard and test.
- `minimappr-core/src/mesh/call_attribution.rs`: server-equivalent window context, lazy budgeted
  rerun.
- `minimappr-core/src/mesh/call_pipeline_types.rs`: `ClipSetLocalizer::masked_cell_fraction`.
- `minimappr-core/src/mesh/config.rs`: `MeshAttributionConfig`, `max_retained_call_outcomes`,
  documentation for the collinear threshold.
- `minimappr-core/src/mesh/host_entity_tracking.rs`: capped outcomes, wire-id pruning, finite-fix
  guard, per-cluster reporting node, unit test.
- `minimappr-core/src/mesh/host_state_machine.rs`: per-cluster reporting nodes for multi-cluster
  windows.
- `minimappr-ingest-sidecar/src/attribution_oracle.rs`: emits `reason` and `mask_seeds`.
- `minimappr-mesh-sim/src/entity_metrics.rs`: `echo_calls_surfaced`, non-vacuous invariants.
- `minimappr-mesh-sim/src/scenarios/entity_scenarios.rs`: config path.
- `tests/attribution_trigger_parity_cases.py`, `tests/wildlife_tracking_parity_cases.py` and both
  `.npz` goldens: new cases and outputs (regenerated from Python).

## Test results (after the fixes)

- **`cargo test --workspace`:** 535 passed, 0 failed, 4 ignored. That is the stream's 532, plus 3
  new unit tests:
  - absorption guard;
  - NaN log-odds guard;
  - host outcome cap and finite-fix guard.

  The 24 mesh PR-seed tests are included.
- **`pytest tests/ --ignore=tests/test_soak_harness.py`:** 2054 passed, 18 skipped.
  - My first full run under heavy machine load had 21 failures in the onset, server-gate and trigger
    scene tests, several with `FileNotFoundError`.
  - All of them pass when rerun on their own (16/16 in the four files), and a full rerun was clean.
  - No Python source changed.
- **`pytest -k parity`:** 45 passed, 1 skipped. The new and regenerated goldens:
  - wildlife: 81 cases, including the degenerate-covariance case;
  - attribution: the batch now holds 166 windows and every branch.
- **Entity scenario sweeps** (release), 60 seeds from 5000: all pass except the known
  `same_band_two_species` seed 5007.
- **300 seeds from 10000:** the same as 7a8f093, plus `flock_echo` 10200, which only the new echo
  metric catches (see "Left as is").
- **Cross builds:** `cargo build -p minimappr-core --lib` succeeds for `wasm32-unknown-unknown`,
  `aarch64-linux-android` and `aarch64-apple-ios`.
