# W2-onset-srp: onset-weighted short-frame SRP with spatial clustering (server track S2)

Branch `fm/onset-srp` (from `field-mode` @ bca0407). Flag: `localization_onset_clustering_enabled`
(default `false` = legacy behaviour). Measured 2026-11-22.

## Step 0: why every fix sat on the first-triggering node

Wave 1 (`server_baseline.md`) saw every replay fix within ~3 cm of a node and the classifier
receiving audio ~100x quieter than at the mics. This is **not a replay-harness artefact**. The
harness feeds node positions, timestamps and the site frame correctly. Four separate pipeline
causes stack up:

1. **Compact-array solver collapse (real bug, fixed unflagged in fb0fa31).** With one 5 cm
   tetra, four mics give only three independent TDOAs, and a 3-D point always fits three
   numbers exactly. `solve_cartesian_tdoa` therefore settled in a near-field basin a few
   centimetres from the array. Neither radial-refinement trigger fired there (range < 2
   apertures, small Fisher radial std), so the point was published as `range_refined` with
   range observability 1.0. A 400 m source with ideal TDOAs landed at 6 cm. This is why the
   fixes collapsed onto the node, and onto the 8-sensor centroid in the old 2-node harness
   (the "6 m off" case).
   *Fix:* `core/far_field_model_selection.py`. For compact arrays (aperture <= 0.5 m) whose
   solve lies within 4 apertures of the centroid, the point is replaced by the best plane-wave
   projection at the far-field default range, unless the near fit beats it by more than one
   interpolated sample (RMS). The result is `range_asymptotic`, which becomes
   `range_bearing_projected`, as the range contract intends. Multi-node layouts and near-field
   fixtures outside the zone are unchanged.
   It goes in unflagged because the old answer claimed an observable range that the data did
   not contain. Two tests had only passed vacuously or because of the collapse, and were
   updated:
   - `test_rust_tdoa_solve::test_bridge_applies_confidence_cap_when_range_unobservable`: its
     `if unobservable:` assertions never ran.
   - `test_phase2_localization_soundscape` near-field cases at 1.6 m and 4.6 m: their
     tolerance was about the source range, so they passed only because the fix collapsed onto
     the array.
2. **Causal trigger window (design limitation, fixed by the flag).** `event_time_ns` is the
   end of the triggering node's frame minus half the 80 ms window. So the localization window
   ends where the trigger frame ends. A node 20 m farther away has not heard the call yet
   (58 ms later), its energy sits at the noise floor, and energy selection drops it (cross-node
   admission is off by default and would fail its relative-energy floor anyway).
3. **Coverage race (live and replay).** `_await_window_coverage` returns as soon as
   `min_sensors_for_2d` sensors are covered. One tetra's 4 mics satisfy that, so nodes whose
   frames arrive even slightly later are silently left out. In replay, node-a's frame k is
   ingested before node-b's and node-c's frame k, so the first bout only ever saw node-a. This
   probably also explains the live box's "18991 single-node 2D solves, 0 cross-node pairs".
4. **Classifier window ends at the trigger (root cause of the "100x quieter" audio).** The
   classification window is the trailing `classification_window_seconds` ending at
   `event_time_ns`. For a call that starts at the trigger it holds about 8 ms of fading-in
   call: RMS 0.001 against 0.1 at the mics, with the spectral peak in noise. Beamformer
   steering was not the cause.

Causes 2-4 change live behaviour by design, so they are fixed only on the flagged path (below).
The flag-off path keeps them.

**A separate lead checked (from W2-core-solver):** `cartesian_tdoa` passes residuals in
seconds, and scipy's absolute `gtol=1e-9` stops LM after 4-6 evaluations (status 1). Confirmed.
It is **not** the cause of the collapse: with residuals rescaled to samples, LM converges
properly (status 2) and *still* lands at 6 cm for the 400 m ideal-TDOA case, and at 0.36 m /
0.64 m for a 13.9 m source (noise-free / 3 us noise). Implications for the core-solver stream:

- (a) The step-0 guard changes Python output for single-tetra solves, so the `.npz` parity
  goldens need regenerating at merge.
- (b) If the gtol fix lands, converged compact-array collapses can sit at 0.4-0.6 m, outside
  this guard's 4-aperture (0.2 m) zone, so the zone should be revisited together with that fix.

### Baseline after step 0 (flag off)

Same command as `server_baseline.md`:
`python scripts/measure_replay_entity_baseline.py --seeds 0 1 --radii-m 15 5`.

| Scenario | Detections | Label acc. | Median / P90 err. | L&C recall @15 / @5 m |
| --- | --- | --- | --- | --- |
| one_bird_two_bouts | 2 | 0.00 | 44.9 / 51.5 m | 0.00 / 0.00 |
| two_species_overlap | 3 | 0.25 | 40.2 / 50.4 m | 0.00 / 0.00 |
| one_bird_trunk_echo | 2 | 0.00 | 45.0 / 51.5 m | 0.00 / 0.00 |

Fixes are now honest bearings at the 50 m far-field range rather than collapsed points. The
median error therefore rises from about 10 m (the bird-to-node distance) to about 45 m. Recall
@15 m falls from the inflated 0.25 to 0. Labels are still `background`, because cause 4 is
unchanged on this path.

## Step 1: the onset-SRP stage

| Module | Role |
| --- | --- |
| `core/onset_detection.py` | 32 log sub-bands, 10 ms hop (frames sized so each band holds >= 2 bins). Positive spectral flux counted only in active bands and beyond each band's own rise-noise. Onsets are peaks above median + 4 MAD. Each onset's band is the sub-bands that *rose* at that onset, so a note starting under another bird gets its own band. No absolute level anywhere |
| `core/onset_frame_correlation.py` | Per onset: an 80 ms template on the node's loudest channel against delay-compensated search windows `[t0 - tau_max, t0 + L + tau_max]` on every other sensor. PHAT is band-limited to the onset band, with Knapp-Carter ML soft weights q^2/(1+2q) from per-sensor noise floors (percentile bias corrected to the mean). Frames narrower than 300 Hz RMS bandwidth use the correlation envelope for SRP on cross-node pairs; fine structure is kept for LM votes, capped at half a carrier period. Compact pairs always keep fine structure |
| `core/onset_srp_grid.py` | Direction grid (single compact array) or polar spatial grid (40 log rings, 1-300 m, plus a far ring, az 2 deg, el -30..60 deg) around the layout centroid. SRP sums full pair curves at the predicted lags; peaks come from greedy NMS |
| `core/onset_srp_clustering.py` | Orchestration: onsets per node, frames, per-frame SRP peaks (weight = onset strength x height x contrast), greedy weighted clustering (radius 3 m + 0.2 x range; 10 deg for bearings), echo checks, refinement |
| `core/onset_cluster_refinement.py` | Per-pair TDOA votes near the cluster seed (weighted median). Multi-node: local LM from the seed (soft-L1) with Jacobian covariance. Single array: `solve_cartesian_tdoa` (reused), cone rescaled to the far-field range. LM output is kept only if it holds at least 0.9 of the cluster's SRP score |
| `core/onset_cluster_echo_checks.py` | (1) Independent onsets: a secondary cluster not led by onsets of its own, or whose onsets trail a stronger cluster's by 3-30 ms in the same band, is an echo. (2) Multi-node level consistency on per-node noise-normalised band SNR, nearest vs farthest node with an 8 dB margin; failure means tentative, never deletion. (3) Persistence (>= 2 independent onsets), applied to single-array windows only. The strongest cluster in a window is always accepted |
| `core/onset_srp_localization_stage.py` | FusionNode seam. Analysis span `[event - 0.3 s, event + 1.2 s + max inter-node delay]`, waited for on every node with a non-blocking poll (fixes cause 3). Clustering runs in `asyncio.to_thread`. One LocalizedCandidate per accepted cluster, with timestamp = the cluster's onset. Its classification window starts at that onset on the cluster's nearest node (fixes cause 4). The candidate goes through the existing `_build_localization_branch` gates and haircut. A memory of consumed onsets stops overlapping spans from re-surfacing notes |
| `core/onset_srp_types.py` | Shared value types |

Config: 33 keys in one `# --- Field Mode W2-onset-srp ---` block in each of `config.py`
(fields, env, validation, `OnsetSrpClusteringConfig` + `onset_srp_clustering_config()`),
`settings_store.py`, `main.py`, `core/config_groups.py` (localization group), and
`core/pipeline_graph.py`. The pipeline graph gets a node `loc:site:onset_srp` with
window/cluster metrics, backed by four new `FusionMetrics` counters.

## Results

### Simulator scenes (Rust `minimappr-sim`, 3 tetra nodes at 24 kHz unless noted)

Scenario TOMLs: `minimappr-sim/scenarios/onset_*.toml`.
Command: `python scripts/measure_onset_srp_scenarios.py --reporting-windows 30 1`.
Entity radius 5 m; events farther than `far_field_distance_m` are scored on bearing (15 deg).
Echo events are generated from the simulator's point reflectors
(`minimappr/sim/sim_scenario_scoring.py`). Classifier: frequency oracle built from each scene's
generators.

| Scene | Window | Flag | Dets | L&C recall | Swap | Dup | Echo prom. | Unassoc. | Label acc. | Median / P90 err. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overlap_two_species | 1 s | off | 4 | 0.00 | — | 0 | 0 | 4 | 0.50 | 23.4 / 28.1 m |
| overlap_two_species | 1 s | **on** | 5 | **1.00** | 0.00 | 0 | 0 | 0 | 1.00 | **0.65 / 0.80 m** |
| trunk_echo | 1 s | off | 2 | 0.00 | — | 0 | 0 | 2 | 0.00 | 40.4 / 58.5 m |
| trunk_echo | 1 s | **on** | 2 | **1.00** | — | 0 | **0** | 0 | 1.00 | **0.44 / 0.57 m** |
| faint_far_bird | 1 s | off | 3 | 0.00 | — | 1 | 0 | 1 | 0.00 | 56.3 m |
| faint_far_bird | 1 s | **on** | 4 | **1.00** | — | 0 | 0 | 0 | 1.00 | **2.2 m** |
| same_band_pair | 1 s | off | 2 | 0.00 | — | 0 | 0 | 2 | 0.00 | 50.8 / 53.9 m |
| same_band_pair | 1 s | **on** | 4 | **1.00** | — | **0** | 0 | 0 | 1.00 | **0.95 / 1.16 m** |
| single_tetra_overlap (1 node) | 1 s | off | 2 | 0.00 | — | 0 | 0 | 2 | 0.00 | — |
| single_tetra_overlap (1 node) | 1 s | on | 2 | 0.50 | 0.00 | 0 | 0 | 0 | 0.50 | — |
| overlap_two_species | 30 s | off / on | 3 / 2 | 0.00 / 0.50 | — / 0.00 | 0 / 0 | 0 / 0 | 3 / 0 | 0.50 / 1.00 | 42.5 / 0.66 m |
| trunk_echo | 30 s | off / on | 2 / 1 | 0.00 / 0.50 | — | 0 / 0 | 0 / 0 | 2 / 0 | 0.00 / 1.00 | 40.4 / 0.28 m |
| faint_far_bird | 30 s | off / on | 2 / 3 | 0.00 / 0.67 | — | 0 / 0 | 0 / 0 | 1 / 0 | 0.00 / 1.00 | 56.3 / 2.2 m |
| same_band_pair | 30 s | off / on | 1 / 2 | 0.00 / 0.50 | — | 0 / 0 | 0 / 0 | 1 / 0 | 0.00 / 1.00 | 41.6 / 0.92 m |
| single_tetra_overlap | 30 s | off / on | 1 / 1 | 0.00 / 0.25 | — / 0.00 | 0 / 0 | 0 / 0 | 1 / 0 | 0.00 / 0.50 | — |

- With the default 30 s reporting window, `ReportingFusionPolicy` merges a node's second
  same-label bout into the first (`report_window_duplicate`). That alone halves recall on
  two-bout scenes for either flag; it is W3-entity's scope. The 1 s rows isolate localization
  plus classification.
- **faint_far_bird:** the ingest trigger is still an absolute `trigger_rms` gate (pre-existing,
  not in this stream's scope). The scene sets it about 3x the rendered noise floor, or the
  faint bird never opens a window on either path. Once a window is open, the onset stage has
  no level gate. The far bird (about 70 m out) is placed within the bearing gate and gets the
  right label.
- **single_tetra_overlap:** the bearing is right, but the second bird is held tentative by
  persistence (one onset per window), and the frequency oracle labels the shared window with
  the louder trill. Separating two species on one compact array needs attribution (W3 LOCO).

### Python synthetic scenes (`synthetic_entity_scenarios.py`, 30 s reporting window, seeds 0 and 1)

| Scenario | Flag | Dets | Label acc. | Median err. | L&C recall @15 / @5 m | Swap | Dup | Echo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one_bird_two_bouts | off | 2 | 0.00 | 44.9 m | 0.00 / 0.00 | — | 0 | 0 |
| one_bird_two_bouts | on | 1 | 1.00 | 0.6-0.9 m | 0.50 / 0.50 | — | 0 | 0 |
| two_species_overlap | off | 3 | 0.25 | 40.2 m | 0.00 / 0.00 | — | 0 | 0 |
| two_species_overlap | on | 2-3 | 0.50 | 6.5-7.4 m | 0.25 / 0.25 | 0.00 | 0 | 0 |
| one_bird_trunk_echo | off | 2 | 0.00 | 34-45 m | 0.00 / 0.00 | — | 0 | 0 |
| one_bird_trunk_echo | on | 1 | 1.00 | 0.7-1.4 m | 0.50 / 0.50 | — | 0 | 0 |

Here 0.50 is the reporting-window merge of bout 2. In two_species_overlap the merge removes one
bout of each species, and one remaining tanager detection is placed between the birds.

**Gate (`tests/test_onset_srp_scenarios.py`):** flag on, 1 s reporting window, all four
multi-node scenes must reach L&C recall >= 0.75, zero swaps, zero duplicates and zero echo
promotions. Flag on must also beat flag off on overlap. Cost per 1.5 s window, 3 tetras at
16 kHz: about 0.3-0.5 s on one M2 core, off the event loop.

## Open issues

- **Reporting-window merge** (30 s, per node and label) hides second bouts. W3-entity replaces it.
- **Absolute `trigger_rms` ingest gate** still decides whether a faint bird opens a window.
  It should become noise-floor-relative (outside S2's scope).
- **Attribution:** on one compact array, or when two species share a classification window,
  the window still goes to whichever classifier label wins. W3 (SNR gradient, LOCO) builds on
  the per-cluster bands and onsets exported in `extra_classification_features`.
- **Confidence:** multi-node LM confidence is mean peak coherence x fit quality. It is low (down
  to about 0) when envelope votes widen residuals. It is not used for gating (the reportable
  confidence gate ships disabled). It needs calibration on golden bundles together with
  likelihood contours (W2-core-srp).
- **Bearing fusion** (`_maybe_fuse_multi_node_bearing`) is not applied on the onset path. Each
  window already uses every node.
- **2-D solver** (`localize_2d`, 3-sensor tier): it starts at the centroid and can collapse in
  the same way. The step-0 guard covers only the 3-D Cartesian solver.
- **Echo ground truth from the simulator** is built at scoring time from point reflectors
  (apparent source = trunk). Ground reflection is not scored as an echo.

## Deviations

- The plan says clustering runs "on the classification worker pool". It runs from the
  localization worker through `asyncio.to_thread`, the same pattern as the legacy localizer,
  under that queue's existing drop-oldest backpressure. The event loop is never blocked.
- Persistence is applied to single-array windows only. On multi-node windows the level check
  replaces it, so a second bird's single note is not held back.
- `cartesian_tdoa.py` was already over 500 lines before this stream; the guard lives in its own
  module and adds about 27 lines there.
