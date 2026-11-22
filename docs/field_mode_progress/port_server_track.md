# W4-port-server-track: server-track designs ported to `minimappr-core`

Branch `fm/port-server-track` (from `field-mode` @ 2eb0a18, "Merge review/sim-gates fixes into
field-mode"). This ports the designs that passed the S gate (`server_track_gate.md`, fixes in
`s_fixes_tracking.md` / `s_fixes_onset.md`) from the Python reference into the sans-IO core, adds
Python↔Rust parity gates, and wires them into the mesh host so Layer 2 exercises them.

The S2 onset-clustering port ran as a parallel sub-stream (`fm/port-onset`, merged here at 14ebf24).
Its notes are in `port_server_track_onset.md`. S4 (partial binding, two-threshold) is NEEDS-WORK and
is not ported.

## What was ported

| Verdict | Rust | Python reference |
| --- | --- | --- |
| S2 onset-SRP clustering + echo checks | `onset_clustering/*` (see `port_server_track_onset.md`) | `core/onset_*.py` |
| S3 perched random-walk filter + relocation snap | `wildlife_tracking/perched_filter.rs` | `track_filters_perched.py` |
| S3 relocation gate, ghost revert, revert notification | `wildlife_tracking/perched_relocation.rs` | `track_perched_relocation.py` |
| S3 concurrent-perch guard | `wildlife_tracking/perched_concurrency.rs` | `track_perched_concurrency.py` |
| S3 capped species log-odds | `wildlife_tracking/species_log_odds.rs` | `track_species_belief.py` |
| S3 species association gate | `wildlife_tracking/species_association_gate.rs` | `track_species_association_gate.py` |
| S3 wildlife path of `TrackManager.update` | `wildlife_tracking/perched_tracker.rs` (+ `nearest_neighbor_association.rs`, `tracker_config.rs`) | `tracking.py`, `track_field_mode_extensions.py`, `tracking_field_mode_methods.py`, `track_associators.py` |
| S5 SNR-gradient tier + LOCO decision | `onset_clustering/attribution_decision.rs` | `onset_cluster_attribution.py` |
| S5 attribution cascade (LOCO rerun budget; masking + rerun is a host callback, `LocoRerunHost`) | `onset_clustering/attribution_cascade.rs` | `onset_cluster_attribution_stage.py` |
| S5 noise-floor-relative trigger (fluctuation allowance, LRU cap) | `stage1/noise_floor_relative_trigger.rs` | `ingest_relative_trigger.py` |
| S6 entity record, states, bearing fans, ellipse | `bird_entity/model.rs`, `config.rs` | `bird_entity_model.py`, `entity_reporting_config.py` |
| S6 covariance-gated absorption, concurrent fixes = two birds, bearing evidence | `bird_entity/absorption_geometry.rs`, `matching.rs` | `bird_entity_absorption_geometry.py`, `bird_entity_matching.py` |
| S6 registry state machine, maintenance | `bird_entity/registry.rs`, `registry_spatial.rs` | `bird_entity_registry.py` |
| S6 reporting decision (insert / enrich / upgrade) | `bird_entity/reporting_decision.rs` | `entity_reporting_policy.py` (decision half) |
| FusionNode order: decide → track → surface → observe, fused row position, revert listener | `bird_entity/entity_pipeline.rs` (`WildlifeEntityPipeline`) | `fusion_node.py`, `entity_reporting_coordinator.py`, `entity_canonical_row_position.py` |

Not ported, on purpose:

- dormant reacquire and classifier fingerprints (not part of the verdicts; the perched and entity
  layers keep identity);
- S4 partial binding and two-threshold hidden tracks;
- the notification grace window and the API/WS projection (shell timers and geo frames);
- LOCO masking itself: the classifier rerun is a host callback (`LocoRerunHost` in core,
  `ClipSetLocalizer::classify_cluster_window` in the mesh).

Every module is sans-IO (caller nanoseconds, no clocks, threads or randomness), under 500 lines,
with settings in config structs whose defaults are the server's.

## Parity (Python is the reference)

The pattern is the existing oracle one. A Python cases module writes `.npz` goldens:
`case_inputs_json` and `expected_outputs_json`. The sidecar runs a hidden subcommand on the same
inputs. A second test re-runs Python against the goldens, so a server-side change fails there
rather than as a mysterious Rust mismatch. A shared comparator is in
`tests/parity_json_comparison.py`.

| Gate | Oracle | Cases | Tolerance (why) | Result |
| --- | --- | --- | --- | --- |
| `test_wildlife_tracking_cross_language_parity.py` | `wildlife-tracking-oracle` | 80: belief ×4, perched filter ×4, species gate ×4 (80 pairs), entity geometry ×4 (48 samples), association ×24, concurrency ×16, pipeline scenarios 8 × 3 seeds | Decisions exact: ids, association, relocation and revert events, gate verdicts, actions, change kinds, states, labels, counts. Numbers 1e-9 relative: the same closed-form f64 expressions on both sides, so only LU/eigen rounding differs. | pass; pipeline outputs bit-identical in spot checks |
| `test_attribution_trigger_cross_language_parity.py` | `attribution-oracle` | relative trigger ×4 (600 frames, 3 nodes, calls, gusts, NaN), LRU eviction (1 030 nodes), LOCO decision ×2 (160 pairs), attribution cascade batch (160 windows through the real `OnsetClusterAttributor`, one shared rerun budget) | Decisions exact: trigger booleans, eviction, tier, reasons, labels, collapsed labels, budget. Dominance compared as the reference annotates it (0.01 dB). Floors and confidences 1e-9: the one-pole recursion is contracting. | pass |
| `test_onset_clustering_cross_language_parity.py` (sub-stream) | `onset-clustering-oracle` | 6 windows | see `port_server_track_onset.md` | pass |

What the pipeline scenarios exercise (decision level, real `TrackManager` + `BirdEntityRegistry` +
`EntityReportingPolicy` in FusionNode order, `tests/wildlife_pipeline_reference.py`):

- countersinging pair 12.6 m apart, a rotating flock 10–11 m apart;
- a hop then a ghost revert, with entity fixes released;
- two species 7.2 m apart, including a weak claim off the perch and a misclassified bout on it;
- omni then localized;
- a far bird on its bearing (bearing evidence, bearing-only entities);
- expiry and step-back through maintenance;
- random mixes.

## Mesh host wiring

- `MeshConfig::entity_tracking` (`EntityTrackingConfig`, enabled by default) runs every completed
  call through `WildlifeEntityPipeline`, one pipeline per island (own frame and reference clock).
  The code is `mesh/host_entity_tracking.rs`.
  - `BirdEntityUpdate` ids are stable per bird, and still `term << 32`-unique.
  - Updates carry the entity's state, fused position, bearing fan (σ = half width / k), the contour
    of the fused covariance, and the belief.
  - Maintenance runs on the island clock, extrapolated from the last event, and emits step-backs
    and `Expired`.
  - Disabled, the host keeps the W3 one-update-per-call behaviour.
- **Multi-cluster windows.** `ClipSetLocalizer` gains `localize_clusters` (default: the single
  fix) and `classify_cluster_window` (default: none). `LocalizedCallRecord::clusters` holds every
  accepted source; `fix` stays the strongest.
- **S5 in the host.** `mesh/call_attribution.rs` runs the ported cascade per cluster:
  - the cluster's loudest node classifies;
  - LOCO goes through the localizer's classifier seam;
  - per-call rerun budget, `MeshConfig::attribution`;
  - unsettled clusters keep their position with no species.
- **Revert re-pointing (host bookkeeping beyond the server).** When a perched hop is undone, the
  call outcomes taken on that track since the hop are re-pointed to the split track's entity once
  it has one. This is the S-fixes open issue "a row stored during a reverted hop keeps the original
  track id". The core pipeline itself stays server-identical.
- **Layer 2 stand-ins** (`minimappr-mesh-sim/src/synthetic_clusters.rs`):
  - every call several clips share is a cluster;
  - a follower time-locked 3–30 ms behind a cluster in the same band at most nodes is an echo (the
    S2 independent-onset rule, with `OnsetClusteringConfig` thresholds);
  - evidence is arrival SNR per node with the ported `spreading_fit`;
  - the classifier stand-in grades scores by SNR like the server's `SnrGradedSpeciesOracle`, with
    masked regions removed.
- The relative trigger is not wired into Layer 2: Stage-1 there is an SNR threshold, not audio.
  The phone shell's Stage-1 and ingest can use `stage1::noise_floor_relative_trigger`.

## Layer 2 scenarios (`minimappr-mesh-sim/src/scenarios/entity_scenarios.rs`)

Each scenario runs the common invariants plus the entity metrics (`entity_metrics.rs`):

- duplicate entities (extra entities per bird);
- merged birds, i.e. swaps across birds (entities holding calls of more than one bird);
- label swaps (an entity showing another bird's species);
- echo promotions (entities built only from echoes).

Ten PR seeds per scenario. Each PR run takes 0.74 s for all 24 mesh tests in the debug profile
(0.6 s in release). Extra sweep: 60 seeds from 5000 per new scenario with
`nightly_mesh_seeds --seeds 60 --start-seed 5000 --scenario <name>`.

| Scenario | Setup | PR seeds (10): entities per bird / dup / merged / swaps / echo | Other assertions | 60-seed sweep |
| --- | --- | --- | --- | --- |
| `countersinging_pair` | 6-phone ring of 16 m; two cardinals 22 m apart answering at 0.5 s (one window, two clusters) and 1.6 s | [1, 1] / 0 / 0 / 0 / 0 on all | both pinpointed; healthy path (fixes ≤ 3 m gate, worst 0.06–1.29 m) | 60/60 |
| `flock_echo` | 6-phone ring of 22 m; 4 chickadees 20–28 m apart rotating with 5–7 s gaps; a trunk echo 4 ms later and 8 dB down behind each perch | [1, 1, 1, 1] / 0 / 0 / 0 / 0 | all 4 pinpointed; region check. The perched relocation gate hops one track per pair and the ghost revert undoes it; revert re-pointing keeps each call on its own bird | 60/60 |
| `asymmetric_bouts` | one robin, bout 1 score 0.25 (localize-to-track lowered to 0.2), bout 2 score 0.85 with speech on 2 of 5 phones | [1] / 0 / 0 / 0 / 0 | both bouts localized, one pinpointed entity (log-odds fuses 0.25 + 0.85) | 60/60 |
| `same_band_two_species` | loud robin (90 dB) and quieter tanager (82 dB) 24 m apart, 0.6 s apart, ×4 | [1, 1] / 0 / 0 / 0 / 0 | ≥ 2 clusters in a window; SNR-gradient labels the robin and LOCO the tanager; LOCO cap 4 per window; healthy path | 59/60 (seed 5007: Open issues) |
| `omni_to_localized_dedup` | wren 1.8 m from phone 0; 5 bouts, 2 quiet ones only phone 0 hears (no fix) | [1] / 0 / 0 / 0 / 0 | exactly 1 creation (one notification); the first quiet bout creates a heard-nearby entity that is promoted, the second is absorbed (`enrich_existing`); pinpointed | 60/60 |
| `collinear_trail` | 5 phones in single file, 8 m apart; robin 15 m to one side, 4 bouts | [1] / 0 / 0 / 0 / 0 | fixes inside their reported region; mirror-ambiguous geometry never pinpoints, and no pinpointed update lands on the mirror side | 60/60 |
| `couples` | 2 phones 6 m apart; a robin and a tanager each near one phone | [1, 1] / 0 / 0 / 0 / 0 | no fix is possible, so never pinpointed; one heard-nearby entity per bird | 60/60 |

Deviations from the test strategy rows:

- **Couples.** No stereo `SubArray` bearing exists in the mesh: DOA nodes are not in Layer 2, and
  bird bearings need the onset/DOA path on real clips. The scenario checks the entity half: one
  bird is one entity across a two-phone couple, and it is never placed.
- **Collinear trail.** The fix is not "both mirror candidates reported". The host withholds a
  mirror-ambiguous fix from the entity layer, so the bird stays heard-nearby. Drawing both
  candidates needs a two-candidate map object, which does not exist yet.
- **Same-band two species.** One shared window gives two clusters through the arrival-time
  stand-in, not clip SRP.

## Open issues

- **Collinear trails pinpointed the mirror side (found here, guarded in the host).** Before the
  guard, 3 of 10 PR seeds pinpointed the robin on the wrong side of the trail.
  - Cause: a near-collinear node set gives TDOA fixes with a mirror twin. The fix's local
    covariance cannot express the twin, so the perched filter fused the two candidates into one
    "confident" track.
  - Guard: `mesh::host_entity_tracking::node_set_is_collinear`, with
    `EntityTrackingConfig::collinear_mirror_max_aspect_ratio` = 0.2. Such fixes never place an
    entity.
  - The server has no equivalent: its arrays are 3-D tetras on a triangle. Any server deployment
    with collinear nodes has the same failure.
- **Interferer labels leak through the SNR-gradient tier** (`same_band_two_species` seed 5007, 1 of
  60, faithful to the server). A gradient-settled cluster keeps the window's full classifier answer.
  - Effect: the robin's track also holds a 0.83 tanager belief.
  - The tanager's own fix had σ 7.7 m. Its 2-dof covariance gate (9.21) and the 0.3
    species-belief floor then absorb it into the robin entity 23 m away: a merged bird.
  - Candidate fixes, each needing a server gate run:
    - drop labels claimed by another settled cluster of the same window from a gradient-settled
      cluster's evidence;
    - or require the absorbing label to be the entity's top label when the fix is more than k σ
      from the entity.
- **Revert re-pointing is host bookkeeping beyond the server.** The core pipeline matches Python,
  so the server still draws a reverted hop's row at the old track's entity (S-fixes open issue).
  The host re-points the call outcome to the split track's entity once that track binds one. The
  wire history still shows the hop: the entity moved to the other perch and then back.
- **Perched tracks start overconfident** (server open issue, ported as is). A new track's filter
  variance is the 4 m² initial variance, not the fix covariance.
- **Contours of fused positions.** Entity updates draw the contour of the bound track's fused
  covariance. The 0.52 σ contour multiplier was fitted on raw call fixes (W4-sim-gates). The fused
  covariance is floored at the 1.5 m² measurement noise, but it has not been recalibrated.
- **Not exercised in Layer 2:**
  - the relative trigger (Layer 2 Stage-1 is an SNR threshold);
  - bearing-only, far-field entities from the mesh, because the arrival-time localizer reports no
    range-projection mode. They are covered by the parity scenarios.
- **S2 on real clips is not yet behind the host seam.** `onset_clustering::localize_onset_clusters`
  is ported and parity-gated, but no `ClipSetLocalizer` decodes FLAC clips, aligns them on the
  island clock and calls it. That is the production phone localizer that `mesh_sim.md` lists as
  still to write. Layer 2 exercises the same seam (`localize_clusters`, `classify_cluster_window`,
  per-cluster `ClusterAttributionEvidence`) through the arrival-time stand-in.
- **The successor host restarts entity tracking.** After a takeover, ids continue above the
  restored highest id, but the tracker and registry start empty, so a bird heard before and after
  a host change gets a second entity.
- `bird_entity_registry.py`, the Python reference, is at 499 lines, and the Rust split
  (`registry.rs` + `registry_spatial.rs`) follows the same seam.

## Commands

```
CARGO_TARGET_DIR=<repo>/target cargo test --workspace                      # 532 passed, 0 failed, 4 ignored
CARGO_TARGET_DIR=<repo>/target cargo test -p minimappr-mesh-sim            # 24 PR-seed tests, 0.74 s (debug)
python -m tests.wildlife_tracking_parity_cases                             # regenerate S3/S6 goldens
python -m tests.attribution_trigger_parity_cases                           # regenerate S5 goldens
CARGO_TARGET_DIR=<repo>/target python -m pytest tests/ -k parity -q -p no:cacheprovider   # 45 passed, 1 skipped
cargo build -p minimappr-core --lib --target {wasm32-unknown-unknown,aarch64-linux-android,aarch64-apple-ios}
MINIMAPPR_SIM_ENTITY_DEBUG=1 MINIMAPPR_SIM_SEED=<seed> cargo test -p minimappr-mesh-sim <scenario> -- --nocapture  # per-call clusters/entities
```

Full Python suite (`--ignore=tests/test_soak_harness.py`): 2050 passed, 18 skipped. One earlier `-x` run under heavy load failed `test_onset_attribution_scenarios.py::test_single_tetra_overlap_labels_both_birds` with "Event loop is closed". It passes alone, and the full rerun was green.
