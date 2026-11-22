# W3-attribution: SNR-gradient + LOCO species attribution, relative trigger (server track S5)

Branch `fm/attribution` (from `field-mode` @ 2b1f0c6). Flags, all off by default (off = W2
behaviour exactly):

- `onset_attribution_enabled` (needs `localization_onset_clustering_enabled`)
- `trigger_noise_floor_relative_enabled` (separate commit)

Measured 2026-11-22.

## What was built

| Piece | File | Notes |
| --- | --- | --- |
| Per-cluster level evidence | `core/onset_cluster_attribution_evidence.py` | For every cluster (accepted, tentative, echo) and node: band SNR above that node's own floor at each led onset's predicted arrival and in that onset's band (reuses `node_band_snr_db`); the arrivals themselves (span sample + band); a 1/r spreading fit of the source level with an RMS residual. Nodes under `audible_snr_db` are censored (excluded from the fit, penalised only if the fit says they should clearly have heard it). Built inside the clustering worker thread, only when the flag is on |
| Decisions (pure) | `core/onset_cluster_attribution.py` | Interferers = other non-echo, non-co-located clusters audible on the classifying node with an onset inside the window (or up to `loco_mask_max_seconds` before it). Onsets within the echo lag of a target onset, in the same band, are the target's own note/echo and are skipped. **Tier 1 SNR gradient:** target out-reads every interferer on the classifying node by `dominance_margin_db` *and* both clusters fit spreading within `spreading_tolerance_db` *and* the measured dominance matches the positions' prediction. Single-array windows never settle here (no cross-node gradient). **LOCO decision:** rerun's top species >= `loco_min_confidence`, leads the runner-up by `loco_label_margin`, and if the label changed the old winner must have collapsed (<= `loco_collapse_ratio` of its score). Otherwise unsettled |
| Masking | `core/onset_cluster_masking.py` | STFT (20 ms, 75 %); interferer regions start at each onset arrival (minus pre-roll), in its band +/- padding, and run while the band stays `loco_mask_active_snr_db` above the floor (release 50 ms, cap 1 s). Cells are replaced by noise at the window's own per-bin floor (20th percentile, Rayleigh-corrected). The fill is the STFT of real white noise shaped per bin, because independent random cells are not a consistent STFT and the inverse transform drops ~4 dB of them (caught by the "at the floor" unit test). Never zeros |
| Classifier-worker seam | `core/onset_cluster_attribution_stage.py` + 4-line hook in `FusionNode._classify_and_assemble` | Runs after the cluster's window is classified. Masking (`asyncio.to_thread`) and the rerun (`ClassificationOrchestrator.rerun_classifier_on_prepared_signal`, worker thread, same timeout) are off the event loop. Rerun budget is per analysis window (shared by its sibling clusters), `loco_max_reruns_per_window` (4). Unsettled / cap hit / LOCO off -> label `unknown`, confidence 0, position kept, original scores kept as `unattributed_scores` in the `onset_attribution` feature |
| Band-distinct persistence promotion | `OnsetSrpLocalizationStage._band_distinct_tentative_clusters` | Single array only: a cluster held *only* by persistence (`not_persistent`) whose band overlaps no accepted cluster's band (< 0.5, the echo check's own same-band rule) cannot be an echo or sidelobe of it (both repeat the source's spectrum), so it gets its own product and attribution labels it. Sub-flag `onset_attribution_promote_band_distinct_tentative` (on under the master flag) |
| Partial-binding levels | `core/track_level_gradient_constraint.py`, `track_partial_constraint_evidence.py`, `assembly_track_resolution.py` | The onset stage writes `onset_cluster_node_band_snr_db` (+ the audible threshold) into classification features. When >= 2 registered nodes carry readings, `MeasuredLevelGradientConstraint` (1/r fit of the measured gradient, RMS misfit / `snr_level_margin_db`, hard bound like the stand-in) replaces the W2 "reporting node ~ nearest" stand-in |
| Relative trigger | `core/ingest_relative_trigger.py`, hook in `IngestProcessor` | Per-node floor of frame RMS in dB, asymmetric one-pole (fall 0.5 s, rise 20 s), warm-up 1 s, clamp 1e-5. A frame triggers when it is `trigger_noise_floor_relative_snr_db` (8) over the floor *before* the frame, OR'd with the absolute `trigger_rms` gate (every window that opens today still opens). O(1) per frame, no I/O |

Masked audio exists only in local numpy arrays inside the worker thread; the
detection's `classification_signal` (the snippet source) stays the original
window. A unit test monkeypatches `open` (write modes), `os.open`, `np.save` and
`np.savez` to raise and runs a full LOCO attribution.

### Config keys (restart-required; one `# --- Field Mode W3-attribution ---` block per file)

`onset_attribution_enabled` (False), `..._audible_snr_db` (3), `..._dominance_margin_db` (10),
`..._spreading_tolerance_db` (6), `..._loco_enabled` (True), `..._loco_max_reruns_per_window` (4),
`..._loco_min_confidence` (0.3), `..._loco_label_margin` (0.1), `..._loco_collapse_ratio` (0.5),
`..._loco_stft_frame_seconds` (0.02), `..._loco_noise_floor_percentile` (20),
`..._loco_mask_active_snr_db` (6), `..._loco_mask_release_seconds` (0.05),
`..._loco_mask_max_seconds` (1.0), `..._loco_mask_band_padding_hz` (150),
`onset_attribution_promote_band_distinct_tentative` (True);
`trigger_noise_floor_relative_enabled` (False), `trigger_noise_floor_relative_snr_db` (8),
`trigger_noise_floor_rise_seconds` (20), `trigger_noise_floor_fall_seconds` (0.5),
`trigger_noise_floor_warmup_seconds` (1), `trigger_noise_floor_minimum_rms` (1e-5).

Registered in `config.py` (fields, env `MINIMAPPR_<KEY>`, validation,
`OnsetClusterAttributionConfig` / `RelativeTriggerConfig` + builders), `settings_store.py`,
`main.py` (GET config), `core/config_groups.py` (attribution keys in Localization next to
the onset keys; trigger keys in Gates), `core/pipeline_graph.py` (node
`cls:site:onset_attribution` with tier counters and mean rerun time; relative-trigger
param on each RMS gate). FusionMetrics: `onset_attribution_{window,single_cluster,
snr_gradient_settled,loco_settled,unlabelled,loco_rerun,loco_rerun_cap_hit,
promoted_tentative}_count`, `onset_attribution_loco_rerun_{total,max}_ms`.

## Results

### Attribution, simulator scenes (onset clustering on, 1 s reporting window)

`python scripts/measure_attribution_scenarios.py --runs 2` (both runs identical). Oracle
classifier = spectral peak in a species band (`DominantFrequencySpeciesClassifier`).

| Scene | Attribution | Dets | Label acc. | L&C recall | Swaps | Dup | Echo | Tiers (per detection) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_tetra_overlap (1 node) | off | 2 | 0.50 | 0.50 | 0 | 0 | 0 | — |
| single_tetra_overlap (1 node) | **on** | **4** | **1.00** | **1.00** | 0 | 0 | 0 | LOCO 4 |
| overlap_two_species | off / on | 5 / 5 | 1.00 / 1.00 | 1.00 / 1.00 | 0 / 0 | 0 / 0 | 0 / 0 | gradient 1, LOCO 4 |
| same_band_pair | off / on | 4 / 4 | 1.00 / 1.00 | 1.00 / 1.00 | — | 0 / 0 | 0 / 0 | gradient 2, LOCO 2 |
| same_band_two_species (new) | off / on | 4 / 4 | 1.00 / 1.00 | 1.00 / 1.00 | 0 / 0 | 1 / 1 | 0 / 0 | single 1, gradient 1, LOCO 2 |
| trunk_echo | off / on | 2 / 2 | 1.00 / 1.00 | 1.00 / 1.00 | — | 0 / 0 | 0 / 0 | single 2 |
| faint_far_bird | off / on | 4 / 4 | 1.00 / 1.00 | 1.00 / 1.00 | — | 0 / 0 | 0 / 0 | single 3, LOCO 1 |

- **single_tetra_overlap** was not what the W2 notes assumed. The accepted cluster was
  the *tanager* (correct bearing, correct label); the *robin* (42 dB band SNR against the
  tanager's 10-14 dB) was the cluster held tentative by persistence (a sweep has one
  onset, a trill many) and vanished. Promotion of the band-distinct hold surfaces the
  robin; LOCO then settles both windows (robin window: tanager masked; tanager window:
  the robin's still-sounding tail masked). Recall 0.5 -> 1.0, accuracy 0.5 -> 1.0.
- **Multi-node scenes:** no change in any metric. With per-cluster nearest-node
  classification the server already gets a gradient for free, and the oracle labels each
  window correctly; attribution confirms (gradient or LOCO) rather than corrects. No
  unlabelled detections remain after the shared-onset fix (below).
- **faint_far_bird, first attempt:** one cardinal detection came out `unknown`. The window
  held a second accepted cluster 4.7 m away (z = 0.3 m, 10 dB down, same band, onset 10 ms
  behind: a ground image W2 did not flag as an echo). Masking it erased the cardinal, so
  the rerun abstained and the cluster was left unlabelled, correctly by the rules but at a
  real cost. Fix (2nd commit): interferer onsets within the echo lag of a target onset in
  the same band are the target's own note and are never masked.
- **same_band_two_species** (new scene, the test strategy's "loud robin + quieter tanager
  in 2-5 kHz, 25 m apart, one window"): the duplicate is a W2 clustering artefact present
  with attribution off too. Harder variants (robin 92-96 dB vs tanager 80-84 dB) lose the
  tanager *in clustering* (its onset frames are swamped by the robin's ongoing note, so it
  never forms its own cluster); attribution cannot label a cluster that does not exist.

### Rerun cost

- Masking: ~5 ms per 1 s window at 24 kHz (two seeds), worker thread.
- Rerun = one classifier call on the masked 1 s window. Oracle: < 1 ms. BirdNET v2.4
  (local TFLite, padded to 3 s): ~50 ms per call measured on the dev Mac while another
  replay was running. So a capped window costs at most 4 x ~55 ms of worker-pool time.
- Reruns per detection on the scenes: 0-1 (4 reruns for 4 detections on the single
  tetra, 4 for 5 on overlap). The cap (4 / window) was never hit on the scenes; it is
  exercised by a unit test.

### Relative trigger, faint_far_bird

`python scripts/measure_relative_trigger_scenarios.py --only faint --trigger-rms 0.001`
(0.001 is the env default; the scene's own 0.00015 was hand-set to ~3x its floor).

| trigger_rms | Relative trigger | Dets | L&C recall | Far pewee found |
| --- | --- | --- | --- | --- |
| 0.001 | off | 2 | 0.33 | no |
| 0.001 | **on** | 4 | **1.00** | yes |
| 0.00015 (hand-tuned) | off / on | 4 / 4 | 1.00 / 1.00 | yes / yes |

With the dataclass default `trigger_rms = 0.015` (all scenes, attribution off,
`--trigger-rms 0.015`), the absolute gate opens nothing on six of the seven scenes (one
`unknown` detection on overlap). The relative trigger brings back exactly what each scene
gives with its hand-tuned threshold: faint 0 -> 4 dets (recall 1.0), overlap 1 -> 5 (1.0),
same_band_pair 0 -> 4 (1.0), same_band_two_species 0 -> 4 (1.0), trunk_echo 0 -> 2 (1.0),
single_tetra 0 -> 2 (0.5, the W2 figure without attribution). No duplicates or echo
promotions appear from the extra triggers on any scene.

## Tests

- `tests/test_onset_cluster_attribution.py` (13): gradient settles / refuses (interferer
  louder, levels not fitting positions, single array), interferer audibility + window,
  shared-onset skip, LOCO settle with collapsed label, LOCO refusals (abstained, ambiguous,
  original survived, not confident), masking keeps the target within 1 dB and refills the
  interferer region within 3 dB of the noise floor (never zeros), stale pre-window notes
  not masked, evidence levels/arrivals/spreading residual, promotion rule, defaults.
- `tests/test_onset_cluster_attribution_stage.py` (7): LOCO relabels a shared window
  (spectral-peak oracle), original signal kept as the snippet source, per-window rerun cap
  (beyond it: `unknown`), unsettled -> unlabelled, **no disk writes**, untouched single-
  cluster windows, failed classification skipped, measured level constraint picks the
  right track and needs >= 2 nodes.
- `tests/test_onset_attribution_scenarios.py` (5, simulator): single-tetra both birds
  labelled; multi-node scenes keep recall >= 0.75, zero swaps, zero echo promotions.
- `tests/test_ingest_relative_trigger.py` (5), `tests/test_relative_trigger_scenarios.py`
  (1, simulator): faint bird found only with the relative trigger at trigger_rms 0.001.

## Harness fix (affects W2's numbers too)

`replay_and_score_scenario` stopped collecting once detections were unchanged for 3 s,
and `FusionNode.stop()` joins each queue for only 5 s. A late window still localizing
(buffer wait + clustering) was silently lost, so the same scene scored recall 0.5 or 1.0
depending on machine load (onset_overlap_two_species flipped between runs). It now
drains the localization, classification and rules queues (`drain_fusion_queues`, 120 s
cap) before collecting. The W2 gate's scenes all still pass.

## Open issues

- **Oracle limits the evidence.** The spectral-peak oracle returns one label; the plan's
  LOCO evidence (BirdNET: tanager 0.82 -> 0.04, robin stays 0.88) needs a multi-label
  classifier. The collapse rule is implemented and unit-tested with multi-label scores, but
  no simulator scene shows a corrected *swap*. A band-energy multi-label oracle (or
  BirdNET on sim bundles) would make a sharper gate. W3-birdnet-cutover owns BirdNET tuning.
- **Clustering, not attribution, loses a quiet same-band second bird** when its onsets sit
  under a louder ongoing note (see same_band_two_species variants). W2 / W4 port scope.
- **Ground images as accepted clusters** (faint scene): handled for attribution by the
  shared-onset rule, but W2's echo check still accepts them as separate clusters.
- **Partial binding on the onset path** rarely fires: onset products are localized
  (2d/full_3d) and go straight to `tracker.update`. The measured-level constraint applies
  to any detection carrying `onset_cluster_node_band_snr_db` without a full fix; the omni
  scanner and legacy paths do not carry per-node levels yet.
- **Rust ingest sidecar** has its own `trigger_rms` gate (`ingest_sidecar_runtime.py`);
  the relative trigger covers the Python ingest path only.
- `unknown` detections from unsettled attribution are persisted with a position (the
  plan's "position without a species"); the entity model (W3-entity) decides how to show them.

## Deviations

- **Band-distinct persistence promotion** changes which single-array clusters reach the
  pipeline. It is not in the S5 text, but without it the single-tetra case had nothing to
  attribute (the second bird never became a candidate). It is gated by the attribution
  flag and its own sub-flag, and applies only to persistence holds whose band cannot be an
  echo of an accepted cluster.
- **Co-located clusters** (within one clustering radius) are never treated as interferers
  of each other: masking one would erase the other.
- **Classification runs on the per-cluster beamformed render of the nearest node**, so
  LOCO masks that rendered + preprocessed signal (same length and timeline as the window;
  the preprocessor is filters only) instead of re-rendering masked raw channels. This avoids
  a second beamform and an extra label upsert per rerun.
- Config blocks: the trigger keys sit in the Gates group of `config_groups.py` and in the
  RMS-gate params of `pipeline_graph.py`, so those two files carry a second small labelled
  block next to the main one.
- `tests/test_onset_srp_scenarios.py` now uses the shared `tests/sim_scenario_binary.py`
  helper (third simulator-gate file).
