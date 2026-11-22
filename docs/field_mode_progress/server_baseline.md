# Server baseline — entity metrics on the synthetic harness

Measured 2026-11-22 on `fm/replay-metrics` (base `field-mode` @ fb0c0d3), before any
server-track algorithm change. Metric definitions: `bundle_format_v2.md` §5.

## Method

- **Bundles.** No field bundles exist (`tests/data/calibration/` is empty), so the
  baseline uses synthetic bundles only:
  1. *Existing harness* — `tests/test_calibration_harness_synthetic.py`: 2 tetra nodes
     (`node-a` at origin, `node-b` at (12, 4, 0) m), one 900 Hz tone 0.7 m from `node-a`,
     built through the production `CalibrationPipeline` + bundle writer, stub classifier
     (always `calib_tone`).
  2. *Entity scenarios* — new `minimappr/sim/synthetic_entity_scenarios.py`, rendered
     straight to ground-truth v2 bundles: 3 tetra nodes (5 cm regular tetrahedra) at
     (0,0,0), (20,0,0), (10,18,0) m; 16 kHz; free-field point sources with exact FFT
     fractional delay and 1/r spreading; white noise floor 0.002 RMS; 0.5 s linear-FM
     calls (robin 2.0→2.6 kHz at (6,12,2) m, tanager 2.9→3.5 kHz at (18,9,3) m).
     Scenarios: `one_bird_two_bouts` (bouts at 1 s and 7 s), `two_species_overlap`
     (robin + tanager overlapping by 0.2 s, twice), `one_bird_trunk_echo` (two bouts,
     each with an image source at (−8,24,2) m, −8 dB, marked `is_echo_source`).
     Classifier: `DominantFrequencySpeciesClassifier` (labels the species whose band
     holds the spectral peak of the window it is given — an oracle whenever the
     window actually contains the call).
- **Pipeline.** `build_fusion_for_bundle` → `replay_bundle` (Python ingest, 1024-sample
  frames, default settings plus `trigger_rms=0.006`, `classification_window_seconds=1.0`,
  as in the existing harness) → poll detections until stable for 3 s → `evaluate_bundle`.
- **Command.** `python scripts/measure_replay_entity_baseline.py --seeds 0 1 2 --radii-m 15 5`
  (≈ 45 s total). Seeds 0–2 gave identical numbers (the noise floor is far below the
  calls), so one row per scenario is shown. Entity metrics are scored at two association
  radii: 15 m (default) and 5 m.

## Results

Per-event (v1) metrics:

| Scenario | Calls | Detections | Tracks | Event recall | Label acc. | Median / P90 pos. error |
| --- | --- | --- | --- | --- | --- | --- |
| existing harness (1 tone, 2 nodes) | 1 | 1 | 1 | 1.00 | 1.00 | 6.0 / 6.0 m |
| one_bird_two_bouts | 2 | 2 | 1 | 1.00 | 0.00 | 10.5 / 12.9 m |
| two_species_overlap | 4 | 3 | 2 | 1.00 | 0.25 | 13.0 / 18.3 m |
| one_bird_trunk_echo | 2 (+2 echo) | 2 | 1 | 1.00 | 0.00 | 10.5 / 12.9 m |

Entity metrics (radius 15 m | radius 5 m):

| Scenario | Localized-and-classified recall | Label swap rate | Duplicate entities | Echo promotions | Unassociated localized entities |
| --- | --- | --- | --- | --- | --- |
| existing harness | 1.00 \| — | n/a | 0 | 0 | 0 |
| one_bird_two_bouts | 0.00 \| 0.00 | n/a (no rival) \| n/a | 0 \| 0 | 0 \| 0 | 0 \| 1 |
| two_species_overlap | 0.25 \| 0.00 | 0.00 (0/3) \| n/a | 1 \| 0 | 0 \| 0 | 0 \| 2 |
| one_bird_trunk_echo | 0.00 \| 0.00 | n/a \| n/a | 0 \| 0 | 0 \| 0 | 0 \| 1 |

(The existing harness was scored at the default 15 m only.)

## Findings

1. **Fixes collapse onto the first-triggering node.** In every entity scenario the
   detection's `position_m` sits within ~3 cm of a node centre (`gcc_phat`,
   `range_projection_mode=range_refined`, `range_observability≈1.0`), so position error
   equals the bird-to-node distance (13.5 m for the robin from `node-a`). A one-node
   bundle gives the identical result, so the other two nodes add nothing: no
   cross-node TDOA reaches the solver on this replay path. Probes with tones (900 Hz)
   and wide sweeps (1–4 kHz) and with the existing harness's 2-node geometry gave the
   same collapse; a source 0.7 m from a node gives 0.7 m error for the same reason.
   (The existing harness lands 6 m off rather than collapsing; why it differs was not
   investigated.) The synthetic audio's per-channel and cross-node delays were checked
   against geometry (sub-0.01-sample agreement), so this is the pipeline, not the
   renderer. This is the main thing W2-onset-srp has to beat.
2. **A collapsed fix degrades the classifier input.** When the fix collapses, the
   window handed to the classifier carried ~0.001 RMS against ~0.1 RMS of call at the
   mics, and the oracle classifier returned `background` for 6 of the 7 detections. Likely the
   beamformed render steered at a point 3 cm from the array (near-field gain
   normalisation); not confirmed. Localization failures thus also cost labels, which
   is exactly the coupling `min_localized_and_classified_recall` measures.
3. **Overlap loses calls.** In `two_species_overlap` the tanager never produced a
   detection (3 detections for 4 calls, none labelled tanager); robin bout 2 was split
   into a second track, giving 1 duplicate entity at 15 m.
4. **Echoes are not promoted today** — but only because fixes collapse to the node and
   never reach the image-source position; the 0 is not evidence of echo rejection.
5. **Radius sensitivity.** At 15 m the collapsed node-a fixes still pass the gate
   (13.5 m < 15 m), which inflates recall and duplicate counts; at 5 m every scenario
   fix is unassociated. Gates for these scenarios should use ≤ 5 m (or the bird-to-
   nearest-node distance minus a margin) so a fix collapsed onto a node cannot count.

## Suggested starting gates (to tighten once W2 lands)

For the synthetic entity scenarios at `association_radius_m = 5`: today's values are
recall 0.0, swap rate n/a, duplicates 0, echo promotions 0, so any positive recall is an
improvement. Proposed W2 acceptance: `min_localized_and_classified_recall ≥ 0.75`,
`max_label_swap_rate = 0`, `max_duplicate_entities = 0`, `max_echo_promotions = 0`, and
median position error no worse than today (≤ 13 m) — ideally < 3 m inside the layout.

## Limits

Free-field, single shared clock, no drift, absorption, ground reflection or real calls;
the classifier is a frequency oracle. These numbers measure the localization →
association → tracking chain, not BirdNET. Re-measure on Rust-simulator
(`minimappr-sim render-bundle`) and speaker-playback bundles as they arrive; both load
through the same `load_bundle` / `evaluate_bundle` path.
