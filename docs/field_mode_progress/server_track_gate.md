# Server-track S gate: every server stream together (Wave 4 "Me: S gate")

Branch `fm/s-gate` (from `field-mode` @ 292e2dd, "Merge fm/attribution into field-mode").
Measured 2026-11-22. Machine heavily loaded by parallel agents throughout (see Cost).

Each server-track stream was measured alone, often on a branch without the others. This gate
runs them together for the first time, on one harness, and decides which designs get ported
to `minimappr-core` in W4-port-server-track.

## Verdicts

| Design (flag) | Verdict | Evidence (sim matrix: 10 scenes; real-audio fixture scene with BirdNET v2.4) |
| --- | --- | --- |
| S2 onset-SRP clustering + echo checks (`localization_onset_clustering_enabled`) | **PASS: port** | The only stream that makes anything localize and classify. Removing it from ALL-ON drops summed individual recall from 7.83 to 0.00 over 10 scenes and the median error from 0.6 m to 26 m. Real audio: median error 8.9 m → 0.03 m, label accuracy 0.75 → 1.00. Fix the three issues under "Reported, not fixed" in the port (ground-image height flip, overclaimed far range, loud near-node mislocalization). |
| S3 perched filter + relocation/ghost (`tracking_perched_filter_enabled`) | **PASS: port** | Removing it from ALL-ON gives 2 duplicates and 1 merged entity (countersinging pair: the CV tracker hops between perches 12.6 m apart). No scene got worse with it. It still merges a same-species flock rotating every 1.2 s, and it merges two species 7.2 m apart on real audio; port with a species-aware association (see Reported). |
| S3 species log-odds (`tracking_species_log_odds_enabled`) | **PASS: port, together with S3 perched** | Neutral on the sim matrix. Removing it changes no recall, swap or merge count; the oracle seldom gives confusable evidence. Two-threshold needs it (W2 finding). Its one interaction cost is that two-threshold rows surface through log-odds tracks, which is a two-threshold issue. |
| S4 partial-constraint binding (`tracking_partial_constraint_binding_enabled`) | **NEEDS-WORK: don't port yet** | Zero effect in every scene and configuration (all_on vs minus: identical on all 10 scenes). Onset products are fully localized, and tetra omni detections do not carry their DOA, so the binder never gets a candidate. The design is untested end to end. Port it once a replay scene exercises it, which needs omni/bearing-only products that carry DOA and per-node levels. |
| S4 two-threshold BirdNET (`birdnet_two_threshold_enabled`) | **NEEDS-WORK: keep off for the trial; port behind the flag after the association fix** | Sim: no swaps, but sub-display rows surface through visible tracks with their own poorer fixes (collinear trail p90 0.75 → 18.75 m; it helps produce the faint bird's one duplicate). Real audio (after the cap fix): neutral to slightly positive. Without it, individual recall is the same (0.75), accuracy 0.80 → 0.67 and p90 4.4 → 9.6 m. The sub-display rows should feed belief, not surface as rows (Reported 8). Thresholds still need real-site data (cutover open issue). |
| S5 SNR-gradient + LOCO attribution (`onset_attribution_enabled`) | **PASS: port** (LOCO optional) | Sim: removing it from ALL-ON gives a label swap and halves recall on the collinear trail (two species, overlapping bouts). This is the only swap in the whole ALL-ON−X set. The SNR-gradient tier alone prevents it: ALL-ON with `onset_attribution_loco_enabled=false` still has 0 swaps and recall 1.0 there. Real audio (after the cap fix): neutral. Individual recall is 0.75 with attribution, without it, and with LOCO off. Each BirdNET LOCO rerun costs about 175 ms on this machine (13 per replay). LOCO's value shown in W3 (single tetra 0.5 → 1.0) does not reproduce with all flags on (0.5 with and without LOCO). |
| S5 noise-floor-relative trigger (`trigger_noise_floor_relative_enabled`) | **PASS: port** | Removing it from ALL-ON loses half the faint far bird (individual recall 1.0 → 0.5) and worsens countersinging median error (0.54 → 1.31 m). No duplicates or echoes attributable to it; O(1) per frame. |
| S6 bird-entity reporting (`entity_reporting_enabled`) | **PASS: port the entity model; NEEDS-WORK on absorption and row update** | Real audio: the clearest win after S2. ALL-ON minus entity drops individual recall 0.75 → 0.25 (the tracker merges birds into shared tracks: 2 merged, 2 duplicate entities), and S6 brings it back. Sim: no duplicates in any ALL-ON scene; it removes the tracker's merged entity in countersinging and flock. Entity positions (the bound track's filtered position) are as good as or better than raw fixes, e.g. omni-then-localized 1.4 m. Three problems: (1) a same-species flock with perches 10-11 m apart collapses into one entity (individual recall 1.00 → 0.33), because `entity_absorption_max_fix_distance_m` = 12 m is above the perch spacing and the bound track is shared; (2) the canonical DB row is overwritten with each new bout's *raw* fix, so stored history and API detections jump to the latest, possibly bad, fix (overlap scene median 0.4 → 15.9 m; omni scene 1.8 → 52 m); (3) it follows a track binding even across species when the tracker has merged two birds (real audio: the Scarlet Tanager never gets its own entity). |

**ALL-ON gate (plan criteria) on the sim matrix: FAIL, narrowly.** Summed individual recall
0.50 → 8.83, swaps 0, echo promotions 0, and position error no worse on 9 of 10 scenes.
The gate fails on one duplicate entity, in `onset_faint_far_bird`. Its cause: the far bird's
second bout is solved as a full fix at 391 m on the correct bearing (true range about 80 m). That
fix starts its own track and entity, and it only surfaces when log-odds, two-threshold and the
relative trigger are all on. This is an S2 range-observability issue, not a combination bug.
Scored on DB rows (entity positions ignored), ALL-ON also fails "position no worse" on
`gate_flock_echo` (S6 problem 1).

On real audio (fixture scene, BirdNET v2.4, after the pre-render cap fix) **ALL-ON passes.**
Against baseline:

- individual recall 0.00 → 0.75;
- duplicates 1 → 0, unassociated 4 → 0;
- median / p90 error 8.85 / 13.4 → 0.42 / 4.4 m;
- no swaps.

It also beats S2 alone (0.50, with 4 duplicates and 2 merged entities). The one missing bird is
the Scarlet Tanager (Reported 1). Before the cap fix the same scene gave ALL-ON 0.50, with 28
candidates skipped pre-classification: fixed bug 2 is the largest combination effect this gate
found.

## Method

**Command.**
```
python scripts/server_track_gate.py run --json out.json [--scenes ...] [--configs ...] [--work-dir d]
python scripts/server_track_gate.py summarize out*.json [--markdown t.md] [--rescore]
```
Library: `minimappr/sim/server_track_gate.py`; pytest gate: `tests/test_server_track_gate.py`.
The pytest gate runs ALL-ON on the seven scenes listed in `ALL_ON_SCENE_EXPECTATIONS` plus
ALL-ON vs baseline on one (about 75 s on this machine when it covered four scenes). Unit tests: `tests/test_server_track_gate_harness.py`.

**Scenes** (Rust `minimappr-sim`, seed 0, 24 kHz, three tetras on the 20 m triangle unless noted):
the six W2/W3 scenes `onset_{overlap_two_species, trunk_echo, faint_far_bird, same_band_pair,
same_band_two_species, single_tetra_overlap}`, plus four new ones in `minimappr-sim/scenarios/`:

| Scene | What it tests |
| --- | --- |
| `gate_countersinging_pair` | Two Northern Cardinals 12.6 m apart answering every 1.5 s, four bouts each. The name has "car" in it; see bug 1 |
| `gate_flock_echo` | Three chickadees on perches 10-11 m apart, bouts rotating every 1.2 s, two trunks inside the layout |
| `gate_omni_then_localized` | A wren 2.3 m from tetra-c: two quiet bouts only tetra-c hears, then two loud bouts. (A single-mic phone node in place of tetra-c gave no detections on any path: every phone-triggered candidate ended as `empty_classification`. That is a pre-existing POINT-node path problem, not investigated.) |
| `gate_collinear_trail` | Three tetras on a straight 40 m line, one bird of a different species on each side, one overlapping bout pair |
| `fixture_birdnet_four_birds` | Real xeno-canto fixtures (American Robin, Scarlet Tanager, Rose-breasted Grosbeak, White-throated Sparrow), two bouts each, on the species-gate perches, 48 kHz, ground and trunk echoes. Classified by the production BirdNET v2.4 member (3 s windows). Not in the default scene list, because it is slow: run it with `--scenes fixture_birdnet_four_birds` |

**Configurations.** `baseline` (all eight flags off), `all_on`, `all_on-minus:<stream>` for
each stream, and `only:S2_onset_clustering`. S5 attribution requires S2, so dropping S2 drops
attribution too, and `only:S5_attribution` carries S2. Other `only:` rows were measured in a
first pass on four scenes. On the baseline path they are identical to baseline, because there
nothing reaches them: the legacy classifier window ends at the trigger, so no labelled localized
detections exist. They were dropped from the final matrix to save time.

**Classifier for sim scenes.** The new `SnrGradedSpeciesOracle`
(`minimappr/sim/graded_species_oracle.py`):

- Each species' band SNR is its band power over the window's median bin power.
- Confidence is a logistic of that SNR (50 % at 12 dB, capped at 0.95).
- It is multi-label and reports `model = birdnet_v2m4`.
- It applies the BirdNET floor (results below `trigger_min_confidence` become `unknown`). With
  two-threshold on, it is wrapped exactly as the factory wraps the BirdNET member.

The old spectral-peak oracle always answered 0.9 for one label, which left log-odds,
two-threshold and LOCO nothing to act on.

**Settings shared by all configurations.**

- `trigger_rms = 0.001`, the env default. The scenes' hand-tuned 0.003 / 0.00015 hide what the
  relative trigger is for.
- `classification_window_seconds = 1.0` (3 s for the fixture scene).
- The production 30 s reporting window. Entity reporting replaces it for wildlife.

**Scoring (operator view).** `replay_entity_metrics`, exact label match, radius 5 m,
far-field (> `far_field_distance_m`) events on bearing (15°).

- Spatial gate and position error use **horizontal** distance. This adds
  `entities.horizontal_only` and `localization.horizontal_position_error`, both opt-in. Every
  layout is planar (all nodes at 1.5 m), which barely constrains height. A ground-image height
  flip should not count as a miss on a 2-D map. A side effect is that ground echoes cannot be
  "promoted", since they share (E, N) with the source.
- Rows are filtered and placed as the UI shows them:
  - unknown / zero-confidence rows are dropped (never broadcast);
  - sub-display rows are kept only through a visible track;
  - with S6, a row belongs to its entity (as the canonical row, or through the entity's bound
    track) and sits at the entity's position. `all_on(entity-pos)` rows; the matrix `all_on`
    and minus-X rows with S6 on were recorded before entity positions were captured, so they use
    the DB row's fix.
- Headline recall is **individual recall** (new `localized_and_classified_individual_recall`):
  the fraction of birds with at least one localized, correctly labelled event in an entity.
  Per-event recall undercounts any one-row-per-bird design: both the 30 s bucket policy and S6
  keep one row per bird and update it in place.
- Each result row stores the raw DB rows, visible tracks and entity maps, so `summarize
  --rescore` can re-apply changed scoring rules without replaying.

## Results

### Summary over the 10 sim scenes, and per-stream deltas

Sums over scenes (individual recall: 10 is perfect). `all_on(entity-pos)` is ALL-ON re-run with
entity positions captured; every other row with S6 on is scored at DB-row positions.

| Config | Indiv. recall (sum) | Event recall (sum) | Swaps | Duplicates | Echo prom. | Merged entities | Unassoc. localized | Median of scene medians (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 4 | 13.93 |
| only:S2 | 8.50 | 3.50 | 1 | 0 | 0 | 2 | 0 | 0.70 |
| all_on | 7.83 | 4.11 | 0 | 1 | 0 | 0 | 2 | 0.62 |
| **all_on(entity-pos)** | **8.83** | **4.47** | **0** | 1 | **0** | **0** | 1 | **0.62** |
| all_on − S2 onset | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 10 | 26.10 |
| all_on − S3 perched | 7.83 | 4.36 | 0 | 3 | 0 | 1 | 2 | 0.62 |
| all_on − S3 log-odds | 7.83 | 3.61 | 0 | 0 | 0 | 0 | 2 | 0.54 |
| all_on − S4 partial binding | 7.83 | 4.11 | 0 | 1 | 0 | 0 | 2 | 0.62 |
| all_on − S4 two-threshold | 7.83 | 3.61 | 0 | 0 | 0 | 0 | 2 | 0.54 |
| all_on − S5 attribution | 7.33 | 3.78 | 1 | 1 | 0 | 0 | 2 | 0.62 |
| all_on − S5 relative trigger | 7.33 | 3.44 | 0 | 0 | 0 | 0 | 2 | 1.31 |
| all_on − S6 entity | 9.50 | 4.42 | 0 | 1 | 0 | 1 | 0 | 0.70 |

Adding each stream to the rest (all_on − all_on-minus-stream; DB-row positions on both sides):

| Stream | Indiv. recall | Swaps | Dup | Echo | Merged | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| S2 onset clustering | +7.83 | 0 | +1 | 0 | 0 | Everything depends on it |
| S3 perched filter | 0 | 0 | −2 | 0 | −1 | Helps (countersinging) |
| S3 species log-odds | 0 | 0 | +1 | 0 | 0 | Neutral; the +1 is the faint-bird far fix that it surfaces together with two-threshold |
| S4 partial binding | 0 | 0 | 0 | 0 | 0 | Never fires |
| S4 two-threshold | 0 | 0 | +1 | 0 | 0 | Hurts slightly (same faint-bird fix; collinear p90) |
| S5 attribution | +0.50 | −1 | 0 | 0 | 0 | Helps (prevents the collinear swap) |
| S5 relative trigger | +0.50 | 0 | +1 | 0 | 0 | Helps (faint far bird) |
| S6 entity reporting | −1.67 | 0 | 0 | 0 | −1 | Hurts on DB rows (flock merge and row overwrite); at entity positions −0.67, all from the flock |

### Real audio: `fixture_birdnet_four_birds` (BirdNET v2.4)

After the pre-render cap fix (commit 94e3226). Rows with S6 on are placed at entity positions.

| Config | Dets | Indiv. recall | Event recall | Label acc. | Swaps | Dup | Merged | Unassoc. | Median / P90 err (m) | LOCO reruns (mean ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 12 | 0.00 | 0.00 | 0.75 | 0 | 1 | 0 | 4 | 8.85 / 13.44 | — |
| only:S2 | 13 | 0.50 | 0.25 | 1.00 | 0 | 4 | 2 | 2 | 0.03 / 0.93 | — |
| **all_on** | 6 | **0.75** | 0.50 | 0.80 | **0** | **0** | **0** | **0** | 0.42 / 4.40 | 13 (174) |
| all_on, LOCO off | 7 | 0.75 | 0.50 | 0.83 | 0 | 0 | 0 | 0 | 1.44 / 9.69 | — |
| all_on − two-threshold | 5 | 0.75 | 0.50 | 0.67 | 0 | 0 | 0 | 0 | 0.42 / 9.59 | 13 (177) |
| all_on − attribution | 9 | 0.75 | 0.62 | 0.83 | 0 | 0 | 0 | 0 | 0.48 / 3.65 | — |
| all_on − entity | 12 | 0.25 | 0.25 | 1.00 | 0 | 2 | 2 | 0 | 0.10 / 0.76 | 13 (165) |

The only:S2 duplicates are the omni scan's per-node "heard nearby" rows; its merges are the CV
tracker joining birds. In ALL-ON the Scarlet Tanager never surfaces as its own entity. A trace
taken before the cap fix showed the mechanism: the perched tracker accepts the grosbeak's fix
7.2 m away onto the tanager's track, and S6 follows the binding (Reported 1).

Before the cap fix, the same table read: ALL-ON 0.50 (4 detections, 28 candidates skipped by
`report_window_saturated`), minus-entity 0.50 (37 skipped). Those numbers were wrong only
because of fixed bug 2. One seed and one scene: treat these rows as indicative.

### Scenes × configurations

All rows, sim matrix (seed 0). Indiv. = individual recall; errors horizontal.

| Scene | Config | Dets | Indiv. recall | Event recall | Swaps | Dup | Echo | Merged | Unassoc. | Median err (m) | P90 err (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onset_overlap_two_species | baseline | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 16.33 | 18.66 |
| onset_overlap_two_species | all_on | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S2_onset_clustering | 2 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 2 | 26.10 | 35.95 |
| onset_overlap_two_species | all_on-minus:S3_perched_filter | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S3_species_log_odds | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S4_partial_binding | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S4_two_threshold | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S5_attribution | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S5_relative_trigger | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.87 | 37.49 |
| onset_overlap_two_species | all_on-minus:S6_entity_reporting | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.41 | 0.49 |
| onset_overlap_two_species | only:S2_onset_clustering | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.31 | 0.47 |
| onset_overlap_two_species | all_on(entity-pos) | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 1 | 15.48 | 37.49 |
| onset_trunk_echo | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_trunk_echo | all_on | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S2_onset_clustering | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_trunk_echo | all_on-minus:S3_perched_filter | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S3_species_log_odds | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S4_partial_binding | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S4_two_threshold | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S5_attribution | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S5_relative_trigger | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.07 | 0.07 |
| onset_trunk_echo | all_on-minus:S6_entity_reporting | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.12 |
| onset_trunk_echo | only:S2_onset_clustering | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.12 |
| onset_trunk_echo | all_on(entity-pos) | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.08 | 0.08 |
| onset_faint_far_bird | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_faint_far_bird | all_on | 4 | 1.00 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S2_onset_clustering | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_faint_far_bird | all_on-minus:S3_perched_filter | 4 | 1.00 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S3_species_log_odds | 2 | 1.00 | 0.67 | 0 | 0 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S4_partial_binding | 4 | 1.00 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S4_two_threshold | 2 | 1.00 | 0.67 | 0 | 0 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S5_attribution | 3 | 1.00 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S5_relative_trigger | 1 | 0.50 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on-minus:S6_entity_reporting | 3 | 1.00 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | only:S2_onset_clustering | 1 | 0.50 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_faint_far_bird | all_on(entity-pos) | 4 | 1.00 | 1.00 | 0 | 1 | 0 | 0 | 0 | 0.62 | 0.62 |
| onset_same_band_pair | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_same_band_pair | all_on | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S2_onset_clustering | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_same_band_pair | all_on-minus:S3_perched_filter | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S3_species_log_odds | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S4_partial_binding | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S4_two_threshold | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S5_attribution | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S5_relative_trigger | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.19 |
| onset_same_band_pair | all_on-minus:S6_entity_reporting | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.98 | 1.47 |
| onset_same_band_pair | only:S2_onset_clustering | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.98 | 1.47 |
| onset_same_band_pair | all_on(entity-pos) | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.78 | 1.32 |
| onset_same_band_two_species | baseline | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | 11.98 | 18.66 |
| onset_same_band_two_species | all_on | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S2_onset_clustering | 2 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 2 | 40.70 | 58.65 |
| onset_same_band_two_species | all_on-minus:S3_perched_filter | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S3_species_log_odds | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S4_partial_binding | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S4_two_threshold | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S5_attribution | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S5_relative_trigger | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_same_band_two_species | all_on-minus:S6_entity_reporting | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 11.54 | 22.76 |
| onset_same_band_two_species | only:S2_onset_clustering | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 11.54 | 22.76 |
| onset_same_band_two_species | all_on(entity-pos) | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.38 |
| onset_single_tetra_overlap | baseline | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | — | — |
| onset_single_tetra_overlap | all_on | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S2_onset_clustering | 2 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 2 | — | — |
| onset_single_tetra_overlap | all_on-minus:S3_perched_filter | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S3_species_log_odds | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S4_partial_binding | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S4_two_threshold | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S5_attribution | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S5_relative_trigger | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on-minus:S6_entity_reporting | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | only:S2_onset_clustering | 2 | 0.00 | 0.00 | 1 | 0 | 0 | 0 | 0 | — | — |
| onset_single_tetra_overlap | all_on(entity-pos) | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_countersinging_pair | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_countersinging_pair | all_on | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.54 | 9.87 |
| gate_countersinging_pair | all_on-minus:S2_onset_clustering | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_countersinging_pair | all_on-minus:S3_perched_filter | 4 | 1.00 | 0.50 | 0 | 2 | 0 | 1 | 0 | 0.61 | 12.65 |
| gate_countersinging_pair | all_on-minus:S3_species_log_odds | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.54 | 9.87 |
| gate_countersinging_pair | all_on-minus:S4_partial_binding | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.54 | 9.87 |
| gate_countersinging_pair | all_on-minus:S4_two_threshold | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.54 | 9.87 |
| gate_countersinging_pair | all_on-minus:S5_attribution | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.54 | 9.87 |
| gate_countersinging_pair | all_on-minus:S5_relative_trigger | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 1.31 | 9.61 |
| gate_countersinging_pair | all_on-minus:S6_entity_reporting | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.57 | 9.91 |
| gate_countersinging_pair | only:S2_onset_clustering | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 1 | 0 | 0.57 | 9.91 |
| gate_countersinging_pair | all_on(entity-pos) | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.53 | 9.96 |
| gate_omni_then_localized | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_omni_then_localized | all_on | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S2_onset_clustering | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_omni_then_localized | all_on-minus:S3_perched_filter | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S3_species_log_odds | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S4_partial_binding | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S4_two_threshold | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S5_attribution | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S5_relative_trigger | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 51.94 | 51.94 |
| gate_omni_then_localized | all_on-minus:S6_entity_reporting | 1 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 1.81 | 1.81 |
| gate_omni_then_localized | only:S2_onset_clustering | 1 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 1.22 | 1.22 |
| gate_omni_then_localized | all_on(entity-pos) | 1 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 1.38 | 1.38 |
| gate_collinear_trail | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_collinear_trail | all_on | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 1.69 | 18.75 |
| gate_collinear_trail | all_on-minus:S2_onset_clustering | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_collinear_trail | all_on-minus:S3_perched_filter | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 1.69 | 18.75 |
| gate_collinear_trail | all_on-minus:S3_species_log_odds | 2 | 1.00 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.13 | 0.18 |
| gate_collinear_trail | all_on-minus:S4_partial_binding | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 1.69 | 18.75 |
| gate_collinear_trail | all_on-minus:S4_two_threshold | 2 | 1.00 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.13 | 0.18 |
| gate_collinear_trail | all_on-minus:S5_attribution | 3 | 0.50 | 0.17 | 1 | 0 | 0 | 0 | 0 | 0.99 | 15.99 |
| gate_collinear_trail | all_on-minus:S5_relative_trigger | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 1.76 | 18.75 |
| gate_collinear_trail | all_on-minus:S6_entity_reporting | 2 | 1.00 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.70 | 0.75 |
| gate_collinear_trail | only:S2_onset_clustering | 2 | 1.00 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.70 | 0.75 |
| gate_collinear_trail | all_on(entity-pos) | 3 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.23 | 16.44 |
| gate_flock_echo | baseline | 3 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 2 | 13.93 | 15.49 |
| gate_flock_echo | all_on | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S2_onset_clustering | 4 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 4 | 14.87 | 38.36 |
| gate_flock_echo | all_on-minus:S3_perched_filter | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S3_species_log_odds | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S4_partial_binding | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S4_two_threshold | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S5_attribution | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S5_relative_trigger | 2 | 0.33 | 0.11 | 0 | 0 | 0 | 0 | 0 | 20.51 | 23.84 |
| gate_flock_echo | all_on-minus:S6_entity_reporting | 3 | 1.00 | 0.33 | 0 | 0 | 0 | 1 | 0 | 5.06 | 10.25 |
| gate_flock_echo | only:S2_onset_clustering | 3 | 1.00 | 0.33 | 0 | 0 | 0 | 1 | 0 | 4.90 | 9.38 |
| gate_flock_echo | all_on(entity-pos) | 2 | 0.33 | 0.22 | 0 | 0 | 0 | 0 | 0 | 8.90 | 9.88 |

## Integration bugs

### Fixed (with tests)

1. **Birds named with "car" / "fire" / "cat" substrings were not wildlife**
   (`core/classification.py`, `core/taxonomy.py`; commit 8451654). `_resolve_label_category`
   let the *name heuristics* override a species model's provenance. So BirdNET's "scarlet
   tanager", "northern cardinal", "canada warbler" and 87 other v2.4 species (common names)
   resolved to `vehicle` ("car"), 19 (e.g. "common firecrest") to `security` ("fire",
   "gun") and 4 ("screamer") to `human`. Every
   wildlife-scoped stage then skipped them: the perched tracker, species log-odds, entity
   reporting and wildlife rules. Now an operator-configured category (taxonomy config file)
   still wins and every other BirdNET label takes BirdNET's own vocabulary category
   (`core/birdnet_label_categories.py`): species are wildlife; v2.4's and V3's sound classes
   have an explicit table (Gun/Siren/Fireworks/Explosion/Fire → security, Engine/Aircraft/
   Motor vehicle/Boat/Train/Vehicle horn → vehicle, Human vocal/non-vocal/whistle, Power
   tool(s)/Chainsaw/Hammer/Sawing/Music → human, Noise/Environmental → unknown, Dog →
   wildlife); V3's `Homo sapiens` (filed under Mammalia) → human. Tests:
   `tests/test_birdnet_label_categories.py` (exhaustive over all 6522 v2.4 labels, the V3
   names in the v2.4→V3 map, the 16 V3 sound classes and all 521 YAMNet labels; the
   full-file V3 check runs when the labels CSV is provisioned or
   `MINIMAPPR_TEST_BIRDNET_V3_LABELS_CSV` points at it), `tests/test_taxonomy_provider.py`.
   **Existing deployments (review fix):** learned label rows (`register_label`,
   `merge_labels` from the `labels` table) no longer count as configured, so persisted wrong
   rows cannot re-impose the heuristic answer. `FusionNode.start` also re-files `labels` rows
   whose name is a known v2.4/mapped-V3 animal species and whose category is not wildlife
   (skipping operator-configured labels; idempotent; logged). Historical
   `detections.label_category` values are left as recorded.
2. **The pre-render report-window cap skipped real birds once onset clustering was on**
   (`core/fusion_node.py` `_should_skip_saturated_report_window`; commit 94e3226). The cap
   (`fusion_report_window_localized_emission_cap` = 3 per node per 30 s bucket) assumes one
   trigger yields several candidates for the same sound. Onset-SRP products are one per
   accepted cluster, i.e. distinct birds, and the consumed-onset memory stops re-surfacing, so
   the cap dropped real birds before classification: 28 per replay on the four-bird real-audio
   scene with ALL-ON, 37 without entities, and 1-5 on three sim scenes in legacy reporting mode.
   In entity mode the cap was also inconsistent. Its check is keyed on legacy buckets, but
   wildlife emissions are noted under entity windows, so three non-entity rows from a node
   blocked every candidate from that node for the rest of the bucket. W3-entity had called this
   gate "harmless". Onset-SRP products are now exempt; the legacy path is unchanged. Real-audio
   ALL-ON individual recall 0.50 → 0.75. Test: `tests/test_hybrid_reporting.py` (1 new).
3. **Replay harness lost late windows in the BirdNET gate**
   (`scripts/birdnet_version_replay_gate.py`). It collected after a settle poll without
   draining the stage queues, which is the same bug W3-attribution fixed in
   `sim_scenario_scoring`. It now calls `drain_fusion_queues` (900 s budget). It also gained
   `--variants` and `--settings`.
4. **Harness:** a replay DB left behind by an interrupted run was scored together with the
   rerun. `replay_scene_configuration` now clears its replay directory first. This one only
   affected my own early matrix.

### Reported, not fixed (bigger than a small fix)

1. **Cross-species track merge → entity relabel (S3 perched × S6).** Real audio, all flags on.
   The grosbeak's fix at (10, 13) associates to the tanager's track at (14, 7), 7.2 m away. The
   fix is beyond `tracking_perched_relocation_snap_distance_m` (6 m), so the filter restarts
   there with the same identity. The association gate has no species check: only the category
   gate, wildlife vs wildlife. S6 then follows the track binding (`find_absorbing_entity`:
   "bound track wins outright"; `_entity_for_track_if_same_species` skips the label check
   whenever the track has a log-odds belief). It also absorbs by `believes_label`, which accepts
   any species *anywhere* in the belief, including prior-level secondary BirdNET scores. The
   tanager entity becomes the grosbeak and the tanager is gone. Proposed fix, in the port: for
   wildlife with log-odds on, refuse association (and relocation or snap) when the detection's
   top label is below the display belief on the track while the track's top label is
   confidently different. Gate S6 absorption on a belief threshold instead of mere presence.
2. **Same-species flock collapses (S3 + S6).** In `gate_flock_echo` the perched tracker keeps
   one track for three birds rotating every 1.2 s. The relocation gate's minimum silence (2 s)
   and the ghost logic do not trigger, so association simply accepts fixes 10 m away. S6's 12 m
   absorption radius then merges the three into one entity. Needs a perch-spacing-aware gate,
   and an absorption radius tied to the fix covariance rather than a fixed 12 m.
3. **S6 overwrites the canonical row with each raw fix.** `upgrade_existing` replaces the
   stored row's position with the latest bout's fix, while the entity itself follows the
   filtered track. Stored history and `/detections` then jump: 0.4 → 15.9 m median (overlap),
   1.8 → 52 m (omni scene). Keep the entity position, or only replace the row's position when
   the new fix's covariance is tighter.
4. **S2 height flips to the ground image in planar layouts.** All nodes at one height:
   onset LM solutions land at z < 0, near the ground-reflection image (countersinging: z −2.0
   for a bird at 4.0 m). The grid's elevation range starts at −30° and LM is unconstrained. Add
   a ground-plane prior (z ≥ ground) in the port.
5. **S2 overclaims range for far birds.** In the faint scene the 70-80 m pewee's second bout
   is a full multi-node fix at 391 m on the right bearing (range unobservable from a 20 m
   layout), published as a position. This causes the gate's single duplicate. The range
   observability and haircut need to reach the onset path's multi-node LM output.
6. **S2 mislocalizes a loud source 2.3 m from one node** (`gate_omni_then_localized`). The loud
   bouts give one `position_out_of_range` drop and one fix 52 m out on roughly the right bearing,
   while the quiet bouts localize to 1.8 m. Not investigated further (near-field of one array
   vs. the polar grid around the layout centroid is the first suspect).
7. **LOCO adds cost without a measured gain with all flags on.** Each BirdNET rerun costs about
   175 ms here (13 per replay on the fixture scene; 50 ms measured idle in W3). ALL-ON with LOCO
   off scored the same individual recall on the fixture, collinear and single-tetra scenes. An
   earlier reading that "LOCO leaves real birds unlabelled" was an artefact of fixed bug 2.
8. **Two-threshold rows surface through tracks with their own fixes** (S4). Sub-display rows
   attached to a visible track keep their own, often poor, fix (collinear trail: a 0.34-confidence
   thrush row 3 m off). They surface as rows, adding error and noise rather than evidence.
   Showing them only as track/entity evidence (not rows) would match the design intent.
9. **Single-mic (phone/POINT) nodes produce nothing on replay**: every phone-triggered
   candidate ended as `empty_classification`, with or without onset clustering. Not investigated
   (the phone path goes through Rust core later), but it blocks a true omni-then-localized
   server scene.

## Compute cost with all flags on

Measured under heavy parallel load (6-10 concurrent replays plus other agents), so the numbers
are upper bounds. W2 measured 0.3-0.5 s per window on an idle M2 core.

| Run | Onset windows | Clustering per window (median / p90 / max) | LOCO reruns (mean) | Replay wall / audio |
| --- | --- | --- | --- | --- |
| ALL-ON, 10 sim scenes, 3 tetras at 24 kHz (lighter load) | 45 | 1.49 / 3.33 / 4.47 s | 22 (17 ms, oracle) | 166 s / 111 s |
| ALL-ON, same scenes, peak load | 45 | 4.71 / 9.95 / 12.85 s | 22 (27 ms) | 299 s / 111 s |
| ALL-ON, fixture scene, 3 tetras at 48 kHz, BirdNET (after cap fix) | 46 | 1.45 / 2.68 / 6.46 s | 13 (174 ms, BirdNET) | 99 s / 60 s |

All clustering runs in a worker thread (`asyncio.to_thread`), and LOCO runs on the
classification worker, so the event loop is never blocked. At 48 kHz with three tetras,
clustering is the dominant cost. Under load it falls behind real time when windows overlap.
The localization queue's drop-oldest backpressure then sheds work. Re-measure idle on the
Bluefin box before a live trial. Per frame, the relative trigger and the tracker / entity
updates are negligible.

## BirdNET cutover re-gate on the merged branch

`python scripts/birdnet_version_replay_gate.py --seeds 0 1 2` (default flags, as before):

| Variant | Dets per seed | Exact acc. | Event recall | L&C @15 m (per seed) | Swaps @15 m | Dup @15 m | Merged @15 m | L&C @5 m | Echo @5 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2.4 | 36 / 34 / 33 | 0.864 (38/44) | 0.917 | 0.31 / 0.31 / 0.50 | 0 | 31 | 21 | 0 / 0 / 0.06 | 1 |
| 3.0 | 34 / 34 / 36 | 0.851 (40/47) | 0.979 | 0.38 / 0.31 / 0.25 | 5 | 34 | 17 | 0 | 1 |
| 3.0+regional (gated) | 33 / 31 / 35 | **0.915** (43/47) | 0.979 | 0.38 / 0.38 / 0.38 | 5 | 34 | 17 | 0 | 1 |

**Verdict unchanged: FAIL, keep 2.4.**

- The swap blocker is unchanged: V3 still calls the robin/tanager/grosbeak caroled songs
  "hermit thrush" 6 times, which gives 5 swaps against 0 for v2.4.
- A second condition now fails: duplicates 34 > 31.
- V3+regional's accuracy edge widened (0.915 vs 0.864; it was 0.894 vs 0.864).
- The node-collapse fix did not make the entity metrics discriminating. With default flags the
  legacy path now gives honest far-field bearings instead of collapsed points, but those are
  still 40-50 m from the birds, so L&C @5 m stays 0 for every variant.

**With all server-track flags on** (`--variants 2.4 3.0+regional --settings <all-on flags>`,
after the cap fix), the entity metrics finally discriminate (L&C @5 m 0.3-0.6 instead of 0):

| Variant | Dets per seed | Exact acc. | Event recall | L&C @15 m | Swaps @15 m | Dup @15 m | Merged @15 m | L&C @5 m | Dup @5 m | Echo @5 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2.4 | 25 / 25 / 22 | **0.738** (31/42) | 0.875 | 0.50 / 0.62 / 0.56 | 0 | 10 | 6 | 0.44 / 0.56 / 0.50 | 5 | 0 |
| 3.0+regional | 28 / 23 / 27 | 0.659 (27/41) | 0.854 | 0.44 / 0.56 / 0.50 | 0 | 9 | 4 | 0.38 / 0.50 / 0.31 | 5 | 0 |

**Verdict: FAIL, keep 2.4** (V3 exact accuracy 0.659 < 0.738). V3's hermit-thrush swaps
disappear here: per-cluster classification windows separate the caroled songs. But it loses
accuracy, and v2.4 is better on every spatial column.

An all-on run made *before* the cap fix gave "INSUFFICIENT EVIDENCE (sim passes)": V3 0.737
vs 0.722, recall 0.792 vs 0.750. Those numbers were distorted by the pre-render cap dropping
candidates, so the version comparison flips once real birds stop being skipped.

**Net: the cutover verdict does not change.** Keep 2.4 under default flags (swaps and
duplicates) and under the server-track flags (accuracy). Real-site (Bluefin) evidence is still
missing.

## Bluefin snippet stability (box down)

`nc -z -G 2 127.0.0.1 2222` returned exit 1 at 2026-11-22 10:02 CST. To run once it is up:

```bash
nc -z -G 2 127.0.0.1 2222 && ssh -p 2222 colin@127.0.0.1
cd ~/MinimapPR && git fetch && git checkout fm/s-gate
# 0. labels persisted before the category fix are re-filed at startup; check what is left
sqlite3 data/minimappr.db "SELECT name, category FROM labels WHERE category IN ('vehicle','security','human') ORDER BY name;"
# 1. snippet stability: same snippets, same labels, before vs after the flag set
python scripts/compare_birdnet_versions.py --audio data/snippets --max-clips 500 --output-json data/birdnet_v3_compare.json
ls data/snippets | wc -l; du -sh data/snippets     # before; repeat after 1 h of live running
# 2. live trial with the recommended flags (below), python ingest; watch the onset
#    window time and queue drops in the pipeline graph / metrics API (not server.log)
curl -s localhost:8080/api/v1/pipeline/graph | python -m json.tool | grep -iA3 "onset\|drop"
# 3. cutover gate with real-site evidence
cargo build --release -p minimappr-sim
python scripts/birdnet_version_replay_gate.py --seeds 0 1 2 --bluefin-report data/birdnet_v3_compare.json --json data/birdnet_v3_gate.json
```

Bring back the two JSON files, the `labels` query output, and the snippet counts and sizes.

## Recommended flag set for a first live trial

| Flag | Trial value | Why |
| --- | --- | --- |
| `localization_onset_clustering_enabled` | **on** | Required for anything else to matter; watch per-window cost |
| `trigger_noise_floor_relative_enabled` | **on** | Faint birds; OR'd with the absolute gate, so no window that opens today closes |
| `tracking_perched_filter_enabled` | **on** | Fixes CV-tracker perch hopping; watch for cross-species merges (bug report 1) |
| `tracking_species_log_odds_enabled` | **on** | Belief exposure; harmless alone |
| `onset_attribution_enabled` | **on, with `onset_attribution_loco_enabled=false`** | The SNR-gradient tier is what prevented the collinear swap. LOCO cost about 175 ms per BirdNET rerun with no measured gain with all flags on (fixture, collinear and single-tetra scenes replayed with LOCO off: same recall, 0 swaps) |
| `entity_reporting_enabled` | **on** | Biggest real-audio gain after S2 (0.25 → 0.75 individual recall). Known: same-species flocks with perches ≤ 12 m apart merge, and the canonical DB row carries the latest raw fix (the map uses the entity position). Set `entity_notification_grace_seconds=0` on single-node sites |
| `birdnet_two_threshold_enabled` | off | No measured gain on real audio. It surfaces weak rows with poor fixes; revisit after the association fix |
| `tracking_partial_constraint_binding_enabled` | off | Never fires on this path |

Before the trial, re-measure per-window onset cost idle on the box (48 kHz × 3 tetras was
1.5-2.4 s median per window under load here). If it falls behind real time, drop-oldest
backpressure sheds windows.

## Limits

- One seed per scene. The replay is deterministic given the seed apart from machine-load
  timing, which the queue drain removes.
- The sim scenes use a synthetic oracle, not BirdNET. Only one real-audio scene, one seed.
- Planar layouts only; no moving birds; no clock error (all tetras disciplined).
- Some of the matrix's S6-on rows predate entity-position capture (noted per table).

## S-fixes tracking (follow-up, branch `fm/s-fixes-tracking`)

Reported items 1-3 and the plan-gate near-miss are fixed behind the existing flags. Details,
tables and tests are in `s_fixes_tracking.md`. In summary:

- **Plan gate on the sim matrix: FAIL → PASS.** Individual recall sum 8.83 → 9.50, duplicates
  1 → 0, swaps / echoes / merges 0, unassociated 1 → 0, median of scene medians 0.62 → 0.43 m.
- **Real audio: 0.75 → 1.00 individual recall.** The Scarlet Tanager is recovered (3 of 3 replays)
  and label accuracy is 1.00.
- **S6 now only helps:** all-on − minus-S6 is 0 recall, −1 duplicate, −1 merged (it was −1.67 recall).

## Recorded baseline: localization core overhaul (2026-11-25)

Branch `loc/relocation-fix` (on `localization-core-overhaul`), after the overhaul review fixes
(hold-rule release paths, point-network worker gates). `baseline` and `all_on` on the 10 sim
scenes, seed 0, scored as above with entity positions. This is the baseline to compare later
runs against; the matrix tables above are the Wave 4 history.

**Repeatability.** Three runs on the fixed head and one on the head before the review fixes
(4a6d3b0) gave identical values on all 20 rows, so no row needed a rerun this time. Some rows
have depended on a cross-reporter dedupe race before: rerun a row that moves before reading
anything into it. The review fixes changed no gate row.

**ALL-ON gate: PASS.** Sums: individual recall 9.50, event recall 4.42, swaps 0, duplicates 0,
echo promotions 0, merged 0, unassociated 0, median of scene medians 0.43 m (baseline: 0.00,
0.00, 1 echo promotion, 3 unassociated, 6.79 m).

| Scene | Config | Dets | Indiv. recall | Event recall | Swaps | Dup | Echo | Merged | Unassoc. | Median err (m) | P90 err (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onset_overlap_two_species | baseline | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 6.42 | 7.46 |
| onset_overlap_two_species | all_on | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.44 | 0.59 |
| onset_trunk_echo | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_trunk_echo | all_on | 1 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.65 | 0.65 |
| onset_faint_far_bird | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_faint_far_bird | all_on | 3 | 1.00 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0.15 | 0.15 |
| onset_same_band_pair | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| onset_same_band_pair | all_on | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.43 | 0.52 |
| onset_same_band_two_species | baseline | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | 13.70 | 18.12 |
| onset_same_band_two_species | all_on | 2 | 1.00 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0.47 | 0.68 |
| onset_single_tetra_overlap | baseline | 1 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 1 | — | — |
| onset_single_tetra_overlap | all_on | 1 | 0.50 | 0.25 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_countersinging_pair | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_countersinging_pair | all_on | 2 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.32 | 9.99 |
| gate_flock_echo | baseline | 3 | 0.00 | 0.00 | 0 | 0 | 1 | 0 | 0 | 6.79 | 14.31 |
| gate_flock_echo | all_on | 3 | 1.00 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.56 | 9.94 |
| gate_omni_then_localized | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_omni_then_localized | all_on | 1 | 1.00 | 0.25 | 0 | 0 | 0 | 0 | 0 | 0.18 | 0.18 |
| gate_collinear_trail | baseline | 0 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0 | — | — |
| gate_collinear_trail | all_on | 2 | 1.00 | 0.33 | 0 | 0 | 0 | 0 | 0 | 0.08 | 0.11 |

Two rows are below the pre-milestone-3 values and are accepted (`docs/tracking.md`, "Hold rule
for perched tracks"):

- `onset_overlap_two_species` median 0.44 m (0.33 m before). The pytest gate now locks it at
  recall 1.0 and median ≤ 0.6 m.
- `gate_flock_echo` event recall 0.33 (0.56 before, at 62cbb2a). Individual recall, error,
  swaps and duplicates are unchanged. Per-event associations show one of the two lost events was
  a misassociation scored as a hit, and the other a correct row that no longer surfaces on its
  own.
