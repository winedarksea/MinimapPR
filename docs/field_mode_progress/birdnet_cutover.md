# W3-birdnet-cutover: v2.4 → V3.0 on the server (2026-11-22)

Branch `fm/birdnet-cutover` (base `field-mode` @ 8eca3b4). **Default stays `birdnet_model_version = "2.4"`.**
With "2.4", behaviour is unchanged: every new path either keys on `"3.0"` or sits behind a flag whose default keeps today's behaviour.

## Summary

- **Label migration (V3 only, in memory).** At load time, rules, retention policy rules, `retention_permanent_labels`, taxonomy `label_to_category` and classifier routing get the V3 English names added next to the v2.4 ones. BirdNET-only thresholds are carried onto the V3 confidence scale. Files on disk are never rewritten. Labels with no V3 class are logged as a WARNING and listed in the migration report. Flag: `birdnet_v3_label_migration_enabled` (default True, only acts under "3.0").
- **Thresholds per model version.** `Settings.birdnet_threshold_set()` returns the live set: v2.4 keys, or the new `birdnet_v3_*` keys. The factory, two-threshold wrapper, retention "confident" bucket, tracker display belief and the sim emulation all read it, so switching the version swaps every threshold at once. V3 defaults were derived by quantile matching on the same audio (table below).
- **Tracker constraint.** Turning on `birdnet_two_threshold_enabled` without `tracking_perched_filter_enabled` + `tracking_species_log_odds_enabled` logs a WARNING at FusionNode start and shows a "Two-threshold warning" on the pipeline graph. It becomes a Settings/PATCH validation error when `birdnet_two_threshold_require_tracker_companions=true` (default false, to keep current behaviour).
- **Gate verdict: **FAIL — keep "2.4".** On the sim gate, V3 with a regional species list beats v2.4 on exact label accuracy (0.894 vs 0.864) and on event recall (0.979 vs 0.917). It fails on label swaps (5 vs 0): V3 calls the caroled-song triad (robin, tanager, grosbeak) "hermit thrush" 6 times. There is also no real-site (Bluefin) evidence. Either reason alone is enough to keep the default.**

## 1. Label migration

`minimappr/classifiers/birdnet_v3_label_migration.py`. It builds a translator from `assets/birdnet/v24_to_v3_label_map.csv`, mapping each lower-cased v2.4 common name to its V3 name(s). Lumps can give more than one name.

| What | Where it hooks | What happens under V3 |
| --- | --- | --- |
| Rules (`rules.json` and default rules) | `ConfigRuleEngine(rule_transform=...)` (`core/rules.py`, built in `fusion_node.py` and the `main.py` pipeline-graph fallback) | `labels` gain the V3 name. If every label is BirdNET-only (not in the YAMNet vocabulary), `min_confidence` goes through the confidence map, e.g. the default coyote rule 0.40 → 0.57. If a rule mixes BirdNET and YAMNet labels (e.g. "dog"), its threshold is left alone and reported (`confidence_not_remapped`). |
| Retention policy rules (`retention_policy.json`) | `load_cleanup_policy` (`cleanup_service.py`) | A `label` rule is cloned as `<id>@birdnet-v3:<v3 label>` with the same priority and actions. A `classifier: birdnet` rule's `confidence_min` / `confidence_max` go through the map. Builtins are not re-mapped: `builtin-birdnet-confident` already uses the V3 trigger. |
| `retention_permanent_labels` | `RetentionPolicy` in `fusion_node.py` | V3 names are added. |
| Taxonomy `label_to_category` | `fusion_node.py` (new `RuntimeTaxonomyProvider.label_category_items()`) | Each V3 name inherits the category configured for its v2.4 label, unless the V3 name is already configured. |
| Classifier routing | `routing.apply_settings` (end) | The BirdNET member's `min_confidence` goes through the map (0.40 → 0.572). Chains and triggers hanging off a BirdNET member get expanded labels and mapped thresholds. Other members are untouched. The routing file and `GET /classifier-routing` stay v2.4. |

**Unmappable labels.** 20 v2.4 labels have no V3 class: 18 birds (e.g. "nihoa finch") plus Noise and Environmental. A rule or retention rule that references one keeps its v2.4 spelling, so it still matches other classifiers, and a WARNING names the rule and label. Expansion is additive, so YAMNet "dog" keeps matching a rule keyed on "dog".

**Stored DB rows are not migrated.** Detections stored under v2.4 keep their v2.4 labels. History queries by label therefore split at the cutover. A one-off `UPDATE detections SET label=...` from the translator would be trivial, but it would also rewrite YAMNet rows that share a spelling ("dog"), so it was not done. The rows carry `classifier_scores_json` and the winner member, which is enough to do it correctly later if wanted.

## 2. Threshold re-tune

### Method

`scripts/calibrate_birdnet_v3_thresholds.py` and `minimappr/classifiers/birdnet_confidence_calibration.py`. Both models score the same clips. Each produces a clip-level vector, taking the maximum over its own windows, exactly as the classifiers aggregate: v2.4 uses 3 s segments with 1.5 s overlap, V3 uses 2–6 s windows with 3 s overlap. A v2.4 threshold *t* becomes `t_v3 = Q_v3(1 − P(top1_v24 ≥ t))`, which keeps the same fraction of clips passing. A 500-sample paired bootstrap gives a 90% interval. The full curve is committed as `assets/birdnet/v24_to_v3_confidence_map.json`, which is used for routing, rule and retention thresholds.

**Clips** (551 clips, 65.4 audio minutes; `--seeds 0 1 2`):

| Source | Clips | Content |
| --- | --- | --- |
| `fixture` | 18 | The 8 committed XC bird clips, whole, plus the 30 s ambience bed in 3 s pieces. The ambience is treated as **unlabelled, not negative**: it holds faint real birds that V3 scores 0.5–0.72 and v2.4 scores < 0.25. |
| `sim` | 171 | Three seeds of `build_distance_ladder_scenario` (`minimappr/sim/fixture_audio_scenarios.py`), rendered by `minimappr-sim`. Each of the 8 fixtures is played as a `wav_file` source at 2, 10, 30, 60, 120 and 250 m from one tetra node: 90 dB SPL at 1 m, ISO 9613 absorption, ground and 2–4 trunk echoes, 35 dB SPL ambient. There is one window per call on channel 0, aligned for propagation delay, plus 9 noise-only windows per seed from a 30 s tail. |
| `local` | 362 | 300 random real site clips from `drone_dataset/wav/ambient` (unlabelled), 8 coyote clips (labelled *Canis latrans*), 40 drone clips, `tests/house_finch.wav` (labelled), and 6 `~/Downloads/det-*_listening.wav` exports. |

### Command

```bash
D=../drone_dataset/wav   # main checkout's drone_dataset
python scripts/calibrate_birdnet_v3_thresholds.py --seeds 0 1 2 \
  --local-audio site_ambient=$D/ambient:300 --local-audio coyote=$D/coyote --local-species "coyote=Canis latrans" \
  --local-audio drone=$D/drone:40 --local-audio house_finch=tests/house_finch.wav \
  --local-species "house_finch=Haemorhous mexicanus" --local-audio downloads_detections=$HOME/Downloads \
  --output-json /tmp/calib.json --write-map      # ~2.5 min on an M2
```

### Threshold table

Recall is measured on 161 labelled clips (true-class score ≥ threshold). The noise pass rate is measured on 27 pure-noise windows.

| Threshold | v2.4 | V3 pooled match (90% CI) | V3 default chosen | Pass fraction (both) | Labelled recall v2.4 → V3 | Noise pass v2.4 → V3 |
| --- | --- | --- | --- | --- | --- | --- |
| `birdnet_trigger_min_confidence` → `birdnet_v3_trigger_min_confidence` (display floor, retention confident bucket) | 0.50 | 0.658 (0.606–0.716) | **0.65** | 0.394 | 0.83 → 0.91 | 0 → 0 |
| `tracking_species_display_min_belief` → `birdnet_v3_species_display_min_belief` | 0.50 | 0.658 | **0.65** | 0.394 | as above | as above |
| routing `classifiers.birdnet.min_confidence` (through the map) | 0.40 | 0.572 (0.511–0.613) | **0.572** (map) | 0.445 | 0.85 → 0.95 | 0 → 0 |
| `birdnet_localize_to_track_min_confidence` → `birdnet_v3_localize_to_track_min_confidence` | 0.15 | 0.172 (0.133–0.207) | **0.17** | 0.646 | 0.89 → 0.99 | 0 → 0.04 |
| `tracking_species_log_odds_prior_probability` (reported only, no V3 key) | 0.10 | 0.109 (0.082–0.128) | unchanged | 0.715 | 0.91 → 0.99 | 0 → 0.04 |

Cross-checks, all from the same run:

- **Matching by source moves the answer.** For the 0.50 threshold: real site clips alone give 0.605, sim alone 0.764, fixtures alone 0.85. The pooled value sits between them, and real site audio is the largest share of it.
- **Recall-matching on labelled clips gives much higher values** (0.76 for 0.50, 0.71 for 0.15). This is because V3 is *right* on more of the labelled clips, so equal recall needs a higher V3 bar. Those values would throw away V3's gain, so they were not used.
- **Model behaviour by sim distance.** V3 is level-invariant and does not lose top-1 accuracy with distance:

  | Distance | v2.4 top-1 accuracy | V3 top-1 accuracy |
  | --- | --- | --- |
  | 2–3 m | 0.96 | 1.00 |
  | 30 m | 0.83 | 1.00 |
  | 120 m | 0.79 | 1.00 |
  | 250 m | 0.75 | 1.00 |

  The median true-class score is 0.81–0.98 for v2.4 and 0.83–0.88 for V3. V3 saturates lower, which is why its matched thresholds sit higher at the top end.
- **Earlier run.** A first run (344 clips, ambience counted as negative) gave 0.625 / 0.57 / 0.244 / 0.148. The 0.15 threshold is the least stable value.

### Wiring

- Defaults are `BIRDNET_V3_DEFAULT_*` in `config.py`.
- The keys are restart-required. They are covered by the existing `birdnet_v3_` and `birdnet_two_threshold_` prefix sets in `main.py`.
- They are registered in `settings_store.py`, `config_groups.py` (Tracking group, next to the two-threshold keys) and `pipeline_graph.py` (BirdNET member: "V3 trigger", "V3 localize-to-track", "V3 label migration", "Two-threshold warning").
- The rust ingest's Python classifier subprocess rebuilds `Settings` from env, so it picks the set from `MINIMAPPR_BIRDNET_MODEL_VERSION` and `MINIMAPPR_BIRDNET_V3_*`. `ingest_sidecar_runtime.py` is unchanged.

## 3. Replay gate

`scripts/birdnet_version_replay_gate.py`.

**Bundles.** `build_species_gate_scenario` uses three tetra nodes (the 20 m triangle) and the 8 fixture birds on fixed perches inside or just outside the array. Each bird sings twice, giving 16 calls. The rest of the setup matches the ladder: 90 dB SPL, ground and trunk echoes, 35 dB ambient. Seeds 0, 1 and 2.

**Replay.** Each bundle goes through `build_fusion_for_bundle` / `replay_bundle` with Python ingest, `trigger_rms=0.003` and `classification_window_seconds=3`. The classifier is the BirdNET member alone, built by the production factory from `apply_settings(default_routing)`, so each version's thresholds and the V3 routing migration are live.

**Scoring.** `evaluate_bundle` with `match = "exact"`, plus `replay_entity_metrics` at 15 m and 5 m.

**Variants.**

- **v2.4.** The production path, including the v2.4 geo filter at the site lat/lon.
- **3.0.** All 11,725 classes.
- **3.0+regional.** V3 restricted, through `birdnet_v3_species_list_path`, to the v2.4 geo model's site list mapped to V3 names. This is the gated variant, because production v2.4 also runs its geo filter.

I added the regional variant after a first seed-0 run. There, plain V3 produced out-of-range species (European Golden-Plover, Grace's Warbler, Hermit Thrush): accuracy 0.81 vs 0.86 for v2.4, and 2 swaps.

**Gate.** All four conditions must hold, summed over seeds:

1. V3 exact accuracy ≥ v2.4.
2. V3 event recall ≥ v2.4 − 0.05.
3. V3 swaps ≤ v2.4 and V3 duplicates ≤ v2.4, both at 15 m.
4. A Bluefin snippet comparison exists with mapped top-1 agreement on confident clips ≥ 0.8.

**Command.** `python scripts/birdnet_version_replay_gate.py --seeds 0 1 2 --json /tmp/gate.json`. It took about 40 minutes: 9 replays, each paced in real time.

**Results** (3 seeds, 16 calls each, 48 calls in total):

| Variant | Detections per seed | Exact label accuracy | Event recall | L&C recall at 15 m (per seed) | Swaps at 15 m | Duplicates at 15 m | Multi-individual entities at 15 m | L&C recall at 5 m | Echo promotions at 5 m | Wrong species labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2.4 (geo filter) | 32 / 35 / 32 | **0.864** (38/44) | 0.917 | 0.31 / 0.31 / 0.50 | **0** | 39 | 19 | 0 / 0 / 0.06 | 5 | Harris's Sparrow ×1 |
| 3.0 (all classes) | 35 / 32 / 31 | 0.851 (40/47) | 0.979 | 0.31 / 0.31 / 0.19 | 5 | 37 | 17 | 0 | 6 | Hermit Thrush ×6, Grace's Warbler ×5, European Golden-Plover ×1 |
| **3.0+regional** (gated) | 32 / 33 / 34 | **0.894** (42/47) | **0.979** | 0.31 / 0.38 / 0.38 | **5** | 38 | 17 | 0 | 5 | Hermit Thrush ×6 |

**Verdict: FAIL (keep 2.4).** The only failed condition was "label swaps V3 5 > v2.4 0". The real-site condition (4) was never reached.

**How the variants differ.**

- V3 picks up Blue Jay, which v2.4 never labelled in any seed: v2.4's top-1 for that fixture was 0.09.
- V3 drops fewer events.
- V3's only regional confusion is Hermit Thrush on the robin/tanager/grosbeak caroled songs. That confusion is what produces the swaps: 5 of the swap-eligible detections land on an entity whose label is another bird.
- Without a regional list, V3 also emits out-of-range species. A species list is required before V3 can be deployed.

The verdict matches the expectation: there is not enough evidence to switch, and the swap result is a concrete blocker to look at with real audio.

**Caveat on the entity metrics.** Entity metrics are dominated by the known replay localization collapse (`server_baseline.md`; W2-onset-srp is fixing it). At 5 m, recall is 0 for every variant and duplicates are high at 15 m for every variant. They do not discriminate between the BirdNET versions until fixes stop collapsing onto nodes. Re-run after W2-onset-srp lands.

### Bluefin (down)

`nc -z -G 2 127.0.0.1 2222` returned exit 1 at 2026-11-22 ~01:30. Re-run on the box once it is up:

```bash
nc -z -G 2 127.0.0.1 2222 && ssh -p 2222 colin@127.0.0.1
cd ~/MinimapPR && git fetch && git checkout fm/birdnet-cutover
python scripts/fetch_birdnet_v3.py
# 1. same-audio comparison on real site snippets (labels mapped through the map)
python scripts/compare_birdnet_versions.py --audio data/snippets --max-clips 500 \
    --output-json data/birdnet_v3_compare.json
# 2. re-derive thresholds with the snippets as the dominant real-audio source
python scripts/calibrate_birdnet_v3_thresholds.py --seeds 0 1 2 \
    --local-audio bluefin_snippets=data/snippets:500 \
    --output-json data/birdnet_v3_threshold_calibration.json      # add --write-map to replace the curve
# 3. the gate, now with real-site evidence (needs target/release/minimappr-sim)
cargo build --release -p minimappr-sim
python scripts/birdnet_version_replay_gate.py --seeds 0 1 2 \
    --bluefin-report data/birdnet_v3_compare.json --json data/birdnet_v3_gate.json
```

Bring back the three JSON files. Before reading the comparison, also check `birdnet_v3_species_list_path`: the Bluefin site should get a regional list. The gate script's `write_v3_regional_species_list` makes one from the v2.4 geo model.

## Tests

| File | Tests | Covers |
| --- | --- | --- |
| `tests/test_birdnet_v3_label_migration.py` | 9 | Translator renames and unmapped labels; rule expansion, threshold remap, mixed-vocabulary report, inputs untouched; a rules engine matching a V3 label only under V3, WARNING for unmappable labels, file not rewritten; default coyote threshold; retention clone, bound remap and builtin not double-mapped; permanent labels; routing member, chain and trigger; flag off; taxonomy |
| `tests/test_birdnet_cutover_thresholds.py` | 14 | Per-version set selection; tracker display belief swap; factory two-threshold floors under V3; constraint WARNING, enforcement and PATCH merged-view error; V3 range validation; quantile-matching math; map monotonicity and round trip; committed map agrees with the defaults; scenario TOML; gate bundle render (skips without the sim binary); gate verdict logic; calibration analysis on synthetic scores |

## Open issues

- **Real-site evidence is the blocker** (Bluefin down). All thresholds come from fixtures, sim and the drone-dataset site clips, with no ground truth on real site audio.
- **Calibration is unfiltered.** The calibration compares unfiltered top-1 for both models, while production v2.4 applies its geo filter. Re-derive with the regional list once Bluefin is up.
- **No geomodel list yet.** V3 needs a regional species list in production. A V3.0.5 geomodel list is still left to a later stream; the gate uses the v2.4 geo list as a stand-in.
- **The omni context's 21 s window** (`omni_scan_window_seconds`) is diluted under V3, which has a 6 s maximum window and takes the maximum over windows, so it is handled. Scores there were not separately calibrated.
- **Not swapped per version:** `tracking_species_log_odds_prior_probability` (0.1 → 0.109 matched, effectively unchanged) and the log-odds caps.
- **`rules.py`** grew from 517 to 531 lines; it was already over 500.
- **Stored rows** keep their v2.4 labels (see §1).
