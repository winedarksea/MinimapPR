# W1-birdnet-v3: BirdNET+ V3.0 on the server (2026-11-22)

Branch `fm/birdnet-v3`. The default behaviour has not changed: `birdnet_model_version` defaults to `"2.4"`.

## Summary

- **Model:** BirdNET+ V3.0.1 ONNX, fetched from Zenodo with pinned MD5s. Weights and labels are Apache 2.0.
- **Classifier:** `minimappr/classifiers/birdnet_v3.py` runs when `birdnet_model_version="3.0"`. It resamples to 32 kHz, uses variable 2–6 s windows (any length from 2 to 12 s can be configured), has a variance/NaN silence guard and a regional species-list hook. It returns the same result shape as v2.4, so labels go through the existing BirdNET result path.
- **v2.4 → V3 label map:** 6,502 of 6,522 v2.4 labels map to V3. 20 do not: 18 birds, plus `Noise` and `Environmental`, which have no V3 class.
- **INT8 verdict: REJECT (provisional, local audio only).** Every variant that quantizes the convolution stack changes the scores far too much. Quantizing only the 11,725-class head keeps accuracy but brings no speedup. **Keep the FP16 pruned model.**
- **Bluefin was down** (`nc -z -G 2 127.0.0.1 2222` failed), so all comparisons ran on local audio. The re-run command is in [Re-running on Bluefin](#re-running-on-bluefin).

## Model source, licence, checksums

Zenodo record [24362980](https://zenodo.org/records/24362980), "BirdNET+ V3.0 model files", version 3.0.1, published 2026-11-19, DOI 10.5281/zenodo.24362980. It supersedes record 24162518 (3.0). The 3.0.1 release re-exported the models with conversion fixes; weights and labels are unchanged. Release notes: all-zero input no longer returns NaN, and the FP16 ONNX files are now mixed precision.

| File | Size | MD5 |
| --- | --- | --- |
| `BirdNET+_V3.0_Global_12K_FP16_pruned.onnx` | 72,317,224 | `6b42d8905f25f856fb8ad91544138914` |
| `BirdNET+_V3.0_Global_12K_FP16.onnx` (optional) | 173,871,766 | `3f284bdeae7ffb38718c391839b58b60` |
| `BirdNET+_V3.0_Global_12K_Labels.csv` | 8,007,144 | `bb25847af8a581e1382081382f01e762` |
| `MODEL_LICENSE.txt` | 11,836 | `77b299ba50d3119cd28c5a2b6fc0ff4b` |
| `ACCEPTABLE_USE.txt` | 2,986 | `92f52349d7a08f74167f096702bac26a` |
| `BirdNET+_Taxonomy_v0.5-Nov2026.zip` (optional, used only to rebuild the label map) | 69,831,066 | `0310312e852326e17e45e5b2dcf64655` |

**Licences**

- Model weights and labels are Apache 2.0, per `MODEL_LICENSE.txt`.
- The taxonomy data is CC BY-SA 4.0. The label map uses its synonym list.
- `ACCEPTABLE_USE.txt` is guidance, not a licence condition. It asks for:
  - no "BirdNET" naming or branding in derived apps;
  - no use in covert surveillance of people;
  - no use in "weapons systems, military targeting, or military surveillance intended to cause harm".

  MinimapPR is a surveillance COP, so **someone should read that section against the intended deployments.**

Files go to `data/models/birdnet_v3/`, which is gitignored. `scripts/fetch_birdnet_v3.py` downloads each file to a `.part` file and renames it only after the MD5 and size match. Model files are only loaded as data.

## Model I/O (measured)

- **Input:** `input` float32 `[batch, samples]` at 32 kHz. The batch and length dimensions are symbolic. Lengths of 1, 2, 3, 7, 12 and 20 s all ran.
- **Outputs:**
  - `predictions` `[batch, 11725]`: probabilities. The sigmoid is applied inside the graph.
  - `embeddings` `[batch, 1280]`.
- **Graph:** opset 18. The spectrogram is computed with DFT MatMuls (`preprocess.dft*`), followed by an EfficientNetV2-style CNN with SE blocks.
- **Labels:** the CSV has 11,725 rows with columns `ix`, `birdnet_id`, `scientific_name`, `common_name_en` and `taxon_group`. Taxon groups:

  | Group | Classes |
  | --- | --- |
  | Aves | 9,788 |
  | Insecta | 802 |
  | Amphibia | 649 |
  | Mammalia | 419 |
  | Actinopterygii | 35 |
  | Reptilia | 15 |
  | Anthropogenic | 14 |
  | Arachnida | 1 |
  | Geophony | 1 |
  | Music | 1 |
- **Silent input:** all-zero input returns finite scores in 3.0.1 (top score 0.25). The guard still never feeds zeros, because 3.0.0 exports returned NaN on silence.
- **Level invariance (house finch):** the top-1 score stays at about 0.91 from full scale down to -80 dB (gain ×1e-4). Faint, distant calls are not penalised by level.
- **Window dilution:** the model pools over the whole window, so a short call loses score as the window grows. A 4.5 s house finch call centred in noise:

  | Window | Score | Latency per run |
  | --- | --- | --- |
  | 3 s | 0.92 | 60 ms |
  | 4 s | 0.90 | 75 ms |
  | 6 s | 0.83 | 127 ms |
  | 8 s | 0.72 | 182 ms |
  | 12 s | 0.48 | 227 ms |

  This is why the default maximum window is **6 s with 3 s overlap** (50%), not 12 s: every call of 3 s or less lands whole in at least one window. The W3 cutover should tune this.

## Implementation

| Piece | File |
| --- | --- |
| Resample, silence guard, padding, windowing (pure numpy) | `minimappr/classifiers/birdnet_v3_audio.py` |
| Labels CSV, regional filter, v2.4→V3 map builder and loader | `minimappr/classifiers/birdnet_v3_labels.py` |
| ONNX classifier | `minimappr/classifiers/birdnet_v3.py` |
| Version switch, `_build_birdnet_v3` | `minimappr/classifiers/factory.py` |
| Wildlife fallback only for animal `taxon_group` | `minimappr/core/classification.py` |
| Config keys (each in one `BirdNET+ V3.0 (Field Mode W1-birdnet-v3)` block) | `config.py` (fields, env loader, validation, `BirdNETV3Config` sub-config with `Settings.birdnet_v3_config()`), `settings_store.py`, `main.py` (GET config, PATCH cross-field validation, restart-required sets), `core/config_groups.py`, `core/pipeline_graph.py` (BirdNET node parameters) |

**Behaviour**

- **Silence guard.** It only rejects input that is numerically degenerate:
  - more than 1% of samples are NaN/inf (`birdnet_v3_max_nonfinite_sample_fraction`); a smaller fraction is repaired with noise;
  - the variance is zero relative to the clip's own DC level. The absolute floor is float32 `tiny`, which is a numerical zero, not a level gate.

  There is **no RMS threshold**: a 1e-6-amplitude tone in 1e-7 noise passes. Windows inside a long clip that have zero variance (for example a zero-filled packet gap) are dropped one by one. Rows of model output that contain NaN are dropped and counted.
- **Length handling** (shortest to longest):
  - Shorter than 2 s: the clip is centred in Gaussian noise at its own noise floor (10th percentile of 50 ms frame RMS, seeded) up to 2 s. It is never zero-padded.
  - 2 to 6 s: one window, sent whole.
  - Longer than 6 s: 6 s windows at a 3 s hop, with the last window flush to the end of the clip.

  All windows go to the model in a single batched `run`. The clip score is the maximum over windows, the same as the v2.4 path.
- **Result fields:**
  - `label` is the lower-cased English common name (the v2.4 convention, so rules keyed on "house finch" still match);
  - `features.model="birdnet_v3"`;
  - `raw_species` is `Scientific_Common`;
  - also included: `scientific_name`, `birdnet_id`, `taxon_group`, `model_release`, `model_precision` and `window_count`.
- **Taxonomy fallback.** V3 includes Gun, Siren, Engine and similar classes, so the model-wide wildlife fallback applies only when `taxon_group` is an animal group.
- **Threading.** Inference is synchronous and runs in the fusion classification worker threads, never on the event loop. A single `InferenceSession` is shared (ORT `run` is thread-safe), with `birdnet_v3_num_threads` intra-op threads.
- **Regional subset.** `birdnet_v3_species_list_path` points to a file with one scientific name, common name or BirdNET id per line. A list that matches no labels raises an error instead of silently disabling the classifier. Generating that list from the V3.0.5 geomodel is left to a later stream.
- **Missing model file with `"3.0"` selected.** The member is dropped with a WARNING, the same policy as the drone head.

**New keys** (all restart-required)

| Key | Default |
| --- | --- |
| `birdnet_model_version` | `"2.4"`; allowed values are `2.4` and `3.0` |
| `birdnet_v3_model_path` | env only, not PATCHable |
| `birdnet_v3_labels_path` | env only, not PATCHable |
| `birdnet_v3_species_list_path` | `""` (all classes) |
| `birdnet_v3_min_window_seconds` | 2.0 |
| `birdnet_v3_max_window_seconds` | 6.0 |
| `birdnet_v3_window_overlap_seconds` | 3.0 |
| `birdnet_v3_max_nonfinite_sample_fraction` | 0.01 |
| `birdnet_v3_pad_noise_floor_percentile` | 10 |
| `birdnet_v3_num_threads` | 4 |

V3 reuses `birdnet_trigger_min_confidence` and the routing spec's `min_confidence`. These thresholds are calibrated for v2.4 and must be re-tuned in W3-birdnet-cutover before the default changes.

## v2.4 → V3 label map

Build it with `python scripts/build_birdnet_label_map.py`. It writes `minimappr/assets/birdnet/v24_to_v3_label_map.csv`, which is committed (6,522 rows) and listed in package-data.

Each v2.4 label is matched by the first tier that succeeds:

1. Manual override for v2.4 pseudo-species: Dog, Human vocal / non-vocal / whistle, Power tools.
2. Exact scientific name.
3. A synonym listed in the BirdNET+ Taxonomy `scientific_name_aliases`: genus moves, and lumps such as the redpolls into *Acanthis flammea*.
4. A unique English common name.

| Match method | Count |
| --- | --- |
| scientific_name | 6,203 |
| taxonomy_alias | 272 |
| common_name_en | 22 |
| override | 5 |
| **unmapped** | **20** |

**Unmapped labels**

- Birds (18):
  - *Aerodramus bartschi*, *Aerodramus maximus*
  - *Anous ceruleus*
  - *Ardenna bulleri*
  - *Circus melanoleucos*
  - *Coracina caesia*
  - *Corythaixoides concolor*
  - *Fregata andrewsi*, *Fregata ariel*
  - *Pachycephala fulvotincta*
  - four *Phylloscartes* bristle-tyrants
  - *Pterodroma cervicalis*
  - *Telespiza ultima*
  - *Trachyphonus purpuratus*
  - *Turdus poliocephalus*
- Classes with no V3 counterpart: `Noise_Noise`, `Environmental_Environmental`.

**Notes on the table**

- 27 V3 classes receive more than one v2.4 label. These are lumps, plus three v2.4 human classes that all map to *Homo sapiens*.
- 5,254 V3 classes have no v2.4 source.
- I tried matching by specific epithet plus the last word of the common name and dropped it, because it produced false pairs (*Sturnia blythii* → *Onychognathus blythii*).

## Comparison: FP16 vs INT8 vs v2.4 (local audio)

### INT8 accept gate (set before measuring)

INT8 is accepted only if all of these hold against FP16 pruned:

- top-1 agreement ≥ 0.95 on clips where FP16's top score is ≥ 0.5;
- median per-clip Spearman ≥ 0.90 over the union of both models' top-50 classes;
- median latency speedup ≥ 1.2×.

With fewer than 30 confident clips, any verdict is provisional.

### Audio

**Evaluation set:** 63 unique clips, 16.5 minutes in total:

- 13 MinimapPR detection exports from `~/Downloads` (`det-*_listening.wav`, 30 s, 32 kHz);
- 1 local detector recording;
- `tests/house_finch.wav`;
- 4 coyote clips, 40 ambient clips and 10 drone clips from `drone_dataset/`.

**Calibration set** (static INT8, does not overlap the evaluation set): 36 files, 119 windows of 6 s:

- 30 `drone_dataset/ambient` clips;
- 4 coyote clips;
- 2 detection exports from the repo root.

**Weakness:** there is very little real bird audio (about 15 clips, mostly house finch and backyard birds) and no ground truth. Only 21 clips are confident under FP16. The Bluefin snippet store should replace both sets.

### Results

| FP16 pruned vs … | Top-1 agreement, confident clips (n=21) | Top-1 agreement, all clips | Median Spearman (top-50 union) | Median max \|Δp\| | Speedup | Size | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INT8 dynamic (Conv+MatMul+Gemm, per-channel) | 0.62 | 0.40 | −0.12 | 0.51 | 0.1–0.5× (ConvInteger is slow) | 37.4 MB | REJECT |
| INT8 static QDQ (percentile 99.99) | 0.33 | 0.22 | −0.42 | 0.73 | 1.4–3.3× | 39.4 MB | REJECT |
| INT8 static, SE + depthwise + stem kept FP32 (minmax) | 0.00 | 0.00 | −0.70 | 0.85 | 1.6× | 50.3 MB | REJECT |
| INT8 dynamic, SE kept FP32 | 0.67 | 0.38 | −0.12 | 0.54 | 0.26× | 48.4 MB | REJECT |
| INT8 dynamic, head only (MatMul/Gemm) | **1.00** | 0.98 | **0.998** | 0.002 | 0.87× | 97.5 MB (convs upcast to FP32) | REJECT (no speed or size gain) |

Latency was measured on an Apple-silicon laptop with 4 threads; the median is per clip. Two runs gave noticeably different figures (FP16 pruned: 112 ms per clip, 30 ms per audio-second in the first run; 55 ms and 22 ms in the second), so treat the speedups as ranges.

For reference:

- **FP16 full vs FP16 pruned** on the house finch clip: max |Δp| 0.049, same top-1.
- **The FP32 upcast is exact** (Δ = 0), so the conversion step is not the cause of the INT8 error.

**v2.4 vs V3 FP16:**

- v2.4 TFLite latency: 54 ms per clip, 11 ms per audio-second. V3 FP16 pruned is about 2–3× slower per audio second at the default 6 s / 50% overlap windows.
- Confident clips (top score ≥ 0.5): 17 for v2.4, 21 for V3.
- On the 12 clips where both are confident, top-1 agreement after label mapping is **0.75**.
- There is no ground truth, so this measures disagreement, not which model is right.

### Verdict

**INT8: REJECT (provisional, only 21 confident clips).**

- The failure is structural, not a calibration-sample issue. Weight-only dynamic quantization of the convolutions already gives max |Δp| of about 0.2–0.5 per clip, and static activation quantization makes it worse.
- This EfficientNet-style network with SE blocks and SiLU quantizes poorly without QAT, a known pattern.
- Head-only INT8 is accurate but gives nothing on CPU.

**Next steps for INT8:**

1. A per-layer sensitivity sweep (ORT `qdq_loss_debug`) to find the few convolutions that break, keeping those in FP16.
2. An FP16 trunk with an INT8 head, to save size on phones.
3. Asking upstream for a QAT or INT8 export.

Until then, ship FP16 pruned (72 MB) on the server, and treat the phone and PWA INT8 assumption in the plan as at risk.

## Re-running on Bluefin

```bash
git pull && git checkout fm/birdnet-v3
python scripts/fetch_birdnet_v3.py
python scripts/quantize_birdnet_v3.py --variants dynamic static --calibration-audio <half of the snippet dirs>
python scripts/quantize_birdnet_v3.py --variants dynamic --op-types MatMul Gemm --output-suffix _head
python scripts/compare_birdnet_versions.py --audio <other half of the snippet dirs> --max-clips 500 \
    --model int8_dynamic_head=data/models/birdnet_v3/BirdNET+_V3.0_Global_12K_pruned_INT8_dynamic_head.onnx \
    --output-json data/birdnet_v3_compare.json
```

`compare_birdnet_versions.py` reads wav, flac, mp3 and ogg, de-duplicates clips by content hash, runs v2.4 through the installed `birdnet` package (pass `--skip-v24` to leave it out), and prints the gate verdict for each INT8 variant.

## Tests

- `tests/test_birdnet_v3.py` (22 tests). They use a fake ONNX session, so no model file is needed. They cover:
  - the silence guard: zero, DC, NaN-heavy and NaN-repair input, and a faint (1e-6) but valid signal;
  - zero-gap windows;
  - lengths of 1, 2, 7, 12 and 20 s;
  - that padding is noise, not zeros;
  - batching and the result shape;
  - NaN model output;
  - the regional filter;
  - the animal-only wildlife fallback;
  - version selection: the default builds v2.4, `"3.0"` builds V3, a missing model drops the member, and an invalid version raises.

  Seven more tests use the real model and are skipped with the fetch command when it is absent: output width, house finch at full scale and at −60 dB, and all five lengths.
- `tests/test_birdnet_label_map.py` (5 tests): the match tiers, the ambiguous-name guard, write/load round trip, coverage of the committed table (at most 20 unmapped), and that every target exists in the V3 labels when the labels have been fetched.

## Open issues

- **Thresholds.** V3 and v2.4 confidences are on different scales. W3-birdnet-cutover must re-tune `birdnet_trigger_min_confidence` and the routing `min_confidence` before the default changes.
- **Not yet migrated.** Rules, snippet-retention buckets and stored labels still use v2.4 labels. The map table is ready, but nothing applies it yet.
- **Backend availability.** `classifier_backends_available` still reports `birdnet` as needing the `birdnet` package, even when V3 only needs `onnxruntime`. This only affects the status display.
- **Rust sidecar.** It passes `MINIMAPPR_*` env to its Python classifier subprocess by inheriting the environment. V3 keys set only through PATCH reach it through `config.yml` on restart. This has not been tested with the sidecar running.
- **Acceptable use.** The military-surveillance clause in `ACCEPTABLE_USE.txt` needs a human decision for MinimapPR deployments.
- **Licensing of the map table.** It mixes v2.4 labels (CC BY-NC-SA 4.0) with taxonomy synonyms (CC BY-SA 4.0). Fine for internal, non-commercial use; check before redistributing.
- **Embeddings.** The `embeddings` output (1280-d) is not exposed. Calibration-capture embedding persistence could use it later.
