# Closeout Phase D: real `ClipSetLocalizer` and SRP contour calibration (D1–D3)

Branch `field-mode-closeout`, one commit per step:

| Step | Commit | Result |
| --- | --- | --- |
| D1 `OnsetClipSetLocalizer` | `8fac19c` | New localizer plus 6 unit tests on real FLAC clips; builds for wasm32, Android and iOS |
| D2 rendered-audio Layer 2 mode | `e3e12b0` | Opt-in mode plus nightly sweep. **Multi-bird entity scenarios mostly fail with the real localizer** (table below) |
| D3 σ-aware SRP scoring | `500b28d` | Max-pool each pair over ±k·σ_pair, with k = 1.25 |
| D3 temperature refit | `5b312a8` | T 0.002 → 0.0013. `srp_contour_coverage` un-ignored; it fails before and passes after |

Suites at `5b312a8`:
- **Rust** (`cargo test --workspace`): 581 passed, 8 ignored.
  - The baseline was 573 passed, 2 ignored.
  - New: 6 D1 tests, 1 pooling test, and `srp_contour_coverage` now runs.
  - Still ignored: `graceful_degradation` plus the 7 opt-in audio entity scenarios.
- **Python**: see the end of this note.

## D1: `minimappr-core/src/mesh/onset_clip_set_localizer.rs`

**What it does.** It implements `ClipSetLocalizer` (`localize`, `localize_clusters`, `classify_cluster_window`). Behind the `flac-codec`
feature, as `clip_codec` is.

1. **Decode and align** (`onset_clip_alignment.rs`).
   - `decode_field_clip`, then one windowed-sinc pass (`srp_layout::clock_resampling`) onto the call's shared window on the island
     reference clock.
   - `NodeClockMapping` becomes a resampler clock: rate ratio 1/(1 − ε), anchored at the clip's first sample.
   - A clip that fails to decode, or covers less than half the window, is listed as rejected. It does not fail the call.
2. **Cluster.** `OnsetClusteringInput` (one mono sensor per phone, sensor id = node id) goes to `localize_onset_clusters`. Only
   `Accepted` clusters are kept.
3. **Audibility gate** (added after a measurement). A cluster must have at least one node at or above
   `minimum_loudest_node_band_snr_db`.
   - The threshold is 6 dB, the onset detector's own `active_band_snr_db`, well under the 10 dB Stage-1 trigger.
   - Why: pure-noise windows produce "accepted" clusters at 2.5–5 dB, and those still fit a point at χ²/dof ≈ 2. So a goodness-of-fit
     test cannot reject them.
4. **LM polish.**
   - Each cluster's reference-relative `tdoa_s` becomes `PairTdoaObservation`s for `JointLmSeedRefiner`.
   - σ_pair = √(floor² + σ_clock,a² + σ_clock,b²), with a 25 µs floor. Each node's layout σ comes from its clip.
   - Height band: the mean phone height minus 1.4 m, plus 0–20 m.
   - The covariance is scaled by χ²/dof when that exceeds 1.
5. **Attribution evidence.** It is moved from 24 kHz window samples onto the host's 48 kHz reference samples, the convention
   `call_attribution` reads.
6. **Classification.** A host-supplied `Box<dyn Fn(&ClusterClassificationInput)>` receives the classifying node's aligned audio.
   The aligned window is cached per call, so LOCO reruns do not decode again.

All constants live in `OnsetClipLocalizerConfig`.

**Tests.** They render analytic FM calls on drifting phone clocks, then decimate and FLAC-encode them exactly as a phone does:
- one bird: under 1 m, within 4σ, and evidence arrivals land on the true arrival;
- two birds 0.5 s apart in one window: two clusters, each within 1.5 m of its bird;
- **overlapping calls** (the second starts 0.15 s into a 0.35 s call): measured to **merge into one cluster on bird 0**. This is the
  clustering failure the arrival-time stand-in cannot have. The test pins only that no phantom fix lands between the birds;
- noise-only windows give no fix, and a truncated clip is left out;
- the classifier callback sees the right node's audio.

**Deviations, both from D2 findings (shipped in the D2 commit)**
- **A window fails only when no accepted cluster yields a fix.** The stand-in fails the window when its strongest call fails. Here
  the strongest onset cluster is often a two-node or bearing-only cluster, with a full-layout cluster behind it.
- **Seed-only clusters.** When onset refinement keeps the SRP seed (no TDOAs, because its LM point scored below 0.9 × the seed), the
  cluster still has a full-layout position, often a correct one. It is published with the pipeline's own covariance plus the mean
  layout σ², not dropped. JointLmSeedRefiner cannot run without TDOAs.

## D2: rendered-audio mode (`minimappr-mesh-sim`)

**What runs in audio mode**
- `AcousticMode::RenderedAudio` (`audio_world.rs`) renders every scripted call once with the Layer 1 renderer, using the same
  device clocks, trajectories and atmosphere as the arrival world, and ground reflection on.
- Layer 1 broadband ambient is set so its share of the call band equals the arrival world's in-band ambient (30 dB). That is what
  Stage-1, readiness and the classifier SNRs are computed against.
- Clips come from `FieldClipDecimator` plus `encode_field_clip`, the phone path.
- The host gets `OnsetClipSetLocalizer`, wrapped in `audio_clip_localizer.rs`.
- Sync chirps, Stage-1, readiness and the classifier stand-in stay on arrival times. The classifier stand-in is fed from truth for
  the classifying node's window.
- Refused rather than rendered differently: capture stalls and placement attenuation.

**Dependency direction.** Mesh-sim already depended on `minimappr-sim` for its physics, so there is no cycle and no feature flag.

**Scoring (sim only; the localizer never sees truth)**
- A cluster's candidates are the calls still sounding at a template node when one of its led onsets fired, with 50 ms slack. Of
  those, the one nearest the fix wins, after a proper rigid fit of the host frame onto the true phone positions.
- A first version matched onsets to the arrival time only. It misread late sub-band onsets of a sweep as echoes.

**How to run it**
- `cargo test --release -p minimappr-mesh-sim --test audio_entity_scenarios -- --ignored`: ignored, for nightly.
- `nightly_mesh_seeds --audio [--audio-waveform sweep|trill|harsh] [--no-ground] --scenario a,b,…`.
- About 2.5 s per run, against about 5 ms for the stand-in.
- The mode reaches the scenarios through a thread-local override at the single `run_checked` funnel
  (`run_scenario_by_name_in_mode` / `_with_audio`).

### Entity scenarios: stand-in vs real localizer

50 seeds from 1000; FM-sweep calls of 0.3 s:

| Scenario | Stand-in pass | Real localizer pass | Dominant real failure |
| --- | --- | --- | --- |
| countersinging_pair | 50 / 50 | **9 / 50** | duplicate entities (27), fix > 3 m gate (14) |
| flock_echo | 49 / 50 | **7 / 50** | spurious secondary clusters scored as echoes surfaced or promoted |
| asymmetric_bouts | 50 / 50 | 49 / 50 | – |
| same_band_two_species | 50 / 50 | **3 / 50** | duplicate entities (20), a bird reaching no entity (11), fix > 3 m |
| omni_to_localized_dedup | 50 / 50 | 26 / 50 | "0 unlocalized bouts": the real localizer *does* localize the 55 dB bouts the scenario expects to stay unlocalized |
| collinear_trail | 50 / 50 | 28 / 50 | 20–30 m errors outside the 4σ + 0.5 m region |
| couples | 50 / 50 | 50 / 50 | – |

**Other waveforms** (seeds 1–20, under the earlier arrival-time matching rule, which mainly affects flock_echo):

| Scenario | Sweep | Trill | Harsh |
| --- | --- | --- | --- |
| countersinging_pair | 1 | 0 | 0 |
| same_band_two_species | 0 | 0 | 0 |
| omni_to_localized_dedup | 9 | 20 | 7 |
| collinear_trail | 14 | 10 | 17 |
| asymmetric_bouts | 20 | 19 | 9 |
| couples | 20 | 20 | 20 |

Without the ground reflection the results barely move (countersinging 3/20, flock 1/20, same-band 0/20, collinear 14/20), so the
ground path is not the cause.

**What fails, from per-cluster dumps**
- **Single-bird windows localize well:** usually 0.3–1 m.
- **Wrong basins.** In windows holding two birds (0.5–1.6 s apart), onset-frame SRP often puts a frame's peak in a wrong basin
  (12–60 m off). The TDOA votes are then taken near that wrong seed, so LM confirms the wrong point with a small σ.
- **The correct cluster gets flagged as an echo.** The same onsets cluster at the right position too, but that cluster is often marked
  `Echo / no_independent_onsets` behind the wrong one.
- **Spurious secondary clusters** (phantoms at the height-band edge) become extra entities.
- The likely root cause is the 300 Hz narrowband onset frames on a widely spaced single-mic layout: a fringe-ambiguous SRP map. The
  pipeline was built around 5 cm tetra arrays.
- This is the Python server's algorithm (a parity port), so the server would behave the same on phone layouts.

**Needs a decision.** Before relying on the onset localizer for phone meshes, choose one:
1. clip SRP (`srp_layout`) per window, then onset attribution;
2. wideband or envelope frames for inter-node pairs in onset clustering;
3. keep the arrival-time path (per-node onset picks feeding `ArrivalTimeLocalizer`) as the phone localizer.

The audio mode is the harness to judge any of them.

## D3: σ-aware SRP scoring and the temperature refit

**What changed**
- `LayoutNode` gained `position_std_m` and `clock_offset_std_s`.
- `WeightedPair::declared_lag_tolerance_s` = k·σ_pair. σ_pair = √((σ_a² + σ_b²)/c² + σ_clk,a² + σ_clk,b²); intra-node pairs get 0.
- Exact scores max-pool the curve over ±tolerance. Envelope scores widen each interval by it.
- `RobustScoringConfig::declared_sigma_pooling_multiplier` k = **1.25**.
- Zero declared σ reproduces the old scores exactly; every existing caller passes 0.
- The gate's trials declare the σ they are perturbed with: max(0.10 m layout, 0.10 m height) and 10 µs.

**Choosing k** (calibration training seeds, at the temperature floor):

| k | Trials covered | Notes |
| --- | --- | --- |
| 1.0 | 83 % | the rest need T ≥ 0.005, back into the old fan trap (11/24 inside fans) |
| 1.25 | 93 % | |
| 1.5 | 97 % | |
| 2.0 | 99 % | |

At k = 2 the temperature is meaningless (it fits to the ladder floor). k = 1.25 is the value that leaves the temperature a real job.
k is a fitted parameter, chosen on training seeds only.

**Temperature refit**
- `cargo run --release -p minimappr-sim-gates --example calibrate_region_parameters -- --only srp --srp-trials 400 --held-out-srp 48`
- Result: **0.0013**; training 0.945, held-out 0.917, held-out inside 1/24 fans, median region 1.0 m².
- The ladder now reaches 1e-4.
- **To rerun** (if the declared σ, the gate perturbation or k change; for example a change to C3's phone-height σ that is carried into
  `RealisticPerturbationConfig::phone_height_std_m`): run the same command, copy the FITTED value into
  `RobustScoringConfig::likelihood_temperature`, then check with the commands below.
- The SRP gate's perturbation uses its own `phone_height_std_m = 0.10`, not the mesh's `assumed_phone_height_sigma_m`. So the
  pending C3 decision does not touch this fit unless it is carried over.

**Gate `srp_contour_coverage`**

| | Before (k = 0, T = 0.002) | After (k = 1.25, T = 0.0013) |
| --- | --- | --- |
| PR, 24 seeds | **FAIL** 3/24 = 0.125; inside area 0.02 m² | **PASS** 23/24 = 0.958; 1 inside fan (0.08); median 1.0 m²; inside error median 0.44 m, P95 2.07 m |
| Nightly 200 @ 100000 | 0.180 (sim_gates.md) | **FAIL** 197/200 = 0.985 (over the 0.98 ceiling) |
| Nightly 200 @ 300000 | – | PASS 0.970 |
| Nightly 200 @ 500000 | – | PASS 0.960 |
| Nightly 200 @ 700000 | – | PASS 0.950 |
| Nightly 200 @ 1300000 | – | PASS 0.960 |

- Nightly pooled: 965/1000 = 0.965. Inside fans 0–7 %, median region 0.6–0.9 m².
- Inside error median is 0.34–0.37 m; it was 0.27–0.34 m before. Pooling flattens the peak.
- Commands:
  - `cargo test --release -p minimappr-sim-gates --test statistical_coverage srp_contour_coverage`
  - `scripts/nightly_sim_gates.sh 200 <start> srp_contour_coverage`
- The LM-covariance fallback from the plan was not needed.

**Open: the gate over-covers sometimes.** It fails about 1 nightly sweep in 5, from over-coverage (0.985). The 400-trial fit lands
exactly on 0.945, so this is sampling spread (SE ≈ 0.016 at n = 200). A slightly lower T would trade it for more under-coverage risk;
it was left at the fitted value.

## Left open

- **D2 multi-bird failures of the onset localizer on phone layouts** (above): needs the decision listed there.
- The audio mode renders only bird calls. Stalls, pockets and speech are not rendered.
- The D1 25 µs pair-timing floor and the seed-only covariance were not calibrated separately. In D2, `collinear_trail` errors fall
  outside 4σ + 0.5 m, which suggests the σ is optimistic when the basin is wrong.
- SRP gate nightly: about a 1-in-5 over-coverage fail.

## Python suite

`python -m pytest tests/ --ignore=tests/test_soak_harness.py`: 2083 passed, 9 skipped (baseline 2083 / 9). Phase D touches no Python.
