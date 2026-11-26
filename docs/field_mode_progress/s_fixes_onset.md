# S-fixes onset: S2 onset-SRP localization fixes from the S gate

Branch `fm/s-fixes-onset` (from `field-mode` @ 753f5ea, "Merge fm/s-gate into field-mode").
Everything is on the onset path (`localization_onset_clustering_enabled`). Flag-off behaviour is
unchanged. New config keys live under `# --- Field Mode S-fixes onset ---` in `config.py`,
`settings_store.py`, `main.py` and `core/config_groups.py`. The main ones are also shown on the
`loc:site:onset_srp` pipeline node.

## Root causes

1. **Loud source 2.3 m from one node placed ~52 m out** (`gate_omni_then_localized`). Two causes
   stacked:
   - *Weight share.* ML pair weights q²/(1+2q) grow roughly linearly with SNR. For frames
     templated on the near node, both mics of each intra-node pair are about 20 dB hotter than
     any cross-node pair, so the three compact pairs carried 95–100% of the frame weight.
     Compact pairs are range-blind: every point on the node's bearing ray scored the same (SRP at
     the truth was 0.930, versus 0.928 at points 50–2000 m out).
   - *Sampling.* The layout-centred polar grid samples a source 2 m from one node at about 10°
     (azimuth) and 46° (elevation) steps as seen from that node. The truth scored 0.71 while the
     best nearby grid point scored 0.48, so the far end of the ray won. LM then ran off to
     100–2000 m (`range_refined` with range observability around 0.001, published as a position).
   - The far-field model-selection guard (`far_field_model_selection.py`) is **not** involved:
     it acts only on compact single-array solves, not on multi-node LM.
2. **Far-field range overclaim** (faint bird, second bout at 391 m against a true ~70–80 m). At
   70 m from a 20 m layout, the difference between range 70 m and range ∞ is a ~0.1–0.3 ms
   *differential* across pairs, about one carrier cycle. Narrowband frames use envelope SRP
   (ms-wide), so the SRP profile along the bearing is flat from about 35 m to 1000 m (≥0.95 of
   peak). LM on fine-structure votes therefore lands anywhere along the bearing, depending on
   which cycle it picks. Its residual-scaled Jacobian covariance called that observable. Seed-only
   multi-node results also carried a 0.7 m covariance (peak spread) with no range observability.
3. **Height flips to the ground image** in planar layouts. Cross-node TDOAs are symmetric about
   the node plane, the grid's elevation started at −30°, and LM was unconstrained. With 3 nodes
   the TDOAs pin a 3-D *curve*. That made one bird split into twin clusters on either side of the
   plane, or into points along the curve.
4. **Single-mic phone nodes give no detections.** The trigger path classifies only beamformed
   renders, and the classifier refuses fewer than `beamformed_classification_min_sensor_count`
   (2) sensors. Onset-SRP classified from the nearest node, which for a bird next to a phone is
   the phone's single channel. Result: `empty_classification`, every time.

## Fixes

| Module | What |
| --- | --- |
| `onset_cluster_geometry_resolution.cap_intra_node_weight_share` | Multi-node frames: intra-node pairs capped at `onset_frame_max_intra_node_weight_share` (0.35) of the frame weight. Normal frames (10–20%) are untouched. A uniform SNR ceiling was tried first and dropped: it fixed the near-node case but biased canopy heights (z 9 → 5.3 m) by starving the compact arrays' elevation evidence |
| `onset_srp_geometry_prior.node_near_field_points` (+ `onset_srp_grid.augment_spatial_grid`) | Polar shells around every node (0.5–8 m, 8 rings, 5°/10°) |
| `onset_near_node_level_check` | A fix in a node's near field must (a) be louder there than at the farthest node by the spreading prediction minus `onset_near_node_level_tolerance_db` (10 dB), and (b) not be led only by that node's onsets. Secondary clusters failing either check become tentative (shells otherwise attracted mixed-frame ghosts next to nodes) |
| `onset_range_observability` | Cluster SRP sampled along its bearing ridge (max over a ±3° cone, 0.5–1000 m, 96 log steps). Supported interval = ranges ≥ 0.9 × peak. **Observable** (far/near ≤ 2, or narrower than 0.5 × layout aperture): fix kept (moved to the peak if LM left the interval), radial variance ≥ interval half-width². **Unobservable**: `range_bearing_projected` at the profile peak, radial std = interval half-width, `range_observability` = 0.05, interval exported as `LocalizedOnsetCluster.range_interval_m`. **Far-open** (supported set reaches the last profile sample, i.e. still rising at the search limit): never observable regardless of width; the near edge is exported as `range_lower_bound_m` (admissibility-blanked far samples close the interval instead). It never upgrades an unobservable range |
| `onset_srp_geometry_prior` + `onset_height_resolution` | Node plane fitted through node centroids. Ground = `onset_node_height_above_ground_m` (1.5) below it; canopy ceiling `onset_source_max_height_above_ground_m` (50). Planar layouts (≤3 nodes or flatness < 0.1): frame peaks are folded above the plane before clustering. After LM, the mirror twin is chosen by the prior (exactly one admissible), then by SRP (ratio ≥ 1.05). Otherwise the height is reported as **uncertain**: the fix sits on the plane and the vertical variance covers both twins. A vertical SRP column (ground..ceiling) bounds the height variance and pulls z back if it left the band. Non-planar layouts: LM is z-bounded to the slab. The prior deliberately does **not** trim the grid: that split birds whose frames peaked at the below-ground twin (two_species_overlap) |
| `onset_cluster_geometry_resolution.apply_window_level_holds` | The weaker accepted cluster becomes tentative when it: is on the same bearing with overlapping range interval and one range is unobservable; predicts every cross-node TDOA within 1.5 ms of a stronger one (`onset_same_source_max_tdoa_difference_seconds`, planar-ridge twins); or is a single-node, range-unobservable cluster with fewer than `persistence_min_independent_onsets` independent onsets (W2's single-array persistence rule, extended) |
| `onset_classification_node` | If the classification node has too few sensors for the beamformed classifier, use the nearest node that has enough |
| `onset_cluster_attribution.snr_gradient_decision` (S5) | The SNR-gradient tier no longer settles when the target's band covers an interferer's onsets unless geometry alone (spreading, equal source levels) gives the dominance margin minus the spreading tolerance. Band SNR is measured over each cluster's band, so a thrush cluster whose band reached the vireo's trill read 13 dB "dominant" and "consistent", and took the vireo's label. This was latent before: that product's row was overwritten by a later split cluster's row. It surfaced once the split was correctly marked echo. LOCO now decides |

The window's strongest cluster is still never suppressed (the W2 rule). The new checks only
annotate it.

## Results

### S gate (sim matrix, 10 scenes, seed 0)

Command:
`python scripts/server_track_gate.py run --configs all_on only:S2_onset_clustering`.
"Before" is `field-mode` @ 753f5ea on the same machine and day; it reproduces the S gate doc
(all_on 8.83 / dup 1).

| Config | Indiv. recall (sum /10) | Event recall | Swaps | Dup | Echo | Merged | Unassoc. | Median of scene medians | Median of scene p90 | Max scene p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_on, before | 8.83 | 4.47 | 0 | **1** | 0 | 0 | 1 | 0.62 m | 1.38 m | 37.5 m |
| all_on, after | **9.17** | 3.97 | **0** | **0** | 0 | 0 | **0** | **0.33 m** | **0.57 m** | **10.2 m** |
| only:S2, before | 8.50 | 3.50 | 1 | 0 | 0 | 2 | 0 | 0.70 m | 1.22 m | 22.8 m |
| only:S2, after | 7.83 | 3.28 | 1 | 0 | 0 | 1 | 1 | 0.71 m | 0.75 m | 11.8 m |

Per scene (individual recall / median error), before → after:

| Scene | all_on | only:S2 |
| --- | --- | --- |
| onset_overlap_two_species | 1.00 / 15.5 → 1.00 / **0.33 m** (spurious far tanager track gone) | 1.00 / 0.31 → 1.00 / 0.73 m |
| onset_faint_far_bird | 1.00, **dup 1** → 1.00, **dup 0**, far row at 71 m (true 69 m) instead of 433 m | 0.50 → 0.50 |
| onset_same_band_two_species | 1.00 / 0.25 → 1.00 / 0.47 | 1.00 / 11.5 → 1.00 / **0.42 m** |
| gate_countersinging_pair | 1.00 / 0.53 → 1.00 / 0.32 | unchanged (1 merged) |
| gate_omni_then_localized | 1.00 / 1.38 → 1.00 / **0.18 m** | 1.00 / 1.22 → 1.00 / 1.57 |
| gate_flock_echo | 0.33 / 8.9 → **0.67** / 5.0 | **1.00 → 0.33** (see open issues) |
| gate_collinear_trail | 1.00 / 0.23 (p90 16.4) → 1.00 / **0.15 (p90 0.24)**, 0 swaps | 1.00 / 0.70 → 1.00 / 0.25 |
| others | unchanged within 0.5 m | unchanged within 0.5 m |

Far-bird range (surfaced rows of truth > 50 m from the centroid): before 433 m, 81 m, 81 m
(true 69 m); after 71 m, `range_bearing_projected` with radial std of several hundred metres.
**ALL-ON now meets every plan criterion on the sim matrix**: summed individual recall 9.17
(baseline 0.50), 0 swaps, 0 duplicates, 0 echo promotions, 0 merged, 0 unassociated. The S gate
FAILed only on the far-bird duplicate.

Height |dz| of surfaced rows (within 5 m horizontally): all_on median 0.89 → 0.85 m, p90
1.77 → 2.48 m. No surfaced row is below ground. The p90 grows because planar heights are now
reported with their real (wide) uncertainty instead of a flip-prone point.

### Direct localization on the three new scenes (`localize_onset_clusters`, per call)

| Scene | Before | After |
| --- | --- | --- |
| `sfix_loud_bird_near_node` (wren 2.0 m from tetra-c) | 4.9–5.3 m error, plus a second accepted ghost per call at 125–1457 m | one cluster per call, 1.0–2.5 m error (radial bias along tetra-c's ray; radial std covers it) |
| `sfix_far_bird_small_baseline` (bird ~76 m, 10 m triangle) | 2–3 accepted per call; some confidently wrong (48 m with 0.3 m std) | one per call, `range_bearing_projected`, bearing within 10°, interval contains the truth |
| `sfix_planar_canopy_bird` (bird 9 m up) | z 7.7–8.2 m | z 9.5 / 11.0 m with vertical std covering the truth; never below ground |
| `gate_omni_then_localized` loud bouts | 1.2–3.1 m plus ghosts 112–1400 m | 0.2–1.6 m, no ghosts |

### Single-mic phone node (tetra-c → phone in the omni scene, only:S2)

Before: 2 candidates, both `empty_classification`, no localized row. After: classified from
tetra-b's render, one localized House Wren row (0.95). Its position is 4.3 m off: a phone adds
only one mic to the layout.

### Cost

ALL-ON onset clustering: median 0.62 s / max 2.0 s per window (before 1.07 / 5.0 s; the machine
was more heavily loaded then). The shells add about 4.6k grid points per node. The range profile
and the vertical column are vectorised (~5k points per cluster).

## Tests

Full suite: `python -m pytest tests/ --ignore=tests/test_soak_harness.py -q -p no:cacheprovider`
gives **2013 passed, 18 skipped** (with `MINIMAPPR_SIM_BINARY` pointing at a built
`minimappr-sim`). That includes the S gate pytest (`tests/test_server_track_gate.py`), which
failed on collinear trail before the attribution guard.

- `tests/test_onset_s_fixes_units.py` (12): the attribution band-overlap guard, weight cap, node-plane prior / mirror / bounds,
  shells, mirror resolution (prior, SRP, ambiguous), range profile (flat → projected,
  peaked → kept and LM pulled back), the window holds, classification-node choice.
- `tests/test_onset_s_fixes_scenarios.py` (3): the three `sfix_*` scenes rendered and run
  directly.
- `tests/test_onset_srp_clustering.py`: two position assertions changed from 3-D to horizontal.
  The layout is planar (nodes at z = 0), so height may now be reported as uncertain; the first
  test also checks that the vertical std covers the truth.

## Open issues

- **Attribution band levels** (S5): the fix above is a guard. A proper fix measures each
  cluster's level only in the bands of its own onsets.
- **gate_flock_echo only:S2 1.00 → 0.33.** Two rows: a strongest-in-window cluster projected
  along a high-elevation bearing (35 m up; the 50 m canopy ceiling admits it), and one bout
  tracked onto another bird's CV track. ALL-ON improves 0.33 → 0.67. A lower ceiling or a
  persistence rule for the *strongest* single-node bearing-only cluster would help, but both
  are judgement calls left for review.
- **Near-node radial bias.** For a bird 2 m from a node, some calls still land ~2.5 m farther
  out along the node's ray. The range interval (from the layout centroid) is honest but wide
  (radial std 5–16 m). A profile from the dominant node would be tighter.
- **Ground height** is a config assumption (node plane minus 1.5 m). There is no per-node height
  above ground in `NodeSpec`.
- **Range interval semantics.** It is measured from the layout centroid along the ridge, with
  support ratio 0.9 of the SRP peak. Calibrate on golden bundles together with confidence.
- Collinear layouts have a full rotational ambiguity about the node line. Only the vertical
  mirror is handled; ring twins are left to the echo checks (they were marked echo in the trail
  scene).
