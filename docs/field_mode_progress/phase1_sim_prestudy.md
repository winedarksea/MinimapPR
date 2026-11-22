# Phase 1 sim pre-study: chirp band, waveform, slot, rate compensation, self-recording mic

Stream W2-core-sync (`build_out_plan.md`). Inputs to the Phase 1 sync spike (`field_mode_test_strategy.md`, roadmap row "1 · Sync spike"): which chirp to put on real phones first. Every number here comes from the Layer 1 simulator (`minimappr-sim`, see `sim.md`), so read it as a ranking of options and a list of failure modes, not as field performance.

## How to reproduce

```
cargo run --release -p minimappr-core --example phase1_sync_prestudy            # 8 seeds per case, ~40 s on an M2
cargo run --release -p minimappr-core --example phase1_sync_prestudy -- --quick # 2 seeds per case
```

Seeds are fixed (1000–1007). The example and the scenario tests (`minimappr-core/tests/sync_sim_scenarios.rs`) share one harness, `minimappr-core/tests/support/`. Each trial:

1. builds a sim scene of phones (default: 6 phones over about 28 m, heights 0.9–1.5 m, ground plane, 3–6 random trunks, 35 dB SPL ambient, clocks random within ±20 ms and ±60 ppm, d_ii drawn from 1–15 cm, phone limiter on);
2. replaces the simulator's chirps with `minimappr-core` waveforms at 85 dB SPL at 1 m (so the matched-filter template is exactly what was played), one phone per slot, plus a closing chirp from phone 0;
3. picks every arrival on each phone's own sample clock with `SplitPartDetector` / `FirstArrivalDetector`. The phone knows its slot only to ±5 ms (the coarse clock anchor);
4. ranges every pair (BeepBeep) three ways: no rate compensation, tracked rates (truth plus 0.5 ppm noise, standing in for a heartbeat-fed clock filter), and the closing chirp (double-sided);
5. removes the speaker–mic geometry bias (`correct_for_speaker_mic_offsets`, using true speaker vectors), solves the clock graph in two passes, solves the layout (MDS + robust LS, true heights), and scores everything against simulator truth.

Metric definitions:

- **Remote miss**: the share of phone-to-phone arrivals not detected (14 dB envelope SNR over the median).
- **Pick spread**: p95 of |pick − median pick| over remote arrivals. Every pick also carries a common ~30–40 µs receive-chain group delay (the sim's 20 kHz order-4 mic low-pass). That delay is the same on the self and remote picks of one phone, so it cancels in both distance and clock offset.
- **Wrong lock**: |pick − median| > 200 µs (a reflection, sidelobe or precursor picked).
- **Range error**: the estimate minus the mean of the two true speaker→mic flights (what two-way ranging measures).
- **Graph offset error**: the solved clock offset minus truth at the round midpoint, per non-reference phone. The plan's gate is < 10 µs clean and < 50 µs with chaos.
- **Layout error**: horizontal error per phone after the best rigid fit, mirror allowed.

## Recommendation for the real-phone spike

1. **Waveform: up-down hyperbolic FM, 50 ms (25 ms up then 25 ms down across the full band), with each half picked separately and the two arrivals averaged.** A single LFM or HFM sweep turns walking Doppler into a ranging bias of 25–28 cm (p50) at 1.2 m/s with random headings. HFM keeps its peak height under Doppler but still shifts it. The up and down halves shift in opposite directions, so their mean cancels the bias: 2.5 cm p50 while walking. Still phones lose nothing (range p95 1.2 cm either way), and the half-to-half difference (`doppler_skew_samples`) measures relative radial speed for free.
2. **Band: 13–16 kHz first, 15–18 kHz as the comparison; drop 18–20 kHz.** 13–16 and 15–18 kHz are equally accurate up to 45 m spacing. 13–16 kHz ranges further: at 80 m one pair misses 25% of arrivals against 79% at 15–18 kHz, and 18–20 kHz already misses 42% at 60 m. 18–20 kHz has only 2 kHz of bandwidth, so its main lobe is wider and multipath biases it more: 2–3× the pick spread, 1–4% wrong locks and a 3–8 cm range p95 even at 28 m. The sim's only roll-off is the mic low-pass, and real OEM paths often roll off above 16–18 kHz, which would widen the gap in favour of 13–16 kHz.
3. **Slot: 0.25 s, or 0.20 s when spacing stays under ~35 m.** With still phones and any rate compensation, slot length does not matter: graph offset p95 is 10–11 µs for 0.20, 0.25 and 0.40 s. What short slots buy is robustness to unseen motion. With a 5 cm/s body sway, offset p95 grows from 66–82 µs at 0.20 s to 139–162 µs at 0.40 s. Without rate compensation, drift alone takes it from 30 to 75 µs.
4. **Rate compensation: tracked rates by default, the closing chirp only for phones without a valid rate.** Both reach about 10 µs offset p95 with still phones, against 30–75 µs uncompensated. The closing chirp measures rate across the whole round, so any motion corrupts it: under sway its range p95 is 4–8 cm, against 1.2–1.7 cm with tracked rates. This matches the plan's "keep the lower residual".
5. **Self-recording mic: the near mic (bottom speaker to bottom mic).** With the matched-filter onset and precursor rejection, both mics pick within about 25 µs spread on foam, in the hand and on a hard table. A raw-threshold onset locks onto the body-conduction precursor on every surface: its median error is 360–460 µs. The far mic needs a larger speaker–mic correction (next point) and gains nothing in the sim, because the limiter did not distort timing at 85–95 dB SPL at 1 m. Check that on real phones, which have AGC and harder limiters.
6. **New requirement: per-model speaker→mic *vector* plus phone attitude.** The BeepBeep clock offset assumes equal flights each way, but the speaker sits d_ii from the mic. The two flights then differ by about (v_A + v_B)·û, which biases the offset by up to d_ii/c. In the sim the raw pair offset error is about 295 µs p95 on the near mic and about 700 µs on the far mic, far outside the 50 µs gate. `correct_for_speaker_mic_offsets` fixes it, given each phone's speaker offset in the world frame (the per-model vector rotated by `TYPE_ROTATION_VECTOR`) and the solved layout. That brings the pair offset error down to 13–25 µs p95. Phase 1 should therefore calibrate the speaker→mic **vector** per model (the plan only lists the distance d_ii) and log attitude with every round.
7. **Leading-edge timing for remote picks.** Reading the arrival at the 50% rising edge plus the template's own rise time, instead of at the envelope peak, halves the pick spread (77 → 40 µs p95) and range p95 (2.6 → 1.2 cm). The reason is that ground bounces 0.3–0.5 ms late fall inside the main lobe and pull the peak, not the edge. Self picks stay on the peak.
8. **Walking rounds give distances, not clocks.** While walking, up-down ranging still works, but every two-way clock offset is off by milliseconds (graph p95 about 2.8 ms): the geometry changes between slots. The plan's rule stands: walking phones are skipped for clock sync, and the layout from a walking round only holds for that instant.

Headline (still phones, core 28 m layout, up-down HFM, tracked rates): no missed detections, range p50 0.4 cm and p95 1.2 cm, clock-graph offset p95 7 µs (13–16 kHz) and 10 µs (15–18 kHz), layout p95 0.7 cm. At 45 m spacing: range p95 2.3–4.9 cm, offset p95 24–51 µs, layout p95 1.9–2.3 cm.

## Caveats (what the simulator cannot tell us)

- **No AGC and no OEM DSP.** The sim's receive chain is a fixed mic response, a tanh limiter and a hard clip. Real phones may apply gain changes during the self-chirp. Self-recording headroom and limiter timing distortion must be measured per model (strategy, "Self-chirp surfaces").
- **No speaker directivity and no real OEM roll-off.** Chirps radiate omnidirectionally at the full 85 dB SPL at 1 m in every band, so the absolute ranges in table F (60 m or more) are far too optimistic. Only the relative band ordering is meaningful. The plan's 5–25 m published indoor ranging and the 18–20 kHz beaming argument are not modelled.
- **No frequency-dependent ground and no foliage.** The ground is one broadband reflection (coefficient 0.6), and trunks are point reflectors. There is no scattering from leaves and no excess attenuation, so wrong-lock rates in real woods will be higher.
- **Ambient noise is independent per mic and spectrally tilted (800 Hz corner)**, so there is almost no noise energy at 13–20 kHz. SNR margins are therefore optimistic: no insects, wind or stream hiss in-band.
- **Perfect side information.** The tests give the solver the true d_ii, speaker vectors, barometric heights and speed of sound. Tracked rates are truth plus 0.5 ppm.
- **Constant-velocity motion only.** Walking is a straight line at 1.2 m/s in a random heading per phone, and sway is 5 cm/s. The sim's micro-motion and walking chaos knobs and the accelerometer-weighted clock graph are not exercised: no motion-energy input was given, so all constraints carry the same motion weight.
- **One round per trial.** The clock filter's heartbeat behaviour (rate from spaced rounds, silent-gap detection) is covered by unit tests, not by multi-round sim scenarios.
- **The receive group delay is the same on every simulated phone.** Real phone models differ. The delay cancels in two-way ranging and is absorbed into each phone's clock offset, which is consistent for TDOA because bird calls go through the same capture path. Mixed-model groups should still be checked in Phase 1.

## Results

Tables as printed by the example (first seed 1000, 8 seeds per case).
#### A. Band × waveform, still phones, core ~28 m (tracked rates)

| Case | Remote miss | Wrong lock | Pick spread p95 (µs) | Range p50 / p95 (cm) | Graph offset p95 (µs) | Layout p50 / p95 (cm) |
|---|---|---|---|---|---|---|
| 13–16 kHz lfm | 0.0% | 0.0% | 36 | 0.4 / 1.1 | 25 | 0.4 / 0.7 |
| 13–16 kHz hfm | 0.0% | 0.0% | 38 | 0.4 / 1.2 | 25 | 0.4 / 0.8 |
| 13–16 kHz updown_hfm | 0.0% | 0.0% | 33 | 0.4 / 1.2 | 7 | 0.4 / 0.7 |
| 15–18 kHz lfm | 0.0% | 0.0% | 46 | 0.3 / 1.2 | 29 | 0.3 / 0.6 |
| 15–18 kHz hfm | 0.0% | 0.0% | 48 | 0.4 / 1.2 | 29 | 0.3 / 0.7 |
| 15–18 kHz updown_hfm | 0.0% | 0.0% | 40 | 0.4 / 1.2 | 10 | 0.3 / 0.7 |
| 18–20 kHz lfm | 0.0% | 0.7% | 110 | 1.4 / 3.4 | 47 | 1.1 / 2.3 |
| 18–20 kHz hfm | 0.0% | 1.4% | 115 | 1.3 / 3.7 | 46 | 1.2 / 3.4 |
| 18–20 kHz updown_hfm | 0.0% | 3.2% | 108 | 1.4 / 8.1 | 19 | 1.3 / 4.1 |

#### A. Band × waveform, still phones, wide ~45 m (tracked rates)

| Case | Remote miss | Wrong lock | Pick spread p95 (µs) | Range p50 / p95 (cm) | Graph offset p95 (µs) | Layout p50 / p95 (cm) |
|---|---|---|---|---|---|---|
| 13–16 kHz lfm | 0.0% | 0.4% | 77 | 1.2 / 2.0 | 26 | 0.9 / 1.7 |
| 13–16 kHz hfm | 0.0% | 1.1% | 80 | 1.1 / 2.1 | 26 | 1.1 / 4.1 |
| 13–16 kHz updown_hfm | 0.0% | 1.8% | 82 | 1.1 / 2.3 | 24 | 1.0 / 1.9 |
| 15–18 kHz lfm | 0.0% | 1.1% | 74 | 0.8 / 2.2 | 28 | 0.7 / 1.4 |
| 15–18 kHz hfm | 0.0% | 2.5% | 82 | 0.9 / 2.3 | 28 | 0.7 / 2.7 |
| 15–18 kHz updown_hfm | 0.0% | 4.6% | 94 | 1.0 / 4.9 | 51 | 0.7 / 2.3 |
| 18–20 kHz lfm | 0.0% | 1.1% | 123 | 2.3 / 3.9 | 49 | 2.0 / 6.1 |
| 18–20 kHz hfm | 0.0% | 1.1% | 123 | 2.4 / 4.0 | 52 | 1.9 / 4.9 |
| 18–20 kHz updown_hfm | 0.0% | 3.9% | 138 | 2.5 / 8.3 | 26 | 2.3 / 5.2 |

#### B. Walking 1.2 m/s (random headings), core layout, 15–18 kHz (tracked rates)

| Case | Remote miss | Wrong lock | Pick spread p95 (µs) | Range p50 / p95 (cm) | Graph offset p95 (µs) | Layout p50 / p95 (cm) |
|---|---|---|---|---|---|---|
| 13–16 kHz lfm | 0.0% | 85.7% | 1314 | 24.9 / 63.5 | 2777 | 38.0 / 51.4 |
| 15–18 kHz lfm | 0.0% | 85.7% | 1459 | 27.7 / 71.6 | 2754 | 35.3 / 51.7 |
| 13–16 kHz hfm | 0.0% | 85.7% | 1302 | 24.9 / 63.7 | 2794 | 39.9 / 51.5 |
| 15–18 kHz hfm | 0.0% | 85.7% | 1450 | 27.7 / 71.9 | 2768 | 36.6 / 52.2 |
| 13–16 kHz updown_hfm | 0.0% | 1.4% | 143 | 2.7 / 6.4 | 2803 | 47.8 / 66.6 |
| 15–18 kHz updown_hfm | 0.0% | 0.7% | 132 | 2.5 / 6.5 | 2779 | 46.9 / 66.6 |
| 13–16 kHz coded_updown_hfm | 0.0% | 40.4% | 665 | 25.7 / 31.4 | 2751 | 43.3 / 59.7 |
| 15–18 kHz coded_updown_hfm | 0.0% | 48.9% | 724 | 25.9 / 36.8 | 2232 | 44.2 / 61.4 |

#### C. Slot length × rate compensation, core layout, 15–18 kHz up-down HFM

| Slot | Motion | Compensation | Range p95 (cm) | Pair offset p95 (µs) | Graph offset p95 (µs) | Layout p95 (cm) |
|---|---|---|---|---|---|---|
| 0.20 s | still | none | 1.6 | 16 | 30 | 1.1 |
| 0.20 s | still | tracked | 1.2 | 16 | 11 | 0.8 |
| 0.20 s | still | closing | 1.2 | 16 | 11 | 0.8 |
| 0.20 s | sway 5 cm/s | none | 1.6 | 96 | 82 | 2.0 |
| 0.20 s | sway 5 cm/s | tracked | 1.5 | 96 | 79 | 2.2 |
| 0.20 s | sway 5 cm/s | closing | 4.2 | 96 | 66 | 1.7 |
| 0.25 s | still | none | 1.6 | 15 | 41 | 1.0 |
| 0.25 s | still | tracked | 1.2 | 15 | 10 | 0.7 |
| 0.25 s | still | closing | 1.2 | 15 | 10 | 0.7 |
| 0.25 s | sway 5 cm/s | none | 1.9 | 121 | 100 | 2.5 |
| 0.25 s | sway 5 cm/s | tracked | 1.7 | 121 | 100 | 2.6 |
| 0.25 s | sway 5 cm/s | closing | 4.9 | 121 | 85 | 1.5 |
| 0.40 s | still | none | 2.1 | 14 | 75 | 1.6 |
| 0.40 s | still | tracked | 1.2 | 14 | 10 | 0.7 |
| 0.40 s | still | closing | 1.2 | 14 | 10 | 0.7 |
| 0.40 s | sway 5 cm/s | none | 2.1 | 187 | 157 | 4.1 |
| 0.40 s | sway 5 cm/s | tracked | 1.2 | 187 | 162 | 4.3 |
| 0.40 s | sway 5 cm/s | closing | 7.6 | 187 | 139 | 2.5 |

#### D. Self-recording: near vs far mic × grip × level, core layout, 15–18 kHz up-down HFM

| Mic | Grip | Level (dB SPL @1 m) | Self miss | Self pick |error| p50 / p95 / max (µs) | Raw-threshold onset |error| p50 (µs) | Raw pair offset p95 (µs) | Corrected pair offset p95 (µs) |
|---|---|---|---|---|---|---|---|
| near | Foam | 85 | 0.0% | 38 / 40 / 40 | 432 | 294 | 13 |
| near | Foam | 95 | 0.0% | 38 / 40 / 40 | 424 | 294 | 13 |
| near | Hand | 85 | 0.0% | 38 / 44 / 45 | 430 | 293 | 15 |
| near | Hand | 95 | 0.0% | 38 / 44 / 45 | 422 | 293 | 14 |
| near | HardTable | 85 | 0.0% | 35 / 62 / 64 | 427 | 298 | 25 |
| near | HardTable | 95 | 0.0% | 36 / 62 / 65 | 362 | 298 | 25 |
| far (+15 cm) | Foam | 85 | 0.0% | 37 / 39 / 39 | 460 | 697 | 25 |
| far (+15 cm) | Foam | 95 | 0.0% | 37 / 39 / 39 | 460 | 697 | 24 |
| far (+15 cm) | Hand | 85 | 0.0% | 37 / 40 / 41 | 460 | 699 | 24 |
| far (+15 cm) | Hand | 95 | 0.0% | 37 / 40 / 41 | 460 | 699 | 24 |
| far (+15 cm) | HardTable | 85 | 0.0% | 39 / 46 / 48 | 401 | 698 | 23 |
| far (+15 cm) | HardTable | 95 | 0.0% | 39 / 46 / 48 | 401 | 698 | 23 |

#### E. Arrival timing: envelope peak vs leading edge, core layout, 15–18 kHz up-down HFM

| Case | Remote miss | Wrong lock | Pick spread p95 (µs) | Range p50 / p95 (cm) | Graph offset p95 (µs) | Layout p50 / p95 (cm) |
|---|---|---|---|---|---|---|
| envelope peak | 0.0% | 0.0% | 77 | 0.5 / 2.6 | 12 | 0.6 / 1.3 |
| leading edge 50% | 0.0% | 0.0% | 40 | 0.4 / 1.2 | 10 | 0.3 / 0.7 |

#### F. Range ladder: one pair, up-down HFM, 85 dB SPL @1 m, 35 dB SPL ambient

| Band | Distance | Detection miss | Range |error| p50 / p95 (cm) | Miss with weak mic (16 kHz, order-8 roll-off) |
|---|---|---|---|---|
| 13–16 kHz | 10 m | 0.0% | 0.1 / 0.3 | 0.0% |
| 13–16 kHz | 20 m | 0.0% | 0.6 / 1.0 | 0.0% |
| 13–16 kHz | 30 m | 0.0% | 2.0 / 2.2 | 0.0% |
| 13–16 kHz | 45 m | 0.0% | 1.0 / 1.8 | 0.0% |
| 13–16 kHz | 60 m | 0.0% | 1.1 / 1.7 | 0.0% |
| 13–16 kHz | 80 m | 25.0% | 3.0 / 6.8 | 25.0% |
| 15–18 kHz | 10 m | 0.0% | 0.2 / 0.2 | 0.0% |
| 15–18 kHz | 20 m | 0.0% | 0.2 / 0.9 | 0.0% |
| 15–18 kHz | 30 m | 0.0% | 1.7 / 9.7 | 0.0% |
| 15–18 kHz | 45 m | 0.0% | 1.5 / 2.0 | 0.0% |
| 15–18 kHz | 60 m | 0.0% | 1.8 / 5.6 | 0.0% |
| 15–18 kHz | 80 m | 79.2% | NaN / NaN | 93.8% |
| 18–20 kHz | 10 m | 0.0% | 0.3 / 0.5 | 0.0% |
| 18–20 kHz | 20 m | 0.0% | 2.5 / 3.2 | 0.0% |
| 18–20 kHz | 30 m | 0.0% | 1.2 / 2.5 | 0.0% |
| 18–20 kHz | 45 m | 0.0% | 6.3 / 16.2 | 0.0% |
| 18–20 kHz | 60 m | 41.7% | 0.6 / 1.5 | 89.6% |
| 18–20 kHz | 80 m | 100.0% | NaN / NaN | 100.0% |

NaN means no pair was ranged in any seed (every arrival missed).

### Reading the tables

- **A (still).** 13–16 and 15–18 kHz are interchangeable up to 45 m. The up-down chirp's graph offset p95 is lower (7–10 µs against 25–29 µs for the plain sweeps at 28 m) because averaging two half picks halves the multipath scatter in the offset. Its range p95 is the same. At 45 m with 15–18 kHz it shows 4.6% wrong locks. Each half carries half the energy (time-bandwidth 75 instead of 150), so a far, weak arrival with a strong trunk echo can pull one half off. The robust layout and graph absorb this (layout p95 2.3 cm).
- **B (walking).** The coupling bias of a single sweep is about T·f·v/(c·B): 1 ms, or 30 cm of range, at 1.2 m/s for a 50 ms, 3 kHz sweep at 16.5 kHz. The up-down chirp removes it. **Session-coded hopped codes fail under Doppler** (40–49% wrong locks). Each hopped segment is only about 750 Hz × 6 ms (time-bandwidth about 5), so under a Doppler shift the segments' correlation peaks add with different carrier phases and the code's envelope breaks up. The coded variant ranges still phones about as well as the plain up-down chirp (spot check: range p50 0.4 cm, with one outlier in three seeds), but it is not fit for walking phones. Slots already separate phones within an island, so per-phone codes are only needed to tell overlapping islands apart. That is an open item (core_sync.md).
- **C (slot × compensation).** The pair offsets ("Pair offset p95", 14–16 µs still) do not depend on compensation, because a single two-way offset refers to its own exchange midpoint. Compensation matters when the offsets are carried to one reference time for the graph: uncompensated, the drift between pair midpoints shows up directly in the graph. Under sway, the closing chirp's own rate estimate is corrupted (range p95 4–8 cm).
- **D (self-recording).** Self picks include the common ~37 µs receive-chain delay. Their spread is what matters: about 2 µs on foam, 6–7 µs in the hand and 25–27 µs on a hard table (near mic). Before the precursor fix (any earlier peak inside the body/air lead window plus one main-lobe width is rejected, whatever its level), near-mic hard-table picks with d_ii ≈ 15 cm locked onto the precursor (−460 µs). Interference with the airborne main lobe moved the apparent precursor peak just outside a lead-only window.
- **F (range ladder).** This is one pair on flat ground at 1.2 m height. The bumps (for example 15–18 kHz at 30 m, p95 9.7 cm) come from the ground bounce arriving inside the main lobe at that geometry and from random trunks, not from SNR. The weak-mic column (16 kHz, order-8 roll-off) only matters for 15–18 and 18–20 kHz at the longest ranges.
