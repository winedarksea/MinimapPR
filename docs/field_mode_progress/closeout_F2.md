# Closeout F2: post-membership-change fix errors (H1) and the flaky nightly gates

Branch `field-mode-closeout`. User decisions from `review_closeout.md`:
- H1: fix the errors, rather than restoring σ 0.25 m or inflating σ after membership changes.
- Flaky gates: fix the root cause when it is simple; widen a band to the measured spread when that is
  enough and the gate still catches its defect; put anything bigger in TODO.md.

| Step | Commit | Result |
| --- | --- | --- |
| H1 clock σ, post-change hold, mid-round stall | `26d13e0` | mesh nightly 114 → 8 failures in 19,200 runs; none "outside reported region" |
| `anchor_persistent_jump` band | `5e722f0` | per-run pass 77 / 100 → 100 / 100 |
| `srp_contour_coverage` band | `43b41c0` | per-run pass 90 / 100 → 100 / 100 (and 40 / 40 fresh runs) |
| `mesh_contour_world_coverage` | none | already passes 157 / 160 runs (98 %); left unchanged |

How the numbers were taken:
- "Mesh nightly" means `nightly_mesh_seeds --seeds 200 --start-seed S`, which runs all 24 scenarios × 200
  seeds = 4,800 runs per start. It was run at four starts (100000, 500000, 700000, 1300000), so 19,200
  runs in all.
- "Before" is a scratch worktree of `70feba4` (the branch head before F2), run on the same seeds.
- The "pre-C3" reference is the scratch build of `3cd6f78` that Phase C used (23 scenarios, so 4,600
  runs per start).
- Gate pass rates come from 100 disjoint default-size runs per gate: `nightly_sim_gates --seeds 200
  --start-seed 200000 + 1000·k --gate G` for k = 0…99.

## H1: what the failures were

**Reproduction.** At `70feba4`, `nightly_mesh_seeds --seeds 200 --start-seed 500000` gave 29 failures
in 4,800 runs, the same count as closeout C3.

All four starts together: 114 failures in 19,200 runs. 92 of them were "error outside reported region
(4σ + 0.5 m)":

| Scenario | "Outside region" failures, 4 starts |
| --- | --- |
| `island_split_merge` (call after the merge) | 47 |
| `offline_host_loss` (call after the wanderers rejoin) | 36 |
| `thermal_shedding` (call at the host handover) | 8 |
| `transport_fallback` | 1 |

**Diagnosis.** A new example, `minimappr-mesh-sim/examples/diagnose_fix_errors.rs`, re-solves every
arrival-time fix in a run with the truth swapped in for one input at a time: clock/timing, phone
heights, horizontal layout.
- To support it, `SolveTrace` gained the solver inputs (`localizer_inputs`) and the node σ.
- `inspect_scenario` now prints each used node's layout residual after a rigid fit to truth.

Example: `island_split_merge` seed 500023, call 2 (error 2.77 m, σ 0.25 m):

| Input swapped for truth | Error | σ |
| --- | --- | --- |
| none (as solved) | 2.77 m | 0.25 m |
| timing | 0.15 m | 0.35 m |
| heights | 2.67 m | 0.23 m |
| horizontal layout | 2.76 m | 0.25 m |

The layout residuals were about 2 cm. The timing residuals were +1.1, −3.2, −2.1 and +4.2 ms, against
a declared σ of 4–15 µs. **So the post-merge error was not a layout error. It was the clock model.**

### Root cause 1: a one-round clock model, extrapolated with the wrong σ

- A merge or rejoin starts a fresh clock domain, so every member's filter restarts. That is by design.
- Next, a `ClockInvalid` round is scheduled, then nothing until the 90 s heartbeat.
- A call in that gap is solved on a filter holding **one** update. Its rate is unknown (prior
  ±100 ppm), and it is extrapolated across the gap: 80 s in the example, which is milliseconds.
- `NodeClockMapping` carried the filter's σ at the *round* time (about 15 µs), not at the call time.
- The post-detection round is scheduled 0.5 s after the detection, but it lands about 3.5 s after the
  call is received, which is after the call has already been localized.
- In the example the rounds themselves were accurate to about 1 µs (checked against the true clocks).

**Fixes:**
1. `IslandSyncEngine::clock_mapping_at(node, reference_time_s)`. It is the same line as before, anchored
   at the call's reference time, so its σ is the filter's own prediction σ there, rate uncertainty
   included. `CallPipeline::localize` uses it for every clip. This is the filter's covariance, not a
   heuristic inflation.
2. `CallPipeline::hold_for_clock_model`. A call with all its clips in is held while both are true:
   - some clip's clock σ at the call exceeds `clock_hold_sigma_pick_multiple` (3) × the round pick σ
     (15 µs);
   - no round has landed after the call.

   The post-detection round that is already scheduled then pins the offsets. The wait is bounded by
   `clock_hold_maximum_wait_ms` (6 s). At expiry the call is solved with its honest σ.
   - A steady model a full 90 s heartbeat from its last round sits at 2.0×, from the filter
     covariance, so it is never held.
   - A first version triggered on "filter has no valid rate", meaning less than 20 s of baseline. It
     held a call whose 12 s-baseline model was fine 3 s out. That call's 3-node solve then came out
     ill-conditioned (`host_death` 700123), so the trigger became the σ itself.

**What each half does** (PR seeds, with the 4 new H1 seeds):
- Either half alone makes the end-to-end seeds pass.
- With both reverted, all 4 fail.
- Measured on start 500000 alone, honest σ without the hold gave 7 failures in 4,800. The fixes were
  then loose (σ about 4 m), because the model still had no rate. The hold makes them precise again:
  0.15 m in the example.

**How often it fires.** Over 40 seeds each:

| Scenario | Calls held | Waits expired |
| --- | --- | --- |
| `island_split_merge` | 80 | 0 |
| `offline_host_loss` | 68 | 0 |
| `host_death` | 0 | 0 |
| `thermal_shedding` | 109 | 23 |

In `thermal_shedding` the expired waits are rounds lost to the opener's own stall (root cause 2).

### Root cause 2: a capture stall inside a chirp round (`thermal_shedding`)

Example: `thermal_shedding` seed 500075, call 10 at 400 s.
- The host handed over at about 400 s.
- The hot phone (node 2) had an unreported 87 ms stall at 400.51 s, inside the new host's first round.
- That phone's intervals were biased. The round's layout placed it 9.5 m off and pulled the other four
  2.2–2.6 m, all at σ 0.13 m.
- Its offset jumped 27 ms, and the clock filter flagged the silent gap only at the next round.
- The existing silent-gap re-solve (`nodes_with_silent_gap`) needs a valid filter, and a fresh
  post-handover domain has none.
- With the true horizontal layout swapped in, the call errs 0.25 m.

**Fix.** `solve_island_round` reads each participant's closing-chirp rate ratio, which was already
computed for rate compensation but never checked.
- A ratio more than `maximum_plausible_round_rate_offset` (2,000 ppm) from 1 means samples were lost
  or gained inside the round. For comparison:
  - crystals are ±100 ppm;
  - 10 cm of handling motion over a round is about 200 ppm;
  - a 5 ms dropped buffer inside a 2.5 s round is 2,000 ppm or more;
  - simulated stalls are 50–300 ms.
- That participant's measurements leave the round's clock graph and layout. It keeps its previous
  position and filter state, and the next round's silent-gap check re-seeds it.
- If a majority is flagged, the opener's own span is broken and the round is skipped.
- Call 10 then errs 0.25 m (it was 2.53 m).

### Results

**Mesh nightly**, 200 seeds × 24 scenarios per start:

| Start | Pre-C3 (`3cd6f78`, 4,600 runs) | Before (`70feba4`) | After (`26d13e0`) |
| --- | --- | --- | --- |
| 100000 | 5 | 24 | 2 |
| 500000 | 6 (closeout C) | 29 | 3 |
| 700000 | 6 | 28 | 1 |
| 1300000 | 6 | 33 | 2 |
| **Total** | 23 / 18,400 | **114 / 19,200** | **8 / 19,200** |

- "Outside reported region" failures went from 92 to **0**.
- The 8 left all fail identically at `70feba4`:
  - `network_faults`: 4. Three are a detection that never reached the host log; one has only 1 of 5
    calls localized.
  - `flock_echo`: 1 (an echo reached the entity layer).
  - `island_split_merge` 100157: a mirrored layout at confidence 0.991.
  - `successor_restores_entities` 700182: 1 duplicate.
  - `transport_fallback` 1300154: a 3-node singular solve.

  None involves σ or layout.
- `assumed_phone_height_sigma_m` stays at 0.115, and no gate or invariant threshold changed.

**Regression seeds.** `PULL_REQUEST_SEEDS` gained six seeds:
- `island_split_merge` 500023, 500073 and 500178;
- `offline_host_loss` 500121;
- `thermal_shedding` 500075 and 700043.

**Sabotage**
- With both clock-model changes reverted, the four clock seeds fail in `pr_seeds` (2.77, 10.58, 12.52
  and 12.48 m outside their regions). The new unit tests fail too
  (`mesh::call_pipeline::clock_model_tests`, 4 tests).
- With `maximum_plausible_round_rate_offset` set to ∞:
  - both thermal PR seeds fail again, and so do 100105, 700072 and 1300069;
  - `mesh::island_round_solver::tests::a_phone_that_stalls_inside_the_round_is_left_out_of_it`
    fails;
  - every change was restored afterwards.

**Effect on the C3-calibrated mesh gates** (100 runs × 200 sessions, before → after):

MESH_| `anchor_persistent_jump` band | `5e722f0` | per-run pass 77 / 100 → 100 / 100 |
| `srp_contour_coverage` band | `43b41c0` | per-run pass 90 / 100 → 100 / 100 (and 40 / 40 fresh runs) |
| `mesh_contour_world_coverage` | none | already passes 157 / 160 runs (98 %); left unchanged |

## Flaky gates

Each gate was measured on 100 disjoint default-size nightly runs (200 seed units, so 200 trials or sessions each).

### `anchor_persistent_jump` (commit `5e722f0`): band widened, root cause deferred

**Measured.** 77 of 100 runs passed. This is the same before and after H1, because the gate does not use the mesh.
- The rule was per-seed: the response must stay under 1 m.
- Per-run maximum response: p50 0.84 m, p90 1.18 m, p95 1.54 m, max 1.90 m (20,000 seeds in all).
- The median per seed is 0.19 m.

**Cause.** This is not a holding defect; the jump is held. The cause is the one closeout C1 named:
- On a 10 m layout under 6 m GPS, the rotation is barely observable.
- Each round's fit is close to chaotic.
- The no-jump twin's filtered rotation wanders, and the twins drift apart through it.

**The known proper fix was tried and rejected.** While still, it fits each phone's running-mean fix instead of the raw
one.
- It helped the gate: persistent median 0.19 → 0.11 m, and 1 failure in 3,000 seeds against several at baseline.
- It also cut `anchor_stability` still-window drift: median 0.55 → 0.40 m.
- But on seed 200837 it made a new 11 m still-window excursion. With the rotation unobservable, the fit to averaged fixes
  is bistable: it sat in one basin (−2.9 rad) for about 100 s, flipped (to about +0.3 rad), and the filter drifted after
  it.
- So it trades independent per-round noise for long correlated excursions. A real fix needs rotation-information
  weighting or a rotation prior. That is in TODO.md.

**Change** (`AnchorGateConfig`):
- per-seed cap `persistent_jump_maximum_response_m` = 2.5 m (measured max 1.90 m);
- the strategy's 1 m kept as an aggregate: at most `persistent_jump_maximum_fraction_over_response` = 1 % of seeds over
  it, enforced from `persistent_jump_minimum_trials_for_fraction` = 100 seeds. The measured rate is about 0.17 %.

**Results**
- 100 / 100 runs pass.
- The PR subset passes (32 seeds, max 0.62 m).

**Sabotage.** With jump holding disabled (`still_jump_threshold_sigma` = ∞) at starts 100000, 200000 and 300000:
- 74–84 % of seeds are over 1 m, against 1 % allowed;
- the maxima are 7.2–9.3 m, above the 2.5 m cap.

So both rules still catch the defect the gate exists for.

### `mesh_contour_world_coverage`: no change

**Measured.**
- 99 / 100 runs pass on starts 200000–299000.
- 58 / 60 pass on starts spread over the nightly range (37 + 79000·k).
- Mean 0.928, sd 0.012–0.014; the failing runs read 0.888–0.895.
- H1 leaves it where it was (mean 0.9285 before, 0.9282 after).

**Why no change.**
- The "1 in 5" came from closeout C3's five starts, one of which read 0.898. The large sweep puts the failure rate at
  about 2 %, which already meets the ≥ 95 % target.
- The low mean (C3: the world gauge has a heavier-than-Gaussian tail) is a calibration question for real sessions, not
  a band problem. It is noted in TODO.md.

### `srp_contour_coverage` (commit `43b41c0`): bands widened to the measured spread

**Measured.** 90 / 100 runs passed. The two failure modes split evenly:

| Failure | Runs | Per-run statistics |
| --- | --- | --- |
| Coverage over the 0.98 ceiling | 5 | mean 0.961, sd 0.012, max 0.995 |
| Inside P95 over the 3 m accuracy bound | 5 | mean 1.90 m, sd 0.53 m, max 3.42 m |

The median inside error (0.375 m, sd 0.047 m) and sharpness (fan fraction ≤ 0.05, median area ≤ 0.93 m²) are stable.

**Can the temperature fix the coverage? No.** A 2,000-trial training refit with a 161-rung ladder (2.5 % per rung; the
example's old ladder was 41 rungs, 26 % apart):
- coverage is 0.955 already at the floor T = 1e-4, and 0.963 at the shipped 0.0013;
- the fitted value for 0.945 is the floor itself;
- D3's 400-trial fit read 0.9375 at the floor, a low sample.

So coverage here is set by the σ-pooling multiplier k = 1.25, not by T. Refitting k reopens D3's fan trade-off: below
k ≈ 1.25 the remaining trials need T ≥ 0.005, back in the fan trap. That is not a simple fix, and it is in TODO.md for
when `srp_layout` gets a production caller.

The temperature was left at 0.0013. The example keeps the finer ladder and prints the fit with 6 decimals.

**Inside P95.** This is the 95th percentile of 100 inside trials, so it is set by the 5th-worst trial. The population
P95 is about 1.9 m, well inside 3 m. `monte_carlo_accuracy` keeps the strategy's 3 m.

**Change** (`CoverageGateConfig`):
- `srp_maximum_coverage` = 0.995;
- `srp_inside_p95_error_m` = 3.5 m, about 3 run sd above the mean.

**Results**
- 100 / 100 of the measured runs pass.
- 40 / 40 fresh runs on starts spread over the nightly range pass (start 1037 + 117000·k): coverage mean 0.9615,
  three runs at 0.985 that would have failed before.
- The PR subset passes.

**Sabotage**
- T = 0.05 (smearing) fails at all three starts tried: coverage 1.000 (200/200 is over 0.995), inside fans in 100 % of
  trials (limit 10 %), median area about 1.2e5 m².
- The under-coverage floor (0.90) is unchanged. It is the rule D3's k = 0 sabotage failed (coverage 0.125).

## What was deferred (TODO.md)
- Anchor rotation on GPS-scale layouts: rotation-information weighting or a prior. Time-averaged fixes are bistable.
- SRP pooling multiplier k: refit with T on ≥ 2,000 training trials before a production `srp_layout` caller.
- World-frame coverage sits low in its band (mean 0.928). Revisit with the real-session recalibration (review M2).
- The mesh nightly residue (8 / 19,200 runs, all from before F2): `network_faults` detection loss, `flock_echo`, an
  unflagged mirror, a successor duplicate, a 3-node singular solve.

## Uncertain or not done
- The hold's 6 s cap is based on rounds landing about 3.5 s after the call is received in the 8-phone scenario.
  `thermal_shedding` expires 23 of 109 holds, from rounds lost to the opener's stall. Those calls are solved with an
  honest, wide σ rather than a pinned clock.
- The stall check only sees a stall that falls between a participant's opener and closing-chirp arrivals. A stall at
  the very end of a round is still left to the next round's silent-gap check.
- Holding a call delays its result by up to about 3.5 s after a membership change. No scenario gates latency.
- `docs/field_mode_progress/summary.md` still describes H1 and the flaky gates as open; only TODO.md was updated.

## Suites

| Suite | Baseline (`70feba4`) | After F2 |
| --- | --- | --- |
| `cargo test --workspace` | 586 passed, 0 failed, 8 ignored | 592 passed, 0 failed, 8 ignored (+4 clock-model, +2 round-solver unit tests) |
| `pytest tests/ --ignore=tests/test_soak_harness.py` | 2084 passed, 9 skipped | 2084 passed, 9 skipped (no Python changes) |
