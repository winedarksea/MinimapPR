# Field Mode non-phone build-out: summary

Status as of 2026-11-22. The plan is in `build_out_plan.md`. Each stream has its own notes file in
this directory, and each promoted stream has a matching `review_*.md` or reviewer notes.

All four waves are built, merged into `field-mode`, independently reviewed by Opus, and promoted to
`main`. The final full suites on `field-mode` (4a69971) were Rust 554 passed / 0 failed and Python
2056 passed / 18 skipped (soak harness excluded). Nothing has been pushed.

## What landed

| Area | Streams | Notes |
| --- | --- | --- |
| Shared Rust core | core_extract, core_solver, core_srp, core_sync, core_codec_proto | Cargo workspace with `minimappr-core`, which builds for wasm32, aarch64-linux-android and aarch64-apple-ios: LM TDOA solver with covariance, SRP over arbitrary layouts, chirp sync, codec and messages |
| Simulators | sim (Layer 1), mesh_sim (Layer 2), sim_gates, phase1_sim_prestudy | Physics renderer that writes calibration bundles (ground truth v2); seeded mesh host/node state machines; statistical gate suite |
| Replay and fixtures | bundle_format_v2, server_baseline, fixtures | Entity-aware replay metrics (recall, swaps, duplicates, echo promotions) and CC xeno-canto fixtures with attribution |
| Server track (S) | onset_srp (S2), tracker (S3/S4), attribution (S5), entity (S6), s_fixes_*, server_track_gate | All behind flags that default to off. Gate verdicts are in `server_track_gate.md` |
| Rust port of S track | port_server_track, port_server_track_onset | S2, S3, S5 and S6 in `minimappr-core`, with Python↔Rust parity goldens and 7 Layer 2 entity scenarios |
| BirdNET | birdnet_v3, birdnet_cutover | V3 behind `birdnet_model_version`. The default stays `2.4` because the V3 gate failed. INT8 was rejected |
| Shells | relay, pwa, android_skeleton | Cloudflare relay (not deployed), Leptos PWA, KMP Android skeleton (smoke-tested on an emulator) |
| Live-bug fixes | coverage_race, sidecar_coverage_race, plus fixes found by the gate | See below |

## Live bugs found and fixed (all on main)

- **Compact-tetra solver collapse.** A far-field guard in `core/far_field_model_selection.py`, promoted without a flag at the user's choice.
- **Python coverage race.** `core/sensor_span_coverage_wait.py`, on by default. Lagging nodes were dropped from localization windows.
- **Rust sidecar coverage race.** Same wait in the sidecar, plus a window-clamp fix and decisions made at batch boundaries. This is the likely cause of "0 cross-node pairs" on the Bluefin box. The reviewer also fixed a sidecar crash that happened when a node joined a cluster while a solve was parked.
- **BirdNET category bug.** Species whose names contain "car" or "fire" were categorised as vehicle or security, and a startup step now corrects categories already persisted in the DB.
- **Smaller fixes.** The report-window cap, V3 "Human" now maps to human, and the onset count settings are typed as int.

## Recommended first live trial (server)

Turn on these flags:

- `localization_onset_clustering_enabled`
- `trigger_noise_floor_relative_enabled`
- `tracking_perched_filter_enabled`
- `tracking_species_log_odds_enabled`
- `onset_attribution_enabled`, with `onset_attribution_loco_enabled=false`
- `entity_reporting_enabled`

Leave `birdnet_two_threshold_enabled` and `tracking_partial_constraint_binding_enabled` off; both
are NEEDS-WORK.

### Bluefin box deploy (192.168.8.103)

1. Pull `main`.
2. Run `cargo build --release -p minimappr-ingest-sidecar`.
3. Restart the server with setsid and check for an orphaned ingest child.
4. Check `GET :<sidecar_port>/api/v1/dsp/status`:
   - `localization_coverage_wait_stop_reasons` should be mostly `all_covered`;
   - localization results should carry cross-node `pair_tdoas`.
5. To roll back, set `localization_coverage_wait_all_streaming_nodes_enabled=false` and restart.

## Closeout (branch `field-mode-closeout`, 2026-11-23)

The open issues below were worked by the closeout plan. Per-phase notes: `closeout_A.md`,
`bluefin_live_trial.md`, `closeout_B1.md`, `closeout_B2.md`, `closeout_C.md`, `closeout_D.md`, and the
independent review in `review_closeout.md`. Final suites: Rust 581 passed / 8 ignored, Python 2084
passed / 9 skipped. New flags all default off, except the B3 collinear guard (ratio key only, 0 disables).

### Resolved
- **Same-band two-species merges** (B2): post-window label reconcile in Python and Rust behind
  `onset_window_label_reconciliation_enabled` (default off). Seeds 5007, 10015, 10187 pass; server
  gate unchanged with the flag (the drop never fires in a server scene).
- **Collinear mirror guard on the server** (B3): point-node-only collinear fixes become bearing rays.
  Rigid-array sites are exempt.
- **Duplicate `localization_result` on parked trigger frames** (B4): the render rides the deferred
  payload. Deployed to Bluefin; the live 1.0-per-trigger check waits for node traffic.
- **`dsp_worker.rs` split** (B5): 573 lines plus eight child modules.
- **NaN species log-odds in Python** (B1).
- **S6 items** (12 m absorption, raw-fix row overwrite, following cross-species merges): already fixed
  behind default-on flags by `fm/s-fixes-tracking` (see `s_fixes_tracking.md`); this list was stale.
  Remaining minor S6 items: perched tracks start overconfident, and reverted-hop rows keep the old
  track id.
- **Anchor jumps** (C1), **anchor error in map polygons** (C2), **mesh LM covariance** (C3, NEES median
  now about 1.1–1.3), **successor host restores entities** (C5).
- **SRP contour calibration** (D3): `srp_contour_coverage` un-ignored and passing.
- **Real `ClipSetLocalizer`** (D1) and a rendered-audio Layer 2 mode (D2) now exist.
- **BirdNET V3 vs 2.4 on real data** (A5): V3 fails again (5 label swaps, mapped top-1 agreement
  0.455). Default stays `2.4`.

### Still open
- **Phone localizer choice** (D2): the real onset localizer fails most multi-bird scenes on single-mic
  meshes (e.g. same_band_two_species 3/50, countersinging_pair 9/50). Server tetra sites are not
  affected, because intra-node pairs anchor the fringe. Options: clip SRP per window, wideband or
  envelope frames for inter-node pairs, or arrival-time picks into `ArrivalTimeLocalizer`.
- **Late or pocketed phone rejection** (C4): improved (rejection of decisive pockets went from 58% to
  92%), but `graceful_degradation` stays ignored.
- **Mesh-sim nightly invariant failures** rose from 6 to 29 per 4,800 runs after the phone-height σ
  went from 0.25 to 0.115 m. These are existing errors after merges, rejoins and handovers that the
  old σ hid.
- **C5 restored covariance** is about 5× too tight in variance for entities outside the layout
  (review M1).
- **C3 contour scales and the D3 temperature** were fitted to sim noise; recalibrate on the first real
  sessions. The D3 temperature needs declared node σ (review M3).
- **Flaky nightly gates**: `anchor_persistent_jump` (~1.7/1000 seeds), world-frame coverage (1 in 5
  sweeps), `srp_contour_coverage` (1 in 5 sweeps, over-coverage).
- **Bluefin live trial** (A3/A4): blocked; no node has streamed since 2026-08-14. Startup retention
  purged about 7,300 expired August snippets on the first start (all bird clips). The 22 fixture clips
  were exported first and need a speech listen before `speech_checked = true`.
- A never-streamed cluster member still empties the omni render (B4 follow-up).

### Deferred (user decisions)
- Phase 1 real-phone chirp sync and iPhone/Xcode work.
- Relay rate limiting and deploy.
- Replay-window persistence: no shell constructs `RelayClientSession` yet.
- Speaker→mic clock bias: needs per-model vectors calibrated in Phase 1 and a wire change.

## Machine changes made during the build-out

- Android emulator and API 35 image (AVD `field_smoke_api35`, about 3 GB).
- Temurin 21 JDK in the gitignored `field-app/.local-jdk`.
- Playwright WebKit installed, which replaced the older `chromium-1223` browsers.
