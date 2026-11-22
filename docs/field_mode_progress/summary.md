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

## Open issues carried forward

**Needs a decision or hardware**
- Phase 1 real-phone chirp sync. This still gates the phone app; iPhone is untested and the iOS build needs Xcode.
- Same-band two-species merges, at seeds 5007, 10015 and 10187 in Layer 2. Server and Rust behave the same. The reviewer recommends the attribution-side fix: a gradient-settled cluster drops labels that are another settled cluster's top label in the same window. It needs a server gate run.
- The relay needs Cloudflare rate limiting before any deploy.
- The Bluefin tunnel was down, so the snippet pull and the real-data BirdNET V3 vs 2.4 comparison were not run.

**Engineering**
- No `ClipSetLocalizer` yet decodes real clips and calls `localize_onset_clusters`. The Layer 2 stand-in groups arrivals by their true call, so it cannot fail on clustering mistakes.
- The collinear-node mirror guard exists only in the mesh host. A server site with collinear nodes would pinpoint mirror fixes.
- When a classifier render falls on a parked trigger frame, the sidecar publishes two `localization_result`s for it, about 1 trigger in 35.
- Map uncertainty polygons don't include the GPS anchor error, and the SRP contour cannot be calibrated.
- Mesh LM covariance is 4–7× too large.
- Late-phone rejection is weak, and an anchor jump moves the anchor too far.
- Clock offsets have a speaker→mic vector bias.
- Replay-window state is lost across restarts, and a successor mesh host starts entity tracking empty.
- S6 issues: absorption distance (12 m) is above flock perch spacing, the DB row is overwritten with each bout's raw fix, and entities follow cross-species track merges.
- NaN classifier scores poison species log-odds in Python. This is fixed in Rust only.
- `dsp_worker.rs` is about 2,850 lines and should be split.

## Machine changes made during the build-out

- Android emulator and API 35 image (AVD `field_smoke_api35`, about 3 GB).
- Temurin 21 JDK in the gitignored `field-app/.local-jdk`.
- Playwright WebKit installed, which replaced the older `chromium-1223` browsers.
