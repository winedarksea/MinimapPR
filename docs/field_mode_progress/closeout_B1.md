# Closeout B1: NaN log-odds, duplicate localization_result, dsp_worker split

Branch `field-mode-closeout`. Plan steps B1, B4, B5. B2 and B3 belong to the next agent.

## B1: NaN guard in `SpeciesLogOddsBelief.update`
- `minimappr/core/track_species_belief.py`: a label is skipped when `math.isnan(probability)`.
  This matches the Rust `probability.is_nan()` guard. `isnan` was used rather than `isfinite`,
  as the plan asks.
- Test: `tests/test_track_species_belief.py::test_nan_score_skipped_without_poisoning_label`.
  - It feeds a NaN for an existing label and a NaN for a new label.
  - It checks that the existing belief is unchanged, that no NaN label appears, and that a later
    real score still updates the label.
  - It failed before the fix.

## B4: duplicate `localization_result` on a parked trigger frame (review M3)
- `dsp_worker.rs` `dispatch_buffered_manifest` (now in `dsp_dispatch_gates.rs`): when a coverage
  deferral exists, the original payload no longer keeps `run_srp` or the render. If a render is
  due, it becomes a `DeferredClassifierRender { end_ns, window_seconds }`. The original frame
  emits only its heartbeat.
- `deferred_localization.rs`:
  - `defer()` takes `Option<DeferredClassifierRender>` and stores it on `PendingLocalization`.
  - `finalize_pending_payload` handles both release and supersede. At release it calls
    `attach_classifier_render_windows`, which re-extracts the trailing render windows, the
    gap-concealed windows, coverage and bounds from the fuller buffers. It truncates to the
    parked geometry, as the localization windows are, and sets `run_classifier_render`.
  - So the single deferred result carries the render's `derived_handle` (from
    `dsp_render_output.rs`) to Python.
  - `localization_only_template` is unchanged apart from its doc comment. The render is attached
    at release, not when the template is built.
- Test: `dsp_worker_coverage_race_tests.rs::render_on_a_parked_trigger_frame_publishes_one_result_carrying_the_render_handle`.
  - Renders are set to the trigger cooldown, overriding `race_config`'s 1e6 s.
  - It asserts one real `localization_result` per trigger segment, that each one after frame 0
    carries `derived_handle`, and that `total_localization_results` equals the number of
    triggers.
- **Before the fix the test failed.** Triggers at frames 25 and 50 each published two `srp_phat`
  results, and no render was published at all.
- **Side finding:** before the fix, a render on the trigger frame of a cluster with a late member
  was always skipped as an empty PCM render. `omni_mix` truncates to the shortest channel, and the
  late node's channels were still empty. Rendering at release fixes this as a side effect, so
  renders on those frames now include the late node.
- **Frame 0 caveat (pre-existing, not changed):** a cluster member that has never streamed still
  empties the omni render, for the same truncation reason. The test leaves frame 0 out of the
  handle assertion for this reason. If the user wants, a later fix could mix only the channels
  that have data.
- **Kept on purpose:**
  - The original frame still emits its heartbeat `localization_result`
    (`localization_cadence_skipped`, which `total_localization_results` does not count). Dropping
    it would delay node liveness by up to the coverage wait.
  - Render timing: the render is still *scheduled* by the trigger frame. Only its compute and
    publish move to the release, which waits at most the coverage stall budget or total cap.

## B5: mechanical split of `dsp_worker.rs`
There is no behaviour change. The new files are child modules of `dsp_worker`, declared with
`#[path]` (the pattern the test modules already use):
- Moved code can still see `DspWorker`'s private fields, and moved methods sit in
  `impl DspWorker` blocks.
- Items needed across modules or by tests are `pub(super)`. Crate-wide items keep their
  `pub(crate)`/`pub` and are re-exported from `dsp_worker`, so every external `crate::dsp_worker::X`
  path is unchanged.
- The test files were not modified.

I checked the split with a line multiset diff against the pre-split file. The only differences are
module glue, rustfmt re-wrapping and three dropped section-banner comments.

| file | lines | contents |
|---|---|---|
| `dsp_worker.rs` | 573 | imports, constants, `DspWorkerState`, the `DspWorker` struct, builders, run loop / `process_pending` / `ingest_one`, bookkeeping (`note_*`, `purge_stale_streams`), `system_now_ns` |
| `dsp_preprocess.rs` | 430 | `DspWorkerConfig` + Default, `PreprocessStage`, `NodeAudioConfig`, `CompiledStage`, `NodeAudioState` |
| `dsp_manifest_buffering.rs` | 429 | `OwnedManifestAudio`/`BufferedManifestAudio`/`ManifestBufferRouting`, prepare/GPS/stale-skip/buffer-append methods |
| `dsp_dispatch_gates.rs` | 463 | `BufferedManifestTimingGates`, large-array / buffered dispatch, `should_*` cadence gates |
| `dsp_coverage_integration.rs` | 117 | `coverage_deferral_span`, `take_ready_deferred_localizations`, test seams |
| `dsp_window_geometry.rs` | 248 | window, coverage and timestamp pure helpers |
| `dsp_array_geometry.rs` | 303 | `AutomaticGpsGeometry`, mic positions, GPS→local, cluster routing |
| `dsp_manifest_publish.rs` | 306 | `RawAudioFramePublishRequest`, source-manifest consumption, publish/consume free helpers |
| `dsp_compute_payload.rs` | 63 | `ComputePayload`, `LocalizationChannelState`, `ClassificationRequest` |

**Deviations from the plan's module list, and why:**
- The plan's `dsp_window_geometry.rs` range (`:1932-2640`, about 700 lines) was over 500 lines. I
  split it three ways: `dsp_window_geometry.rs`, `dsp_array_geometry.rs` and
  `dsp_manifest_publish.rs`.
- Moving only the listed ranges left `dsp_worker.rs` at about 770 lines. To get under 600 I also
  moved:
  - the per-manifest data structs, next to their users;
  - the source-manifest consumption trio (`defer_`/`mark_`/`flush_`), into
    `dsp_manifest_publish.rs` with the other consume helpers;
  - the compute hand-off structs, into `dsp_compute_payload.rs`.
- `dsp_worker_coverage_race_tests.rs` is 554 lines, just over 500, because of the new B4 test. I
  left it as one file rather than split the shared harness for a single test.

## Test numbers
| | Rust (`cargo test --workspace`) | Python (`pytest tests/ --ignore=tests/test_soak_harness.py`) |
|---|---|---|
| Baseline (from the brief, main@df95986) | 554 passed | 2056 passed / 18 skipped |
| After B5 | 555 passed / 0 failed / 4 ignored | 2066 passed / 9 skipped |

- **Rust:** the +1 is the new B4 test.
- **Python:** the +1 is the B1 test. Nine fewer tests skip in this environment: the Rust sidecar
  binary and fixtures are available now. No test was skipped or removed.
- **B4 regression found and fixed** (second B4 commit):
  `tests/test_hybrid_settings.py::test_birdnet_hybrid_production_rust_sidecar_uses_reported_environment`
  failed after B4.
  - Its `_wait_for_rust_localization_manifest` helper took whichever `localization_result`
    iterated last in the newest-first `/api/v1/dsp/results` list, which is the oldest row.
  - Before B4 that row was the trigger-frame solve plus render. Now it is the trigger-frame
    heartbeat, while the solve with the render handle is the newer row.
  - The helper now pairs the result with its render by `derived_handle`, which is the contract
    Python's render bundle loader uses.
  - I checked live: the sidecar published exactly one `srp_phat` result with a handle,
    `total_localization_results` = 1, and stop reason `all_covered`.
- **Environment note:** the first workspace run failed 8 `minimappr-sim` tests (`bundle_writer`,
  `scenario_knobs`) with `NotFound`.
  - Cause: stale test binaries in the shared `target/` had `env!("CARGO_MANIFEST_DIR")` baked in
    as a deleted `.claude/worktrees/agent-…` path.
  - Fix: `cargo clean -p minimappr-sim`. Unrelated to these changes.
  - Other agents that see this should do the same.

## Left open
- The omni render is emptied by never-streamed cluster members (the frame 0 caveat above).
- Review L1 (deferred results never persisted on the disk-query path) is unchanged, and has no
  live impact.
- The live B4 check (Bluefin `total_localization_results` per trigger, from about 1.03 down to
  1.0) waits on Agent A redeploying the sidecar.
