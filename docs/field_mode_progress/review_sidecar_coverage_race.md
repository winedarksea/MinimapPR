# Promotion review: sidecar coverage race (`fm/sidecar-coverage-race`)

Reviewed: `git diff 430659f..fm/sidecar-coverage-race` (tip 5bea2b4, merged into
`field-mode` at 30838e5). Review branch: `review/sidecar-coverage-race`.
Only the sidecar coverage-race commits (4801d4c, 5bea2b4) were reviewed. The
merged-in onset/wildlife/attribution port work in the same range is out of scope.

Verdict: **promote, with the fixes below.** The design is sound. It defers and
never sleeps, keeps one pending window per buffer, and decides at batch
boundaries. One crash bug and one untested core claim were fixed.

## Findings

### High: fixed

**H1. A parked window could abort the sidecar if its cluster grew before release.**
`finalize_pending_payload` re-read *every* buffer channel at release time, but
the parked payload keeps the `mic_positions_m` from park time. `buffer_owned_manifest_audio`
grows a cluster buffer in place (`buffers.push`) when a member joins under the
same `cluster::<id>::<stream>` key. If the new member covered the window
(>= 0.85 coverage, which is one 32 ms frame), its channel index went into
`active_channels`. SRP then indexed `mic_positions_m[ch]` out of bounds
(`srp_phat.rs:371`). The DSP rayon pool has no `panic_handler`, so a panic in
`spawn_fifo` aborts the process.
Fix: re-read only the first `mic_positions_m.len()` channels. Cluster buffers
grow by appending and never reorder, so those channels are the parked geometry.
Test: `cluster_growth_while_pending_solves_with_the_parked_geometry`. It panics
at `srp_phat.rs:371` with the fix reverted.

### Medium: fixed (test gap)

**M1. The batch-boundary rule, the key change in 5bea2b4, had no Rust test.**
Every worker test went through `process_one`, a batch of one that already ends
at a boundary. Only the e2e test exercised it, and only probabilistically
(about 1 in 5 before the fix).
Fix: the real batch loop is extracted from `process_pending` into
`ingest_batch_and_dispatch(pending, depth, dispatch)`. Production behaviour is
unchanged: each payload is still dispatched to the rayon pool right after it is
built. A `#[cfg(test)] process_batch_inline` drives that loop with
`merge_pending_manifests_for_batch` (newest-first) and runs compute inline.
Test: `backlog_batch_decides_only_after_newest_first_backfill`.
- The batch is processed newest-first: b26 (silent, so below the RMS gate),
  then a25 (the trigger), then b25 (the backfill).
- With the cluster "always park until boundary" rule disabled, the solve has
  4 channels and the test fails. With the rule on, it has 8.

**M2. No test showed that packet loss doesn't burn the stall budget.**
By reading the code the policy is correct. Progress is the buffer end
*advancing*, a channel whose end has passed the window end is not waited on,
and an inner gap is left to the coverage ratio.
Test added: `lossy_late_node_costs_one_frame_gap_not_the_stall_budget`.
- Node b runs one frame late and loses 5 of every 11 frames (~45%), including
  both trigger frames.
- Result: 0 `stalled`, 0 `total_cap`, and a max wait of 0 < wait <= 64 ms (two
  frames). A completeness-based check would wait the full 0.5 s and fail.

### Medium: left (documented)

**M3. A classifier render that coincides with a cadence trigger produces two
`localization_result`s.**
When a render is due on a parked trigger frame, the original payload keeps
`run_srp` for render steering. It publishes its own (possibly single-node)
solve, and the deferred payload publishes the cross-node one later.
- Renders are about 28 s apart, so this affects about 1 trigger in 35 on a
  render-enabled box.
- Effects: `total_localization_results` double-counts, and Python sees one
  extra, poorer fix for the same event.
- Not fixed: the render's `localization_result` carries the `derived_handle`
  that Python needs (the "Bug A" fix), so it can't just be suppressed. The
  clean fix is to move the render onto the deferred payload too. That changes
  render timing, so it should be its own change.

### Low: left

- **L1. Deferred results are never persisted to the manifest store.**
  `localization_only_template` marks the source as memory-path
  (`raw_payload = Some(empty)`), so on the disk-query ingest path
  (`query_persisted_raw_manifests = true`) the deferred `localization_result` is
  only broadcast. Production sets that flag false, and channel-delivered
  manifests were never persisted, so there's no live impact.
- **L2. The latency bound is cap plus one batch, not cap plus 20 ms.**
  Pendings are decided at the end of each poll cycle. When idle, that's the
  20 ms sleep. Under a full backlog it's the duration of one batch (up to 128
  manifests). That is still bounded and still never blocks ingest, but the
  claim in the notes is slightly optimistic.
- **L3. Some paths skip or drop pending localizations.**
  - The `query_pending_limited` error path returns before the batch boundary.
    The next cycle services it.
  - Shutdown drops any pending localization (at most one per cluster, at most
    1 s of solves).
- **L4. Deferred results publish out of order.**
  A deferred result is published after later frames' heartbeats, but it carries
  the trigger frame's `created_ns`. The delay is at most cap plus one batch. I
  didn't audit Python consumers for an arrival-order assumption.
- **L5. Pre-existing, not introduced here: a cluster that *shrinks* under the same key.**
  `buffers` stay longer than `mic_positions_m` and the legacy path reads every
  buffer. In practice the removed member stops covering new windows, so it is
  never eligible.
- **L6. AGENTS.md file size.** `dsp_worker.rs` was already about 2,650 lines
  and the fix adds about 200. The new modules are small: 299, 228 and 498
  lines, the last being the tests file, which is now close to the limit.
  Splitting `dsp_worker.rs` is beyond this review.
- **L7. Duplicated defaults in `sidecar_coverage_wait_environment`.**
  `getattr(..., 0.5/1.0)` repeats the Settings defaults. This follows the
  existing convention in that file, and Settings always defines the keys.

## Verified, no change needed

- **Hot path.**
  - Parking is a struct move into a per-buffer `Option`, with no sleep and no lock held.
  - Observe is O(channels) per append. `service_all` is O(buffers) per cycle.
  - Memory is bounded: one pending per buffer key, the parked manifest has its
    audio bytes stripped, and state is evicted with the buffer in
    `purge_stale_streams`.
  - Heartbeats and renders on the trigger frame are still emitted at once.
    Other clusters are unaffected because state is keyed per buffer.
- **Release rules versus Python.**
  - The stall memory persists per channel, so a dead node costs one stall
    budget once (worker test). A rejoining node is waited for again as soon as
    it advances.
  - A node that never streamed is ignored, a documented divergence.
  - A node lagging by more audio than the remaining cap is skipped immediately.
  - The clock is a monotonic `Instant`.
- **Ring-buffer eviction or reanchor between park and release.**
  `coverage_centered_at` and `window_centered_at` return `None` when the window
  start falls before the buffer start or after a reanchor. The channel is then
  ineligible, so the solve never uses wrong samples. In any case the 32 s ring
  is far longer than the 1 s cap.
- **Superseding.** The older pending window is finalized against the current
  buffers and dispatched, and the newer one is parked. With a 0.8 s cooldown and
  a 0.5 s stall budget, this happens only for a node that lags but keeps advancing.
- **Flag off.**
  - Observe and service return early, `coverage_deferral_span` returns `None`,
    and the clamp index is 0, the same as `buffers.first()`.
  - No outcomes are recorded, so the state lock is never taken.
  - The only difference is the two additive fields on `/dsp/status`.
- **Clamp fix.** `reference_buffer_index` is the arriving node's first cluster
  channel, and it is 0 for single-node streams, so those are unchanged.
- **Config registration.** The three keys predate this branch (Python fix) and
  are not duplicated. `tests/test_config_structured.py` passes (5).
- **Tests that test.**
  - The legacy worker test and the flag-off e2e test reproduce the bug.
  - The flag-on e2e test passed 5 of 5 runs with the widened budget, and 3 of 3
    with the *default* 0.5 s / 1.0 s budget on an idle machine. The widened
    budget is only load headroom, not a hidden failure.
  - Release timing is covered by the manual-clock Rust tests, not the e2e test.

## Test results (review branch)

- `cargo test --workspace`: 551 passed, 0 failed. The sidecar coverage tests
  are 10 policy tests and 9 worker tests.
- `pytest tests/ --ignore=tests/test_soak_harness.py`: 2055 passed, 19 skipped, 0 failed.

## Deploy (Bluefin 192.168.8.103)

The deploy steps don't change. Defaults are on, the env vars flow from Settings,
and in rust mode a change to the three keys reports `restart_required`. After
deploy, check `GET /api/v1/dsp/status`:

- `localization_coverage_wait_stop_reasons` should be mostly `all_covered`.
- A steady stream of `stalled` means one member is dead or its timeline is
  offset by more than the cap.
- `localization_coverage_wait_max_ms` should stay below the cap.
