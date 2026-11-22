# Coverage race fix in the Rust ingest sidecar (Mode 2)

Branch `fm/sidecar-coverage-race` (based on `field-mode` @ 04f491c). This is the Rust
counterpart of `coverage_race.md`. The fix is on by default.

## Confirmed cause

The analysis in `coverage_race.md` § "Rust ingest sidecar" holds. In
`minimappr-ingest-sidecar/src/dsp_worker.rs`:

1. `resolve_cluster_buffer_routing` puts every cluster member (explicit cluster or
   automatic GPS array) into one shared buffer, `cluster::<id>::<stream>`.
2. Every arriving manifest builds its localization window centred on *its own* frame end.
3. `eligible_localization_channels` drops channels below `min_coverage_ratio` (0.85).
   Whichever node delivers frame k first therefore solves with only its own channels.
4. `should_run_localization` (trigger cooldown 0.8 s plus `localization_cadence_ms`) is
   keyed on the cluster buffer key, so the later node's arrival cannot re-run the solve
   with both nodes. Even with cooldown 0 it could not help: the window of the early node's
   *next* frame ends past the late node's buffer end again.

A detail that made this look intermittent: `resolve_localization_center_time_ns` clamped
the window against cluster channel 0. When the node that joined first (channel 0) was the
*late* one, the clamp pulled the window back into its coverage by accident. When the early
node owned channel 0, which is the usual case, every solve was single-node. This matches
"0 cross-node pairs" on the live box.

Reproduction: `dsp_worker_coverage_race_tests::legacy_dispatch_localizes_the_lagging_cluster_alone`.
Two tetras share one explicit cluster, node b's frame k is delivered after node a's frame
k+1, and production cadence and cooldown are used. With the flag off, which is
byte-for-byte the pre-fix dispatch, every cadence solve has 4 active channels. The same
scenario through the real binary is
`tests/test_rust_sidecar_coverage_race_e2e.py::test_live_sidecar_without_coverage_wait_reproduces_single_node_solves`:
every solve has 0 cross-node pairs.

## Fix

The fix defers the dispatch and never sleeps, so the ingest/audio hot path never blocks.

- `src/localization_coverage_wait.rs` holds the policy, `evaluate_window_coverage`, which
  has the same meaning as `sensor_span_coverage_wait.py`:
  - A localization is held while any member channel that has not yet delivered audio up to
    the window end could still do so. Such a channel is streaming (has buffered audio), its
    buffer start is at or before the window start, it trails the window end by no more
    audio than the remaining cap, and it advanced within the stall budget.
  - Stall budget: a channel whose buffer end has not advanced for
    `localization_coverage_stall_budget_seconds` is released.
  - Total cap: no window is held longer than `localization_coverage_total_cap_seconds`.
  - `ChannelStreamActivityTracker` persists per cluster buffer. It is updated in O(channels)
    on every append. A dead node costs at most one stall budget, once. Later windows skip it
    immediately.
- `src/deferred_localization.rs` holds `DeferredLocalizationScheduler`:
  - At most one pending localization per buffer key. A newer candidate dispatches the older
    one first (stop reason `superseded`).
  - Pending entries are dropped with the stream buffers in `purge_stale_streams`.
  - Pendings are re-evaluated only at batch boundaries: at the end of every DSP poll cycle,
    which repeats every 20 ms when idle. Channel progress is still observed on every
    append. A cadence solve on a cluster buffer is always parked until the batch boundary,
    even when it already looks covered.
  - Batch boundaries matter because `merge_pending_manifests_for_batch` processes a backlog
    **newest-first**. Mid-batch, a member can show frame k+1 while frame k is still a hole
    that the same batch backfills a moment later. Deciding mid-batch made the first e2e
    version flaky: about 1 in 5 runs dispatched a 4-channel solve because node b's hole
    looked "settled". On the live box, where backlog was the norm, newest-first ordering
    was probably a second contributor to single-node solves.
  - Parking costs no wall time. Single-node buffers keep the immediate path.
  - When released, the pending payload re-reads the buffers at the **original window
    centre**. The solve therefore covers the trigger frame's span, using every channel that
    arrived, and keeps each node's own packet-GPS timeline (TDOA precision is unchanged).
- `dsp_worker.rs` changes (small):
  - A cadence-triggered solve whose window is not yet covered by the streaming members is
    moved into a localization-only payload, built by `localization_only_template`. The
    triggering frame still emits its heartbeat and its classifier render, if one is due.
  - The localization-only copy marks its manifest as memory-path, so only the original
    payload consumes a persisted manifest.
  - With the flag on, the window clamp uses the arriving node's own channel instead of
    cluster channel 0.
  - Single-node buffers are unaffected. Their own channels always cover the window, so the
    stop reason is `all_covered` and the wait is 0.
- Divergences from Python:
  - Channels that never buffered audio are "not streaming" and are ignored, not counted as
    stalled.
  - Rust has no `required_covered_count` drop contract, because Rust never dropped
    under-covered candidates. It solves with whatever is covered, as before.
  - A gap inside an already-delivered window is not waited on past the batch boundary.
    Within a batch, newest-first ordering backfills it. After the boundary, the gap is a
    lost packet, and waiting on it would burn the cap on every loss (the Pico nodes lose
    ~45% of frames). The coverage ratio still gates the channel at solve time.
- The classifier render is unchanged. A render due on a trigger frame (about every 28 s)
  still steers with that frame's coverage.

### Config (Python ↔ Rust parity of meaning)

| Python Settings key | Sidecar env (set by `ingest_sidecar_runtime.build_ingest_sidecar_environment`) | default |
|---|---|---|
| `localization_coverage_wait_all_streaming_nodes_enabled` | `MINIMAPPR_LOCALIZATION_COVERAGE_WAIT_ALL_STREAMING_NODES_ENABLED` | `true` |
| `localization_coverage_stall_budget_seconds` | `MINIMAPPR_LOCALIZATION_COVERAGE_STALL_BUDGET_SECONDS` | `0.5` |
| `localization_coverage_total_cap_seconds` | `MINIMAPPR_LOCALIZATION_COVERAGE_TOTAL_CAP_SECONDS` | `1.0` |

The sidecar CLI defaults match, so a sidecar started by hand behaves the same way. The three
keys were added to `_SIDECAR_RESTART_REQUIRED_KEYS`, so in rust mode a PATCH reports
`restart_required`.

### Observability

`GET /api/v1/dsp/status` (sidecar) now reports `localization_coverage_wait_stop_reasons`
(`all_covered` / `stalled` / `total_cap` / `superseded`) and
`localization_coverage_wait_max_ms`, under the same names as the Python `FusionMetrics`
fields.

## Added latency

- **All nodes present, arriving together: 0 ms.** `all_nodes_present_adds_no_wait` uses
  lockstep delivery. Every solve has 8 channels, the max wait is 0.0 ms, and every stop
  reason is `all_covered`. The deferral lasts only until the other node's frame is processed
  in the same DSP batch, which is microseconds.
- **Late but streaming node: adds the inter-node arrival skew.** In the one-frame-late test
  the wait is under one frame (32 ms) and every solve has 8 channels.
- Real debug binary, measured end to end: node b posted one frame late over HTTP, and the
  harness polled `/api/v1/dsp/results` after every post.
  - Cadence solves had 28 pairs (8 channels). The status showed
    `{"all_covered": 3}` and `localization_coverage_wait_max_ms` = 95.7.
  - The 95.7 ms is the wall time between node a's frame k+1 POST and node b's frame k POST,
    which includes that results GET. It is the arrival skew the harness created, not
    processing cost.
  - With the flag off, the same replay has 0 cross-node pairs in every solve.
- **Dead node: at most one stall budget, once.**
  `stalled_node_is_released_within_the_stall_budget_and_then_costs_nothing`: node b dies at
  frame 10. The first trigger after death waits 64 ms, which is the 0.5 s budget minus the
  time already elapsed since b's last frame. The next two triggers wait 0 ms. No trigger is
  lost and nothing is left pending.
- **Worst case for one window:** `total_cap` (1.0 s) plus one poll interval (20 ms).

## Tests

Rust:

- `localization_coverage_wait_tests` (10 tests): all covered, late-but-advancing node,
  stall release, dead node free on later windows, total cap while advancing, lagging beyond
  the cap, never-streamed channels ignored, intra-window gap not waited on, disabled flag,
  and config clamping.
- `dsp_worker_coverage_race_tests` (6 tests, deterministic manual clock):
  - legacy repro (single-node)
  - late node included in every solve
  - all present adds no wait
  - stalled node released within budget, then free
  - total cap releases while the node is still advancing
  - poll tick releases when every member goes quiet

Python:

- `tests/test_rust_sidecar_coverage_race_e2e.py` (2 tests, real sidecar binary): with the
  coverage wait on, every cluster solve contains cross-node pairs. With it off, none do.
  The test widens the stall budget and cap through env. It checks inclusion, not timing,
  and under full-suite CPU load sequential HTTP posts can be more than 0.5 s apart.
  Passed 10/10 consecutive runs.
- `tests/test_http_api.py`: the spawn env asserts the three new env keys.
