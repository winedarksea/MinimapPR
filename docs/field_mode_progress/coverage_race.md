# Coverage race fix (default localization path)

Branch `fm/coverage-race` (based on `field-mode`). This is cause #3 in `onset_srp.md`, fixed on
the default (unflagged) path. The fix is on by default.

## Confirmed cause

`FusionNode._await_window_coverage` returned as soon as `min(len(sensors), min_sensors_for_2d)`
sensors covered the window. One tetra has 4 mics, so the trigger node's own channels were enough,
and any node whose matching frame landed a few ms later was silently left out of the solve.

Reproduction (`test_calibration_harness_synthetic`, two tetras, source 0.7 m from node-a): the
test wraps `_await_window_coverage` to log the covered nodes and sleeps 30 ms before every node-b
frame.

| run | covered at localization | detection | L&C recall @15 m |
|---|---|---|---|
| unslowed (usual case) | node-a, node-b | (5.96, 2.18, -1.94) | 1.0 |
| node-b slowed 30 ms/frame, pre-fix | **node-a only** | (49.5, -4.0, 6.1), single-tetra solve projected by the far-field guard | **0.0**, `unassociated_localized_entities=1` |
| node-b slowed, with fix | node-a, node-b (waited 43 ms) | (5.96, 2.18, -1.94) | 1.0 |

The roughly 1-in-6 flake in `test_synthetic_bundle_full_replay_loop` is the same thing without
the injected delay. Under CPU load, the localization worker sometimes ran between node-a's frame k
and node-b's frame k.

## Fix

New module `minimappr/core/sensor_span_coverage_wait.py`:

- `await_sensor_span_coverage` is the shared non-blocking poll, now also used by the onset-SRP
  stage's `_await_span`. It keeps waiting while any uncovered sensor that can still cover the
  span is advancing. A sensor counts as able to cover when it is present, its buffer start is at
  or before the span start, and it trails the span end by less audio than the remaining cap.
  - A sensor whose buffer has not advanced for `stall_budget` is released.
  - A hard `total_cap` bounds the whole wait.
  - The legacy drop contract is kept: below `required_covered_count` the wait runs to the old
    `localization_buffer_wait_max_seconds` deadline, and the caller then drops the candidate as
    `buffer_lag_timeout` / `event_too_old`.
- `SensorStreamActivityTracker` is held on the FusionNode and shared across waits. It remembers
  when each sensor's buffer last advanced, so a dead node costs one stall budget once and nothing
  on later events. It is updated only from coverage-wait polls, so the ingest hot path pays
  nothing.
- `FusionNode._await_window_coverage` is now a thin wrapper (fusion_node.py shrank). Turning the
  flag off gives stall budget 0 and cap = `timeout_s`, which is exactly the pre-fix behaviour.
- New metrics: `localization_coverage_wait_stop_reasons` (all_covered / stalled / total_cap /
  required_coverage_deadline / span_older_than_buffer) and `localization_coverage_wait_max_ms`.

Config is under `# --- coverage race fix ---` in config.py, the env loader, settings_store.py,
main.py, config_groups.py, and pipeline_graph.py (on the Localization Solve node):

| key | default | meaning |
|---|---|---|
| `localization_coverage_wait_all_streaming_nodes_enabled` | `true` | wait for every streaming node |
| `localization_coverage_stall_budget_seconds` | `0.5` | release a node after this long with no buffer progress (about 8 frames of 64 ms) |
| `localization_coverage_total_cap_seconds` | `1.0` | hard cap on the whole wait |

Onset-SRP side effects, both improvements:

- Progress now counts per uncovered sensor. Before, any advancing sensor (including covered ones)
  kept the wait alive, so with a dead node and live peers the wait ran to the 15 s hard deadline.
- Sensors that can never cover (buffer starts after the span, or too far behind) are skipped.

## Added latency

- All nodes present (the usual live case): **about 0 ms**. Three unslowed replays measured
  0.0–0.1 ms, and `test_all_nodes_present_adds_no_wait` asserts that a single snapshot is taken.
- A late but streaming node adds the inter-node arrival skew: 43 ms with one node delayed 30 ms
  per frame.
- A dead node adds at most `stall_budget` (0.5 s) on the first event after it dies, then 0.
- A node lagging by more audio than the cap is skipped immediately.
- The worst case for one candidate is `max(localization_buffer_wait_max_seconds, total_cap)` = 1.0 s.
- The wait is `asyncio.sleep` polling, so the event loop is never blocked.

## Rust ingest sidecar

There is no coverage wait, but the same class of race exists in a different form, and it was not
fixed here. In `minimappr-ingest-sidecar/src/dsp_worker.rs`:

1. Cluster-routed nodes share one buffer (`resolve_cluster_buffer_routing`).
2. Each arriving manifest localizes immediately on a window centred at that manifest's end.
3. Channels below `min_coverage_ratio` (0.85) are dropped by `eligible_localization_channels`, so
   the first member node to deliver frame k localizes alone.
4. `should_run_localization` (trigger cooldown plus `localization_cadence_ms`) is keyed on the
   *cluster buffer key*, so the later node's arrival usually cannot re-run the solve that would
   include both nodes.

Fixing it needs a deferral, such as holding the dispatch until active member channels cover the
window (with a stall budget), or centring on the minimum end time across actively streaming
channels. That is a design change in the manifest dispatch loop, so it is reported rather than
patched.

## Tests

- `tests/test_sensor_span_coverage_wait.py` (10 tests):
  - Deterministic stepping-buffer reproduction of the race (the pre-fix semantics exclude
    node-b), plus the waited-for late node.
  - All nodes present → no wait.
  - Stall release, total cap respected, dead node free on the next wait, lagging node skipped,
    legacy required-count deadline.
  - Two FusionNode-level tests with a real `MultiSensorBuffer`: a late frame is waited for, and a
    stalled node is released within the bound and costs nothing on the next event.
- `tests/test_calibration_harness_synthetic.py::test_synthetic_bundle_replay_with_slowed_node_keeps_both_nodes`
  is the end-to-end regression (node-b slowed 30 ms/frame). It fails with the flag off and passes
  with it on.
