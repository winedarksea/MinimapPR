# Effectors (PTZ cameras)

This page covers driver behavior and tuning. For the install extra and the endpoint list, see
the [README](../README.md): the "PTZ camera slew-to-track" feature, `pip install
"minimappr[full]"`, and the `/api/v1/nodes/{node_id}/effector/...` routes.

## Scope

- The only driver is ONVIF PTZ (`minimappr/core/effectors/onvif_ptz.py`). A node becomes
  an effector when it has the `ptz_camera` capability. No other effector hardware has a
  driver yet.
- ONVIF support is the optional `onvif-zeep-async` dependency, from the `effector` or `full`
  extra. It is imported lazily, so a base install runs without it, and PTZ nodes report
  that the package is missing.
- There is no camera discovery. You set the camera by hand in the node's `transport`:
  `host` (required), `port` (default `80`), `username`, `password`, and an optional
  `rtsp_url`.
- The subsystem stays dormant until a PTZ node is registered. `effectors_enabled` is only a
  kill switch.

## Driver behavior

- **Move mode.** At connect, the driver picks the first move mode the camera supports, in
  reliability order: `AbsoluteMove`, then `RelativeMove` (a delta from the `GetStatus`
  position), then `ContinuousMove` with a computed timed `Stop`. Several Reolink models
  reject `AbsoluteMove` but always support `ContinuousMove`.
- **Snapshots.** The driver fetches the `GetSnapshotUri` HTTP URL. If the camera has no
  snapshot URI, it falls back to an ffmpeg single-frame grab from the RTSP stream.
  `capture()` writes the JPEG under `effector_snapshot_dir` (default
  `data/effector_snapshots`) and records it as a `node_artifacts` row linked to the
  triggering track or detection.
- **Reconnect and error state.** A failed connect is retried on each status poll
  (`effector_status_poll_interval_seconds`, default `5.0`). The camera reports `error` when
  a move fails or `GetStatus` gets no answer. It returns to `idle` once `GetStatus` answers
  again.
- **Dwell, then return home.** After a slew, the camera holds for
  `effector_slew_dwell_seconds` (default `10.0`; `0` disables the return). It then calls
  `GotoHomePosition`, or slews to the home bearing (pan 0, tilt 0) if the camera has no home
  preset. A newer slew cancels a pending return.

The `effector_*` settings can only be set through environment variables
(`MINIMAPPR_EFFECTOR_SLEW_DWELL_SECONDS`, and so on). They are not in the PATCH allowlist,
so `data/config.yml` ignores them.

## Safety interlocks

`EffectorManager._check_interlocks` (`minimappr/core/effectors/registry.py`) runs before
every movement command. It checks the node's `safety` block, which you edit with
`PATCH /api/v1/nodes/{node_id}/safety`:

| Check | Rejection | Config |
|-------|-----------|--------|
| Armed | `interlock:disarmed` | `require_arm_for_action` (default `false`); arm with `POST .../effector/arm` |
| Rate limit | `rate_limited` | `min_action_interval_seconds`; falls back to `effector_min_slew_interval_seconds` (default `3.0`) |
| No-go zones | `interlock:no_go_zone` | `no_go_zone_ids`: the slew is refused if the target point falls inside any listed zone |

A rejection returns a `REJECTED` execution result with the reason as `failure_class`, and is
logged at INFO. It is not written to a persistent audit log.
