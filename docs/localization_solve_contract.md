# Localization solve contract (Rust sidecar <-> Python)

Source of truth for the wire shapes: `minimappr-core/src/localization_sensor_entry.rs`
(serde types) plus `PairTdoaDiagnostic` / `LocalizationManifestPayload` in
`minimappr-ingest-sidecar/src/manifests.rs`. The example JSON below is pinned
by the Rust test `contract_doc_example_round_trips_field_for_field`; when a field
changes, update the type, that test and this document together.

Status (milestone 2): the sidecar emits `solve_provenance` on every
`localization_result` and `classifier_render` (built in
`minimappr-ingest-sidecar/src/localization_solve_provenance.rs`), with
`excluded_pairs` from pair admission (R7,
`minimappr-ingest-sidecar/src/dsp_solve_pair_admission.rs`) and
`pair_tdoas[].is_cross_node` / `timing_uncertainty_seconds` on every admitted
pair, timed from `node_timing` (clock evidence, R5,
`minimappr-ingest-sidecar/src/dsp_node_clock_quality.rs`; section 7). A manifest without
`solve_provenance` is a legacy manifest.

## 1. Where it appears

`localization.solve_provenance`, an object inside the `localization` payload of:

- `localization_result` manifests, and
- `classifier_render` manifests (the render carries a copy of the solve's
  `localization` payload, so the same `solve_id` appears on both).

A render does not localize its own trailing frame when its span contains the
latest event-triggered solve on its buffer: it solves that event's window
instead, so the stored detection is the event solve the render covers. When
the buffers still cover exactly the sensors the event solve used, the render
re-runs that solve and repeats its `solve_id`, `window_center_time_ns` and
layout snapshot (the `localization_result` published with the render repeats
it too; one `solve_id` is one solve, however many manifests carry it). When a
member whose audio arrived late now covers the window as well, the render
solves the fuller sensor set under a new `solve_id` (same window centre and
`attempt_kind`), and later covering renders repeat that one.

The key is omitted (not `null`) when the sidecar has no provenance to report.

## 2. `solve_provenance` fields

All fields are always present once the object is emitted. "nullable" means the
value can be JSON `null`.

| Field | JSON type | Meaning |
|---|---|---|
| `schema_version` | integer | `1`. Bumped on any incompatible change. |
| `solve_id` | string | Identifies one solve. A render covering the solve repeats the id. |
| `attempt_kind` | enum string | Why the solve ran (section 3.1). |
| `buffer_scope` | enum string | Which buffer the audio came from (3.2). |
| `cluster_id` | string, nullable | Cluster of the buffer; `null` for node-local solves. |
| `reporting_node_id` | string | Node whose packet produced the manifest. Not necessarily the only contributor; store separately from `cluster_id`. |
| `node_local_routing_reason` | enum string, nullable | Why the reporter solved node-local (section 3.7); `null` for cluster solves. |
| `coordinate_frame` | object | Frame of every `position_m` (section 4). |
| `layout_revision` | integer | Bumps whenever any sensor position in the layout changes, GPS updates included. |
| `geometry_push_revision` | integer, nullable | Revision of the geometry push in effect for this solve; `null` before any push. |
| `sensors` | array of sensor entries | Ordered solve sensors (section 5). Index `i` is buffer channel `i`. |
| `active_sensor_indices` | array of integers | Indices into `sensors` that took part in the solve. |
| `excluded_sensors` | array | `{sensor_index, sensor_id, reason}` for each sensor left out (section 6). |
| `excluded_pairs` | array | `{sensor_index_a, sensor_index_b, reason}` for each pair left out (section 6). |
| `node_timing` | array | Clock evidence per contributing node (section 7). |
| `window_center_time_ns` | integer | Packet-derived centre of the solve window, ns since the Unix epoch. This is the stored observation time; never substitute receipt time. |
| `window_seconds` | number | Solve window length, centred on `window_center_time_ns`: the node-local window, or for an event solve across nodes more than ~half a node-local window of lag apart, the cross-node event span (twice the max cross-node lag, capped by the coverage-wait cap and buffer length; milestone 2 / R4). |
| `solve_sample_rate_hz` | integer | Rate of the solve buffer. |

### Index rule

Every index in the contract — `active_sensor_indices`, `excluded_sensors[].sensor_index`,
`excluded_pairs[].sensor_index_a/_b`, and `localization.pair_tdoas[].ch_a/ch_b` —
indexes **this solve's** `sensors` array. It is never a channel of the reporting
node and never an index into Python's registry. Map through `sensors[i].sensor_id`.

## 3. Enumerations

### 3.1 `attempt_kind`

| Value | Meaning |
|---|---|
| `event_trigger_absolute_rms` | Event trigger: the absolute RMS gate (wins when both gates fire). |
| `event_trigger_noise_floor_relative` | Event trigger: the node's frame stood `localization_trigger_noise_floor_relative_db` over its tracked noise floor while the absolute gate did not fire (milestone 2 / R4). |
| `scheduled_classifier_render` | Solved only because a classifier render was due on that frame. |
| `not_attempted` | No solve (omni fallback, heartbeat, too few channels). |

Only event-triggered solves may start a provisional track (P5).

### 3.2 `buffer_scope`

| Value | Meaning |
|---|---|
| `node_local` | The reporting node's own channels only. |
| `explicit_cluster` | Operator-declared cluster (manifest `cluster_id` + `cluster_sensor_positions`). |
| `automatic_site_cluster` | The sidecar's automatic cluster of every positioned node on the stream (cluster id `auto-gps`). |

### 3.3 `coordinate_frame.kind`

| Value | `origin` / `projection` | Meaning |
|---|---|---|
| `site_enu` | set | East/north/up metres about the pushed `site_origin`; `frame_id` echoes the push. |
| `sidecar_automatic_enu` | both set once the first trusted GPS fix is seen (both `null` before); `projection` is `flat_wgs84_series` | ENU about an origin the sidecar chose (first trusted GPS fix) because nothing was pushed. `frame_id` is `null`, so Python must not accept these positions as site positions. |
| `explicit_cluster_positions` | both `null` | Positions copied verbatim from the manifest's explicit cluster geometry. |
| `node_body` | both `null` | The reporting node's own body-frame mic offsets (node-local solve). |

### 3.4 `projection`

| Value | Python equivalent |
|---|---|
| `flat_wgs84_series` | `LocalCoordinateFrame(mode="flat")` |
| `geodetic_ecef_enu` | `LocalCoordinateFrame(mode="geodetic")` |

### 3.5 `position_source` (sensor entries)

Precedence, highest first: pushed placement, then GPS, then the rest.

| Value | Meaning |
|---|---|
| `operator_override` | Pushed placement an operator entered (hand-measured / surveyed). Always admitted cross-node; its `position_std_m` widens the error. |
| `stationary_estimate` | Pushed placement from Python's filtered stationary-node estimate. |
| `reported_gps` | Raw packet GPS, used only for nodes with no pushed placement. |
| `fallback_static` | Firmware built-in position; untrusted for cross-node baselines. |
| `explicit_cluster_manifest` | From the manifest's `cluster_sensor_positions`. |
| `node_body_offset` | Body-frame offset of a node-local solve (no world position). |

### 3.6 `clock_source` and `node_timing[].time_quality`

`gps_locked`, `gps_holdover`, `ntp_disciplined`, `free_running`, `unknown`.
Firmware / Python `TimeQuality` spellings; legacy `ntp_sync` is read as
`ntp_disciplined` (as Python converts it on load); any other value is reported as `unknown`.

### 3.7 `node_local_routing_reason`

| Value | Meaning |
|---|---|
| `node_array_geometry_unusable` | 2-3 channels without usable offsets: the node cannot be placed in a cluster. |
| `node_position_untrusted_no_override` | No trusted world position (`fallback_static` or no fix) and no pushed placement. |
| `explicit_cluster_incomplete` | The manifest carries only one of `cluster_id` / `cluster_sensor_positions`. |
| `explicit_cluster_below_minimum_sensors` | The declared cluster has fewer sensors than an SRP solve needs. |
| `node_channels_missing_from_explicit_cluster` | Some of the node's channels are not in the declared sensor list. |

## 4. `coordinate_frame`

```json
{"frame_id": "site-7", "kind": "site_enu",
 "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
 "projection": "flat_wgs84_series"}
```

| Field | JSON type | Meaning |
|---|---|---|
| `frame_id` | string, nullable | Echo of the pushed `site_frame_id`; `null` until Python has pushed. Python accepts a solve's positions only when this equals its own frame id; otherwise omni with a stated reason. |
| `kind` | enum string | Section 3.3. |
| `origin` | object, nullable | `{lat, lon, alt_m}` (Python `GeoPoint` names), degrees / metres. |
| `projection` | enum string, nullable | Section 3.4. |

Local axes: x = east, y = north, z = up, metres.

## 5. Sensor entry (`sensors[]`)

| Field | JSON type | Meaning |
|---|---|---|
| `sensor_id` | string | `<node_id>:ch<node_channel_index>`. |
| `node_id` | string | Owning node. |
| `node_channel_index` | integer | Channel on the owning node. |
| `position_m` | `[x, y, z]` numbers | **The snapshot this solve used**, in `coordinate_frame`. Re-solve with these, not with the current registry. |
| `sample_rate_hz` | integer | The sensor's own rate. |
| `position_source` | enum string | Section 3.5. |
| `position_std_m` | number, nullable | 1-sigma position accuracy of the sensor's node placement, per axis (read as isotropic); widens the published covariance (section 8). `null` when nothing states one, and then adds nothing. |
| `clock_source` | enum string | Section 3.6. |
| `clock_domain` | string | Timebase of the packet timestamps; currently always `utc`. |
| `clock_offset_std_s` | number, nullable | 1-sigma clock-offset uncertainty used for this sensor's pairs: its node's `node_timing[].clock_offset_std_s`. |

`clock_source` and `clock_offset_std_s` are filled from the solve's clock
snapshot (section 7); `unknown` / `null` only for a node with no snapshot entry.

Sensor world position = node position + R(body-to-world) x body offset, with
R the identity when the node has no orientation (rotation convention in section 9).

## 6. Exclusions

Every exclusion carries one `reason`:

| `reason` | Applies to | Meaning |
|---|---|---|
| `sample_rate_mismatch` | sensor, pair | Sensor rate differs from `solve_sample_rate_hz`. |
| `pair_baseline_below_minimum` | pair | Separation under `localization_min_pair_baseline_m`. |
| `coverage_below_threshold` | sensor, pair | Window coverage under the minimum ratio. |
| `timeline_residual_over_limit` | pair | A node's measured timeline residual exceeds `localization_cross_node_max_timeline_residual_seconds`; that node's cross-node pairs are dropped; its intra-node pairs stay admitted (they solve when it is the reporter, section 6.1). |
| `node_position_untrusted_no_override` | sensor | `fallback_static` node with no pushed placement. |
| `node_array_geometry_unusable` | sensor | The node's own offsets fail validation for its channel count. |
| `max_solve_sensors_exceeded` | sensor | Beyond `localization_max_solve_sensors`. |

A weak clock class alone never excludes a pair; it raises that pair's
`timing_uncertainty_seconds`.

### 6.1 Pair admission

One admission is computed per solve from its layout snapshot, its active
sensors and the clock snapshot taken at trigger; a parked solve recomputes it
when it is released (coverage changed while it waited), and a render that
repeats an event solve repeats that solve's admission. Rules, applied to
every pair of sensors **not** already in `excluded_sensors` for
`sample_rate_mismatch`, `node_array_geometry_unusable` or
`max_solve_sensors_exceeded` (those sensors contribute no pair entries; the
sensor exclusion is the record), first match wins:

1. either sensor is not in `active_sensor_indices` -> `coverage_below_threshold`;
2. sensor separation under `localization_min_pair_baseline_m` -> `pair_baseline_below_minimum` (same-node pairs included);
3. cross-node pair and either node's measured timeline residual is over
   `localization_cross_node_max_timeline_residual_seconds` -> `timeline_residual_over_limit`
   (that node's intra-node pairs stay admitted);
4. otherwise admitted.

Pairs are listed with `sensor_index_a < sensor_index_b`. An excluded pair is
never correlated, so it is absent from `pair_tdoas` and contributes nothing to
the SRP grid, the reference TDOAs or the covariance. A node whose timeline is
over the limit is also left out of the beamformed classifier render (it
cannot be delay-aligned with other nodes); when that node is the reporter,
the render uses only its own channels.

SRP solves only the active sensors connected to the reporting node through
admitted pairs (its reference fit, centroid, aperture and grid assume one
connected array). So a timeline-rejected node that is not the reporter takes
no part in the solve: its intra-node pairs stay admitted (not listed in
`excluded_pairs`) but are not correlated and are absent from `pair_tdoas`,
and its sensors stay in `active_sensor_indices` (they were covered). When
the rejected node is the reporter, only its own channels solve. A reporter
with no covered sensor solves on the largest connected group.

```json
{"sensor_index": 3, "sensor_id": "point-c:ch0", "reason": "sample_rate_mismatch"}
{"sensor_index_a": 0, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"}
```

## 7. `node_timing[]`

| Field | JSON type | Meaning |
|---|---|---|
| `node_id` | string | |
| `time_quality` | enum string | Section 3.6. |
| `clock_offset_std_s` | number, nullable | Node timing uncertainty (seconds): max of class default, reported holdover error, \|measured residual\|, active step penalty. |
| `timeline_residual_ns` | integer, nullable | Signed packet timestamp minus the node's buffer timeline, aggregated over the node's last `localization_timeline_residual_window_frames` frames (the frame residual at `localization_timeline_residual_percentile` of their magnitudes, sign kept; 50 = median); `null` until measured, or while the node is off packet time. |
| `last_timestamp_step_ns` | integer, nullable | Signed size of the most recent detected timestamp step (positive = clock jumped ahead); `null` if none. |
| `timestamp_step_count` | integer | Steps detected since the node was first heard (tracker entries are evicted after `localization_sensor_registry_stale_seconds` of silence). |
| `timeline_step_reanchor_count` | integer | Confirmed steps the node's buffer timeline was re-anchored on, same lifetime. Absent in older manifests (read as 0). |
| `timeline_drift_slew_count` | integer | Drift slews of the node's buffer timeline origin, same lifetime. Absent in older manifests (read as 0). |

One row per node in `sensors[]`, in sensor order, from the clock snapshot
taken when the solve was **triggered** (a parked / coverage-deferred solve
reports the snapshot it triggered with). Rules (Rust `dsp_node_clock_quality.rs`):

- **Per-frame evidence.** Decode keeps every frame's packet timestamp,
  `time_quality`, holdover predicted error (MMB3 section `0x0010`, only while
  `holdover_active`) and boot session, so a step between two frames of one
  upload is visible. `time_quality` is the latest frame's.
- **Class default.** `localization_clock_class_uncertainty_seconds[time_quality]`;
  `unknown`, or a class missing from the map, takes
  `localization_clock_unknown_uncertainty_seconds` (widened, never excluded).
  A node in the layout that has not streamed yet reports `unknown` with that default.
- **Off packet time.** A node whose latest segment carried no node timestamp
  (its audio sits on server or receipt time) reports `unknown` with
  `localization_clock_unknown_uncertainty_seconds`, a `null` residual and no
  gate, whatever class its frames claim, until a segment is on packet time again.
- **Step.** Between consecutive frames of one boot session (within or across
  uploads), packet-time advance minus sample-index advance / rate beyond
  `localization_timestamp_step_threshold_seconds`. It raises the node's
  uncertainty to `|step|` until the residual has stayed within the threshold
  for `localization_timestamp_step_recovery_holdoff_seconds` of the node's own
  sample time. The penalty is independent of the re-anchor below: a
  re-anchored node's cross-node pairs come back at once, but carry the step's
  widened uncertainty until the holdoff clears.
- **Boot-session change.** Not counted as a step (no step size exists: sample
  indices restart); clears the residual and holds
  `localization_clock_unknown_uncertainty_seconds` under the same recovery rule.
- **Residual.** Measured against the nominal-rate timeline the solve buffer
  placed the audio on (its current anchor extrapolated by sample count), only
  when the buffer runs on packet time. Both terms come from the node itself
  (its packet clock against its own sample clock), so the residual carries no
  acoustic delay and no other node. When the windowed residual exceeds
  `localization_cross_node_max_timeline_residual_seconds` the node's
  cross-node pairs are excluded (`timeline_residual_over_limit`); the flag
  clears only once the windowed residual is back under that limit times
  `localization_timeline_residual_over_limit_clear_ratio` (hysteresis), so a
  single jittery frame never flips it. With the timeline discipline on (the
  default) only an unconfirmed step or a clock flapping faster than the
  discipline can follow reaches the gate.
- **Timeline discipline (owner decision 2026-11-25).** Two corrections move
  the node's buffer timeline origin, on every channel of the node, on both
  ingest paths (Rust `minimappr-core/src/timeline_discipline.rs` driven by
  `dsp_node_clock_quality.rs`; Python `minimappr/core/timeline_discipline.py`
  driven by `SensorStreamBuffer.append`; decisions pinned by
  `tests/fixtures/timeline_discipline_parity_cases.json`). The mapping stays
  one straight line over the sample index, so no buffered audio is dropped or
  zero-filled; every buffered sample, pre-step ones included, is retimed onto
  the corrected clock (the sample clock never jumped, and a step is usually a
  re-sync correcting an offset the earlier packets already carried).
  - *Step re-anchor* (`localization_timeline_step_reanchor_enabled`): a
    flagged step is confirmed once `localization_timestamp_step_confirm_frames`
    frames (the flagged one included) agree with the stepped clock within the
    step threshold; the origin then shifts by their lower-median residual. A
    frame back on the pre-step timeline first cancels it (jitter), and a
    reboot cancels it too.
  - *Drift slew* (`localization_timeline_drift_slew_fraction`, 0 = off): when
    the windowed residual magnitude is above that fraction of the hard limit
    and not above the limit itself, the origin shifts by the windowed residual
    (recorded residuals shift with it), so steady crystal drift (20 ppm passes
    10 ms in about 8 minutes) never reaches the gate. Never while a step is
    pending; above the limit is not drift and is left to the gate.
  - A full buffer reset (correction beyond the buffer length, or an
    out-of-bounds frame) still clears the audio and counts in
    `reanchor_count` / `total_buffer_reanchors`; the discipline's origin
    shifts are counted apart (`timeline_step_reanchor_count`,
    `timeline_drift_slew_count` here and per Python sensor buffer;
    `total_timeline_step_reanchors`, `total_timeline_drift_slews` in
    `/api/v1/dsp/status`). Python has no step penalty or residual gate; only
    its timeline is corrected, and receipt-time frames skip the discipline.

## 8. `localization.pair_tdoas[]` additions

`pair_tdoas` holds **admitted pairs only** (section 6.1). Existing fields are unchanged
(`ch_a`, `ch_b`, `delay_samples`, `lag_seconds`, `confidence`); `ch_a`/`ch_b`
index `solve_provenance.sensors`. `lag_seconds` is arrival at `ch_a` minus
arrival at `ch_b`. New fields (present on every pair the sidecar emits;
optional for legacy manifests):

| Field | JSON type | Meaning |
|---|---|---|
| `is_cross_node` | bool | The two sensors belong to different nodes. |
| `timing_uncertainty_seconds` | number | 1-sigma timing uncertainty of this pair: `0` for two sensors of one node (one clock); for a cross-node pair the two nodes' timing uncertainties in quadrature, `sqrt(u_a^2 + u_b^2)`. A node's uncertainty is its clock assessment (largest of class default, holdover error, measured residual, step penalty); a node without one is timed with `localization_clock_unknown_uncertainty_seconds`. The Python re-solve weights the pair by it and inflates covariance to match. |

**Published covariance.** The Rust SRP solver weights every pair equally;
`localization.position_covariance_m2` adds the clock offsets' effect on the
solved position, propagated through the solve's Jacobian (least-squares
sandwich `G R G^T`, `G = (J^T J)^-1 J^T` over the reference TDOAs). `R` is
the per-node correlated timing noise: node `k`'s offset (1-sigma = its
`clock_offset_std_s`) shifts every reference TDOA between node `k` and the
reference channel's node together, so each pair's variance is `u_pair^2` and
pairs sharing a node co-vary. This grows with geometric dilution (about r/b
laterally and r^2/b in range for a node spread b at range r), unlike a flat
per-axis term. A solve with only same-node pairs is unchanged.

**Node position term.** The covariance also carries each node's stated
`position_std_m` (owner decision: a hand-measured position is used as
entered, and its stated accuracy widens the error; it never excludes the
node). A rigid placement error `d ~ N(0, sigma^2 I)` moves every arrival at
that node's sensors by `-u_s . d / c` (`u_s` the unit vector from sensor `s`
toward the solved position), so like a clock offset it is correlated across
the node's pairs; it enters through the same sandwich, with `R` the
direction-dependent projection (`sigma^2 u_i . u_j / c^2` between two sensors
of one node). For a single-sensor node this equals a clock offset of
`sigma / c`; a solve whose sensors all sit on one node gains `sigma^2 I`
(the whole fix moves with the node). Every stated accuracy counts, whatever
the `position_source` (an operator's entry, a stationary estimate's spread,
a reported GPS accuracy); `null` adds nothing. The Python `python_cartesian`
re-solve adds the same term from `sensors[].position_std_m`, through its own
weighted pair rows (`minimappr/core/node_position_covariance.py`; shared
implementation in Rust `minimappr-core/src/tdoa_node_position_covariance.rs`,
pinned by `tests/fixtures/node_position_covariance_parity_cases.json`). The
Python-ingest path adds it from the node registry's accuracy, which follows
the geometry push (operator override, else stationary estimate).

## 9. Example `solve_provenance`

```json
{
  "schema_version": 1,
  "solve_id": "solve-3f2a",
  "attempt_kind": "event_trigger_absolute_rms",
  "buffer_scope": "automatic_site_cluster",
  "cluster_id": "auto-gps",
  "reporting_node_id": "planar-a",
  "node_local_routing_reason": null,
  "coordinate_frame": {
    "frame_id": "site-7",
    "kind": "site_enu",
    "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
    "projection": "flat_wgs84_series"
  },
  "layout_revision": 12,
  "geometry_push_revision": 3,
  "sensors": [
    {"sensor_id": "planar-a:ch0", "node_id": "planar-a", "node_channel_index": 0,
     "position_m": [1.0177, 2.0177, 1.5], "sample_rate_hz": 16000,
     "position_source": "operator_override", "position_std_m": 0.05,
     "clock_source": "gps_locked", "clock_domain": "utc", "clock_offset_std_s": 1e-6},
    {"sensor_id": "planar-a:ch4", "node_id": "planar-a", "node_channel_index": 4,
     "position_m": [1.0, 2.0, 1.5], "sample_rate_hz": 16000,
     "position_source": "operator_override", "position_std_m": 0.05,
     "clock_source": "gps_locked", "clock_domain": "utc", "clock_offset_std_s": 1e-6},
    {"sensor_id": "point-b:ch0", "node_id": "point-b", "node_channel_index": 0,
     "position_m": [6.0, 0.0, 1.5], "sample_rate_hz": 16000,
     "position_source": "reported_gps", "position_std_m": null,
     "clock_source": "ntp_disciplined", "clock_domain": "utc", "clock_offset_std_s": 0.002}
  ],
  "active_sensor_indices": [0, 1, 2],
  "excluded_sensors": [],
  "excluded_pairs": [
    {"sensor_index_a": 0, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"},
    {"sensor_index_a": 1, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"}
  ],
  "node_timing": [
    {"node_id": "planar-a", "time_quality": "gps_locked", "clock_offset_std_s": 1e-6,
     "timeline_residual_ns": 1200, "last_timestamp_step_ns": null, "timestamp_step_count": 0,
     "timeline_step_reanchor_count": 0, "timeline_drift_slew_count": 0},
    {"node_id": "point-b", "time_quality": "ntp_disciplined", "clock_offset_std_s": 0.002,
     "timeline_residual_ns": 15000000, "last_timestamp_step_ns": -21000000, "timestamp_step_count": 1,
     "timeline_step_reanchor_count": 0, "timeline_drift_slew_count": 2}
  ],
  "window_center_time_ns": 1764000000016000000,
  "window_seconds": 0.032,
  "solve_sample_rate_hz": 16000
}
```

## 10. Geometry push

`POST /api/v1/dsp/geometry` (milestone 1 / R2), and the same body at sidecar
launch in env `MINIMAPPR_SITE_GEOMETRY_JSON`. A malformed launch value fails
sidecar startup.

```json
{
  "site_frame_id": "site-7",
  "geometry_push_revision": 3,
  "site_origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
  "projection": "flat_wgs84_series",
  "node_placements": {
    "planar-a": {"position_m": [1.0, 2.0, 1.5], "position_std_m": 0.05,
                 "placement_source": "operator_override",
                 "body_to_world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
    "point-b":  {"position_m": [6.0, 0.0, 1.5], "position_std_m": 0.4,
                 "placement_source": "stationary_estimate"},
    "planar-c": {"placement_source": "operator_override",
                 "body_to_world_quaternion_wxyz": [0.9537, 0.0, 0.0, 0.3007]}
  }
}
```

| Field | JSON type | Meaning |
|---|---|---|
| `site_frame_id` | string | Echoed as `coordinate_frame.frame_id` on solves that use this push. |
| `geometry_push_revision` | integer | Echoed as `geometry_push_revision`. |
| `site_origin` | `{lat, lon, alt_m}` | Origin of the site ENU frame. |
| `projection` | enum string | Section 3.4; must be the projection Python's `LocalCoordinateFrame` uses. |
| `node_placements` | object keyed by `node_id` | Node origins in site ENU metres. |
| `node_placements.*.position_m` | `[x, y, z]`, optional | Node origin (the point its body offsets are measured from). Omitted = orientation-only placement (below). |
| `node_placements.*.position_std_m` | number, optional | 1-sigma accuracy of `position_m`; only with `position_m`. |
| `node_placements.*.placement_source` | `operator_override` \| `stationary_estimate` | Becomes the sensors' `position_source` (when `position_m` is set). |
| `node_placements.*.body_to_world_quaternion_wxyz` | `[w, x, y, z]`, optional | Body -> world rotation; key omitted = identity. |

**Orientation-only placement**: an entry with `body_to_world_quaternion_wxyz`
and no `position_m` means "position from the node's reported GPS, rotation from
this entry". Python sends one (`placement_source` `operator_override`) for a
node whose operator pinned its mount orientation but neither its position nor
a settled stationary estimate; packets carry no reliable orientation, so
without it the override never reached Rust. The sensors' `position_source` is
then `reported_gps` (or the node is untrusted, as with no entry at all).

Semantics: the body is a **full snapshot**. A node missing from
`node_placements` loses any earlier pushed placement and falls back to GPS.
Rust uses raw GPS only for nodes with no pushed placement. A solve already
parked or queued keeps the layout snapshot taken at its trigger; later solves
use the new geometry and a higher `layout_revision`.

Response:

```json
{"site_frame_id": "site-7", "applied_geometry_push_revision": 3, "node_placement_count": 2}
```

Status codes: `200` applied; `409` when `geometry_push_revision` is lower
than the applied one (an equal revision is accepted as a full replace), body
`{"error": "stale_geometry_push_revision", "detail", "applied_geometry_push_revision"}`;
`422` for a malformed or invalid body (`invalid_site_geometry_body` /
`invalid_site_geometry`: empty `site_frame_id`, origin out of range,
non-finite `position_m`, negative `position_std_m`, zero or non-finite
quaternion, an entry with neither `position_m` nor a quaternion, or
`position_std_m` without `position_m`). The revision is held in sidecar memory only, so a Python process
that restarts while the sidecar keeps running must read `GET` first and push
above the applied revision.

`GET /api/v1/dsp/geometry` reports what a solve started now would use:

```json
{"coordinate_frame": {"frame_id": "site-7", "kind": "site_enu",
                      "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
                      "projection": "flat_wgs84_series"},
 "applied_geometry_push_revision": 3, "node_placement_count": 2}
```

Placement resolution per node: pushed placement with `position_m` (its
`position_std_m` and quaternion) > raw GPS projected into the frame in effect
(`position_std_m` null, the orientation-only placement's quaternion if one is
pushed, else the node's reported orientation) > untrusted
(`node_position_untrusted_no_override`) for a `fallback_static` source
(`gps_fallback` / `static` / `fallback_static`) or no fix at all. An untrusted
node solves node-local with that `node_local_routing_reason`, and leaves the
automatic cluster if it was in it. Every automatic-cluster route re-places every
member in the frame in effect, so a push (or a GPS move) changes the whole
layout at once and bumps `layout_revision`.

Rotation convention: R is the matrix of Python `orientation_rotation_matrix`
(`minimappr/spatial_audio/geometry.py`, Rz(yaw) Ry(pitch) Rx(roll), world =
R x body), expressed as a unit quaternion (w, x, y, z). A pure yaw of `psi`
is `[cos(psi/2), 0, 0, sin(psi/2)]`.

## 11. Sidecar configuration env (Rust-forwarded Settings keys)

Settings key `foo` is forwarded as env `MINIMAPPR_FOO`
(`minimappr/ingest_sidecar_runtime.py`). Rust parses them in
`minimappr-ingest-sidecar/src/localization_solve_config.rs`; its clap defaults
must equal the Python `Settings` defaults.

| Env | Type | Rust default |
|---|---|---|
| `MINIMAPPR_LOCALIZATION_MAX_SOLVE_SENSORS` | integer | `16` |
| `MINIMAPPR_LOCALIZATION_MIN_PAIR_BASELINE_M` | float | `0.02` |
| `MINIMAPPR_LOCALIZATION_CROSS_NODE_MAX_TIMELINE_RESIDUAL_SECONDS` | float | `0.010` |
| `MINIMAPPR_LOCALIZATION_CLOCK_CLASS_UNCERTAINTY_SECONDS` | JSON object `{time_quality: seconds}` | `{"gps_locked":1e-6,"gps_holdover":1e-4,"ntp_disciplined":0.002,"free_running":0.05}` |
| `MINIMAPPR_LOCALIZATION_CLOCK_UNKNOWN_UNCERTAINTY_SECONDS` | float | `0.005` |
| `MINIMAPPR_LOCALIZATION_TIMESTAMP_STEP_THRESHOLD_SECONDS` | float | `0.002` |
| `MINIMAPPR_LOCALIZATION_TIMESTAMP_STEP_RECOVERY_HOLDOFF_SECONDS` | float | `30.0` |
| `MINIMAPPR_LOCALIZATION_TRIGGER_NOISE_FLOOR_RELATIVE_ENABLED` | bool (`true`/`false`) | `true` |
| `MINIMAPPR_LOCALIZATION_TRIGGER_NOISE_FLOOR_RELATIVE_DB` | float | `8.0` |
| `MINIMAPPR_LOCALIZATION_SOLVER_FLOOR_RELATIVE_DB` | float | `3.0` |
| `MINIMAPPR_LOCALIZATION_SENSOR_REGISTRY_STALE_SECONDS` | float | `120.0` |
| `MINIMAPPR_SITE_GEOMETRY_JSON` | JSON (section 10) | unset |

The clock-class map **replaces** the default map (no merge); a class missing
from it uses `MINIMAPPR_LOCALIZATION_CLOCK_UNKNOWN_UNCERTAINTY_SECONDS`. Every
value must be finite and >= 0 or the sidecar refuses to start.
