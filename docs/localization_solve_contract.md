# Localization solve contract (Rust sidecar <-> Python)

Source of truth for the wire shapes: `minimappr-core/src/localization_sensor_entry.rs`
(serde types) plus `PairTdoaDiagnostic` / `LocalizationManifestPayload` in
`minimappr-ingest-sidecar/src/manifests.rs`. The example JSON below is pinned
by the Rust test `contract_doc_example_round_trips_field_for_field`; when a field
changes, update the type, that test and this document together.

Status (milestone 0): the types exist and parse, but the sidecar does not emit
them yet. Every new key is omitted from the JSON until the milestone that fills
it (`solve_provenance`: milestone 1 / R6; `is_cross_node`,
`timing_uncertainty_seconds`: milestones 1-2 / R6-R7). A manifest without
`solve_provenance` is a legacy manifest.

## 1. Where it appears

`localization.solve_provenance`, an object inside the `localization` payload of:

- `localization_result` manifests, and
- `classifier_render` manifests (the render carries a copy of the solve's
  `localization` payload, so the same `solve_id` appears on both).

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
| `coordinate_frame` | object | Frame of every `position_m` (section 4). |
| `layout_revision` | integer | Bumps whenever any sensor position in the layout changes, GPS updates included. |
| `geometry_push_revision` | integer, nullable | Revision of the geometry push in effect for this solve; `null` before any push. |
| `sensors` | array of sensor entries | Ordered solve sensors (section 5). Index `i` is buffer channel `i`. |
| `active_sensor_indices` | array of integers | Indices into `sensors` that took part in the solve. |
| `excluded_sensors` | array | `{sensor_index, sensor_id, reason}` for each sensor left out (section 6). |
| `excluded_pairs` | array | `{sensor_index_a, sensor_index_b, reason}` for each pair left out (section 6). |
| `node_timing` | array | Clock evidence per contributing node (section 7). |
| `window_center_time_ns` | integer | Packet-derived centre of the solve window, ns since the Unix epoch. This is the stored observation time; never substitute receipt time. |
| `window_seconds` | number | Solve window length. |
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
| `event_trigger_absolute_rms` | Event trigger: the absolute RMS gate (today's only live trigger). |
| `event_trigger_noise_floor_relative` | Event trigger: noise-floor-relative gate (milestone 2 / R4). |
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
| `position_std_m` | number, nullable | 1-sigma position accuracy; `null` when nothing states one. |
| `clock_source` | enum string | Section 3.6. |
| `clock_domain` | string | Timebase of the packet timestamps; currently always `utc`. |
| `clock_offset_std_s` | number, nullable | 1-sigma clock-offset uncertainty used for this sensor's pairs. |

Sensor world position = node position + R(body-to-world) x body offset, with
R the identity when the node has no orientation (rotation convention in section 9).

## 6. Exclusions

Every exclusion carries one `reason`:

| `reason` | Applies to | Meaning |
|---|---|---|
| `sample_rate_mismatch` | sensor, pair | Sensor rate differs from `solve_sample_rate_hz`. |
| `pair_baseline_below_minimum` | pair | Separation under `localization_min_pair_baseline_m`. |
| `coverage_below_threshold` | sensor, pair | Window coverage under the minimum ratio. |
| `timeline_residual_over_limit` | pair | A node's measured timeline residual exceeds `localization_cross_node_max_timeline_residual_seconds`; that node's cross-node pairs are dropped, its own channels still solve. |
| `node_position_untrusted_no_override` | sensor | `fallback_static` node with no pushed placement. |
| `node_array_geometry_unusable` | sensor | The node's own offsets fail validation for its channel count. |
| `max_solve_sensors_exceeded` | sensor | Beyond `localization_max_solve_sensors`. |

A weak clock class alone never excludes a pair; it raises that pair's
`timing_uncertainty_seconds`.

```json
{"sensor_index": 3, "sensor_id": "point-c:ch0", "reason": "sample_rate_mismatch"}
{"sensor_index_a": 0, "sensor_index_b": 2, "reason": "timeline_residual_over_limit"}
```

## 7. `node_timing[]`

| Field | JSON type | Meaning |
|---|---|---|
| `node_id` | string | |
| `time_quality` | enum string | Section 3.6. |
| `clock_offset_std_s` | number, nullable | Node timing uncertainty: max of class default, reported holdover error, measured residual. |
| `timeline_residual_ns` | integer, nullable | Measured packet-timeline residual; `null` until measured. |
| `last_timestamp_step_ns` | integer, nullable | Signed size of the most recent detected timestamp step; `null` if none. |
| `timestamp_step_count` | integer | Steps detected since the node was first heard. |

## 8. `localization.pair_tdoas[]` additions

`pair_tdoas` holds **admitted pairs only**. Existing fields are unchanged
(`ch_a`, `ch_b`, `delay_samples`, `lag_seconds`, `confidence`); `ch_a`/`ch_b`
index `solve_provenance.sensors`. `lag_seconds` is arrival at `ch_a` minus
arrival at `ch_b`. New fields, omitted while unset:

| Field | JSON type | Meaning |
|---|---|---|
| `is_cross_node` | bool | The two sensors belong to different nodes. |
| `timing_uncertainty_seconds` | number | 1-sigma timing uncertainty of this pair (clock evidence of both nodes). The Python re-solve weights the pair by it and inflates covariance to match. |

## 9. Example `solve_provenance`

```json
{
  "schema_version": 1,
  "solve_id": "solve-3f2a",
  "attempt_kind": "event_trigger_absolute_rms",
  "buffer_scope": "automatic_site_cluster",
  "cluster_id": "auto-gps",
  "reporting_node_id": "planar-a",
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
     "timeline_residual_ns": 1200, "last_timestamp_step_ns": null, "timestamp_step_count": 0},
    {"node_id": "point-b", "time_quality": "ntp_disciplined", "clock_offset_std_s": 0.002,
     "timeline_residual_ns": 15000000, "last_timestamp_step_ns": -21000000, "timestamp_step_count": 1}
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
                 "placement_source": "stationary_estimate"}
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
| `node_placements.*.position_m` | `[x, y, z]` | Node origin (the point its body offsets are measured from). |
| `node_placements.*.position_std_m` | number | 1-sigma accuracy of the placement. |
| `node_placements.*.placement_source` | `operator_override` \| `stationary_estimate` | Becomes the sensors' `position_source`. |
| `node_placements.*.body_to_world_quaternion_wxyz` | `[w, x, y, z]`, optional | Body -> world rotation; key omitted = identity. |

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
quaternion). The revision is held in sidecar memory only, so a Python process
that restarts while the sidecar keeps running must read `GET` first and push
above the applied revision.

`GET /api/v1/dsp/geometry` reports what a solve started now would use:

```json
{"coordinate_frame": {"frame_id": "site-7", "kind": "site_enu",
                      "origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
                      "projection": "flat_wgs84_series"},
 "applied_geometry_push_revision": 3, "node_placement_count": 2}
```

Placement resolution per node: pushed placement (its `position_std_m` and
quaternion) > raw GPS projected into the frame in effect (`position_std_m`
null, the node's reported orientation) > untrusted
(`node_position_untrusted_no_override`) for a `fallback_static` source
(`gps_fallback` / `static` / `fallback_static`) or no fix at all.

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
