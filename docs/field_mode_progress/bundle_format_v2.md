# Calibration bundle format — ground truth v2 + entity expectations

Status: implemented on `fm/replay-metrics` (W1-replay-metrics, server track S1).
Code: `minimappr/calibration/bundle.py` (container), `minimappr/calibration/ground_truth_events.py`
(event model + validation), `minimappr/sim/replay_scoring.py` (per-event scoring),
`minimappr/sim/replay_entity_metrics.py` (entity metrics). Writers of synthetic
bundles (e.g. `minimappr-sim render-bundle`) should target this document.

## 1. Container (unchanged from v1)

A bundle is one zip:

| Path | Required | Contents |
| --- | --- | --- |
| `manifest.json` | yes | session, site frame, environment, nodes (§2) |
| `ground_truth.json` | yes | ground-truth events (§3) |
| `detections.json` | no | `{"schema_version": 1, "detections": [...]}` — live outputs during capture, reference only |
| `expectations.json` | no | pass thresholds (§5); harness defaults when absent |
| `audio/{node_id}.wav` | yes, per node | PCM16 little-endian, multichannel interleaved; channel `i` is `channel_sensor_ids[i]` |
| `reference_audio/{event_id}.wav` | no | mono PCM16 at the source node, for node-sourced events |

Versioning: `manifest.json` keeps `"schema_version": 1` (its layout did not
change). `ground_truth.json` carries its own `schema_version`: **2** for this
format; loaders accept 1 and 2. Every v1 ground-truth file is a valid v2 file —
all additions are optional and default to v1 behaviour. Unknown keys are ignored.

## 2. manifest.json (what a writer must provide)

```json
{
  "schema_version": 1,
  "kind": "minimappr_calibration_bundle",
  "session_id": "sim-flock-seed42",
  "time_window": {"start_ns": 1800000000000000000, "end_ns": 1800000011000000000},
  "site": {"origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0}, "coordinate_mode": "flat"},
  "environment": {"temperature_c": 20.0, "humidity_fraction": 0.5},
  "nodes": [
    {
      "node_id": "node-a",
      "audio_file": "audio/node-a.wav",
      "sample_rate_hz": 16000,
      "audio_start_time_ns": 1800000000000000000,
      "channel_sensor_ids": ["node-a:ch0", "node-a:ch1", "node-a:ch2", "node-a:ch3"],
      "position_m": [0.0, 0.0, 0.0],
      "position_geo": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
      "sensor_offsets_m": [[-0.0144, -0.0083, -0.0051], [0.0144, -0.0083, -0.0051],
                           [0.0, 0.0167, -0.0051], [0.0, 0.0, 0.0153]],
      "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    }
  ]
}
```

- **Site frame.** `position_m` everywhere (nodes and ground truth) is the
  site-local ENU frame of `minimappr.core.geo.LocalCoordinateFrame` built from
  `site.origin` + `site.coordinate_mode` (`"flat"` or `"geodetic"`): x = east,
  y = north, z = up, metres from the origin. Geo coordinates are WGS-84 degrees,
  `alt_m` metres.
- **Time.** All `*_ns` are integer nanoseconds on one shared epoch (Unix ns in
  live bundles; synthetic bundles may use any epoch as long as audio and ground
  truth agree). Sample `n` of a node's WAV is at
  `audio_start_time_ns + n / sample_rate_hz` seconds.
- **Node type.** 4 `sensor_offsets_m` → tetrahedral node; otherwise a point node.
  Offsets are metres in the node body frame (rotated by `orientation`).
- A node whose `node_id` is some event's `source_node_id` is a ground-truth
  reference node and is excluded from replay by default.

## 3. ground_truth.json v2

```json
{"schema_version": 2, "events": [ <event>, ... ]}
```

Events may overlap freely in time (several birds calling at once).

### 3.1 Event fields

| Field | Type | Req. | Meaning |
| --- | --- | --- | --- |
| `event_id` | string | yes | unique within the bundle |
| `start_ns`, `end_ns` | int ns | yes | emission window at the source (`end_ns >= start_ns`) |
| `label` | string | yes unless `species` given | primary species label (classifier label space, e.g. BirdNET common name) |
| `label_category` | string | no (default `"unknown"`) | coarse category (`"bird"`, `"drone"`, …) |
| `geometry` | object | yes | §3.2 |
| `individual_id` | string | no (default = `event_id`) | events sharing it are bouts of **one** individual (one bird) |
| `species` | list | no | §3.3; default one `true` entry from `label`/`label_category` |
| `is_echo_source` | bool | no (default false) | this event is a virtual image source (reflection / sidelobe position), **not** a bird |
| `echo_of_event_id` | string | required if `is_echo_source` | the real event this echo comes from; must exist in the bundle |
| `reflector` | object | no | free-form reflector metadata for echo events, recommended keys below |
| `source` | string | no | `"manual"`, `"node"`, `"synthetic"` (informational) |
| `source_node_id` | string | no | node placed at the source (reference node) |
| `reference_audio` | string | no | `reference_audio/{event_id}.wav` |
| `notes` | string | no | free text |

Recommended `reflector` keys: `kind` (`"trunk" | "ground" | "wall" | "sidelobe"`),
`position_m` `[x, y, z]` (reflecting point, site frame), `extra_delay_ms`
(path delay vs. the direct arrival at the array centroid), `attenuation_db`
(level below direct, positive number). Scoring does not read these.

Echo events use the **apparent** source position (image source) as their
geometry and usually copy the real event's `label`, `species` and window. They
are excluded from all per-event metrics and from recall denominators; they only
exist so that a localized detection there can be recognized as an echo promotion.

### 3.2 Geometry

Static point (v1 shape, `position_geo`; v2 also accepts `position_m`; if both are
given `position_m` wins):

```json
{"type": "static", "position_geo": {"lat": 45.0001, "lon": -93.0001, "alt_m": 252.0}}
{"type": "static", "position_m": [6.0, 12.0, 2.0]}
```

Moving source (new): time-stamped waypoints, each with `t_ns` and either
`position_m` or `position_geo`. Waypoints must be non-empty and sorted by
`t_ns` (non-decreasing). The truth position at time `t` is the linear
interpolation (in the site ENU frame) between the bracketing waypoints, clamped
to the first/last waypoint outside the span.

```json
{"type": "trajectory", "waypoints": [
  {"t_ns": 1800000001000000000, "position_m": [0.0, 10.0, 8.0]},
  {"t_ns": 1800000003000000000, "position_m": [12.0, 14.0, 9.0]}
]}
```

Scoring evaluates a trajectory at each **detection's** `timestamp_ns`.

### 3.3 Species list

```json
"species": [
  {"label": "American Robin", "label_category": "bird", "status": "true"},
  {"label": "Wood Thrush",    "label_category": "bird", "status": "candidate"}
]
```

- `label` required; `label_category` defaults to the event's `label_category`
  (then `"unknown"`); `status` ∈
  `{"true", "candidate"}`, default `"true"`.
- `true`: a species actually present (several `true` entries = a genuinely
  mixed source, e.g. two speakers at one point). `candidate`: an acceptable
  answer when ground truth is uncertain (operator unsure, look-alike call).
- A detection label is **correct** for the event if it equals any entry's
  `label` (match mode `exact`) or any entry's `label_category` (match mode
  `category`, the default). `true` vs `candidate` is informational for now.
- If `label` is omitted, it is taken from the first `true` entry. If both are
  given, `label` must appear among the entries' labels.

### 3.4 Full example (two birds overlapping, two bouts of one, one echo, one walker)

```json
{
  "schema_version": 2,
  "events": [
    {"event_id": "robin-b1", "individual_id": "robin-1", "label": "American Robin",
     "label_category": "bird", "start_ns": 1800000001000000000, "end_ns": 1800000001500000000,
     "geometry": {"type": "static", "position_m": [6.0, 12.0, 8.0]}},
    {"event_id": "robin-b1-echo", "individual_id": "robin-1-echo", "label": "American Robin",
     "label_category": "bird", "start_ns": 1800000001000000000, "end_ns": 1800000001500000000,
     "is_echo_source": true, "echo_of_event_id": "robin-b1",
     "reflector": {"kind": "trunk", "position_m": [-1.0, 18.0, 2.0],
                   "extra_delay_ms": 12.5, "attenuation_db": 8.0},
     "geometry": {"type": "static", "position_m": [-8.0, 24.0, 8.0]}},
    {"event_id": "tanager-b1", "individual_id": "tanager-1", "label": "Scarlet Tanager",
     "label_category": "bird", "start_ns": 1800000001300000000, "end_ns": 1800000001800000000,
     "species": [{"label": "Scarlet Tanager", "label_category": "bird", "status": "true"},
                 {"label": "American Robin", "label_category": "bird", "status": "candidate"}],
     "geometry": {"type": "static", "position_m": [18.0, 9.0, 12.0]}},
    {"event_id": "robin-b2", "individual_id": "robin-1", "label": "American Robin",
     "label_category": "bird", "start_ns": 1800000007000000000, "end_ns": 1800000007500000000,
     "geometry": {"type": "static", "position_m": [6.0, 12.0, 8.0]}},
    {"event_id": "jay-fly", "individual_id": "jay-1", "label": "Blue Jay", "label_category": "bird",
     "start_ns": 1800000004000000000, "end_ns": 1800000006000000000,
     "geometry": {"type": "trajectory", "waypoints": [
       {"t_ns": 1800000004000000000, "position_m": [30.0, 0.0, 10.0]},
       {"t_ns": 1800000006000000000, "position_m": [30.0, 20.0, 10.0]}]}}
  ]
}
```

### 3.5 Validation (load_bundle raises `ValueError`)

Unsupported `ground_truth.schema_version`; missing or non-integer
`event_id`/`start_ns`/`end_ns`; `end_ns < start_ns`; duplicate `event_id`;
neither `label` nor `species`; empty `species` list, entry without `label`,
unknown `status`, or a `label` not among the `species` labels; geometry type
other than `static`/`trajectory`; point without `position_geo`/`position_m`;
`position_m` not three finite numbers; `position_geo` (when used) without
numeric `lat`/`lon`; trajectory with no waypoints, a waypoint without integer
`t_ns`, or unsorted waypoints; `is_echo_source` without an existing
`echo_of_event_id`, or one that names an echo event (including itself).

## 4. Detections consumed by scoring

Scoring reads replay detections as dicts (`Storage.list_detections` rows): `timestamp_ns`,
`label`, `label_category`, `label_confidence`, `position_m` (site ENU or null),
`reporting_modality` (`"localized"` | `"omni"`), `track_id`, `source_node_id`, `id`,
`feature_summary.resolved_algorithm | localization_method`.

## 5. expectations.json

```json
{
  "schema_version": 2,
  "runtime": {"classifier_backend": "yamnet", "settings_overrides": {}, "replay_reference_nodes": false},
  "classification": {"min_label_accuracy": 0.5, "match": "category", "per_label": {}},
  "localization": {"min_event_recall": 0.5, "max_median_position_error_m": 20.0,
                   "max_p90_position_error_m": 40.0, "far_field_distance_m": 120.0,
                   "far_field": {"max_median_bearing_error_deg": 10.0}, "per_algorithm": {}},
  "entities": {
    "association_time_slop_s": 2.0,
    "association_radius_m": 15.0,
    "far_field_bearing_gate_deg": 15.0,
    "min_localized_and_classified_recall": null,
    "max_label_swap_rate": null,
    "max_duplicate_entities": null,
    "max_echo_promotions": null,
    "max_entities_spanning_multiple_individuals": null
  }
}
```

Sections merge key-by-key over the defaults (`minimappr.sim.replay_scoring.DEFAULT_EXPECTATIONS`).
`classification`/`localization` keep their v1 meaning (per-event, any detection within
±2 s of a real event's window matches it). Entity gates set to `null` are computed and
reported but not enforced — set them per bundle family once a baseline exists.

### 5.1 Association rule (detection → ground-truth event)

Each detection `d` gets at most one event:

1. **Temporal gate.** Event `E` is a candidate when
   `E.start_ns − slop ≤ d.timestamp_ns ≤ E.end_ns + slop`, `slop = association_time_slop_s`.
2. **Localized detections** (`position_m` present and `reporting_modality != "omni"`)
   also need a **spatial gate** against `truth = E.position_at(d.timestamp_ns)`:
   - near-field (`‖truth − array_centroid‖ ≤ far_field_distance_m`):
     `‖d.position_m − truth‖ ≤ association_radius_m`;
   - far-field: bearing error seen from the array centroid ≤ `far_field_bearing_gate_deg`.

   The candidate with the smallest gate ratio (error ÷ gate) wins. Real events are
   tried first; only if none passes are echo events tried (an *echo association*).
   Otherwise `d` is unassociated.
3. **Omni / unpositioned detections** associate by time only: to the
   label-matching candidate with the nearest window midpoint; with no label-matching
   candidate, only if exactly one candidate exists. Omni detections can therefore
   never be counted as swaps (no spatial evidence).

`array_centroid` = mean `position_m` of non-reference nodes (all nodes if every node is a reference).

### 5.2 Entities

What the UI would show as one thing. Entity key per detection:
`("track", track_id)` if it has a track; else `("omni", source_node_id, label)` for
omni/unpositioned detections (one "heard nearby" row per node and species); else
`("detection", id)`. **Entity label** = label of the entity's detection with the
highest `label_confidence` (ties → latest `timestamp_ns`), over the whole replay.

### 5.3 Metrics

| Metric | Gate key | Definition |
| --- | --- | --- |
| `localized_and_classified_recall` | `min_localized_and_classified_recall` | Over real (non-echo) events `E`: fraction for which some detection associated with `E` belongs to an entity that (a) has ≥ 1 detection spatially gated to *any* event of `E`'s individual and (b) whose entity label is correct for `E`. So a weak-ID bout and a weak-fix bout of one bird both count when they share a track. `null` if no real events. |
| `label_swap_rate` | `max_label_swap_rate` | Over spatially gated detections with a label: a detection of individual `I` is *swap-eligible* if another individual whose accepted species set is not a subset of `I`'s is active (temporal gate) at its timestamp. It is a *swap* if its label is wrong for `I` and correct for one of those concurrent individuals. Rate = swaps ÷ eligible; `null` if none eligible. |
| `duplicate_entities` | `max_duplicate_entities` | Σ over individuals of `max(0, distinct entity keys among detections call-associated with that individual − 1)`. A track + an omni row for the same bird = 1; a bird split into two tracks = 1. |
| `echo_promotions` | `max_echo_promotions` | Number of entities with ≥ 1 echo-associated detection and no call-associated detection (a ghost surfaced as its own bird). |
| `entities_spanning_multiple_individuals` | `max_entities_spanning_multiple_individuals` | Number of entities whose call-associated detections cover ≥ 2 individuals (one track absorbing two birds). The dual of `duplicate_entities`: without it, one site-wide track can satisfy recall with zero duplicates. |

`association_radius_m` and `far_field_bearing_gate_deg` must be > 0 and
`association_time_slop_s` ≥ 0 (`ValueError` otherwise).

Label comparisons follow `classification.match`. In the default `category` mode two
birds share the category `"bird"`, so bird-vs-bird swaps are never eligible
(`label_swap_rate` stays `null`); use `"match": "exact"` for species-level swap gating.

Report-only diagnostics on `EntityMetrics`: `call_event_count`, `echo_event_count`,
`individual_count`, `localized_and_classified_event_count`, `label_swap_count`,
`swap_eligible_detection_count`, `entity_count_by_individual`,
`unassociated_localized_entities` (localized entities matching no real or echo event —
false positives, or echoes in bundles without echo metadata).

Choice of radius matters: the gate is what turns "a fix" into "a fix of this bird".
Keep `association_radius_m` well below the inter-bird spacing in the scenario and
below the distance from the birds to the nearest node (a fix collapsed onto a node
must not pass) — see `server_baseline.md`.
