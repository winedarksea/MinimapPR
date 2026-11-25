# Field capture protocol

How to record a calibration capture on site that can score localization
honestly. The goal is to give a real capture the same footing the synthetic
evaluations have: the replay harness (`minimappr/sim/replay*.py`), the E3
scoring tiers (`minimappr/sim/replay_tier_scoring.py`, heard / classified and
bearing-correct / classified and range-resolved, plus covariance coverage) and
the E5 range matrix (`tests/test_localization_range_matrix_realistic.py`).

A capture is scored against what you write down, so what you write down has to
be right. The result is only as honest as three things:

1. **Where each microphone was**, and how sure you are (node position, its
   accuracy and the mount orientation).
2. **What each node's clock said**, frame by frame (`time_quality`, timestamp steps).
3. **Where and when each true sound was made**, and how sure you are.

Finish every session by running the validator (section 9) before you pack up.
It catches most of what would make the capture useless, while you can still
fix it.

## 1. What a capture bundle contains

A calibration session (`POST /api/v1/recordings` with `"capture_kind": "calibration"`)
records every registered audio node into one session buffer. When it stops,
`minimappr/calibration/pipeline.py` writes, per node, a multichannel WAV, a
per-frame timing table and a manifest entry. Ground truth is entered separately
(`POST /api/v1/calibration/{session_id}/ground-truth`) and joined at export
(`GET /api/v1/calibration/{session_id}/bundle`). The format is
`docs/field_mode_progress/bundle_format_v2.md`; in short:

| Member | What the protocol needs from it |
|---|---|
| `manifest.json` `site` | origin and `coordinate_mode` (section 3) |
| `manifest.json` `nodes[]` | `node_type`, `position_m` / `position_geo`, `position_source`, `position_std_m`, `sensor_offsets_m`, `orientation` and/or `body_to_world_quaternion_wxyz`, `sample_rate_hz`, optional `cluster_id` (section 4) |
| `frame_timing/{node_id}.json` | one entry per firmware frame: packet timestamp, `time_quality`, sequence (section 5) |
| `audio/{node_id}.wav` | PCM16, channel *i* = `channel_sensor_ids[i]` |
| `ground_truth.json` | events with positions, times, labels; optional `geometry.position_std_m` (section 7) |

Each session is capped at `calibration_max_duration_s` (Settings, default
120 s). Record one session per range-ladder rung or source type, or raise the
cap for the day.

## 2. Before you go

- Nodes: firmware that stamps every frame with a GPS (PPS-disciplined) or NTP
  timestamp and reports `time_quality`. Bring the GPS antennas.
- A tape (30 m+) and a laser rangefinder; a compass with the local declination,
  or a known landmark bearing; a bubble level.
- A loudspeaker that can play a chirp or band-limited burst and recorded bird
  calls, on a stand of known height. Ideally a spare node to sit at the
  speaker as a reference node (section 7.3).
- For drones: a drone whose flight log records GPS at 1 Hz or faster (5-10 Hz
  is better) and its time base (GPS / UTC).
- A survey peg or a point with a known position to measure from, if any node
  is hand-measured.
- A laptop that can reach the server (the validator CLI or the validation endpoint).

## 3. Site frame: origin and projection

Every position in the bundle is site-local ENU metres (x east, y north, z up)
from `site.origin`, in the projection named by `site.coordinate_mode`:

| `coordinate_mode` | Contract projection | Use |
|---|---|---|
| `flat` | `flat_wgs84_series` | default; fine across the 1 km rung |
| `geodetic` | `geodetic_ecef_enu` | sites spanning several km, or large height relief |

Rules:

- **Fix the origin before the first session and do not change it for the
  day.** An origin rebind changes the site `frame_id`; solves taken under the
  old frame are refused by the handoff, and node positions written in one
  frame and ground truth in another put every fix tens of metres off without
  any error.
- Survey ground truth in the same frame. If you log truth as lat/lon, the
  bundle projects it through the same `LocalCoordinateFrame`; if you log
  `position_m`, measure it from the same origin.
- The validator flags a node whose `position_m` and `position_geo` disagree by
  more than 5 cm (`node_position_frame_mismatch`), and any node or truth point
  more than 20 km from the origin (`node_position_far_from_site_origin`,
  `ground_truth_position_outside_site`).

## 4. Node setup and survey

### 4.1 Positions and their accuracy

Every node needs a position, where it came from (`position_source`), and a
1-sigma accuracy (`position_std_m`). The capture pipeline records the geometry
the live solve used, with the same precedence as the sidecar
(`docs/localization_solve_contract.md` section 3.5):

| `position_source` | How you get it | `position_std_m` | Scoring consequence |
|---|---|---|---|
| `operator_override` | hand-measured (tape, laser, total station) or RTK, entered via `PATCH /api/v1/nodes/{id}` | the accuracy you **enter** (defaults to 0.1 m if you leave it out; don't) | trusted: used as given; where it mixes with GPS or chirp-surveyed nodes, the stated accuracy widens the error |
| `stationary_estimate` | the server's settled filtered estimate of a fixed node's GPS | the estimate's horizontal std | trusted; needs several minutes of GPS before the capture |
| `reported_gps` | raw packet GPS | none (packets carry no accuracy) | usable cross-node; float32 packet GPS alone is quantised to 0.4-0.6 m |
| `fallback_static` | firmware built-in position | none | **untrusted**: the node solves alone and forms no cross-node baseline |

Owner decision: **hand-measured positions are trusted.** A lone hand-measured
node, or a site where every node is hand-measured, uses the entered numbers as
given. Where hand-measured nodes are mixed with GPS or surveyed nodes, the
entered position is still used, and its stated accuracy widens the error. So
state an honest accuracy: E5 checks that hand-measured cells still contain the
truth inside the stated uncertainty, which only holds if the stated number is
real.

How to hand-measure:

1. Measure from a peg whose position you know (survey mark, RTK point, or a
   point you define as the origin).
2. Measure to the node's **body origin** (the point its `sensor_offsets_m` are
   relative to), not the enclosure corner. Record height above the ground at
   the origin, not above the local ground at the node, if the site slopes.
3. Enter it:
   ```json
   PATCH /api/v1/nodes/planar-a
   {"overrides": {"position_m": [12.40, -3.10, 1.52], "position_std_m": 0.05}}
   ```
   A `position_geo` override is projected into the site frame for you.
4. Typical honest accuracies: tape from a peg over 10-30 m, 0.05-0.3 m;
   laser + compass bearing, 0.2-0.5 m at 50 m; phone GPS, 3-5 m (prefer the
   node's own GPS estimate).

### 4.2 Orientation

An array node's bearing depends on how it is mounted, so every node with two or
more distinct microphone offsets (tetra, planar) needs an orientation. Record
it **even when it is identity**: a missing orientation and a level, east-facing
mount look the same otherwise (`node_orientation_missing` is an error).

Convention (the shared rule in `minimappr/core/sensor_world_position.py`,
identical on the Rust side): sensor world position = node position +
R(body→world) × body offset, with R = Rz(yaw) · Ry(pitch) · Rx(roll), angles in
degrees, in the site ENU frame. Measured signs:

- **yaw** rotates about +z (up), counter-clockwise seen from above: at yaw 90
  the body +x axis points north. It is *not* a compass bearing:
  `yaw_deg = 90 − (compass bearing of the body +x axis, true north)`. Correct
  the compass for declination.
- **pitch** rotates about body +y: positive pitch tips body +x **down**.
- **roll** rotates about body +x: positive roll tips body +y **up**.

Enter it as an override (`{"overrides": {"orientation": {"yaw_deg": 35.0,
"pitch_deg": 0.0, "roll_deg": 0.0}}}`). The bundle carries both the
yaw/pitch/roll dict (`orientation`) and the unit quaternion
`body_to_world_quaternion_wxyz` ([w, x, y, z], the sidecar push form, w ≥ 0);
if you write both by hand, they must agree (`node_orientation_conflict`).

Level the array with a bubble level; one degree of tilt on a planar array is a
degree of elevation error. A planar array cannot tell a source above its plane
from one mirrored below it; mount it so the sources of interest are on one
side, and include at least one source well off the plane.

### 4.3 Mixed node types and clusters

- Tetra (4 mics) and planar (5 mics) arrays localize alone; point nodes (1 mic)
  contribute cross-node pairs and otherwise classify omni-only.
- Nodes without a `cluster_id` share the site-wide automatic cluster. Give
  nodes a `cluster_id` only when they are meant to solve separately.
- Spread nodes for the far rungs: a 1 km fix needs baselines of tens of metres
  or more, and four or more nodes with some height relief. Close nodes give
  bearing, not range.

## 5. Clock quality

Precise TDOA rests on packet timestamps, not server receipt time. Each frame
carries the node's `time_quality`:

| `time_quality` | Default timing uncertainty (Settings `localization_clock_class_uncertainty_seconds`) | Use |
|---|---|---|
| `gps_locked` | 1 µs | all rungs; required for cross-node range at 1-10 m |
| `gps_holdover` | 100 µs (or the reported holdover error, if larger) | acceptable briefly |
| `ntp_disciplined` | 2 ms (≈0.7 m of path) | bearing and long-baseline only |
| `free_running` | 50 ms | effectively no cross-node timing |
| unreported | `localization_clock_unknown_uncertainty_seconds` (5 ms) | protocol error: replay cannot reproduce it |

Owner decision: **weak clocks are never excluded by class**; they lower the
pair's weight and widen the stated uncertainty. What *does* drop a node's
cross-node pairs is a measured timeline residual over the hard limit
(`localization_cross_node_max_timeline_residual_seconds`, 10 ms): the timeline
is then known to be wrong, not merely uncertain.

On site:

- Wait for GPS lock (and PPS) on every node before starting; give NTP nodes a
  few minutes to settle.
- Do not reboot, re-flash or move a node during a session. A reboot is a new
  boot session; a GPS re-lock can step the clock.
- **Timestamp steps.** A step is a frame-to-frame packet-time jump that
  disagrees with the sample count by more than
  `localization_timestamp_step_threshold_seconds` (2 ms). The sidecar raises
  the node's uncertainty to the step size until it has stayed clean for
  `localization_timestamp_step_recovery_holdoff_seconds` (30 s), and once
  `localization_timestamp_step_confirm_frames` (3) frames confirm the step it
  re-anchors the node's buffer timeline on it (no audio dropped). The
  validator finds steps in the per-frame table (`timestamp_step_detected`),
  including a step inside a single upload. A session with a step is still
  usable; note what caused it.
- **Drift is slewed live, measured raw here.** A buffer timeline runs at the
  nominal rate from its anchor, so crystal drift accumulates as a timeline
  residual; the sidecar slews the anchor once the windowed residual passes
  `localization_timeline_drift_slew_fraction` (0.5) of the 10 ms limit
  (`docs/localization_solve_contract.md` section 7). The validator reports
  each node's raw drift in ppm and its largest uncorrected residual
  (`node_clock_summary`), and warns when it passes the 10 ms limit
  (`timeline_residual_over_limit`): the point at which the sidecar would drop
  that node's cross-node pairs with the timeline discipline off.

## 6. Sample rates

Run every node of a cluster at **one sample rate**. Sample rates are guarded
now and resampled later: a cluster mixing 48 kHz and 44.1 kHz keeps each rate
group's own solves, but every cross-rate pair is excluded
(`sample_rate_mismatch`), not resampled. The validator reports this as
`cluster_sample_rate_mismatch` with the rate groups. Do not resample WAVs after
the fact to hide it: the packet timing table describes the rate the node sent.

## 7. Ground-truth events

### 7.1 Sources

- **Known-position loudspeaker** playing a chirp or band-limited burst
  (0.4-3.2 kHz, 0.3-1 s): the backbone of the range ladder. Mark the speaker
  cone's position and height.
- **Bird-like calls**: recorded calls played from the speaker. Label with the
  classifier's label space (BirdNET common name) and `label_category: "bird"`;
  list look-alike species as `candidate`. Bouts of one bird share an
  `individual_id`.
- **Continuous sources**: a drone flying a path with its GPS logged
  (`label_category: "drone"`), as a trajectory event.
- Natural sounds you did not make are not ground truth unless you know where
  and when they happened.

### 7.2 Range ladder

Walk the speaker out through the E5 rungs: **1 m, 10 m, 100 m and 1 km**, as
far as the site allows. An event's range is from its position at mid-event to
the nearest localizing node; it belongs to the rung within half a decade
(1 m covers 0.32-3.2 m, 10 m covers 3.2-32 m, and so on). At each rung:

- Record **at least three** events (`minimum_events_per_range_rung`), at
  different azimuths and at least one elevated source (a speaker on a pole or
  held up), so coverage statistics and the coplanar-mirror case are both
  exercised.
- Keep the level natural. Long-range events are faint by nature, and faint is
  the point: do not turn the speaker up until it clips, and do not move the
  speaker closer to "help". Note the level setting in `notes`.
- A rung you cannot reach is reported as empty (`range_ladder_rung_empty`,
  info); a rung with one or two events is a warning.

### 7.3 Times

Truth times are emission times at the source, on the same epoch as the packet
timestamps (UTC nanoseconds).

- Best: a **reference node** at the speaker. Enter the event with
  `source_node_id`; the bundle then slices the reference node's audio over the
  event (`reference_audio/{event_id}.wav`) as a timing oracle, and replay
  leaves that node out of localization.
- Otherwise: a GPS-timed playback schedule, or start/end times read from a
  GPS-synchronised clock. A phone's clock is good to about a second: widen the
  window by that margin and say so in `notes`.
- Every event must lie inside the capture window (`ground_truth_outside_capture`
  is an error; running past the edge is a warning).

### 7.4 Positions and their accuracy

- Static events: `position_geo` (lat, lon, alt) or `position_m` in the site
  frame, at the speaker cone's height.
- State the truth accuracy as `geometry.position_std_m`. Scoring does not read
  it yet, but the validator flags truth too coarse for its range
  (`ground_truth_accuracy_coarse_for_range`: std above 10 % of the range). A
  phone GPS fix (3-5 m) cannot grade a range-resolved fix at the 1 m or 10 m
  rung; tape the speaker position there.

### 7.5 Drone paths

- Log the drone's GPS at 1 Hz or faster; waypoints more than 2 s apart are
  flagged (`ground_truth_trajectory_sparse`), since linear interpolation then
  misplaces a 10 m/s drone by metres.
- The waypoints must span the event (`ground_truth_trajectory_incomplete`);
  outside the logged path truth is clamped to the end points.
- Convert the log's time base to UTC ns and its positions to `position_geo`
  waypoints: `{"type": "trajectory", "waypoints": [{"t_ns": ..., "position_geo": {...}}, ...], "position_std_m": 3.0}`.
- Fly a fly-by, a hover and a crossing path, as in the E6 scenes.

### 7.6 Entering ground truth

Static events go through `POST /api/v1/calibration/{session_id}/ground-truth`
(`label`, `label_category`, `lat`, `lon`, `alt_m` or `source_node_id`,
`start_ns`, `end_ns`, `notes`). Trajectories, species lists and
`geometry.position_std_m` are not stored by that API yet: add them to
`ground_truth.json` in the exported bundle and re-zip, then validate the edited
bundle with the CLI.

## 8. Session procedure

1. Nodes up, GPS locked, positions and orientations entered; check
   `GET /api/v1/nodes` shows the overrides.
2. Start a calibration session; record 10-20 s of quiet first (the noise floor
   the relative trigger and solver floor track).
3. Play the rung's events, logging each start/end and position as you go.
4. Stop the session and wait for processing to finish.
5. Enter ground truth, then run validation (section 9). Fix errors now; note
   warnings in the session notes.
6. Move to the next rung or source and repeat.

## 9. Validation before leaving the site

```sh
python -m minimappr.calibration.bundle_validation calibration_<session>.zip
python -m minimappr.calibration.bundle_validation calibration_<session>.zip --json
```

or `GET /api/v1/calibration/{session_id}/validation`, which exports the bundle
as it stands and returns the same report. Validation is advisory: extraction
and export never depend on it. The CLI exits 1 on errors (`--strict`: on
warnings too); `--minimum-capture-duration-seconds` and
`--minimum-events-per-range-rung` adjust the protocol thresholds (defaults
60 s and 3).

Findings have a stable code and fixed severity
(`minimappr/calibration/bundle_validation_findings.py`):

| Code | Severity | Fix |
|---|---|---|
| `bundle_unreadable` | error | not a calibration bundle zip, or no nodes |
| `site_origin_invalid`, `site_coordinate_mode_unknown`, `capture_window_invalid` | error | site origin, projection or time window missing or malformed |
| `node_type_missing` | error | node has no `NodeType`; replay cannot rebuild its uploads |
| `node_position_missing` | error | survey the node |
| `node_position_frame_mismatch` | error | `position_m` and `position_geo` were written in different frames |
| `node_position_source_missing`, `node_position_source_unknown` | error | record where the position came from (section 4.1) |
| `hand_measured_position_std_missing` | error | enter the measured accuracy with the override |
| `node_position_std_invalid` | error | accuracy must be a finite number ≥ 0 |
| `node_orientation_missing` | error | record the array's mount, even identity |
| `node_orientation_invalid`, `node_orientation_conflict` | error | fix the yaw/pitch/roll or quaternion |
| `node_sensor_offsets_mismatch` | error | offsets must match the channel list |
| `node_audio_missing`, `node_audio_channel_mismatch`, `node_sample_rate_inconsistent` | error | WAV, manifest and timing table disagree |
| `frame_timing_missing`, `frame_timing_unreadable`, `frame_timing_node_mismatch` | error | per-frame timing absent or broken (pre-E4 capture) |
| `frame_timing_not_monotonic`, `frame_timing_overlap`, `frame_timing_empty_frame`, `frame_timing_outside_audio` | error | corrupted timing table |
| `time_quality_unreported`, `time_quality_unknown_value` | error | firmware must report the clock class per frame |
| `ground_truth_invalid`, `ground_truth_missing` | error | enter valid ground truth |
| `ground_truth_outside_capture` | error | event is not in the capture window (wrong epoch or session) |
| `ground_truth_position_outside_site` | error | truth logged in another frame |
| `node_position_far_from_site_origin` | warning | check origin and projection |
| `node_position_untrusted` | warning | `fallback_static`: survey the node or it solves alone |
| `node_position_std_missing` | warning | GPS position without accuracy; coverage treats it as exact |
| `node_window_coverage_partial` | warning | node joined late or dropped out |
| `cluster_sample_rate_mismatch` | warning | cross-rate pairs excluded (section 6) |
| `time_quality_free_running` | warning | wait for lock; pairs carry 50 ms uncertainty |
| `timestamp_step_detected` | warning | clock stepped; note why |
| `timeline_residual_over_limit` | warning | the sidecar drops this node's cross-node pairs while over the limit |
| `node_gap_fraction_high` | warning | more than 2 % of the node's audio lost; check the network |
| `ground_truth_partially_outside_capture` | warning | event runs past the window |
| `ground_truth_source_node_unknown` | warning | reference node is not in the bundle |
| `ground_truth_trajectory_incomplete`, `ground_truth_trajectory_sparse` | warning | log the whole path, at 1 Hz or faster |
| `ground_truth_accuracy_coarse_for_range` | warning | truth too coarse to grade range at this rung |
| `range_ladder_rung_undercovered` | warning | fewer than the minimum events at a rung |
| `capture_too_short` | warning | shorter than the protocol minimum |
| `time_quality_changed`, `node_clock_summary`, `node_gap_statistics` | info | clock class changes, step count, residual, drift ppm, gaps, sequence jumps |
| `ground_truth_position_std_missing` | info | truth has no stated accuracy |
| `range_ladder_coverage`, `range_ladder_rung_empty`, `ground_truth_outside_range_ladder` | info | events per rung |
| `capture_summary` | info | duration, nodes, events, rates |

## 10. What not to do

- Do not resample, trim or re-time WAVs or timing tables after capture.
- Do not mix sample rates inside one cluster and expect cross-node fixes.
- Do not change the site origin or projection between sessions of one day.
- Do not leave a node on `fallback_static`, or an array without an orientation.
- Do not enter an optimistic `position_std_m`; a stated accuracy that is too
  small makes the honest-coverage check fail for the right reason.
- Do not reboot nodes or let GPS drop mid-session.
- Do not boost the speaker for the far rungs; faint long-range events are what
  the capture is for.
- Do not log truth with a phone clock without widening the window.

## 11. Checklist

- [ ] Site origin and `coordinate_mode` set and unchanged for the day
- [ ] Every node: GPS locked (or NTP settled), `time_quality` reported
- [ ] Every node: position with `position_source`; hand-measured nodes have an entered `position_std_m`
- [ ] Every array: orientation entered (yaw = 90 − compass bearing of body +x; levelled)
- [ ] One sample rate per cluster
- [ ] Quiet lead-in recorded
- [ ] ≥ 3 events at each reachable rung (1 m, 10 m, 100 m, 1 km), varied azimuth, one elevated
- [ ] Event times on UTC from GPS or a reference node; positions in the site frame with `position_std_m`
- [ ] Drone flights: GPS log at ≥ 1 Hz covering every flight event
- [ ] Bird calls labelled in the classifier label space, candidates listed
- [ ] Bundle exported and validated: no errors; warnings understood and noted

## 12. Known format gaps

- The ground-truth API stores static events only: trajectories, species lists
  and `geometry.position_std_m` have to be added to the exported bundle by hand.
- The bundle has no per-frame boot session or holdover predicted error (the
  sidecar sees both), so a reboot looks like a step and holdover is judged by
  class only.
- The sidecar replay (`minimappr/sim/replay_frames.py`, `replay_sidecar.py`)
  sends each node's manifest position as a trusted GPS fix and pushes no site
  geometry, so it applies neither `position_std_m` / `position_source` nor the
  orientation the way the live geometry push does. The in-process replay
  (`replay.py`) applies the orientation but not the accuracy.
- Scoring does not yet read truth accuracy (`geometry.position_std_m`).
- A node that registers itself stores the default orientation (all zeros), so
  a mount nobody measured still looks recorded and `node_orientation_missing`
  only fires for a bundle written without one. Until the node row records
  whether an orientation was entered, enter every array's orientation as an
  override, identity included.
