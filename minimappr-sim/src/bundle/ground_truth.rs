//! `ground_truth.json` writer — deliberately the ONLY module that knows the
//! ground-truth schema, so adapting to the extended bundle format (W1
//! replay-metrics: overlapping events, `individual_id`, echo metadata,
//! per-event species lists) is a one-file change.
//!
//! Targets ground-truth schema v2 (`docs/field_mode_progress/bundle_format_v2.md`):
//! `species` entries, `individual_id`, and `trajectory` geometry for moving
//! sources (waypoints on the shared ns epoch). Extra simulator keys
//! (`position_m`, `trajectory_m`, `chirps`, …) are ignored by the loader.

use serde_json::{json, Value};

use super::geo::FlatLocalFrame;
use crate::physics::constants::NANOSECONDS_PER_SECOND;
use crate::scenario::Scenario;
use crate::world::{EmissionRecord, PlannedDevice, PlannedEmission, WorldPlan};

/// Ground-truth file schema (independent of the manifest's schema version).
const GROUND_TRUTH_SCHEMA_VERSION: u32 = 2;

/// Bird calls are scored; their sound source is "manual" (no reference node).
const GROUND_TRUTH_SOURCE_KIND: &str = "manual";

fn true_time_ns(scenario: &Scenario, time_s: f64) -> i64 {
    scenario.site.start_time_ns + (time_s * NANOSECONDS_PER_SECOND).round() as i64
}

fn geo_json(frame: &FlatLocalFrame, position_m: [f64; 3]) -> Value {
    let geo = frame.local_to_geo(position_m);
    json!({"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m})
}

fn emission_window_s(plan: &WorldPlan, emission: &PlannedEmission) -> (f64, f64) {
    (
        emission.start_time_s,
        emission.start_time_s + emission.duration_s(plan.sample_rate_hz),
    )
}

fn bird_event(
    plan: &WorldPlan,
    scenario: &Scenario,
    frame: &FlatLocalFrame,
    emission: &PlannedEmission,
) -> Option<Value> {
    let EmissionRecord::BirdCall(record) = &emission.record else {
        return None;
    };
    let (start_s, end_s) = emission_window_s(plan, emission);
    let position = emission.trajectory.midpoint_position(start_s, end_s);
    let moving = !emission.trajectory.is_stationary();
    let geometry = if moving {
        let waypoints: Vec<Value> = emission
            .trajectory
            .to_rows()
            .iter()
            .map(|[time_s, x, y, z]| {
                json!({"t_ns": true_time_ns(scenario, *time_s), "position_m": [x, y, z]})
            })
            .collect();
        json!({"type": "trajectory", "waypoints": waypoints})
    } else {
        json!({"type": "static", "position_m": position, "position_geo": geo_json(frame, position)})
    };
    let species: Vec<Value> = record
        .species
        .iter()
        .map(|label| json!({"label": label, "label_category": record.label_category, "status": "true"}))
        .collect();
    Some(json!({
        // --- ground-truth schema v2 keys ---
        "event_id": record.event_id,
        "label": record.label,
        "label_category": record.label_category,
        "source": GROUND_TRUTH_SOURCE_KIND,
        "geometry": geometry,
        "start_ns": true_time_ns(scenario, start_s),
        "end_ns": true_time_ns(scenario, end_s),
        "notes": format!("minimappr-sim {} ({})", record.generator_kind, record.origin),
        "individual_id": record.individual_id,
        "species": species,
        "is_echo_source": false,
        // --- simulator extras (ignored by the loader) ---
        "position_m": position,
        "trajectory_m": if moving { json!(emission.trajectory.to_rows()) } else { Value::Null },
        "level_db_spl_at_1m": record.level_db_spl_at_1m,
        "generator": record.generator_kind,
        "origin": record.origin,
    }))
}

fn device_truth(device: &PlannedDevice) -> Value {
    json!({
        "device_id": device.device_id,
        "clock_offset_s": device.clock.offset_s,
        "clock_drift_ppm": device.clock.drift_ppm,
        "true_trajectory_m": device.trajectory.to_rows(),
        "reported_position_m": device.reported_position_m,
        "gps_error_track_m": device.gps_error_track,
        "mic_offsets_m": device.mic_offsets_m,
        "speaker_to_mic_distance_m": device.speaker_to_mic_distance_m,
        "speakers": device.speakers.iter().map(|s| json!({"name": s.name, "offset_m": s.offset_m, "relative_gain": s.relative_gain})).collect::<Vec<_>>(),
        "speaker_muted": device.speaker_muted,
        "body_conduction_relative_gain": device.body_conduction_relative_gain,
        "dropouts": device.dropouts.iter().map(|d| json!({
            "start_sample": d.start_sample,
            "length_samples": d.length_samples,
            "reported": d.reported,
            "cause": d.cause.name(),
        })).collect::<Vec<_>>(),
        "gated_filters": device.gated_filters.iter().map(|g| json!({
            "kind": g.kind, "start_s": g.start_s, "end_s": g.end_s,
        })).collect::<Vec<_>>(),
    })
}

fn non_bird_emissions(plan: &WorldPlan, scenario: &Scenario) -> (Vec<Value>, Vec<Value>) {
    let mut chirps = Vec::new();
    let mut distractors = Vec::new();
    for emission in &plan.emissions {
        let (start_s, end_s) = emission_window_s(plan, emission);
        let position = emission.trajectory.midpoint_position(start_s, end_s);
        match &emission.record {
            EmissionRecord::Chirp(chirp) => chirps.push(json!({
                "emission_id": emission.emission_id,
                "device_id": chirp.device_id,
                "round_index": chirp.round_index,
                "speaker": chirp.speaker,
                "muted": chirp.muted,
                "start_ns": true_time_ns(scenario, start_s),
                "end_ns": true_time_ns(scenario, end_s),
                "band_hz": [chirp.waveform.band_low_hz, chirp.waveform.band_high_hz],
                "waveform": chirp.waveform.waveform,
                "level_db_spl_at_1m": chirp.level_db_spl_at_1m,
                "position_m": position,
            })),
            EmissionRecord::Distractor(distractor) => distractors.push(json!({
                "emission_id": emission.emission_id,
                "kind": distractor.kind,
                "near_device_id": distractor.near_device_id,
                "start_ns": true_time_ns(scenario, start_s),
                "end_ns": true_time_ns(scenario, end_s),
                "position_m": position,
                "level_db_spl_at_1m": distractor.level_db_spl_at_1m,
            })),
            EmissionRecord::BirdCall(_) => {}
        }
    }
    (chirps, distractors)
}

pub fn build_ground_truth(plan: &WorldPlan, scenario: &Scenario, frame: &FlatLocalFrame) -> Value {
    let events: Vec<Value> = plan
        .emissions
        .iter()
        .filter_map(|e| bird_event(plan, scenario, frame, e))
        .collect();
    let (chirps, distractors) = non_bird_emissions(plan, scenario);
    let mut reflectors: Vec<Value> = plan
        .reflectors
        .iter()
        .map(|r| json!({"reflector_id": r.reflector_id, "kind": "point", "position_m": r.position_m, "reflection_coefficient": r.reflection_coefficient}))
        .collect();
    if let Some((height_m, coefficient)) = plan.ground {
        reflectors.push(json!({"reflector_id": "ground", "kind": "ground_plane", "height_m": height_m, "reflection_coefficient": coefficient}));
    }
    json!({
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "events": events,
        "reflectors": reflectors,
        "chirps": chirps,
        "distractors": distractors,
        "simulator": {
            "seed": plan.seed,
            "scenario_name": plan.scenario_name,
            "speed_of_sound_mps": plan.speed_of_sound_mps,
            "sample_rate_hz": plan.sample_rate_hz,
            "devices": plan.devices.iter().map(device_truth).collect::<Vec<_>>(),
        },
    })
}
