//! How much audio an event-triggered solve reads, how long its coverage wait
//! may hold, and which later triggers are the same event, all from the
//! layout's node geometry (`minimappr_core::localization_event_timing`; the
//! Python ingest path mirrors it in `core/localization_event_timing.py`).
//!
//! One array's TDOAs are sub-millisecond, so the short node-local window
//! (`DspWorkerConfig.window_seconds`) holds the whole event at every mic.
//! Across nodes the same call reaches the farthest sensor up to
//! `baseline / c` (plus clock error) before or after the trigger node hears
//! it. Past half a node-local window of lag the span grows by the excess on
//! each side, centred on the trigger, so a trigger-length event keeps at
//! least half of itself inside the span at every sensor; the coverage wait
//! holds the solve until every member has delivered the later half, for at
//! most the lag plus the delivery margin. The wait is a deferral, never a
//! sleep, so a long wait costs no hot-path time.

use super::*;
use minimappr_core::localization_event_timing::{
    derive_event_timing, same_event_trigger_window_seconds,
    worst_two_node_clock_uncertainty_seconds, CrossNodeGeometry, DerivedEventTiming,
    EventTimingConfig,
};
use minimappr_core::localization_sensor_entry::SolveSensorEntry;

pub(crate) fn event_timing_config(config: &DspWorkerConfig) -> EventTimingConfig {
    let solve = &config.localization_solve;
    EventTimingConfig {
        node_local_window_seconds: config.window_seconds,
        coverage_wait_enabled: config.localization_coverage_wait_all_streaming_nodes_enabled,
        explicit_coverage_total_cap_seconds: config.localization_coverage_total_cap_seconds,
        coverage_delivery_margin_seconds: solve.localization_coverage_delivery_margin_seconds,
        clock_margin_sigma: solve.localization_event_span_clock_margin_sigma,
        max_cross_node_lag_seconds: solve.localization_event_span_max_cross_node_lag_seconds,
        max_buffer_seconds: config.max_buffer_seconds,
    }
}

/// Span, coverage cap and lag of an event solve on `layout`. Node-local
/// layouts (every solve-eligible sensor on one node) keep today's window.
pub(crate) fn derived_event_timing(
    layout: &SolveSensorLayout,
    config: &DspWorkerConfig,
    sound_speed_mps: f32,
) -> DerivedEventTiming {
    derive_event_timing(
        max_cross_node_sensor_distance_m(layout).map(|max_cross_node_distance_m| {
            CrossNodeGeometry {
                max_cross_node_distance_m,
            }
        }),
        f64::from(sound_speed_mps),
        &event_timing_config(config),
    )
}

/// Seconds of audio the event solve on `layout` reads (see `derived_event_timing`).
#[cfg(test)]
pub(crate) fn event_solve_window_seconds(
    layout: &SolveSensorLayout,
    config: &DspWorkerConfig,
    sound_speed_mps: f32,
) -> f64 {
    derived_event_timing(layout, config, sound_speed_mps).event_span_seconds
}

/// How far apart two triggers, on the last trigger's node and this one, may
/// be and still be one event: the cooldown plus the sound's travel time
/// between the two nodes (plus clock margin). Same node: the cooldown.
pub(crate) fn same_event_window_seconds(
    layout: &SolveSensorLayout,
    config: &DspWorkerConfig,
    sound_speed_mps: f32,
    last_trigger_node_id: &str,
    trigger_node_id: &str,
    node_clock_uncertainty_seconds: impl Fn(&str) -> f64,
) -> f64 {
    let (node_distance_m, pair_clock_uncertainty_seconds) =
        if last_trigger_node_id == trigger_node_id {
            (0.0, 0.0)
        } else {
            // A node no longer in the layout: assume the layout's largest
            // baseline, so a far node's onset is never taken for a new event.
            let distance_m = node_pair_distance_m(layout, last_trigger_node_id, trigger_node_id)
                .or_else(|| max_cross_node_sensor_distance_m(layout))
                .unwrap_or(0.0);
            (
                distance_m,
                worst_two_node_clock_uncertainty_seconds([
                    node_clock_uncertainty_seconds(last_trigger_node_id),
                    node_clock_uncertainty_seconds(trigger_node_id),
                ]),
            )
        };
    same_event_trigger_window_seconds(
        config.trigger_cooldown_seconds,
        node_distance_m,
        pair_clock_uncertainty_seconds,
        f64::from(sound_speed_mps),
        &event_timing_config(config),
    )
}

fn eligible_sensors(layout: &SolveSensorLayout) -> impl Iterator<Item = &SolveSensorEntry> {
    layout
        .sensors
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            !layout
                .excluded_sensors
                .iter()
                .any(|excluded| excluded.sensor_index == *index)
        })
        .map(|(_, sensor)| sensor)
}

fn sensor_distance_m(sensor_a: &SolveSensorEntry, sensor_b: &SolveSensorEntry) -> f64 {
    (0..3)
        .map(|axis| (sensor_a.position_m[axis] - sensor_b.position_m[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Largest distance between solve-eligible sensors on different nodes;
/// `None` when every eligible sensor is on one node.
fn max_cross_node_sensor_distance_m(layout: &SolveSensorLayout) -> Option<f64> {
    let eligible: Vec<&SolveSensorEntry> = eligible_sensors(layout).collect();
    let mut max_distance_m: Option<f64> = None;
    for (position, sensor_a) in eligible.iter().enumerate() {
        for sensor_b in &eligible[position + 1..] {
            if sensor_a.node_id == sensor_b.node_id {
                continue;
            }
            let distance_m = sensor_distance_m(sensor_a, sensor_b);
            max_distance_m = Some(max_distance_m.map_or(distance_m, |max| max.max(distance_m)));
        }
    }
    max_distance_m
}

/// Largest sensor-to-sensor distance between two nodes of the layout.
fn node_pair_distance_m(layout: &SolveSensorLayout, node_a: &str, node_b: &str) -> Option<f64> {
    let mut max_distance_m: Option<f64> = None;
    for sensor_a in layout.sensors.iter().filter(|sensor| sensor.node_id == node_a) {
        for sensor_b in layout.sensors.iter().filter(|sensor| sensor.node_id == node_b) {
            let distance_m = sensor_distance_m(sensor_a, sensor_b);
            max_distance_m = Some(max_distance_m.map_or(distance_m, |max| max.max(distance_m)));
        }
    }
    max_distance_m
}

#[cfg(test)]
#[path = "localization_event_span_tests.rs"]
mod localization_event_span_tests;
