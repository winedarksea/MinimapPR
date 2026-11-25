//! How much audio an event-triggered solve reads. One array's TDOAs are
//! sub-millisecond, so the short node-local window (`DspWorkerConfig.window_seconds`)
//! holds the whole event at every mic. Across nodes the same call reaches the
//! farthest sensor up to `baseline / c` before or after the trigger node hears
//! it; the trigger usually fires on the nearest node, so the far copy is
//! mostly *later*. The node-local window, centred on the trigger frame,
//! already keeps at least half of a trigger-length event at a sensor lagging
//! by up to half the window (~5 m of baseline at 32 ms). Past that, a 100 m
//! cluster would hold the call at one node and silence at the other, so the
//! span grows by the excess lag on each side, still centred on the trigger,
//! keeping that same "at least half the event" guarantee at every sensor.
//! The coverage wait holds the solve until every member (the trigger node
//! too) has delivered the later half.
//!
//! No clock margin is added: GCC-PHAT only searches lags up to each pair's
//! `baseline / c` (`gcc_phat::pair_max_tau_s`), so audio beyond that could
//! not bring a peak into range.

use super::*;
use minimappr_core::localization_sensor_entry::SolveSensorEntry;

/// The far sensor may hear the event up to the max lag before or after the trigger.
const LAG_EXTENT_SIDES: f64 = 2.0;
/// A sensor lagging by this fraction of the node-local window still holds at
/// least half of a trigger-length event inside it.
const NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION: f64 = 0.5;

/// Seconds of audio the event solve on `layout` reads.
///
/// Node-local (every solve-eligible sensor on one node) and short-baseline
/// layouts keep today's window. So does a cross-node layout with the coverage
/// wait off: the later half only exists once members deliver it, and only
/// the wait collects it. The lag is capped at the wait's total cap (audio
/// arrives at most in real time, so the wait cannot gather more) and the
/// whole span at the sensor buffer length.
pub(crate) fn event_solve_window_seconds(
    layout: &SolveSensorLayout,
    config: &DspWorkerConfig,
    sound_speed_mps: f32,
) -> f64 {
    let node_local_window_seconds = config.window_seconds;
    if !config.localization_coverage_wait_all_streaming_nodes_enabled {
        return node_local_window_seconds;
    }
    let Some(max_cross_node_distance_m) = max_cross_node_sensor_distance_m(layout) else {
        return node_local_window_seconds;
    };
    let max_lag_seconds = (max_cross_node_distance_m / f64::from(sound_speed_mps))
        .min(config.localization_coverage_total_cap_seconds);
    let lag_beyond_node_local_window_seconds = (max_lag_seconds
        - NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION * node_local_window_seconds)
        .max(0.0);
    (node_local_window_seconds + LAG_EXTENT_SIDES * lag_beyond_node_local_window_seconds)
        .min(config.max_buffer_seconds)
        .max(node_local_window_seconds)
}

/// Largest distance between solve-eligible sensors on different nodes;
/// `None` when every eligible sensor is on one node.
fn max_cross_node_sensor_distance_m(layout: &SolveSensorLayout) -> Option<f64> {
    let eligible: Vec<&SolveSensorEntry> = layout
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
        .collect();
    let mut max_distance_m: Option<f64> = None;
    for (position, sensor_a) in eligible.iter().enumerate() {
        for sensor_b in &eligible[position + 1..] {
            if sensor_a.node_id == sensor_b.node_id {
                continue;
            }
            let distance_m = (0..3)
                .map(|axis| (sensor_a.position_m[axis] - sensor_b.position_m[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            max_distance_m = Some(max_distance_m.map_or(distance_m, |max| max.max(distance_m)));
        }
    }
    max_distance_m
}

#[cfg(test)]
#[path = "localization_event_span_tests.rs"]
mod localization_event_span_tests;
