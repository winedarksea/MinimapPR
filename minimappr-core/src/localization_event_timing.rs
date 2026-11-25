//! Event span, coverage-wait cap and same-event window derived from the
//! site's node geometry (Python mirror: `minimappr/core/localization_event_timing.py`;
//! both are pinned by the same test vectors).
//!
//! Why geometry: the same sound reaches every node within `baseline / c` of
//! the first, whatever the source range. So the largest cross-node baseline,
//! not a fixed cap, bounds how much audio an event solve must read, how long
//! the coverage wait must hold for the far node, and how long after one
//! node's onset another node's onset can still be the same event. A compact
//! site keeps a short span and wait; a kilometre point network gets a
//! multi-second span without anyone setting a cap.
//!
//! - lag = max cross-node distance / c, bounded by `max_cross_node_lag_seconds`
//!   (memory/latency safety) and by what the sensor buffer can hold. No clock
//!   margin: GCC-PHAT searches each pair only to `baseline / c`
//!   (`gcc_phat::pair_max_tau_s`), so audio past that lag cannot move a peak;
//!   a span margin would cost latency for nothing;
//! - span = node-local window grown by the lag beyond half that window on
//!   each side (centred on the trigger, so a trigger-length event keeps at
//!   least half of itself inside the span at every sensor);
//! - coverage cap = lag + `coverage_delivery_margin_seconds` (audio arrives
//!   at most in real time, plus the delivery skew between nodes);
//! - same-event window (trigger cooldown) = cooldown + the two trigger nodes'
//!   distance / c + `clock_margin_sigma` x their combined clock uncertainty:
//!   onsets are stamped by each node's own clock, so here a clock error does
//!   move the far node's onset;
//! - an explicit `localization_coverage_total_cap_seconds` > 0 overrides the
//!   cap and also bounds the lag (the pre-derivation behaviour).

/// The far sensor may hear the event up to the lag before or after the trigger.
const LAG_EXTENT_SIDES: f64 = 2.0;
/// A sensor lagging by this fraction of the node-local window still holds at
/// least half of a trigger-length event inside it.
const NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventTimingConfig {
    pub node_local_window_seconds: f64,
    pub coverage_wait_enabled: bool,
    /// `<= 0`: derive the cap from geometry; `> 0`: explicit override.
    pub explicit_coverage_total_cap_seconds: f64,
    pub coverage_delivery_margin_seconds: f64,
    /// Same-event window only (see module docs).
    pub clock_margin_sigma: f64,
    /// Hard safety bound on the derived lag (and so on span and wait).
    pub max_cross_node_lag_seconds: f64,
    pub max_buffer_seconds: f64,
}

/// What the derivation needs from a layout's solve-eligible sensors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossNodeGeometry {
    /// Largest distance between sensors on different nodes.
    pub max_cross_node_distance_m: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerivedEventTiming {
    /// 0 for a node-local layout.
    pub cross_node_lag_seconds: f64,
    pub event_span_seconds: f64,
    pub coverage_total_cap_seconds: f64,
    /// The geometric lag exceeded a bound (safety, buffer or explicit cap):
    /// the far end of the span is cut and the far node may miss the event.
    pub lag_bounded: bool,
}

impl EventTimingConfig {
    fn explicit_cap_seconds(&self) -> Option<f64> {
        (self.explicit_coverage_total_cap_seconds.is_finite()
            && self.explicit_coverage_total_cap_seconds > 0.0)
            .then_some(self.explicit_coverage_total_cap_seconds)
    }

    fn lag_bound_seconds(&self) -> f64 {
        // The buffer must hold the whole span plus the delivery skew while
        // the far node catches up (`required_sensor_buffer_seconds`).
        let buffer_bound_seconds = ((self.max_buffer_seconds
            - self.coverage_delivery_margin_seconds.max(0.0)
            - self.node_local_window_seconds)
            / LAG_EXTENT_SIDES)
            .max(0.0);
        let mut bound_seconds = self
            .max_cross_node_lag_seconds
            .max(0.0)
            .min(buffer_bound_seconds);
        if let Some(explicit_cap_seconds) = self.explicit_cap_seconds() {
            bound_seconds = bound_seconds.min(explicit_cap_seconds);
        }
        bound_seconds
    }

    fn coverage_cap_for_lag(&self, lag_seconds: f64) -> f64 {
        self.explicit_cap_seconds()
            .unwrap_or(lag_seconds + self.coverage_delivery_margin_seconds.max(0.0))
    }
}

/// Lag between two nodes `distance_m` apart, before any bound: the far copy of
/// one sound can be this much later (or earlier) than the near one.
pub fn pair_arrival_lag_seconds(
    distance_m: f64,
    pair_clock_uncertainty_seconds: f64,
    sound_speed_mps: f64,
    clock_margin_sigma: f64,
) -> f64 {
    distance_m.max(0.0) / sound_speed_mps.max(1.0)
        + clock_margin_sigma.max(0.0) * pair_clock_uncertainty_seconds.max(0.0)
}

pub fn derive_event_timing(
    geometry: Option<CrossNodeGeometry>,
    sound_speed_mps: f64,
    config: &EventTimingConfig,
) -> DerivedEventTiming {
    let node_local_window_seconds = config.node_local_window_seconds;
    let Some(geometry) = geometry else {
        return DerivedEventTiming {
            cross_node_lag_seconds: 0.0,
            event_span_seconds: node_local_window_seconds,
            coverage_total_cap_seconds: config.coverage_cap_for_lag(0.0),
            lag_bounded: false,
        };
    };
    let geometric_lag_seconds =
        pair_arrival_lag_seconds(geometry.max_cross_node_distance_m, 0.0, sound_speed_mps, 0.0);
    let lag_bound_seconds = config.lag_bound_seconds();
    let cross_node_lag_seconds = geometric_lag_seconds.min(lag_bound_seconds);
    // Without the coverage wait the later half never arrives before the solve,
    // so the event keeps the node-local window.
    let event_span_seconds = if config.coverage_wait_enabled {
        let lag_beyond_node_local_window_seconds = (cross_node_lag_seconds
            - NODE_LOCAL_WINDOW_TOLERATED_LAG_FRACTION * node_local_window_seconds)
            .max(0.0);
        (node_local_window_seconds + LAG_EXTENT_SIDES * lag_beyond_node_local_window_seconds)
            .min(config.max_buffer_seconds)
            .max(node_local_window_seconds)
    } else {
        node_local_window_seconds
    };
    DerivedEventTiming {
        cross_node_lag_seconds,
        event_span_seconds,
        coverage_total_cap_seconds: config.coverage_cap_for_lag(cross_node_lag_seconds),
        lag_bounded: geometric_lag_seconds > lag_bound_seconds,
    }
}

/// Root-sum-square of the two largest node clock uncertainties: the worst
/// cross-node pair's combined clock uncertainty.
pub fn worst_two_node_clock_uncertainty_seconds(
    node_uncertainties_seconds: impl IntoIterator<Item = f64>,
) -> f64 {
    let mut largest_two = [0.0_f64; 2];
    for value in node_uncertainties_seconds {
        let value = if value.is_finite() { value.max(0.0) } else { 0.0 };
        if value > largest_two[0] {
            largest_two = [value, largest_two[0]];
        } else if value > largest_two[1] {
            largest_two[1] = value;
        }
    }
    largest_two[0].hypot(largest_two[1])
}

/// Sensor buffer length that holds the largest span the safety bound allows,
/// plus the delivery skew while the far node catches up. Startup raises a
/// shorter configured buffer to this (bounded by the safety bound itself).
pub fn required_sensor_buffer_seconds(config: &EventTimingConfig) -> f64 {
    config.node_local_window_seconds
        + LAG_EXTENT_SIDES * config.max_cross_node_lag_seconds.max(0.0)
        + config.coverage_delivery_margin_seconds.max(0.0)
}

/// How far apart (seconds, either order) two triggers on nodes `distance_m`
/// apart may be and still be one event: the cooldown plus the time the sound
/// takes between them. Same node: just the cooldown. Two onsets closer than
/// this cannot be told apart by timing, and the first solve's span already
/// reads the second node's copy. A cooldown of 0 switches the cooldown off
/// entirely (every trigger opens a solve), as before the pair lag existed.
pub fn same_event_trigger_window_seconds(
    trigger_cooldown_seconds: f64,
    node_distance_m: f64,
    pair_clock_uncertainty_seconds: f64,
    sound_speed_mps: f64,
    config: &EventTimingConfig,
) -> f64 {
    if trigger_cooldown_seconds.is_nan() || trigger_cooldown_seconds <= 0.0 {
        return 0.0;
    }
    let pair_lag_seconds = pair_arrival_lag_seconds(
        node_distance_m,
        pair_clock_uncertainty_seconds,
        sound_speed_mps,
        config.clock_margin_sigma,
    )
    .min(config.lag_bound_seconds());
    trigger_cooldown_seconds + pair_lag_seconds
}

#[cfg(test)]
#[path = "localization_event_timing_tests.rs"]
mod localization_event_timing_tests;
