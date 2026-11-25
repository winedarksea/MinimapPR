//! Noise-floor-relative trigger (port of `minimappr/core/ingest_relative_trigger.py`,
//! server S5 verdict "PASS: port").
//!
//! A faint far bird 10 dB over a quiet site's floor can sit far below any
//! fixed RMS gate that also keeps a noisy site calm. Each node tracks its own
//! floor (frame energy in dB, asymmetric one-pole: fast fall into gaps, slow
//! rise so a call barely lifts it) and a frame triggers when it stands
//! `snr_db` above the floor as it was *before* the frame. The bar is raised by
//! `fluctuation_multiplier` × the slow mean excess of non-triggering frames
//! (~0 dB on steady noise, several dB in gusts), so fluctuating ambience does
//! not trigger continuously. Callers OR it with their absolute gate. O(1) per
//! frame; non-finite levels are ignored; per-node state is LRU-capped.

use std::collections::BTreeMap;

const DECIBELS_PER_AMPLITUDE_DECADE: f64 = 20.0;
/// Least-recently-heard node floors are evicted beyond this (memory bound only).
pub const MAX_TRACKED_NODE_FLOORS: usize = 1024;
const MINIMUM_TIME_CONSTANT_SECONDS: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeTriggerConfig {
    pub snr_db: f64,
    pub rise_seconds: f64,
    pub fall_seconds: f64,
    pub warmup_seconds: f64,
    /// Floor/level clamp so digital silence cannot make bit noise "loud".
    pub minimum_rms: f64,
    pub fluctuation_multiplier: f64,
}

impl Default for RelativeTriggerConfig {
    fn default() -> Self {
        Self { snr_db: 8.0, rise_seconds: 20.0, fall_seconds: 0.5, warmup_seconds: 1.0, minimum_rms: 1e-5, fluctuation_multiplier: 1.5 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct NodeFloorState {
    floor_db: f64,
    observed_seconds: f64,
    /// Slow mean of max(level − floor, 0) over non-triggering frames.
    mean_excess_db: f64,
    last_used_sequence: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoiseFloorRelativeTrigger {
    config: RelativeTriggerConfig,
    floor_by_node: BTreeMap<String, NodeFloorState>,
    use_sequence: u64,
}

fn one_pole_smoothing(step_seconds: f64, time_constant_seconds: f64) -> f64 {
    1.0 - (-step_seconds / time_constant_seconds.max(MINIMUM_TIME_CONSTANT_SECONDS)).exp()
}

impl NoiseFloorRelativeTrigger {
    pub fn new(config: RelativeTriggerConfig) -> Self {
        Self { config, floor_by_node: BTreeMap::new(), use_sequence: 0 }
    }

    pub fn floor_rms(&self, node_id: &str) -> Option<f64> {
        self.floor_by_node.get(node_id).map(|state| 10f64.powf(state.floor_db / DECIBELS_PER_AMPLITUDE_DECADE))
    }

    pub fn tracked_node_count(&self) -> usize {
        self.floor_by_node.len()
    }

    /// RMS a frame from this node must reach to trigger now (floor plus the
    /// SNR bar and fluctuation allowance); `None` until the node is tracked.
    pub fn trigger_level_rms(&self, node_id: &str) -> Option<f64> {
        let state = self.floor_by_node.get(node_id)?;
        let required_db = self.config.snr_db + self.config.fluctuation_multiplier * state.mean_excess_db;
        Some(10f64.powf((state.floor_db + required_db) / DECIBELS_PER_AMPLITUDE_DECADE))
    }

    /// Drops the node's floor so it re-learns (and re-warms) from its next
    /// frame: after a reboot or a purge its old floor describes another gain
    /// state or site.
    pub fn forget_node(&mut self, node_id: &str) {
        self.floor_by_node.remove(node_id);
    }

    /// Update the node's floor with this frame; true when the frame stands
    /// `snr_db` (+ the fluctuation allowance) above the floor before it.
    pub fn observe(&mut self, node_id: &str, frame_rms: f64, frame_seconds: f64) -> bool {
        if !frame_rms.is_finite() || !frame_seconds.is_finite() {
            return false;
        }
        let config = self.config;
        let level_db = DECIBELS_PER_AMPLITUDE_DECADE * frame_rms.max(config.minimum_rms).log10();
        self.use_sequence += 1;
        let sequence = self.use_sequence;
        let Some(state) = self.floor_by_node.get_mut(node_id) else {
            self.floor_by_node.insert(
                node_id.to_string(),
                NodeFloorState { floor_db: level_db, observed_seconds: frame_seconds, mean_excess_db: 0.0, last_used_sequence: sequence },
            );
            if self.floor_by_node.len() > MAX_TRACKED_NODE_FLOORS {
                self.evict_least_recently_used();
            }
            return false;
        };
        state.last_used_sequence = sequence;
        let excess_db = level_db - state.floor_db;
        let required_db = config.snr_db + config.fluctuation_multiplier * state.mean_excess_db;
        let triggered = state.observed_seconds >= config.warmup_seconds && frame_rms > config.minimum_rms && excess_db >= required_db;
        let step_seconds = frame_seconds.max(0.0);
        let time_constant = if excess_db < 0.0 { config.fall_seconds } else { config.rise_seconds };
        state.floor_db += one_pole_smoothing(step_seconds, time_constant) * excess_db;
        if !triggered {
            let excess_smoothing = one_pole_smoothing(step_seconds, config.rise_seconds);
            state.mean_excess_db += excess_smoothing * (excess_db.max(0.0) - state.mean_excess_db);
        }
        state.observed_seconds += frame_seconds;
        triggered
    }

    fn evict_least_recently_used(&mut self) {
        let oldest = self.floor_by_node.iter().min_by_key(|(_, state)| state.last_used_sequence).map(|(node, _)| node.clone());
        if let Some(oldest) = oldest {
            self.floor_by_node.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faint_call_over_a_quiet_floor_triggers_and_steady_noise_does_not() {
        let mut trigger = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
        let frame_seconds = 0.02;
        let fired_on_noise = (0..200).filter(|_| trigger.observe("n", 1e-4, frame_seconds)).count();
        assert_eq!(fired_on_noise, 0);
        assert!(trigger.observe("n", 4e-4, frame_seconds));
        assert!(!trigger.observe("n", f64::NAN, frame_seconds));
        assert_eq!(trigger.tracked_node_count(), 1);
    }

    #[test]
    fn trigger_level_is_the_bar_a_frame_must_clear_and_forgetting_rewarms() {
        let mut trigger = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
        let frame_seconds = 0.02;
        assert_eq!(trigger.trigger_level_rms("n"), None);
        for _ in 0..200 {
            trigger.observe("n", 1e-4, frame_seconds);
        }
        let level = trigger.trigger_level_rms("n").unwrap();
        // Steady noise: the bar is the floor plus the 8 dB SNR.
        assert!((level / (1e-4 * 10f64.powf(8.0 / 20.0)) - 1.0).abs() < 1e-6);
        assert!(!trigger.clone().observe("n", level * 0.99, frame_seconds));
        assert!(trigger.clone().observe("n", level * 1.01, frame_seconds));
        trigger.forget_node("n");
        assert_eq!(trigger.tracked_node_count(), 0);
        // The first frame after forgetting seeds the floor and never triggers.
        assert!(!trigger.observe("n", 1.0, frame_seconds));
    }
}
