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
//!
//! Short-block excess (default off at every caller): a call shorter than a
//! frame, or straddling two, is diluted by the frame's quiet remainder. When
//! the caller passes the frame's loudest short block (`peak_short_block_rms`),
//! it is judged against the same floor with its own fluctuation allowance and
//! OR'd with the frame decision; the floor always follows the frame level.
//! Same semantics as the Python reference.

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
    /// Same, for the loudest short block (only fed when the caller passes one).
    mean_short_block_excess_db: f64,
    last_used_sequence: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoiseFloorRelativeTrigger {
    config: RelativeTriggerConfig,
    floor_by_node: BTreeMap<String, NodeFloorState>,
    use_sequence: u64,
}

/// Samples per short block: `floor(seconds * rate + 0.5)`, at least 1 (the
/// Python reference rounds the same way, not half-to-even).
pub fn short_block_samples(sample_rate_hz: f64, block_seconds: f64) -> usize {
    ((block_seconds * sample_rate_hz + 0.5).floor()).max(1.0) as usize
}

fn mean_square(samples: &[f32]) -> f64 {
    samples.iter().map(|sample| f64::from(*sample) * f64::from(*sample)).sum::<f64>() / samples.len() as f64
}

/// Largest RMS over any channel's non-overlapping `block_seconds` blocks.
/// Blocks start at sample 0; a remainder is covered by one extra block ending
/// at the last sample, so every block has full length (equal noise variance).
/// A channel no longer than one block contributes its whole-frame RMS.
pub fn peak_short_block_rms<C: AsRef<[f32]>>(channels: &[C], sample_rate_hz: f64, block_seconds: f64) -> f64 {
    let block_samples = short_block_samples(sample_rate_hz, block_seconds);
    let mut peak_mean_square = 0.0_f64;
    for channel in channels {
        let samples = channel.as_ref();
        if samples.is_empty() {
            continue;
        }
        if block_samples >= samples.len() {
            peak_mean_square = peak_mean_square.max(mean_square(samples));
            continue;
        }
        for block in samples.chunks_exact(block_samples) {
            peak_mean_square = peak_mean_square.max(mean_square(block));
        }
        if samples.len() % block_samples != 0 {
            peak_mean_square = peak_mean_square.max(mean_square(&samples[samples.len() - block_samples..]));
        }
    }
    peak_mean_square.sqrt()
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
        self.observe_with_short_blocks(node_id, frame_rms, frame_seconds, None)
    }

    /// `observe`, OR'd with the frame's loudest short block (when given)
    /// standing `snr_db` + its own fluctuation allowance above the floor.
    pub fn observe_with_short_blocks(
        &mut self,
        node_id: &str,
        frame_rms: f64,
        frame_seconds: f64,
        peak_short_block_rms: Option<f64>,
    ) -> bool {
        if !frame_rms.is_finite() || !frame_seconds.is_finite() {
            return false;
        }
        if peak_short_block_rms.is_some_and(|rms| !rms.is_finite()) {
            return false;
        }
        let config = self.config;
        let level_db = DECIBELS_PER_AMPLITUDE_DECADE * frame_rms.max(config.minimum_rms).log10();
        self.use_sequence += 1;
        let sequence = self.use_sequence;
        let Some(state) = self.floor_by_node.get_mut(node_id) else {
            self.floor_by_node.insert(
                node_id.to_string(),
                NodeFloorState {
                    floor_db: level_db,
                    observed_seconds: frame_seconds,
                    mean_excess_db: 0.0,
                    mean_short_block_excess_db: 0.0,
                    last_used_sequence: sequence,
                },
            );
            if self.floor_by_node.len() > MAX_TRACKED_NODE_FLOORS {
                self.evict_least_recently_used();
            }
            return false;
        };
        state.last_used_sequence = sequence;
        let excess_db = level_db - state.floor_db;
        let warmed_up = state.observed_seconds >= config.warmup_seconds;
        let required_db = config.snr_db + config.fluctuation_multiplier * state.mean_excess_db;
        let mut triggered = warmed_up && frame_rms > config.minimum_rms && excess_db >= required_db;
        let short_block_excess_db = peak_short_block_rms.map(|peak_rms| {
            DECIBELS_PER_AMPLITUDE_DECADE * peak_rms.max(config.minimum_rms).log10() - state.floor_db
        });
        if let (Some(peak_rms), Some(block_excess_db)) = (peak_short_block_rms, short_block_excess_db) {
            let block_required_db = config.snr_db + config.fluctuation_multiplier * state.mean_short_block_excess_db;
            triggered = triggered || (warmed_up && peak_rms > config.minimum_rms && block_excess_db >= block_required_db);
        }
        let step_seconds = frame_seconds.max(0.0);
        let time_constant = if excess_db < 0.0 { config.fall_seconds } else { config.rise_seconds };
        state.floor_db += one_pole_smoothing(step_seconds, time_constant) * excess_db;
        if !triggered {
            let excess_smoothing = one_pole_smoothing(step_seconds, config.rise_seconds);
            state.mean_excess_db += excess_smoothing * (excess_db.max(0.0) - state.mean_excess_db);
            if let Some(block_excess_db) = short_block_excess_db {
                state.mean_short_block_excess_db +=
                    excess_smoothing * (block_excess_db.max(0.0) - state.mean_short_block_excess_db);
            }
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

    #[test]
    fn short_block_peak_tiles_full_blocks_and_covers_the_remainder() {
        // 10 samples, 4-sample blocks: [0..4), [4..8) and the tail block [6..10).
        let mut channel = vec![0.0_f32; 10];
        channel[9] = 2.0;
        let peak = peak_short_block_rms(&[channel.clone()], 4.0, 1.0);
        assert!((peak - (4.0_f64 / 4.0).sqrt()).abs() < 1e-12);
        // A block no shorter than the frame is the whole-frame RMS.
        let whole = peak_short_block_rms(&[channel], 4.0, 5.0);
        assert!((whole - (4.0_f64 / 10.0).sqrt()).abs() < 1e-12);
        assert_eq!(peak_short_block_rms::<Vec<f32>>(&[], 48_000.0, 0.005), 0.0);
    }

    #[test]
    fn short_block_excess_admits_a_chip_the_frame_mean_dilutes_and_is_otherwise_inert() {
        let frame_seconds = 0.02;
        let mut frame_only = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
        let mut with_blocks = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
        for _ in 0..200 {
            assert!(!frame_only.observe("n", 1e-4, frame_seconds));
            // Steady noise: the block peak sits ~1 dB over the frame mean.
            assert!(!with_blocks.observe_with_short_blocks("n", 1e-4, frame_seconds, Some(1.12e-4)));
        }
        // A 5 ms chip 12 dB over the floor: ~6.7 dB diluted to the 20 ms frame mean.
        let chip_block_rms = 1e-4 * 10f64.powf(12.0 / 20.0);
        let chip_frame_rms = (1e-8 * 0.75 + chip_block_rms * chip_block_rms * 0.25).sqrt();
        assert!(!frame_only.observe("n", chip_frame_rms, frame_seconds));
        assert!(with_blocks.observe_with_short_blocks("n", chip_frame_rms, frame_seconds, Some(chip_block_rms)));
        assert!(!with_blocks.observe_with_short_blocks("n", 1e-4, frame_seconds, Some(f64::NAN)));
    }

    /// Per-frame cost for `docs/faint_call_admission_results.md`:
    /// `cargo test --release -p minimappr-core short_block_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing report, not a check"]
    fn short_block_cost_per_frame_report() {
        const FRAMES: usize = 50_000;
        const CHANNELS: usize = 4;
        const FRAME_SAMPLES: usize = 960;
        let mut seed = 1_u32;
        let frame: Vec<Vec<f32>> = (0..CHANNELS)
            .map(|_| {
                (0..FRAME_SAMPLES)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed as i32 as f32) / (i32::MAX as f32) * 1e-4
                    })
                    .collect()
            })
            .collect();
        let frame_rms = |frame: &[Vec<f32>]| frame.iter().map(|channel| mean_square(channel).sqrt()).fold(0.0, f64::max);
        let mut whole_frame = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
        let started = std::time::Instant::now();
        for _ in 0..FRAMES {
            whole_frame.observe("n", frame_rms(std::hint::black_box(&frame)), 0.02);
        }
        let whole_frame_ns = started.elapsed().as_nanos() as f64 / FRAMES as f64;
        let mut short_block = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
        let started = std::time::Instant::now();
        for _ in 0..FRAMES {
            let frame = std::hint::black_box(&frame);
            let peak = peak_short_block_rms(frame, 48_000.0, 0.005);
            short_block.observe_with_short_blocks("n", frame_rms(frame), 0.02, Some(peak));
        }
        let short_block_ns = started.elapsed().as_nanos() as f64 / FRAMES as f64;
        println!("relative trigger per 4x960 frame: whole-frame {whole_frame_ns:.0} ns, with 5 ms short blocks {short_block_ns:.0} ns");
    }
}
