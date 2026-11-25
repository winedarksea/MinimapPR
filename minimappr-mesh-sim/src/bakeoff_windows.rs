//! Trigger model, noise floors and window cutting for the point bake-off.
//!
//! Both candidates see the same triggers, cut from the same stamped
//! timelines, so a difference between them is the localizer's, not the
//! trigger's. Windows are cut on each node's own stamped clock, which is
//! exactly how a shared sensor buffer aligns them.

use crate::bakeoff_config::TriggerModelConfig;
use crate::bakeoff_scenes::BakeoffScene;

const DECIBELS_PER_AMPLITUDE_DECADE: f64 = 20.0;

/// One solve request: the node that crossed first and when, on its timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakeoffTrigger {
    pub node: usize,
    pub stream_time_s: f64,
}

fn frame_rms_series(stream: &[f32], frame_samples: usize) -> Vec<f64> {
    stream
        .chunks(frame_samples.max(1))
        .filter(|frame| frame.len() == frame_samples)
        .map(|frame| (frame.iter().map(|sample| f64::from(*sample).powi(2)).sum::<f64>() / frame.len() as f64).sqrt())
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Median frame RMS of the noise-only lead-in: the node's tracked floor.
pub fn node_noise_floor_rms(scene: &BakeoffScene, node: usize, frame_seconds: f64) -> f64 {
    let frame_samples = (frame_seconds * f64::from(scene.sample_rate_hz)).round() as usize;
    let lead_in_samples = ((scene.lead_in_noise_s * f64::from(scene.sample_rate_hz)) as usize).min(scene.streams[node].len());
    median(&mut frame_rms_series(&scene.streams[node][..lead_in_samples], frame_samples))
}

/// Cluster-level noise-floor-relative trigger: every rising crossing of
/// `relative_db` over a node's floor, earliest first, thinned by the cooldown.
pub fn detect_triggers(scene: &BakeoffScene, config: &TriggerModelConfig) -> Vec<BakeoffTrigger> {
    let frame_samples = (config.frame_seconds * f64::from(scene.sample_rate_hz)).round() as usize;
    let threshold_ratio = 10f64.powf(config.relative_db / DECIBELS_PER_AMPLITUDE_DECADE);
    let mut crossings = Vec::new();
    for node in 0..scene.nodes.len() {
        let threshold = node_noise_floor_rms(scene, node, config.frame_seconds) * threshold_ratio;
        let series = frame_rms_series(&scene.streams[node], frame_samples);
        for (index, pair) in series.windows(2).enumerate() {
            if pair[0] < threshold && pair[1] >= threshold {
                let frame_start_s = ((index + 1) * frame_samples) as f64 / f64::from(scene.sample_rate_hz);
                if frame_start_s >= scene.lead_in_noise_s {
                    crossings.push(BakeoffTrigger { node, stream_time_s: frame_start_s });
                }
            }
        }
    }
    crossings.sort_by(|a, b| a.stream_time_s.total_cmp(&b.stream_time_s));
    let mut accepted: Vec<BakeoffTrigger> = Vec::new();
    for crossing in crossings {
        if accepted.last().is_none_or(|last| crossing.stream_time_s >= last.stream_time_s + config.cooldown_seconds) {
            accepted.push(crossing);
        }
    }
    accepted
}

/// Samples of `stream` stamped in `[start_s, end_s)`, zero-filled where the
/// stream has no audio, so every channel of a solve has the same length.
pub fn cut_window(stream: &[f32], sample_rate_hz: u32, start_s: f64, end_s: f64) -> Vec<f32> {
    let rate = f64::from(sample_rate_hz);
    let first = (start_s * rate).round() as i64;
    let count = ((end_s - start_s) * rate).round().max(0.0) as usize;
    (0..count)
        .map(|offset| {
            let index = first + offset as i64;
            if index >= 0 && (index as usize) < stream.len() {
                stream[index as usize]
            } else {
                0.0
            }
        })
        .collect()
}
