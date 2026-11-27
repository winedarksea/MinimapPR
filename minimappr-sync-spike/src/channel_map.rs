//! Speaker channel map (plan A1): which stereo channel drives which
//! speaker, and where that speaker sits. Two kinds of evidence:
//!
//! - One channel per run: the listener's remote-chirp SNR per (model,
//!   channel, orientation), tabulated across runs by `summarize`.
//! - Channel-cycle runs (emission k plays `channel_cycle[k % n]` inside one
//!   stream, so every channel shares the output latency): per channel, the
//!   self pick − output frame (Δ between channels = self-path difference
//!   d_c − d_ref) and the remote pick − the emitter's output frame mapped
//!   across phones (Δ = cross-path difference). In the colinear tape layout
//!   the cross-path Δ × c is the speaker's axial offset x_c − x_ref. Both use
//!   the output frame, not the self pick: the self pick carries d_c, which
//!   would cancel half of what is being measured.

use std::collections::BTreeMap;

use crate::bundle_schema::BundleEvent;
use crate::pair_ranging_series::{median_of, robust_sigma};
use crate::pick_structure::{pick_structure, PickStructure, PickWithWindow};
use crate::round_pairing::SelfPick;

const MICROSECONDS_PER_SECOND: f64 = 1e6;
const MILLIMETRES_PER_METRE: f64 = 1e3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum PhoneOrientation {
    Portrait,
    PortraitUpsideDown,
    LandscapeLeft,
    LandscapeRight,
    FlatFaceUp,
    FlatFaceDown,
    Unknown,
}

/// Gravity in the device frame from a rotation-vector quaternion
/// (device → world, world z up): g_device = Rᵀ · [0, 0, 1].
fn gravity_in_device_frame(quaternion: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = quaternion;
    // Third row of R (world z axis expressed in device coordinates).
    [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
}

pub fn orientation_from_quaternion(quaternion: [f64; 4]) -> PhoneOrientation {
    let norm = quaternion.iter().map(|c| c * c).sum::<f64>().sqrt();
    if norm < 1e-6 {
        return PhoneOrientation::Unknown;
    }
    let [gx, gy, gz] = gravity_in_device_frame(quaternion.map(|c| c / norm));
    let (ax, ay, az) = (gx.abs(), gy.abs(), gz.abs());
    if az >= ax && az >= ay {
        if gz > 0.0 { PhoneOrientation::FlatFaceUp } else { PhoneOrientation::FlatFaceDown }
    } else if ay >= ax {
        if gy > 0.0 { PhoneOrientation::Portrait } else { PhoneOrientation::PortraitUpsideDown }
    } else if gx > 0.0 {
        PhoneOrientation::LandscapeLeft
    } else {
        PhoneOrientation::LandscapeRight
    }
}

/// Most frequent orientation over the run's attitude samples.
pub fn dominant_orientation(events: &[BundleEvent]) -> PhoneOrientation {
    let mut counts: Vec<(PhoneOrientation, usize)> = Vec::new();
    for event in events {
        if let BundleEvent::Attitude { quaternion, .. } = event {
            let orientation = orientation_from_quaternion(*quaternion);
            match counts.iter_mut().find(|(o, _)| *o == orientation) {
                Some((_, count)) => *count += 1,
                None => counts.push((orientation, 1)),
            }
        }
    }
    counts.into_iter().max_by_key(|(_, count)| *count).map_or(PhoneOrientation::Unknown, |(o, _)| o)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ChannelMapObservation {
    pub emitter_platform: String,
    pub emitter_channel: String,
    pub emitter_orientation: PhoneOrientation,
    pub listener_platform: String,
    pub median_remote_snr_db: Option<f64>,
    pub remote_detected: usize,
    pub remote_expected: usize,
}

/// One channel's path offset against the run's reference channel.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ChannelOffset {
    pub channel: String,
    pub count: usize,
    pub offset_samples: f64,
    pub offset_us: f64,
    /// Offset × c: the path-length difference.
    pub offset_mm: f64,
    pub robust_sigma_samples: f64,
}

fn channel_label(channel: &Option<String>) -> String {
    channel.clone().unwrap_or_else(|| "?".into())
}

/// Per-channel offsets of `values` = (time, value, channel), in time
/// order, relative to `reference`. Each pick is compared with the reference
/// channel interpolated linearly between its nearest reference picks before
/// and after, so slow drift (clock rate, latency creep) cancels exactly to
/// first order; the per-channel median ignores the few picks next to a
/// timeline step.
pub fn per_channel_offsets(values: &[(f64, f64, Option<String>)], reference: &str, speed_of_sound_mps: f64, sample_rate_hz: f64) -> Vec<ChannelOffset> {
    let reference_indices: Vec<usize> = (0..values.len()).filter(|i| channel_label(&values[*i].2) == reference).collect();
    let deviation_at = |index: usize| -> Option<f64> {
        let position = reference_indices.partition_point(|r| *r < index);
        let before = position.checked_sub(1).map(|p| reference_indices[p]);
        let after = reference_indices[position..].iter().copied().find(|r| *r > index);
        let (before, after) = (before?, after?);
        let (x0, v0, _) = &values[before];
        let (x1, v1, _) = &values[after];
        let (x, v, _) = &values[index];
        let fraction = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.5 };
        Some(v - (v0 + fraction * (v1 - v0)))
    };
    let mut by_channel: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for index in 0..values.len() {
        let channel = channel_label(&values[index].2);
        let entry = by_channel.entry(channel).or_default();
        if let Some(deviation) = deviation_at(index) {
            entry.push(deviation);
        }
    }
    by_channel
        .into_iter()
        .map(|(channel, deviation)| {
            // The reference is zero by definition; its spread is its own pick jitter.
            let offset_samples = if channel == reference { 0.0 } else { median_of(&deviation).unwrap_or(0.0) };
            ChannelOffset {
                count: deviation.len(),
                offset_samples,
                offset_us: offset_samples / sample_rate_hz * MICROSECONDS_PER_SECOND,
                offset_mm: offset_samples / sample_rate_hz * speed_of_sound_mps * MILLIMETRES_PER_METRE,
                robust_sigma_samples: robust_sigma(&deviation).unwrap_or(0.0),
                channel,
            }
        })
        .collect()
}

/// Self pick − output frame per detected pick, in emission order.
fn self_latencies(self_picks: &[SelfPick]) -> Vec<(f64, f64, Option<String>)> {
    self_picks
        .iter()
        .filter_map(|p| p.arrival.map(|a| (p.output_frame_index as f64, a.sample_index - p.output_frame_index as f64, p.channel.clone())))
        .collect()
}

fn reference_channel(self_picks: &[SelfPick]) -> String {
    channel_label(&self_picks.iter().find(|p| p.arrival.is_some()).and_then(|p| p.channel.clone()))
}

/// Self picks with each channel's self-path offset removed, so the
/// continuity and onset checks (which expect one fixed loopback latency)
/// see one timeline. Unchanged for a single-channel run.
pub fn channel_aligned_self_picks(self_picks: &[SelfPick], sample_rate_hz: f64) -> Vec<SelfPick> {
    let latencies = self_latencies(self_picks);
    let channel_count = {
        let mut labels: Vec<String> = latencies.iter().map(|(_, _, c)| channel_label(c)).collect();
        labels.sort();
        labels.dedup();
        labels.len()
    };
    if channel_count < 2 {
        return self_picks.to_vec();
    }
    // Speed only scales offset_mm, which is not used here.
    let offsets = per_channel_offsets(&latencies, &reference_channel(self_picks), 343.0, sample_rate_hz);
    self_picks
        .iter()
        .map(|pick| {
            let mut aligned = pick.clone();
            let offset = offsets.iter().find(|o| o.channel == channel_label(&pick.channel)).map_or(0.0, |o| o.offset_samples);
            if let Some(arrival) = aligned.arrival.as_mut() {
                arrival.sample_index -= offset;
            }
            aligned
        })
        .collect()
}

/// One emitting phone's evidence for one of its channels.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ChannelEvidence {
    pub channel: String,
    /// Own self path relative to the reference channel.
    pub self_offset: Option<ChannelOffset>,
    pub self_picks: PickStructure,
    /// The other phone's pick relative to this phone's emission (output
    /// frame), against the reference channel: the axial speaker offset in
    /// the tape layout.
    pub remote_offset: Option<ChannelOffset>,
    pub remote_picks: PickStructure,
    pub median_remote_snr_db: Option<f64>,
    /// The same remote picks split by the channel the *listener* last
    /// played before hearing them: tests whether the listener's own recent
    /// playback (e.g. a still-powered speaker next to its mic) adds copies.
    pub remote_picks_by_listener_last_channel: Vec<(String, PickStructure)>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct EmitterChannelEvidence {
    pub emitter_role: String,
    pub emitter_platform: String,
    pub reference_channel: String,
    pub channels: Vec<ChannelEvidence>,
}

pub struct EmitterPicks<'a> {
    pub role: &'a str,
    pub platform: &'a str,
    pub self_picks: &'a [SelfPick],
    /// The listener's picks of these emissions (same order as `self_picks`).
    pub remote_picks: &'a [Option<crate::arrival_scan::ScannedArrival>],
    /// Emitter capture sample → listener capture sample.
    pub map_to_listener: &'a dyn Fn(f64) -> f64,
    /// Precursor window (samples) for a channel's d_ii.
    pub precursor_window_for_channel: &'a dyn Fn(Option<&str>) -> Option<f64>,
    /// The listener's own self picks (capture samples on the listener).
    pub listener_self_picks: &'a [SelfPick],
}

/// Channel of the listener's latest emission heard (self pick) before `sample`.
fn listener_last_channel(listener_self_picks: &[SelfPick], sample: f64) -> String {
    listener_self_picks
        .iter()
        .filter_map(|p| p.arrival.map(|a| (a.sample_index, &p.channel)))
        .filter(|(own, _)| *own < sample)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map_or("none".into(), |(_, channel)| channel_label(channel))
}

pub fn emitter_channel_evidence(picks: &EmitterPicks, speed_of_sound_mps: f64, sample_rate_hz: f64) -> EmitterChannelEvidence {
    let reference = reference_channel(picks.self_picks);
    let self_offsets = per_channel_offsets(&self_latencies(picks.self_picks), &reference, speed_of_sound_mps, sample_rate_hz);
    // Output frame → emitter capture sample through one constant loopback
    // latency (any constant cancels in the channel Δ; the median keeps the
    // cross-phone mapping on the right capture segment).
    let self_values = self_latencies(picks.self_picks);
    let loopback_latency_samples = median_of(&self_values.iter().map(|(_, latency, _)| *latency).collect::<Vec<_>>()).unwrap_or(0.0);
    let remote_values: Vec<(f64, f64, Option<String>)> = picks
        .self_picks
        .iter()
        .zip(picks.remote_picks)
        .filter_map(|(own, heard)| {
            let emission_on_emitter = own.output_frame_index as f64 + loopback_latency_samples;
            Some((own.output_frame_index as f64, (*heard)?.sample_index - (picks.map_to_listener)(emission_on_emitter), own.channel.clone()))
        })
        .collect();
    let remote_offsets = per_channel_offsets(&remote_values, &reference, speed_of_sound_mps, sample_rate_hz);
    let mut channels: Vec<String> = picks.self_picks.iter().map(|p| channel_label(&p.channel)).collect();
    channels.sort();
    channels.dedup();
    let channels = channels
        .into_iter()
        .map(|channel| {
            let members: Vec<usize> = (0..picks.self_picks.len()).filter(|i| channel_label(&picks.self_picks[*i].channel) == channel).collect();
            let window = (picks.precursor_window_for_channel)(Some(channel.as_str()));
            let own: Vec<PickWithWindow> =
                members.iter().filter_map(|i| picks.self_picks[*i].arrival).map(|arrival| PickWithWindow { arrival, precursor_window_samples: window }).collect();
            let heard: Vec<PickWithWindow> = members
                .iter()
                .filter_map(|i| picks.remote_picks.get(*i).copied().flatten())
                .map(|arrival| PickWithWindow { arrival, precursor_window_samples: None })
                .collect();
            let snrs: Vec<f64> = heard.iter().map(|p| p.arrival.snr_db).collect();
            let mut by_last: BTreeMap<String, Vec<PickWithWindow>> = BTreeMap::new();
            for pick in &heard {
                by_last
                    .entry(listener_last_channel(picks.listener_self_picks, pick.arrival.sample_index))
                    .or_default()
                    .push(PickWithWindow { arrival: pick.arrival, precursor_window_samples: None });
            }
            let remote_picks_by_listener_last_channel =
                by_last.into_iter().map(|(last, group)| (last, pick_structure(&group, speed_of_sound_mps, sample_rate_hz))).collect();
            ChannelEvidence {
                self_offset: self_offsets.iter().find(|o| o.channel == channel).cloned(),
                self_picks: pick_structure(&own, speed_of_sound_mps, sample_rate_hz),
                remote_offset: remote_offsets.iter().find(|o| o.channel == channel).cloned(),
                remote_picks: pick_structure(&heard, speed_of_sound_mps, sample_rate_hz),
                median_remote_snr_db: median_of(&snrs),
                remote_picks_by_listener_last_channel,
                channel,
            }
        })
        .collect();
    EmitterChannelEvidence { emitter_role: picks.role.into(), emitter_platform: picks.platform.into(), reference_channel: reference, channels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrival_scan::ScannedArrival;

    fn cycled_self_picks(offsets_samples: &[(&str, f64)], count: usize, drift_samples_per_pick: f64) -> Vec<SelfPick> {
        (0..count)
            .map(|k| {
                let (channel, offset) = offsets_samples[k % offsets_samples.len()];
                let output = 96_000 + k as u64 * 192_000;
                SelfPick {
                    emission_index: k as u64,
                    output_frame_index: output,
                    emission_t_ns: 0,
                    predicted_sample: None,
                    channel: Some(channel.into()),
                    arrival: Some(ScannedArrival::clean(output as f64 + 1_234.0 + offset + drift_samples_per_pick * k as f64, 50.0)),
                }
            })
            .collect()
    }

    #[test]
    fn channel_offsets_survive_drift_and_alignment_removes_them() {
        let picks = cycled_self_picks(&[("left", 0.0), ("right", 20.9), ("both", 0.4)], 60, 0.37);
        let offsets = per_channel_offsets(&self_latencies(&picks), "left", 343.0, 48_000.0);
        let offset_of = |c: &str| offsets.iter().find(|o| o.channel == c).unwrap().offset_samples;
        assert!(offset_of("left").abs() < 1e-9);
        assert!((offset_of("right") - 20.9).abs() < 0.5, "{offsets:?}");
        assert!((offset_of("both") - 0.4).abs() < 0.5, "{offsets:?}");
        let aligned = channel_aligned_self_picks(&picks, 48_000.0);
        let residual: Vec<f64> = aligned.iter().map(|p| p.arrival.unwrap().sample_index - p.output_frame_index as f64 - 0.37 * p.emission_index as f64).collect();
        let spread = residual.iter().cloned().fold(f64::MIN, f64::max) - residual.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 1.0, "aligned self picks still spread {spread} samples");
    }

    #[test]
    fn single_channel_picks_are_left_untouched() {
        let picks = cycled_self_picks(&[("right", 0.0)], 10, 0.0);
        assert_eq!(channel_aligned_self_picks(&picks, 48_000.0), picks);
    }

    #[test]
    fn identity_quaternion_is_flat_face_up_and_pitched_up_is_portrait() {
        assert_eq!(orientation_from_quaternion([1.0, 0.0, 0.0, 0.0]), PhoneOrientation::FlatFaceUp);
        // Rotate +90° about device x: the top edge points up (held upright).
        let half = std::f64::consts::FRAC_PI_4;
        assert_eq!(orientation_from_quaternion([half.cos(), half.sin(), 0.0, 0.0]), PhoneOrientation::Portrait);
    }
}
