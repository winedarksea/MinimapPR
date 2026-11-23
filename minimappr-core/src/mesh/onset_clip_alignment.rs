//! Decode a call's field clips and place each on the island reference
//! timeline of the call's shared window, ready for onset clustering.
//!
//! A clip arrives as FLAC at 24 kHz on its node's own audio clock (output
//! sample k sits at device-clock index `first + k · decimation`). The node's
//! `NodeClockMapping` gives reference = node − θ − ε (node − anchor); one
//! windowed-sinc pass (`clock_resampling`) applies offset and rate together,
//! so every lag downstream is in island reference time.

use crate::clip_codec::field_clip::decode_field_clip;
use crate::protocol::NodeId;
use crate::srp_layout::clock_resampling::{resample_onto_island_timeline, NodeClockModel};

use super::call_pipeline_types::{CallLocalizationRequest, ClipLocalizationInput};
use super::clock_mapping::NodeClockMapping;

/// PCM16 full scale: decoded samples are divided by this.
const PCM16_FULL_SCALE: f32 = 32_768.0;

/// One node's clip on the window's reference timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedNodeWindow {
    pub node_id: NodeId,
    /// Index into `CallLocalizationRequest::clips`.
    pub clip_index: usize,
    pub samples: Vec<f32>,
    pub coverage_fraction: f64,
}

/// Every usable clip of a call on one timeline starting at
/// `window_start_reference_s`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedCallWindow {
    pub window_start_reference_s: f64,
    pub sample_rate_hz: u32,
    pub nodes: Vec<AlignedNodeWindow>,
    /// (node, reason) for clips that could not be used.
    pub rejected_clips: Vec<(NodeId, String)>,
}

impl AlignedCallWindow {
    pub fn node(&self, node_id: NodeId) -> Option<&AlignedNodeWindow> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }

    pub fn reference_seconds_of_sample(&self, window_sample: f64) -> f64 {
        self.window_start_reference_s + window_sample / f64::from(self.sample_rate_hz)
    }
}

/// The clip clock in the resampler's form: clip-local index k at the clip
/// rate maps to reference time `t(0) + k (1 − ε) / fs_clip`, i.e. a rate
/// ratio of 1 / (1 − ε) anchored at the clip's first sample.
pub fn clip_clock_model(mapping: &NodeClockMapping, first_sample_node_seconds: f64) -> NodeClockModel {
    NodeClockModel {
        anchor_sample_index: 0.0,
        anchor_island_time_s: mapping.reference_seconds_from_node_seconds(first_sample_node_seconds),
        rate_ratio: 1.0 / (1.0 - mapping.rate_offset),
    }
}

fn align_clip(
    clip: &ClipLocalizationInput,
    window_start_reference_s: f64,
    output_length: usize,
    output_sample_rate_hz: u32,
) -> Result<(Vec<f32>, f64), String> {
    let decoded = decode_field_clip(&clip.payload.clip_container).map_err(|error| error.to_string())?;
    let header = decoded.header;
    if header.output_sample_rate_hz != output_sample_rate_hz {
        return Err(format!("clip rate {} Hz, window rate {output_sample_rate_hz} Hz", header.output_sample_rate_hz));
    }
    if header.channel_count != 1 {
        return Err(format!("{} channels (phone clips are mono)", header.channel_count));
    }
    let first_sample_node_seconds =
        header.first_sample_index_device_clock as f64 / f64::from(header.device_clock_sample_rate_hz);
    let samples: Vec<f32> = decoded.samples_output_rate.iter().map(|sample| f32::from(*sample) / PCM16_FULL_SCALE).collect();
    let channel = resample_onto_island_timeline(
        &samples,
        0,
        &clip_clock_model(&clip.clock, first_sample_node_seconds),
        window_start_reference_s,
        output_length,
        f64::from(output_sample_rate_hz),
    );
    Ok((channel.samples, channel.coverage_fraction))
}

/// Decode and align every clip of `request` onto its shared window. A clip
/// that fails to decode, or covers less than `minimum_coverage_fraction` of
/// the window, is listed in `rejected_clips` rather than failing the call.
pub fn align_call_clips(
    request: &CallLocalizationRequest,
    output_sample_rate_hz: u32,
    minimum_coverage_fraction: f64,
) -> AlignedCallWindow {
    let window_start_reference_s = request.centre_reference_s - request.half_window_s;
    let output_length = (2.0 * request.half_window_s * f64::from(output_sample_rate_hz)).round().max(0.0) as usize;
    let mut window = AlignedCallWindow {
        window_start_reference_s,
        sample_rate_hz: output_sample_rate_hz,
        nodes: Vec::new(),
        rejected_clips: Vec::new(),
    };
    for (clip_index, clip) in request.clips.iter().enumerate() {
        match align_clip(clip, window_start_reference_s, output_length, output_sample_rate_hz) {
            Ok((samples, coverage_fraction)) if coverage_fraction >= minimum_coverage_fraction => {
                window.nodes.push(AlignedNodeWindow { node_id: clip.node_id, clip_index, samples, coverage_fraction })
            }
            Ok((_, coverage_fraction)) => {
                window.rejected_clips.push((clip.node_id, format!("clip covers {coverage_fraction:.2} of the window")))
            }
            Err(reason) => window.rejected_clips.push((clip.node_id, reason)),
        }
    }
    window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_clock_model_matches_the_mapping_across_the_clip() {
        let mapping = NodeClockMapping {
            clock_epoch: 1,
            anchor_node_seconds: 80.0,
            offset_s: 0.0123,
            rate_offset: 55e-6,
            offset_sigma_s: 1e-5,
        };
        let first_node_seconds = 100.0;
        let clip_rate = 24_000.0;
        let model = clip_clock_model(&mapping, first_node_seconds);
        for clip_index in [0.0, 12_000.0, 48_000.0] {
            let node_seconds = first_node_seconds + clip_index / clip_rate;
            let expected = mapping.reference_seconds_from_node_seconds(node_seconds);
            let modelled = model.island_time_of_sample(clip_index, clip_rate);
            assert!((modelled - expected).abs() < 1e-9, "{modelled} vs {expected}");
        }
    }
}
