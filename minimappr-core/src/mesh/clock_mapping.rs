//! Linear map between a node's own audio clock and its island's reference
//! clock, in the wire form of `NodeClockEstimate`:
//! `reference_s = node_s − θ − ε · (node_s − anchor_node_s)`.
//!
//! Built from a `PhoneClockFilter` prediction at one reference time; the
//! filter works backward too, so a post-detection round's offset applies to
//! the call's buffered audio (plan: "clock goes backward"). The difference
//! from the filter's exact form is ε²·Δt (≈ 0.4 µs after 100 s at 60 ppm).

use crate::protocol::session_messages::NodeClockEstimate;
use crate::protocol::NodeId;
use crate::sync::clock_filter::ClockPrediction;

use super::config::DEVICE_SAMPLE_RATE_HZ;

const PARTS_PER_MILLION: f64 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeClockMapping {
    /// Host-side epoch: bumps on every invalidation or domain reset, so a
    /// clip requested under one model is never used under another.
    pub clock_epoch: u32,
    pub anchor_node_seconds: f64,
    /// θ = node − reference at the anchor (seconds).
    pub offset_s: f64,
    /// ε = node ticks per reference tick − 1.
    pub rate_offset: f64,
    pub offset_sigma_s: f64,
}

impl NodeClockMapping {
    /// Mapping valid around `reference_time_s` from a filter prediction.
    pub fn from_prediction(
        clock_epoch: u32,
        reference_time_s: f64,
        prediction: &ClockPrediction,
    ) -> Self {
        Self {
            clock_epoch,
            anchor_node_seconds: reference_time_s + prediction.offset_s,
            offset_s: prediction.offset_s,
            rate_offset: prediction.rate_offset,
            offset_sigma_s: prediction.offset_sigma_s,
        }
    }

    /// The island reference node maps onto itself; the anchor records when
    /// (on its own clock) the model was last confirmed by a round.
    pub fn identity(clock_epoch: u32, anchor_node_seconds: f64) -> Self {
        Self {
            clock_epoch,
            anchor_node_seconds,
            offset_s: 0.0,
            rate_offset: 0.0,
            offset_sigma_s: 0.0,
        }
    }

    pub fn reference_seconds_from_node_seconds(&self, node_seconds: f64) -> f64 {
        node_seconds - self.offset_s - self.rate_offset * (node_seconds - self.anchor_node_seconds)
    }

    pub fn node_seconds_from_reference_seconds(&self, reference_seconds: f64) -> f64 {
        (reference_seconds + self.offset_s - self.rate_offset * self.anchor_node_seconds)
            / (1.0 - self.rate_offset)
    }

    pub fn node_sample_index_from_reference_seconds(&self, reference_seconds: f64) -> u64 {
        let node_seconds = self.node_seconds_from_reference_seconds(reference_seconds);
        (node_seconds * DEVICE_SAMPLE_RATE_HZ).round().max(0.0) as u64
    }

    pub fn reference_seconds_from_node_sample(&self, node_sample_index: f64) -> f64 {
        self.reference_seconds_from_node_seconds(node_sample_index / DEVICE_SAMPLE_RATE_HZ)
    }

    pub fn to_wire(&self, node_id: NodeId) -> NodeClockEstimate {
        NodeClockEstimate {
            node_id,
            anchor_node_sample_index: (self.anchor_node_seconds * DEVICE_SAMPLE_RATE_HZ)
                .round()
                .max(0.0) as u64,
            offset_samples: self.offset_s * DEVICE_SAMPLE_RATE_HZ,
            rate_offset_ppm: self.rate_offset * PARTS_PER_MILLION,
            offset_standard_deviation_samples: (self.offset_sigma_s * DEVICE_SAMPLE_RATE_HZ)
                as f32,
        }
    }

    pub fn from_wire(clock_epoch: u32, estimate: &NodeClockEstimate) -> Self {
        Self {
            clock_epoch,
            anchor_node_seconds: estimate.anchor_node_sample_index as f64 / DEVICE_SAMPLE_RATE_HZ,
            offset_s: estimate.offset_samples / DEVICE_SAMPLE_RATE_HZ,
            rate_offset: estimate.rate_offset_ppm / PARTS_PER_MILLION,
            offset_sigma_s: estimate.offset_standard_deviation_samples as f64
                / DEVICE_SAMPLE_RATE_HZ,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_round_trips_and_matches_drifting_clock() {
        let theta = 0.0123;
        let epsilon = 45e-6;
        let reference_anchor = 100.0;
        let mapping = NodeClockMapping::from_prediction(
            1,
            reference_anchor,
            &ClockPrediction {
                offset_s: theta,
                offset_sigma_s: 1e-5,
                rate_offset: epsilon,
                rate_sigma: 1e-7,
            },
        );
        for reference in [40.0, 100.0, 180.0] {
            // True node clock: node = ref + θ + ε (ref − anchor).
            let node = reference + theta + epsilon * (reference - reference_anchor);
            let mapped = mapping.reference_seconds_from_node_seconds(node);
            assert!((mapped - reference).abs() < 1e-6, "{mapped} vs {reference}");
            let back = mapping.node_seconds_from_reference_seconds(mapped);
            assert!((back - node).abs() < 1e-9);
        }
        let wire = mapping.to_wire(NodeId(3));
        let restored = NodeClockMapping::from_wire(1, &wire);
        assert!((restored.offset_s - theta).abs() < 1e-9);
        assert!((restored.rate_offset - epsilon).abs() < 1e-12);
    }
}
