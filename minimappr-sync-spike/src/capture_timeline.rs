//! Capture timeline: monotonic ns ↔ capture sample, split at silent drops.
//!
//! Android can lose capture audio without a failed read: the HAL simply
//! delivers fewer frames than real time (Pixel 10 Pro XL, 2026-11-26: 527 ms
//! lost in one audio-server stall, `frames_read == frame_position`, no
//! `capture_gap` event). The AudioRecord timestamps still show it: between
//! two timestamps the monotonic clock advances further than the frame count
//! explains. One least-squares line through such a run puts every later
//! prediction a whole drop off, so the timeline is fitted per segment and
//! the drops are reported.

use crate::bundle_loader::{fit_timeline, TimelineAnchor};

const NOMINAL_NS_PER_FRAME: f64 = 1e9 / 48_000.0;

/// Audio the capture path lost between two timestamps.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SilentCaptureDrop {
    /// First capture sample after the drop.
    pub capture_frame: f64,
    pub missing_ns: f64,
}

/// One contiguous stretch of the capture timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineSegment {
    pub first_frame: f64,
    pub first_ns: f64,
    pub anchor: TimelineAnchor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureTimeline {
    /// In capture order; never empty.
    pub segments: Vec<TimelineSegment>,
    pub silent_drops: Vec<SilentCaptureDrop>,
}

impl CaptureTimeline {
    /// `points` are (frame_position, nano_time) in log order. A step larger
    /// than `step_threshold_ns` between consecutive points starts a segment.
    pub fn from_points(points: &[(f64, f64)], step_threshold_ns: f64) -> Option<Self> {
        let mut sorted: Vec<(f64, f64)> = points.to_vec();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        sorted.dedup();
        let mut runs: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut silent_drops = Vec::new();
        for point in sorted {
            if let Some(previous) = runs.last().and_then(|run| run.last()).copied() {
                let step_ns = (point.1 - previous.1) - (point.0 - previous.0) * NOMINAL_NS_PER_FRAME;
                if step_ns.abs() > step_threshold_ns {
                    silent_drops.push(SilentCaptureDrop { capture_frame: point.0, missing_ns: step_ns });
                    runs.push(Vec::new());
                }
            } else {
                runs.push(Vec::new());
            }
            runs.last_mut().expect("pushed above").push(point);
        }
        let segments: Vec<TimelineSegment> = runs
            .iter()
            .filter_map(|run| {
                let first = *run.first()?;
                Some(TimelineSegment { first_frame: first.0, first_ns: first.1, anchor: fit_timeline(run)? })
            })
            .collect();
        (!segments.is_empty()).then_some(Self { segments, silent_drops })
    }

    fn segment_for_ns(&self, ns: f64) -> &TimelineSegment {
        self.segments.iter().rev().find(|segment| ns >= segment.first_ns).unwrap_or(&self.segments[0])
    }

    fn segment_for_frame(&self, frame: f64) -> &TimelineSegment {
        self.segments.iter().rev().find(|segment| frame >= segment.first_frame).unwrap_or(&self.segments[0])
    }

    /// Capture sample at monotonic `ns`. Inside a drop there is no audio;
    /// the answer is extrapolated from the segment before it.
    pub fn frame_at_ns(&self, ns: f64) -> f64 {
        self.segment_for_ns(ns).anchor.frame_at_ns(ns)
    }

    pub fn ns_at_frame(&self, frame: f64) -> f64 {
        self.segment_for_frame(frame).anchor.ns_at_frame(frame)
    }

    pub fn anchor_count(&self) -> usize {
        self.segments.iter().map(|segment| segment.anchor.anchor_count).sum()
    }

    /// Worst per-segment fit residual (the step itself no longer counts).
    pub fn residual_rms_ns(&self) -> f64 {
        self.segments.iter().map(|segment| segment.anchor.residual_rms_ns).fold(0.0, f64::max)
    }

    /// Rate from the segment with the most anchors (the best-determined one).
    pub fn rate_offset_ppm(&self) -> Option<f64> {
        self.segments
            .iter()
            .filter(|segment| segment.anchor.anchor_count > 1)
            .max_by_key(|segment| segment.anchor.anchor_count)
            .map(|segment| segment.anchor.rate_offset_ppm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD_NS: f64 = 15e6;

    #[test]
    fn a_silent_drop_splits_the_timeline_and_keeps_both_halves_exact() {
        let drop_ns = 527e6;
        let points: Vec<(f64, f64)> = (0..60)
            .map(|k| {
                let frame = k as f64 * 48_000.0;
                let ns = 1e12 + k as f64 * 1e9 + if k >= 30 { drop_ns } else { 0.0 };
                (frame, ns)
            })
            .collect();
        let timeline = CaptureTimeline::from_points(&points, THRESHOLD_NS).unwrap();
        assert_eq!(timeline.segments.len(), 2);
        assert_eq!(timeline.silent_drops.len(), 1);
        assert!((timeline.silent_drops[0].missing_ns - drop_ns).abs() < 1.0);
        assert_eq!(timeline.silent_drops[0].capture_frame, 30.0 * 48_000.0);
        assert!((timeline.frame_at_ns(1e12 + 10.5e9) - 10.5 * 48_000.0).abs() < 1e-3);
        assert!((timeline.frame_at_ns(1e12 + 40.5e9 + drop_ns) - 40.5 * 48_000.0).abs() < 1e-3);
        assert!((timeline.ns_at_frame(40.5 * 48_000.0) - (1e12 + 40.5e9 + drop_ns)).abs() < 1.0);
        assert!(timeline.residual_rms_ns() < 1.0);
    }

    #[test]
    fn timestamp_jitter_below_the_threshold_stays_one_segment() {
        let points: Vec<(f64, f64)> = (0..40).map(|k| (k as f64 * 48_000.0, 1e12 + k as f64 * 1e9 + if k % 2 == 0 { 5e6 } else { 0.0 })).collect();
        let timeline = CaptureTimeline::from_points(&points, THRESHOLD_NS).unwrap();
        assert_eq!(timeline.segments.len(), 1);
        assert!(timeline.silent_drops.is_empty());
    }
}
