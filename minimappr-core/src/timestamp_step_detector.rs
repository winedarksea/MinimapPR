//! Timestamp steps and timeline residual of one node's packet clock (R5).
//!
//! A node's audio runs at its nominal sample rate, so between two frames the
//! packet timestamps should advance by exactly the sample-index difference
//! divided by the rate. A larger disagreement is a clock *step* (NTP/GPS
//! correction, firmware re-sync), detectable even between two frames of one
//! upload. The *timeline residual* compares each packet timestamp with the
//! nominal-rate timeline the solve buffer actually placed the audio on (its
//! current anchor extrapolated by sample count), so it shows how far the
//! buffered audio sits from where its packets say it belongs: after a step
//! not yet re-anchored, or as crystal drift accumulates between slews
//! (`timeline_discipline.rs`).

const NANOSECONDS_PER_SECOND: i128 = 1_000_000_000;

/// One frame's place on the node's two clocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTimelinePoint {
    pub start_sample_index: i64,
    pub packet_timestamp_ns: i128,
    pub boot_session: Option<u32>,
}

/// Packet-time change the sample clock predicts between two frames.
fn nominal_elapsed_ns(sample_index_delta: i64, sample_rate_hz: u32) -> i128 {
    let sample_rate_hz = i128::from(sample_rate_hz.max(1));
    let scaled = i128::from(sample_index_delta) * NANOSECONDS_PER_SECOND;
    // Round half away from zero, like the buffer's own index-to-time mapping.
    let half = sample_rate_hz / 2;
    if scaled >= 0 {
        (scaled + half) / sample_rate_hz
    } else {
        (scaled - half) / sample_rate_hz
    }
}

/// Signed packet-time jump between consecutive frames beyond what their
/// sample-index difference accounts for (positive: the clock jumped ahead).
pub fn timestamp_step_ns(
    previous: &FrameTimelinePoint,
    next: &FrameTimelinePoint,
    sample_rate_hz: u32,
) -> i128 {
    let packet_elapsed_ns = next.packet_timestamp_ns - previous.packet_timestamp_ns;
    let sample_index_delta = next.start_sample_index - previous.start_sample_index;
    packet_elapsed_ns - nominal_elapsed_ns(sample_index_delta, sample_rate_hz)
}

/// A buffer's nominal-rate timeline: time of any sample index from its anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NominalSampleTimeline {
    pub origin_time_ns: i128,
    pub origin_sample_index: i64,
    pub sample_rate_hz: u32,
}

impl NominalSampleTimeline {
    pub fn time_at_sample_index_ns(&self, sample_index: i64) -> i128 {
        self.origin_time_ns
            + nominal_elapsed_ns(sample_index - self.origin_sample_index, self.sample_rate_hz)
    }

    /// Packet timestamp minus where this timeline put the frame (signed ns).
    pub fn residual_ns(&self, point: &FrameTimelinePoint) -> i128 {
        point.packet_timestamp_ns - self.time_at_sample_index_ns(point.start_sample_index)
    }
}

/// A break in a node's packet timeline found between two frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingDiscontinuity {
    /// Same boot session, timestamp jump over the threshold.
    TimestampStep { step_ns: i128 },
    /// The node rebooted: sample indices restart, so no step size exists, but
    /// its clock may have re-synchronised to anything.
    BootSessionChange,
}

/// Compares each frame with the node's previous one (across uploads too).
#[derive(Clone, Debug)]
pub struct TimestampStepDetector {
    step_threshold_ns: i128,
    previous: Option<FrameTimelinePoint>,
}

impl TimestampStepDetector {
    pub fn new(step_threshold_seconds: f64) -> Self {
        Self {
            step_threshold_ns: (step_threshold_seconds.max(0.0) * NANOSECONDS_PER_SECOND as f64)
                .round() as i128,
            previous: None,
        }
    }

    pub fn observe(
        &mut self,
        point: FrameTimelinePoint,
        sample_rate_hz: u32,
    ) -> Option<TimingDiscontinuity> {
        let previous = self.previous.replace(point)?;
        let boot_session_changed = matches!(
            (previous.boot_session, point.boot_session),
            (Some(before), Some(after)) if before != after
        );
        if boot_session_changed {
            return Some(TimingDiscontinuity::BootSessionChange);
        }
        let step_ns = timestamp_step_ns(&previous, &point, sample_rate_hz);
        (step_ns.abs() > self.step_threshold_ns)
            .then_some(TimingDiscontinuity::TimestampStep { step_ns })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE_HZ: u32 = 16_000;
    const FRAME_SAMPLES: i64 = 512;
    const FRAME_NS: i128 = 32_000_000;
    const START_NS: i128 = 1_764_000_000_000_000_000;
    const THRESHOLD_SECONDS: f64 = 0.002;

    fn frame(index: i64, extra_ns: i128, boot_session: u32) -> FrameTimelinePoint {
        FrameTimelinePoint {
            start_sample_index: index * FRAME_SAMPLES,
            packet_timestamp_ns: START_NS + i128::from(index) * FRAME_NS + extra_ns,
            boot_session: Some(boot_session),
        }
    }

    #[test]
    fn contiguous_frames_on_the_nominal_rate_raise_no_step() {
        let mut detector = TimestampStepDetector::new(THRESHOLD_SECONDS);
        for index in 0..10 {
            assert_eq!(detector.observe(frame(index, 0, 1), RATE_HZ), None);
        }
    }

    #[test]
    fn a_21_ms_jump_between_sample_contiguous_frames_is_one_step() {
        let mut detector = TimestampStepDetector::new(THRESHOLD_SECONDS);
        let mut discontinuities = Vec::new();
        for index in 0..6 {
            let extra_ns = if index >= 3 { 21_000_000 } else { 0 };
            discontinuities.extend(detector.observe(frame(index, extra_ns, 1), RATE_HZ));
        }
        assert_eq!(
            discontinuities,
            [TimingDiscontinuity::TimestampStep {
                step_ns: 21_000_000
            }]
        );
    }

    #[test]
    fn jitter_under_the_threshold_and_sample_gaps_are_not_steps() {
        let mut detector = TimestampStepDetector::new(THRESHOLD_SECONDS);
        assert_eq!(detector.observe(frame(0, 0, 1), RATE_HZ), None);
        assert_eq!(detector.observe(frame(1, 1_500_000, 1), RATE_HZ), None);
        // Five frames lost in transit: indices and timestamps both skip ahead.
        assert_eq!(detector.observe(frame(7, 0, 1), RATE_HZ), None);
    }

    #[test]
    fn a_boot_session_change_is_reported_instead_of_a_step() {
        let mut detector = TimestampStepDetector::new(THRESHOLD_SECONDS);
        detector.observe(frame(400, 0, 1), RATE_HZ);
        // After the reboot sample indices restart near zero.
        assert_eq!(
            detector.observe(frame(0, 900 * FRAME_NS, 2), RATE_HZ),
            Some(TimingDiscontinuity::BootSessionChange)
        );
        assert_eq!(detector.observe(frame(1, 900 * FRAME_NS, 2), RATE_HZ), None);
    }

    #[test]
    fn residual_is_packet_time_against_the_anchored_nominal_timeline() {
        let timeline = NominalSampleTimeline {
            origin_time_ns: START_NS,
            origin_sample_index: 0,
            sample_rate_hz: RATE_HZ,
        };
        assert_eq!(timeline.residual_ns(&frame(5, 0, 1)), 0);
        assert_eq!(timeline.residual_ns(&frame(9, 21_000_000, 1)), 21_000_000);
        assert_eq!(timeline.residual_ns(&frame(9, -400_000, 1)), -400_000);
    }
}
