//! Capture-sample ↔ local-monotonic-time mapping from the OS audio
//! timestamps (`AudioRecord.getTimestamp`: a frame position and the
//! monotonic nanosecond it was captured). Converts a slot's local time into
//! a capture sample index so the listener knows where to search.
//!
//! Timestamps jitter by tens to hundreds of µs and occasionally glitch by
//! milliseconds, so a robust line fit over a sliding window is used: least
//! squares, drop residuals beyond a MAD gate, refit. The search windows only
//! need ~ms accuracy; the chirp picks themselves give the fine timing.

use std::collections::VecDeque;

const NOMINAL_CAPTURE_SAMPLE_RATE_HZ: f64 = 48_000.0;
const NANOSECONDS_PER_SECOND: f64 = 1e9;
const ANCHOR_WINDOW_COUNT: usize = 64;
/// A new anchor this far off the current fit means the frame timeline
/// jumped (capture restart or a silent drop): the fit restarts from it.
const TIMELINE_JUMP_THRESHOLD_NS: f64 = 5e6;
/// MAD → σ for a normal distribution.
const MAD_TO_SIGMA: f64 = 1.4826;
const OUTLIER_GATE_SIGMA: f64 = 3.0;
/// Floor on the outlier gate so a near-perfect window does not reject
/// ordinary scheduler jitter.
const MINIMUM_OUTLIER_GATE_NS: f64 = 50_000.0;
const PARTS_PER_MILLION: f64 = 1e6;

#[derive(Debug, Clone, Copy, PartialEq)]
struct CaptureAnchor {
    frame_position: u64,
    monotonic_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AnchorLine {
    reference_frame: u64,
    ns_at_reference: f64,
    ns_per_frame: f64,
    residual_rms_ns: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CaptureClockAnchor {
    anchors: VecDeque<CaptureAnchor>,
    line: Option<AnchorLine>,
    pub timeline_jump_count: u64,
}

fn fit_line(anchors: &[CaptureAnchor]) -> Option<AnchorLine> {
    let reference = anchors.first()?;
    if anchors.len() == 1 {
        return Some(AnchorLine {
            reference_frame: reference.frame_position,
            ns_at_reference: reference.monotonic_ns as f64,
            ns_per_frame: NANOSECONDS_PER_SECOND / NOMINAL_CAPTURE_SAMPLE_RATE_HZ,
            residual_rms_ns: 0.0,
        });
    }
    let points: Vec<(f64, f64)> = anchors
        .iter()
        .map(|a| {
            (
                a.frame_position as f64 - reference.frame_position as f64,
                (a.monotonic_ns - reference.monotonic_ns) as f64,
            )
        })
        .collect();
    let count = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / count;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / count;
    let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let ns_per_frame = if sxx > 0.0 { sxy / sxx } else { NANOSECONDS_PER_SECOND / NOMINAL_CAPTURE_SAMPLE_RATE_HZ };
    let intercept = mean_y - ns_per_frame * mean_x;
    let residual_rms_ns = (points.iter().map(|p| (p.1 - intercept - ns_per_frame * p.0).powi(2)).sum::<f64>() / count).sqrt();
    Some(AnchorLine {
        reference_frame: reference.frame_position,
        ns_at_reference: reference.monotonic_ns as f64 + intercept,
        ns_per_frame,
        residual_rms_ns,
    })
}

impl AnchorLine {
    fn ns_at(&self, frame: f64) -> f64 {
        self.ns_at_reference + (frame - self.reference_frame as f64) * self.ns_per_frame
    }

    fn frame_at(&self, ns: f64) -> f64 {
        self.reference_frame as f64 + (ns - self.ns_at_reference) / self.ns_per_frame
    }
}

impl CaptureClockAnchor {
    pub fn report(&mut self, frame_position: u64, monotonic_ns: i64) {
        let anchor = CaptureAnchor { frame_position, monotonic_ns };
        let reversed = self.anchors.back().is_some_and(|last| frame_position <= last.frame_position);
        let jumped = self
            .line
            .is_some_and(|line| (line.ns_at(frame_position as f64) - monotonic_ns as f64).abs() > TIMELINE_JUMP_THRESHOLD_NS);
        if reversed || jumped {
            self.timeline_jump_count += 1;
            self.anchors.clear();
        }
        self.anchors.push_back(anchor);
        while self.anchors.len() > ANCHOR_WINDOW_COUNT {
            self.anchors.pop_front();
        }
        self.refit();
    }

    fn refit(&mut self) {
        let all: Vec<CaptureAnchor> = self.anchors.iter().copied().collect();
        let Some(first_pass) = fit_line(&all) else {
            self.line = None;
            return;
        };
        let mut residuals: Vec<f64> = all.iter().map(|a| (a.monotonic_ns as f64 - first_pass.ns_at(a.frame_position as f64)).abs()).collect();
        let mut sorted = residuals.clone();
        sorted.sort_by(f64::total_cmp);
        let mad = sorted[sorted.len() / 2];
        let gate = (OUTLIER_GATE_SIGMA * MAD_TO_SIGMA * mad).max(MINIMUM_OUTLIER_GATE_NS);
        let inliers: Vec<CaptureAnchor> = all
            .iter()
            .zip(residuals.drain(..))
            .filter(|(_, residual)| *residual <= gate)
            .map(|(anchor, _)| *anchor)
            .collect();
        self.line = fit_line(&inliers).or(Some(first_pass));
    }

    pub fn is_valid(&self) -> bool {
        self.line.is_some()
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Fractional capture sample at a local monotonic time.
    pub fn sample_index_at(&self, local_ns: i64) -> Option<f64> {
        self.line.map(|line| line.frame_at(local_ns as f64))
    }

    pub fn local_ns_at(&self, sample_index: f64) -> Option<i64> {
        self.line.map(|line| line.ns_at(sample_index).round() as i64)
    }

    /// Audio clock rate relative to the monotonic clock, ppm.
    pub fn rate_offset_ppm(&self) -> Option<f64> {
        self.line.map(|line| (NANOSECONDS_PER_SECOND / NOMINAL_CAPTURE_SAMPLE_RATE_HZ / line.ns_per_frame - 1.0) * PARTS_PER_MILLION)
    }

    pub fn residual_rms_ns(&self) -> Option<f64> {
        self.line.map(|line| line.residual_rms_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAMES_PER_ANCHOR: u64 = 48_000;

    fn ns_for(frame: u64, rate_ppm: f64, start_ns: i64) -> i64 {
        start_ns + (frame as f64 / (NOMINAL_CAPTURE_SAMPLE_RATE_HZ * (1.0 + rate_ppm / PARTS_PER_MILLION)) * NANOSECONDS_PER_SECOND) as i64
    }

    #[test]
    fn fit_recovers_rate_and_ignores_a_glitch() {
        let mut anchor = CaptureClockAnchor::default();
        let start_ns = 5_000_000_000;
        for step in 0..30u64 {
            let frame = step * FRAMES_PER_ANCHOR;
            let jitter = if step % 2 == 0 { 40_000 } else { -40_000 };
            let glitch = if step == 17 { 3_000_000 } else { 0 };
            anchor.report(frame, ns_for(frame, 40.0, start_ns) + jitter + glitch);
        }
        assert!((anchor.rate_offset_ppm().unwrap() - 40.0).abs() < 2.0, "{:?}", anchor.rate_offset_ppm());
        let frame = 20 * FRAMES_PER_ANCHOR + 1_234;
        let predicted = anchor.sample_index_at(ns_for(frame, 40.0, start_ns)).unwrap();
        assert!((predicted - frame as f64).abs() < 5.0, "predicted {predicted} for {frame}");
        assert_eq!(anchor.timeline_jump_count, 0);
    }

    #[test]
    fn single_anchor_uses_nominal_rate_and_round_trips() {
        let mut anchor = CaptureClockAnchor::default();
        anchor.report(1_000, 2_000_000_000);
        let ns = anchor.local_ns_at(49_000.0).unwrap();
        assert_eq!(ns, 3_000_000_000);
        assert!((anchor.sample_index_at(ns).unwrap() - 49_000.0).abs() < 1e-6);
    }

    #[test]
    fn timeline_jump_restarts_the_fit() {
        let mut anchor = CaptureClockAnchor::default();
        for step in 0..5u64 {
            anchor.report(step * FRAMES_PER_ANCHOR, ns_for(step * FRAMES_PER_ANCHOR, 0.0, 0));
        }
        anchor.report(5 * FRAMES_PER_ANCHOR, ns_for(5 * FRAMES_PER_ANCHOR, 0.0, 0) + 50_000_000);
        assert_eq!(anchor.timeline_jump_count, 1);
        assert_eq!(anchor.anchor_count(), 1);
    }
}
