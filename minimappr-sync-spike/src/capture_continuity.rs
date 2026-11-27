//! Where a phone's capture (or playback) timeline broke, and which rounds
//! that spoils.
//!
//! A round's ranging intervals are differences of capture samples on each
//! phone, so a capture discontinuity between two picks of a round moves the
//! range by the lost time × c (1 ms ≈ 34 cm). Two detectors:
//!
//! - **Timestamp steps** (`capture_timeline.rs`): catch large silent drops
//!   (≥ the step threshold, ~15 ms) even when no self chirp brackets them.
//! - **Self-interval breaks**: consecutive self chirps are a known number of
//!   output frames apart, and on a healthy phone their capture picks agree
//!   to ~0.1 sample (Pixel 7a: σ 0.9 µs). A disagreement over the tolerance
//!   means capture or playback slipped in between. This catches drops far
//!   below timestamp jitter. A pure playback slip does not bias ranging
//!   (the self pick measures the real emission), so leaving those rounds out
//!   is conservative.

use crate::capture_timeline::SilentCaptureDrop;
use crate::pair_ranging_series::median_of;
use crate::round_pairing::{PairedRound, SelfPick};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum BreakKind {
    /// Capture timestamps stepped: audio lost without a failed read.
    SilentDrop,
    /// The self-pick interval changed and stayed changed: playback or
    /// capture slipped between the two picks.
    SelfIntervalStep,
    /// One self pick off by the same amount both ways (the A03s ±240 µs
    /// hops): a bad pick, not a timeline break. Rounds using it are left
    /// out, but θ does not step across it.
    SelfPickOutlier,
}

/// Capture samples `(start, end)` that may hold a discontinuity.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct CaptureBreak {
    pub kind: BreakKind,
    pub start_sample: f64,
    pub end_sample: f64,
    /// Capture − expected interval in samples (self-interval breaks), or the
    /// dropped time in samples (timestamp steps).
    pub discrepancy_samples: f64,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct CaptureContinuity {
    pub breaks: Vec<CaptureBreak>,
    pub self_interval_break_count: usize,
    pub self_pick_outlier_count: usize,
    pub silent_drop_count: usize,
}

impl CaptureContinuity {
    /// True when no break reaches inside the span. A self-interval break
    /// that only touches the span at a shared self pick does not count: the
    /// pick itself is good, the discontinuity is on the far side of it.
    /// Timeline breaks (not pick outliers) entirely before `sample`: the
    /// index of the continuous segment `sample` lies in.
    pub fn timeline_breaks_before(&self, sample: f64) -> usize {
        self.breaks.iter().filter(|gap| gap.kind != BreakKind::SelfPickOutlier && gap.end_sample <= sample).count()
    }

    pub fn span_is_continuous(&self, first_sample: f64, last_sample: f64) -> bool {
        self.breaks.iter().all(|gap| gap.end_sample <= first_sample || gap.start_sample >= last_sample)
    }
}

pub fn capture_continuity(self_picks: &[SelfPick], silent_drops: &[SilentCaptureDrop], tolerance_samples: f64, sample_rate_hz: f64) -> CaptureContinuity {
    let detected: Vec<(f64, f64)> = self_picks
        .iter()
        .filter_map(|pick| pick.arrival.map(|arrival| (pick.output_frame_index as f64, arrival.sample_index)))
        .collect();
    let ratios: Vec<f64> = detected
        .windows(2)
        .filter(|pair| pair[1].0 > pair[0].0)
        .map(|pair| (pair[1].1 - pair[0].1) / (pair[1].0 - pair[0].0))
        .collect();
    let mut steps: Vec<CaptureBreak> = Vec::new();
    if let Some(ratio) = median_of(&ratios) {
        for pair in detected.windows(2).filter(|pair| pair[1].0 > pair[0].0) {
            let discrepancy_samples = (pair[1].1 - pair[0].1) - ratio * (pair[1].0 - pair[0].0);
            if discrepancy_samples.abs() > tolerance_samples {
                steps.push(CaptureBreak { kind: BreakKind::SelfIntervalStep, start_sample: pair[0].1, end_sample: pair[1].1, discrepancy_samples });
            }
        }
    }
    let mut breaks = Vec::new();
    let mut index = 0;
    while index < steps.len() {
        let step = steps[index];
        let cancelling_next = steps.get(index + 1).filter(|next| next.start_sample == step.end_sample && (next.discrepancy_samples + step.discrepancy_samples).abs() <= tolerance_samples);
        match cancelling_next {
            Some(next) => {
                breaks.push(CaptureBreak { kind: BreakKind::SelfPickOutlier, start_sample: step.start_sample, end_sample: next.end_sample, discrepancy_samples: step.discrepancy_samples });
                index += 2;
            }
            None => {
                breaks.push(step);
                index += 1;
            }
        }
    }
    let count_of = |kind: BreakKind| breaks.iter().filter(|gap| gap.kind == kind).count();
    let (self_interval_break_count, self_pick_outlier_count) = (count_of(BreakKind::SelfIntervalStep), count_of(BreakKind::SelfPickOutlier));
    for drop in silent_drops {
        breaks.push(CaptureBreak { kind: BreakKind::SilentDrop, start_sample: drop.capture_frame, end_sample: drop.capture_frame, discrepancy_samples: drop.missing_ns * 1e-9 * sample_rate_hz });
    }
    CaptureContinuity { breaks, self_interval_break_count, self_pick_outlier_count, silent_drop_count: silent_drops.len() }
}

fn sample_span(seconds: &[f64], sample_rate_hz: f64) -> (f64, f64) {
    let samples = seconds.iter().map(|s| s * sample_rate_hz);
    (samples.clone().fold(f64::INFINITY, f64::min), samples.fold(f64::NEG_INFINITY, f64::max))
}

/// Rounds whose picks on both phones lie in continuous capture, each
/// tagged with its capture segment, and the number left out.
pub fn continuous_rounds(rounds: Vec<PairedRound>, a: &CaptureContinuity, b: &CaptureContinuity, sample_rate_hz: f64) -> (Vec<PairedRound>, usize) {
    let total = rounds.len();
    let kept: Vec<PairedRound> = rounds
        .into_iter()
        .filter_map(|mut round| {
            let times = &round.times;
            let mut on_a = vec![times.a_hears_a_s, times.a_hears_b_s];
            let mut on_b = vec![times.b_hears_a_s, times.b_hears_b_s];
            if let Some((a_repeat, b_repeat)) = round.repeat {
                on_a.push(a_repeat);
                on_b.push(b_repeat);
            }
            let (a_first, a_last) = sample_span(&on_a, sample_rate_hz);
            let (b_first, b_last) = sample_span(&on_b, sample_rate_hz);
            let continuous = a.span_is_continuous(a_first, a_last) && b.span_is_continuous(b_first, b_last);
            // Both counts only grow with time, so their sum numbers the segments.
            round.capture_segment = a.timeline_breaks_before(a_first) + b.timeline_breaks_before(b_first);
            continuous.then_some(round)
        })
        .collect();
    let excluded = total - kept.len();
    (kept, excluded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrival_scan::ScannedArrival;

    fn pick(index: u64, capture_sample: f64) -> SelfPick {
        SelfPick {
            emission_index: index,
            output_frame_index: 96_000 + index * 192_000,
            emission_t_ns: 0,
            predicted_sample: None,
            channel: None,
            arrival: Some(ScannedArrival::clean(capture_sample, 60.0)),
        }
    }

    #[test]
    fn a_capture_drop_between_two_self_chirps_is_a_break_and_spoils_only_overlapping_rounds() {
        // 8496 samples lost between emissions 11 and 12 (the 2026-11-26 Pixel 10 stall).
        let picks: Vec<SelfPick> = (0..20u64).map(|k| pick(k, 107_080.0 + k as f64 * 192_000.0 - if k >= 12 { 8_496.0 } else { 0.0 })).collect();
        let continuity = capture_continuity(&picks, &[], 2.0, 48_000.0);
        assert_eq!(continuity.self_interval_break_count, 1);
        let gap = continuity.breaks[0];
        assert!((gap.discrepancy_samples + 8_496.0).abs() < 1e-6);
        assert!(continuity.span_is_continuous(picks[2].arrival.unwrap().sample_index, picks[11].arrival.unwrap().sample_index));
        assert!(!continuity.span_is_continuous(picks[11].arrival.unwrap().sample_index - 10.0, picks[11].arrival.unwrap().sample_index + 10.0));
        assert!(continuity.span_is_continuous(picks[12].arrival.unwrap().sample_index + 1.0, picks[19].arrival.unwrap().sample_index));
    }

    #[test]
    fn a_single_hopped_self_pick_is_an_outlier_not_a_timeline_break() {
        let picks: Vec<SelfPick> = (0..20u64).map(|k| pick(k, 1_000.0 + k as f64 * 192_000.0 + if k == 7 { 11.5 } else { 0.0 })).collect();
        let continuity = capture_continuity(&picks, &[], 2.0, 48_000.0);
        assert_eq!((continuity.self_pick_outlier_count, continuity.self_interval_break_count), (1, 0));
        let hopped = picks[7].arrival.unwrap().sample_index;
        assert!(!continuity.span_is_continuous(hopped - 192_000.0, hopped));
        assert!(!continuity.span_is_continuous(hopped, hopped + 192_000.0));
        assert_eq!(continuity.timeline_breaks_before(hopped + 400_000.0), 0);
    }

    #[test]
    fn a_steady_rate_offset_is_not_a_break() {
        let picks: Vec<SelfPick> = (0..20u64).map(|k| pick(k, 1_000.0 + k as f64 * 192_000.0 * (1.0 + 3.4e-6))).collect();
        assert!(capture_continuity(&picks, &[], 2.0, 48_000.0).breaks.is_empty());
    }
}
