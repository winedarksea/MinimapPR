//! Self-onset stability (plan A2): the self chirp's capture sample against
//! the exact output frame it was written at. Playback and capture usually
//! share the codec crystal, so a line fit removes the fixed loopback latency
//! and any rate difference; what remains is pick jitter plus real onset
//! shifts. Splitting by marker (foam → hand → table) shows whether the
//! placement moves the onset (body precursor, limiter), which would bias
//! every two-way range by the shift × c / 2.

use crate::capture_continuity::CaptureContinuity;
use crate::clock_residuals::{polynomial_fit_residuals, stitch_segments};
use crate::pair_ranging_series::{median_of, robust_sigma};
use crate::round_pairing::SelfPick;

const MICROSECONDS_PER_SECOND: f64 = 1e6;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct OnsetSegment {
    pub label: String,
    pub count: usize,
    pub mean_residual_us: f64,
    pub sigma_us: f64,
    pub robust_sigma_us: f64,
    pub median_self_snr_db: Option<f64>,
    pub max_peak_fraction: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct OnsetStabilityReport {
    pub detected: usize,
    pub missed: usize,
    pub overall_sigma_us: Option<f64>,
    pub loopback_rate_offset_ppm: Option<f64>,
    pub segments: Vec<OnsetSegment>,
    /// Largest |mean(segment) − mean(first segment)|.
    pub max_segment_shift_us: Option<f64>,
    /// Self picks with a peak above 0.98 full scale (limiter/clipping).
    pub clipped_count: usize,
}

const CLIPPED_PEAK_FRACTION: f32 = 0.98;
/// Label of the stretch before the first marker.
const UNMARKED_SEGMENT_LABEL: &str = "before_first_marker";

/// A playback/capture break shifts every later onset by the slip, which is
/// a timeline fault (reported by `capture_continuity`), not onset jitter:
/// the fit stitches the segments between breaks first.
pub fn onset_stability(self_picks: &[SelfPick], markers: &[(i64, String)], continuity: &CaptureContinuity, sample_rate_hz: f64) -> OnsetStabilityReport {
    let detected: Vec<&SelfPick> = self_picks.iter().filter(|pick| pick.arrival.is_some()).collect();
    let detected_count = detected.len();
    let raw_points: Vec<(f64, f64)> = detected
        .iter()
        .map(|pick| (pick.output_frame_index as f64, pick.arrival.expect("filtered").sample_index))
        .collect();
    let segment_ids: Vec<usize> = raw_points.iter().map(|(_, sample)| continuity.timeline_breaks_before(*sample)).collect();
    let (points, _) = stitch_segments(&raw_points, &segment_ids);
    // A one-pick segment has no intercept and is dropped by the stitch.
    let detected: Vec<&SelfPick> = if points.len() == detected.len() {
        detected
    } else {
        let kept: Vec<f64> = points.iter().map(|p| p.0).collect();
        detected.into_iter().filter(|pick| kept.contains(&(pick.output_frame_index as f64))).collect()
    };
    let fit = polynomial_fit_residuals(&points, 1);
    let clipped_count = detected.iter().filter(|pick| pick.arrival.is_some_and(|a| a.peak_fraction >= CLIPPED_PEAK_FRACTION)).count();
    let Some((residual_samples, coefficients)) = fit else {
        return OnsetStabilityReport {
            detected: detected_count,
            missed: self_picks.len() - detected_count,
            overall_sigma_us: None,
            loopback_rate_offset_ppm: None,
            segments: Vec::new(),
            max_segment_shift_us: None,
            clipped_count,
        };
    };
    let residual_us: Vec<f64> = residual_samples.iter().map(|r| r / sample_rate_hz * MICROSECONDS_PER_SECOND).collect();
    let mut sorted_markers = markers.to_vec();
    sorted_markers.sort_by_key(|(t_ns, _)| *t_ns);
    let segment_of = |t_ns: i64| -> String {
        sorted_markers.iter().rev().find(|(marker_ns, _)| *marker_ns <= t_ns).map_or(UNMARKED_SEGMENT_LABEL.to_string(), |(_, label)| label.clone())
    };
    let mut labels: Vec<String> = Vec::new();
    for pick in &detected {
        let label = segment_of(pick.emission_t_ns);
        if !labels.contains(&label) {
            labels.push(label);
        }
    }
    let segments: Vec<OnsetSegment> = labels
        .iter()
        .map(|label| {
            let members: Vec<usize> = detected.iter().enumerate().filter(|(_, p)| &segment_of(p.emission_t_ns) == label).map(|(i, _)| i).collect();
            let values: Vec<f64> = members.iter().map(|i| residual_us[*i]).collect();
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            let snrs: Vec<f64> = members.iter().map(|i| detected[*i].arrival.expect("filtered").snr_db).collect();
            OnsetSegment {
                label: label.clone(),
                count: values.len(),
                mean_residual_us: mean,
                sigma_us: (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len().max(1) as f64).sqrt(),
                robust_sigma_us: robust_sigma(&values).unwrap_or(0.0),
                median_self_snr_db: median_of(&snrs),
                max_peak_fraction: members.iter().map(|i| detected[*i].arrival.expect("filtered").peak_fraction).fold(0.0, f32::max),
            }
        })
        .collect();
    let max_segment_shift_us = segments.first().map(|first| segments.iter().map(|s| (s.mean_residual_us - first.mean_residual_us).abs()).fold(0.0, f64::max));
    OnsetStabilityReport {
        detected: detected_count,
        missed: self_picks.len() - detected_count,
        overall_sigma_us: robust_sigma(&residual_us),
        loopback_rate_offset_ppm: Some((coefficients[1] - 1.0) * 1e6),
        segments,
        max_segment_shift_us,
        clipped_count,
    }
}
