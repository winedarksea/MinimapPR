//! How well the B − A audio-clock offset θ(t) is tracked: residuals of
//! linear and quadratic fits over the whole run (crystal drift and its
//! thermal curvature), and the one-step prediction residual of the core
//! `PhoneClockFilter` fed at every round and at the 90 s heartbeat cadence,
//! which is what a call between heartbeat rounds would actually see
//! (Phase 1 gate: < 50 µs over 30 min).

use minimappr_core::sync::clock_filter::{ClockFilterConfig, ClockUpdateOutcome, PhoneClockFilter};
use minimappr_core::sync::clock_graph::ClockGraphConfig;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ResidualSummary {
    pub count: usize,
    pub rms_s: f64,
    pub p95_abs_s: f64,
    pub max_abs_s: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ClockResidualReport {
    pub source_mode: String,
    pub duration_s: f64,
    pub rate_offset_ppm: Option<f64>,
    pub linear_fit: Option<ResidualSummary>,
    pub quadratic_fit: Option<ResidualSummary>,
    pub filter_every_round: Option<ResidualSummary>,
    pub filter_heartbeat_cadence: Option<ResidualSummary>,
    pub heartbeat_cadence_s: f64,
    pub filter_outliers_rejected: usize,
    pub filter_silent_gaps: usize,
    /// θ steps removed at capture breaks before fitting (see `stitch_segments`).
    pub segment_steps_removed: usize,
}

pub fn summarize(residuals: &[f64]) -> Option<ResidualSummary> {
    if residuals.is_empty() {
        return None;
    }
    let mut absolute: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    absolute.sort_by(f64::total_cmp);
    let p95_index = ((absolute.len() as f64 * 0.95).ceil() as usize).clamp(1, absolute.len()) - 1;
    Some(ResidualSummary {
        count: residuals.len(),
        rms_s: (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt(),
        p95_abs_s: absolute[p95_index],
        max_abs_s: *absolute.last().expect("non-empty"),
    })
}

/// Least-squares polynomial (degree 1 or 2) residuals via normal equations
/// on centred, scaled time (well conditioned for 30-minute runs).
pub fn polynomial_fit_residuals(points: &[(f64, f64)], degree: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let terms = degree + 1;
    if points.len() <= terms {
        return None;
    }
    let mean_t = points.iter().map(|p| p.0).sum::<f64>() / points.len() as f64;
    let scale = points.iter().map(|p| (p.0 - mean_t).abs()).fold(0.0, f64::max).max(1e-9);
    let mut normal = vec![vec![0.0; terms + 1]; terms];
    for (t, y) in points {
        let x = (t - mean_t) / scale;
        let powers: Vec<f64> = (0..terms).map(|p| x.powi(p as i32)).collect();
        for row in 0..terms {
            for column in 0..terms {
                normal[row][column] += powers[row] * powers[column];
            }
            normal[row][terms] += powers[row] * y;
        }
    }
    // Gaussian elimination with partial pivoting; a singular system (all
    // rounds at one instant) yields None instead of NaNs.
    for pivot in 0..terms {
        let best = (pivot..terms).max_by(|a, b| normal[*a][pivot].abs().total_cmp(&normal[*b][pivot].abs()))?;
        normal.swap(pivot, best);
        if normal[pivot][pivot].abs() < 1e-12 {
            return None;
        }
        for row in 0..terms {
            if row != pivot {
                let factor = normal[row][pivot] / normal[pivot][pivot];
                for column in pivot..=terms {
                    normal[row][column] -= factor * normal[pivot][column];
                }
            }
        }
    }
    let coefficients: Vec<f64> = (0..terms).map(|row| normal[row][terms] / normal[row][row]).collect();
    let residuals = points
        .iter()
        .map(|(t, y)| {
            let x = (t - mean_t) / scale;
            y - coefficients.iter().enumerate().map(|(p, c)| c * x.powi(p as i32)).sum::<f64>()
        })
        .collect();
    // Report the physical slope (s/s) as the second coefficient.
    let physical: Vec<f64> = coefficients.iter().enumerate().map(|(p, c)| c / scale.powi(p as i32)).collect();
    Some((residuals, physical))
}

/// Removes the steps between continuous segments of a series that is
/// linear within each segment (θ(t) across silent capture drops, or self
/// onsets across playback slips): one common slope, one intercept per
/// segment, and every segment shifted onto the first one's intercept.
/// Points in a segment of fewer than `MINIMUM_SEGMENT_POINTS` are dropped
/// (their intercept is not determined). Returns stitched points in order
/// and the number of steps removed.
pub fn stitch_segments(points: &[(f64, f64)], segment_ids: &[usize]) -> (Vec<(f64, f64)>, usize) {
    const MINIMUM_SEGMENT_POINTS: usize = 2;
    let mut ids: Vec<usize> = segment_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let members = |id: usize| points.iter().zip(segment_ids).filter(move |(_, s)| **s == id).map(|(p, _)| *p);
    let usable: Vec<usize> = ids.into_iter().filter(|id| members(*id).count() >= MINIMUM_SEGMENT_POINTS).collect();
    let means: Vec<(usize, f64, f64)> = usable
        .iter()
        .map(|id| {
            let count = members(*id).count() as f64;
            (*id, members(*id).map(|p| p.0).sum::<f64>() / count, members(*id).map(|p| p.1).sum::<f64>() / count)
        })
        .collect();
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (id, mean_t, mean_y) in &means {
        for (t, y) in members(*id) {
            sxy += (t - mean_t) * (y - mean_y);
            sxx += (t - mean_t).powi(2);
        }
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept_of = |id: usize| means.iter().find(|m| m.0 == id).map(|(_, mean_t, mean_y)| mean_y - slope * mean_t);
    let Some(reference) = means.first().map(|m| m.0).and_then(intercept_of) else { return (Vec::new(), 0) };
    let stitched = points
        .iter()
        .zip(segment_ids)
        .filter_map(|(point, id)| intercept_of(*id).map(|intercept| (point.0, point.1 - (intercept - reference))))
        .collect();
    (stitched, means.len().saturating_sub(1))
}

/// One-step-ahead prediction residuals: predict at each point before
/// updating with it (the first points only seed the filter).
fn filter_prediction_residuals(points: &[(f64, f64)], pick_sigma_s: f64) -> (Vec<f64>, usize, usize) {
    let mut filter = PhoneClockFilter::new(ClockFilterConfig::default());
    let mut residuals = Vec::new();
    let (mut rejected, mut silent_gaps) = (0, 0);
    for (t, offset) in points {
        if filter.has_valid_rate() {
            if let Some(prediction) = filter.predict(*t) {
                residuals.push(offset - prediction.offset_s);
            }
        }
        match filter.update(*t, *offset, pick_sigma_s) {
            ClockUpdateOutcome::RejectedOutlier { .. } => rejected += 1,
            ClockUpdateOutcome::SilentGapDetected { .. } => silent_gaps += 1,
            _ => {}
        }
    }
    (residuals, rejected, silent_gaps)
}

fn subsample_at_cadence(points: &[(f64, f64)], cadence_s: f64) -> Vec<(f64, f64)> {
    let mut kept: Vec<(f64, f64)> = Vec::new();
    for point in points {
        if kept.last().is_none_or(|last| point.0 - last.0 >= cadence_s) {
            kept.push(*point);
        }
    }
    kept
}

/// `points`: (A-clock time s, θ s), time-ordered; `segment_ids`: the
/// continuous capture stretch of each point (θ steps between them).
pub fn clock_residual_report(points: &[(f64, f64)], segment_ids: &[usize], source_mode: &str, heartbeat_cadence_s: f64) -> ClockResidualReport {
    let (stitched, segment_steps_removed) = stitch_segments(points, segment_ids);
    let points = &stitched[..];
    let pick_sigma_s = ClockGraphConfig::default().pick_offset_sigma_s;
    let linear = polynomial_fit_residuals(points, 1);
    let quadratic = polynomial_fit_residuals(points, 2);
    let (every_round, rejected, silent_gaps) = filter_prediction_residuals(points, pick_sigma_s);
    let (heartbeat, _, _) = filter_prediction_residuals(&subsample_at_cadence(points, heartbeat_cadence_s), pick_sigma_s);
    ClockResidualReport {
        source_mode: source_mode.into(),
        duration_s: match (points.first(), points.last()) {
            (Some(first), Some(last)) => last.0 - first.0,
            _ => 0.0,
        },
        rate_offset_ppm: linear.as_ref().map(|(_, coefficients)| coefficients[1] * 1e6),
        linear_fit: linear.and_then(|(residuals, _)| summarize(&residuals)),
        quadratic_fit: quadratic.and_then(|(residuals, _)| summarize(&residuals)),
        filter_every_round: summarize(&every_round),
        filter_heartbeat_cadence: summarize(&heartbeat),
        heartbeat_cadence_s,
        filter_outliers_rejected: rejected,
        filter_silent_gaps: silent_gaps,
        segment_steps_removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_theta_step_at_a_capture_break_is_stitched_out() {
        let points: Vec<(f64, f64)> = (0..100).map(|k| (k as f64 * 4.0, 0.25 + 12e-6 * k as f64 * 4.0 + if k >= 40 { 0.527 } else { 0.0 })).collect();
        let segments: Vec<usize> = (0..100).map(|k| usize::from(k >= 40)).collect();
        let report = clock_residual_report(&points, &segments, "double_sided", 90.0);
        assert_eq!(report.segment_steps_removed, 1);
        assert!((report.rate_offset_ppm.unwrap() - 12.0).abs() < 1e-6);
        assert!(report.linear_fit.unwrap().rms_s < 1e-12);
    }

    #[test]
    fn linear_drift_has_tiny_linear_residuals_and_recovers_rate() {
        let points: Vec<(f64, f64)> = (0..300).map(|k| (k as f64 * 4.0, 0.25 + 42e-6 * k as f64 * 4.0)).collect();
        let report = clock_residual_report(&points, &vec![0; points.len()], "double_sided", 90.0);
        assert!((report.rate_offset_ppm.unwrap() - 42.0).abs() < 1e-6);
        assert!(report.linear_fit.unwrap().rms_s < 1e-12);
        assert!(report.filter_every_round.unwrap().rms_s < 1e-6);
    }

    #[test]
    fn quadratic_fit_absorbs_curvature() {
        let points: Vec<(f64, f64)> = (0..200).map(|k| {
            let t = k as f64 * 9.0;
            (t, 1e-5 * t + 2e-9 * t * t)
        }).collect();
        let linear = polynomial_fit_residuals(&points, 1).unwrap().0;
        let quadratic = polynomial_fit_residuals(&points, 2).unwrap().0;
        assert!(summarize(&quadratic).unwrap().rms_s < 1e-9);
        assert!(summarize(&linear).unwrap().rms_s > 1e-5);
    }
}
