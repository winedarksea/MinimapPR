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

/// `points`: (A-clock time s, θ s), time-ordered.
pub fn clock_residual_report(points: &[(f64, f64)], source_mode: &str, heartbeat_cadence_s: f64) -> ClockResidualReport {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_drift_has_tiny_linear_residuals_and_recovers_rate() {
        let points: Vec<(f64, f64)> = (0..300).map(|k| (k as f64 * 4.0, 0.25 + 42e-6 * k as f64 * 4.0)).collect();
        let report = clock_residual_report(&points, "double_sided", 90.0);
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
