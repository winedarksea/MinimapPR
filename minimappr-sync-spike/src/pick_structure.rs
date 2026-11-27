//! Arrival-pick structure from the detector's own diagnostics (no
//! re-correlation): how close each pick sat to the earliest-peak acceptance
//! level, whether the pick lead flips between copies, how often a self
//! precursor was rejected, and whether d_ii puts the precursor-window edge
//! right on the measured precursor lead (the 2026-11-26 soak failure: a
//! precursor just outside the window is picked as the airborne arrival).

use minimappr_core::sync::first_arrival::SOUND_SPEED_IN_PHONE_BODY_MPS;

use crate::arrival_scan::ScannedArrival;
use crate::pair_ranging_series::median_of;

/// Earliest-peak acceptance of the core detector (`first_arrival_relative_level_db`).
pub const FIRST_ARRIVAL_ACCEPTANCE_DB: f64 = -10.0;
/// A pick within this of the acceptance level can flip with a few dB of level change.
const MARGIN_AT_RISK_DB: f64 = 3.0;
/// A precursor lead this close to the rejection-window edge flips with d_ii or noise.
pub const WINDOW_EDGE_WARNING_SAMPLES: f64 = 3.0;
/// A lead bin holding at least this share of picks counts as a mode.
const LEAD_MODE_MINIMUM_SHARE: f64 = 0.1;
/// Two lead modes further apart than this are distinct copies, not jitter.
const LEAD_MODE_MINIMUM_SEPARATION_SAMPLES: i64 = 2;

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct PickStructure {
    pub count: usize,
    pub median_margin_db: Option<f64>,
    /// Picks that were not the strongest peak, sitting within 3 dB of the
    /// −10 dB acceptance: a few dB of level change flips them.
    pub margin_at_risk_count: usize,
    /// Picks earlier or later than the strongest peak (a copy was chosen).
    pub not_strongest_count: usize,
    /// (lead before strongest, rounded samples, count), most common first.
    pub lead_histogram: Vec<(i64, usize)>,
    /// Two or more lead modes > 2 samples apart: the pick flips between copies.
    pub lead_bimodal: bool,
    pub precursor_rejected_count: usize,
    pub double_arrival_count: usize,
    /// Self only: picks whose precursor lead lies within ±3 samples of the
    /// rejection-window edge for their d_ii.
    pub window_edge_count: usize,
    pub median_precursor_lead_samples: Option<f64>,
    /// Self only: d_ii the median precursor lead implies,
    /// lead / fs ÷ (1/c − 1/c_body).
    pub suggested_self_path_m: Option<f64>,
}

/// One pick and, for a self pick, its precursor rejection window (samples).
pub struct PickWithWindow {
    pub arrival: ScannedArrival,
    pub precursor_window_samples: Option<f64>,
}

/// Body-path lead over the airborne self path: (1/c − 1/c_body) per metre.
fn lead_seconds_per_metre(speed_of_sound_mps: f64) -> f64 {
    1.0 / speed_of_sound_mps - 1.0 / SOUND_SPEED_IN_PHONE_BODY_MPS
}

/// The precursor's lag behind/ahead of the airborne path, when the pick
/// shows one: a rejected precursor, or (self only) a pick *earlier* than the
/// strongest peak, which is a precursor that fell outside the window.
fn precursor_lead(pick: &PickWithWindow) -> Option<f64> {
    pick.arrival.rejected_precursor_lead_samples.or_else(|| {
        (pick.precursor_window_samples.is_some() && pick.arrival.lead_before_strongest_samples > 0.0).then_some(pick.arrival.lead_before_strongest_samples)
    })
}

pub fn pick_structure(picks: &[PickWithWindow], speed_of_sound_mps: f64, sample_rate_hz: f64) -> PickStructure {
    if picks.is_empty() {
        return PickStructure::default();
    }
    let margins: Vec<f64> = picks.iter().map(|p| p.arrival.first_arrival_margin_db).collect();
    let mut histogram: Vec<(i64, usize)> = Vec::new();
    for pick in picks {
        let bin = pick.arrival.lead_before_strongest_samples.round() as i64;
        match histogram.iter_mut().find(|(lead, _)| *lead == bin) {
            Some((_, count)) => *count += 1,
            None => histogram.push((bin, 1)),
        }
    }
    histogram.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let minimum_mode_count = ((picks.len() as f64 * LEAD_MODE_MINIMUM_SHARE).ceil() as usize).max(2);
    let modes: Vec<i64> = histogram.iter().filter(|(_, count)| *count >= minimum_mode_count).map(|(lead, _)| *lead).collect();
    let lead_bimodal = modes.iter().any(|a| modes.iter().any(|b| (a - b).abs() > LEAD_MODE_MINIMUM_SEPARATION_SAMPLES));
    let precursor_leads: Vec<f64> = picks.iter().filter_map(precursor_lead).map(f64::abs).collect();
    let window_edge_count = picks
        .iter()
        .filter(|pick| match (precursor_lead(pick), pick.precursor_window_samples) {
            (Some(lead), Some(window)) => (lead.abs() - window).abs() <= WINDOW_EDGE_WARNING_SAMPLES,
            _ => false,
        })
        .count();
    let median_precursor_lead_samples = median_of(&precursor_leads);
    let is_self = picks.iter().any(|p| p.precursor_window_samples.is_some());
    PickStructure {
        count: picks.len(),
        median_margin_db: median_of(&margins),
        margin_at_risk_count: margins.iter().filter(|m| **m < 0.0 && **m <= FIRST_ARRIVAL_ACCEPTANCE_DB + MARGIN_AT_RISK_DB).count(),
        not_strongest_count: picks.iter().filter(|p| p.arrival.lead_before_strongest_samples.abs() >= 0.5).count(),
        lead_histogram: histogram,
        lead_bimodal,
        precursor_rejected_count: picks.iter().filter(|p| p.arrival.precursor_rejected).count(),
        double_arrival_count: picks.iter().filter(|p| p.arrival.double_arrival_suspected).count(),
        window_edge_count,
        median_precursor_lead_samples,
        suggested_self_path_m: median_precursor_lead_samples
            .filter(|_| is_self)
            .map(|lead| lead / sample_rate_hz / lead_seconds_per_metre(speed_of_sound_mps)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(lead: f64, margin_db: f64, rejected: Option<f64>, window: Option<f64>) -> PickWithWindow {
        let mut arrival = ScannedArrival::clean(0.0, 40.0);
        arrival.lead_before_strongest_samples = lead;
        arrival.first_arrival_margin_db = margin_db;
        arrival.rejected_precursor_lead_samples = rejected;
        arrival.precursor_rejected = rejected.is_some();
        PickWithWindow { arrival, precursor_window_samples: window }
    }

    #[test]
    fn flipping_remote_copies_are_bimodal_and_at_risk() {
        let picks: Vec<_> = (0..20).map(|i| if i % 2 == 0 { pick(0.0, 0.0, None, None) } else { pick(21.0, -9.0, None, None) }).collect();
        let structure = pick_structure(&picks, 343.0, 48_000.0);
        assert!(structure.lead_bimodal, "{structure:?}");
        assert_eq!(structure.margin_at_risk_count, 10);
        assert_eq!(structure.not_strongest_count, 10);
        assert_eq!(structure.suggested_self_path_m, None, "remote picks imply no d_ii");
    }

    #[test]
    fn earpiece_precursor_lead_suggests_d_ii_and_flags_the_window_edge() {
        // 0.16 m at 343 m/s against 5000 m/s ≈ 20.9 samples at 48 kHz.
        let lead = 0.16 * lead_seconds_per_metre(343.0) * 48_000.0;
        let picks: Vec<_> = (0..10).map(|_| pick(0.0, 0.0, Some(lead), Some(lead + 2.0))).collect();
        let structure = pick_structure(&picks, 343.0, 48_000.0);
        assert!((structure.suggested_self_path_m.unwrap() - 0.16).abs() < 1e-9);
        assert_eq!(structure.window_edge_count, 10);
        assert!(!structure.lead_bimodal);
        let far_window: Vec<_> = (0..10).map(|_| pick(0.0, 0.0, Some(lead), Some(lead + 8.0))).collect();
        assert_eq!(pick_structure(&far_window, 343.0, 48_000.0).window_edge_count, 0);
    }
}
