//! Two-way ranging per round with the three rate compensations, the
//! speaker/mic offset correction when the pose is known, and the d_ii sum
//! a tape-measured run implies.
//!
//! Rate compensations compared (plan, A5): None (nominal rates), the rate
//! ratio a filter would track (here: the slope of the whole run's offset
//! series, i.e. a well-converged filter), and symmetric double-sided
//! ranging using A's next chirp (rate cancels inside the exchange).

use minimappr_core::sync::two_way_ranging::{
    correct_for_speaker_mic_offsets, range_pair, RateCompensation, SelfPathLengths, TwoWayRangingResult,
};

use crate::round_pairing::PairedRound;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompensationMode {
    None,
    TrackedRateRatio,
    DoubleSided,
}

impl CompensationMode {
    pub const ALL: [CompensationMode; 3] = [Self::None, Self::TrackedRateRatio, Self::DoubleSided];

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TrackedRateRatio => "tracked_rate_ratio",
            Self::DoubleSided => "double_sided",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangedRound {
    pub round: PairedRound,
    /// Indexed like `CompensationMode::ALL`.
    pub results: [Option<TwoWayRangingResult>; 3],
    /// Distance with d_ii = 0 (for the tape calibration).
    pub distance_without_self_paths_m: Option<f64>,
}

impl RangedRound {
    pub fn result(&self, mode: CompensationMode) -> Option<&TwoWayRangingResult> {
        let index = CompensationMode::ALL.iter().position(|m| *m == mode).expect("listed mode");
        self.results[index].as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerGeometry {
    pub mic_a_m: [f64; 3],
    pub mic_b_m: [f64; 3],
    pub speaker_offset_a_world_m: [f64; 3],
    pub speaker_offset_b_world_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DistanceStatistics {
    pub mode: String,
    pub count: usize,
    pub median_m: f64,
    pub robust_sigma_m: f64,
    pub standard_deviation_m: f64,
    pub outlier_count: usize,
    pub error_vs_tape_m: Option<f64>,
}

/// Slope of θ(t) from uncompensated rounds: (1 + slope) = B ticks per A tick.
pub fn offset_series_rate_ratio(rounds: &[PairedRound], self_paths: SelfPathLengths, speed_of_sound_mps: f64) -> Option<f64> {
    let points: Vec<(f64, f64)> = rounds
        .iter()
        .filter_map(|round| {
            let result = range_pair(&round.times, self_paths, speed_of_sound_mps, RateCompensation::None).ok()?;
            Some((result.offset_reference_time_on_a_s, result.clock_offset_b_minus_a_s))
        })
        .collect();
    if points.len() < 2 {
        return None;
    }
    let count = points.len() as f64;
    let mean_t = points.iter().map(|p| p.0).sum::<f64>() / count;
    let mean_o = points.iter().map(|p| p.1).sum::<f64>() / count;
    let stt: f64 = points.iter().map(|p| (p.0 - mean_t).powi(2)).sum();
    let sto: f64 = points.iter().map(|p| (p.0 - mean_t) * (p.1 - mean_o)).sum();
    (stt > 0.0).then(|| 1.0 + sto / stt)
}

pub fn range_rounds(
    rounds: &[PairedRound],
    self_paths: SelfPathLengths,
    speed_of_sound_mps: f64,
    speaker_geometry: Option<SpeakerGeometry>,
) -> Vec<RangedRound> {
    let tracked_ratio = offset_series_rate_ratio(rounds, self_paths, speed_of_sound_mps);
    let correct = |result: TwoWayRangingResult| match speaker_geometry {
        Some(geometry) => correct_for_speaker_mic_offsets(
            &result,
            geometry.mic_a_m,
            geometry.mic_b_m,
            geometry.speaker_offset_a_world_m,
            geometry.speaker_offset_b_world_m,
            speed_of_sound_mps,
        ),
        None => result,
    };
    rounds
        .iter()
        .map(|round| {
            let compensation_for = |mode: CompensationMode| -> Option<RateCompensation> {
                match mode {
                    CompensationMode::None => Some(RateCompensation::None),
                    CompensationMode::TrackedRateRatio => tracked_ratio.map(|ratio| RateCompensation::TrackedRateRatio { b_ticks_per_a_tick: ratio }),
                    CompensationMode::DoubleSided => round.repeat.map(|(a_repeat, b_repeat)| RateCompensation::DoubleSided {
                        a_hears_a_repeat_s: a_repeat,
                        b_hears_a_repeat_s: b_repeat,
                    }),
                }
            };
            let results = CompensationMode::ALL.map(|mode| {
                compensation_for(mode).and_then(|compensation| range_pair(&round.times, self_paths, speed_of_sound_mps, compensation).ok().map(correct))
            });
            let zero_paths = SelfPathLengths { a_m: 0.0, b_m: 0.0 };
            let best_compensation = compensation_for(CompensationMode::DoubleSided).or(compensation_for(CompensationMode::TrackedRateRatio)).unwrap_or(RateCompensation::None);
            let distance_without_self_paths_m = range_pair(&round.times, zero_paths, speed_of_sound_mps, best_compensation).ok().map(|r| r.distance_m);
            RangedRound { round: round.clone(), results, distance_without_self_paths_m }
        })
        .collect()
}

pub fn median_of(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 { 0.5 * (sorted[middle - 1] + sorted[middle]) } else { sorted[middle] })
}

/// MAD × 1.4826: σ of a normal distribution, immune to a few gross picks.
pub fn robust_sigma(values: &[f64]) -> Option<f64> {
    let centre = median_of(values)?;
    let deviations: Vec<f64> = values.iter().map(|v| (v - centre).abs()).collect();
    median_of(&deviations).map(|mad| 1.4826 * mad)
}

pub fn distance_statistics(ranged: &[RangedRound], mode: CompensationMode, tape_distance_m: Option<f64>, outlier_threshold_m: f64) -> Option<DistanceStatistics> {
    let distances: Vec<f64> = ranged.iter().filter_map(|r| r.result(mode).map(|res| res.distance_m)).collect();
    let median_m = median_of(&distances)?;
    let inliers: Vec<f64> = distances.iter().copied().filter(|d| (d - median_m).abs() <= outlier_threshold_m).collect();
    let mean = inliers.iter().sum::<f64>() / inliers.len().max(1) as f64;
    let standard_deviation_m = (inliers.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / inliers.len().max(1) as f64).sqrt();
    Some(DistanceStatistics {
        mode: mode.name().into(),
        count: distances.len(),
        median_m,
        robust_sigma_m: robust_sigma(&inliers).unwrap_or(0.0),
        standard_deviation_m,
        outlier_count: distances.len() - inliers.len(),
        error_vs_tape_m: tape_distance_m.map(|tape| median_m - tape),
    })
}

/// (d_AA + d_BB)/2 a tape-measured run implies: only the sum is observable
/// from two-way ranging (plan A3).
pub fn implied_mean_self_path_m(ranged: &[RangedRound], tape_distance_m: f64) -> Option<f64> {
    let implied: Vec<f64> = ranged.iter().filter_map(|r| r.distance_without_self_paths_m.map(|d| tape_distance_m - d)).collect();
    median_of(&implied)
}
