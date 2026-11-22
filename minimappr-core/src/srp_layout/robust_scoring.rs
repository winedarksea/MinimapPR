//! Huber-weighted SRP score over coherence curves.
//!
//! score(x) = −Σₚ wₚ · ρ_δ(1 − cₚ(τₚ(x)) / peakₚ)
//!
//! Each pair is normalized by its own peak so a pair with a loud reflection
//! cannot outvote quieter pairs, and ρ_δ (Huber) is quadratic near zero
//! residual (the score near the optimum is a sum of squares, which is what
//! the LM solver minimizes) and linear beyond the knee (bounded influence of
//! a pair locked onto a reflection). wₚ includes a significance factor so a
//! pair whose peak is no higher than its noise-only expectation adds nothing.

use super::config::RobustScoringConfig;
use super::pair_correlation::PairCoherenceCurve;
use super::polar_grid::{norm, sub, LayoutFrame, SearchCell};

/// Huber loss on a non-negative residual.
pub fn huber_loss(residual: f64, knee: f64) -> f64 {
    if residual <= knee {
        0.5 * residual * residual
    } else {
        knee * (residual - 0.5 * knee)
    }
}

/// Significance in [0, 1]: 0 at the expected noise-only maximum
/// √(2 ln L) / √K, 1 at `full_weight_significance_ratio` times that.
pub fn pair_significance(curve: &PairCoherenceCurve, upsample_factor: usize, full_weight_significance_ratio: f64) -> f64 {
    let independent_lag_count = (curve.values.len() / upsample_factor.max(1)).max(2) as f64;
    let noise_peak_z = (2.0 * independent_lag_count.ln()).sqrt();
    let peak_z = curve.peak_coherence as f64 * (curve.in_band_bin_count.max(1) as f64).sqrt();
    // Ratio ≤ 1 would divide by zero (NaN weights); treat it as a hard step.
    let ramp_width_z = noise_peak_z * (full_weight_significance_ratio - 1.0).max(f64::EPSILON);
    ((peak_z - noise_peak_z) / ramp_width_z).clamp(0.0, 1.0)
}

/// A coherence curve plus its sensor positions and final weight.
#[derive(Clone, Debug)]
pub struct WeightedPair {
    pub curve: PairCoherenceCurve,
    pub sensor_a_position_m: [f64; 3],
    pub sensor_b_position_m: [f64; 3],
    pub sensor_a_node: usize,
    pub sensor_b_node: usize,
    pub weight: f64,
}

impl WeightedPair {
    fn normalized_residual(&self, coherence: f32) -> f64 {
        let peak = self.curve.peak_coherence.max(f32::MIN_POSITIVE) as f64;
        (1.0 - coherence as f64 / peak).max(0.0)
    }
}

/// Which pairs a score sums over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairSubset {
    All,
    /// Pairs inside one rigid multi-mic node: their geometry is known to
    /// millimetres, unlike inter-node baselines from the layout solve.
    IntraNodeOnly,
}

impl PairSubset {
    fn includes(self, pair: &WeightedPair) -> bool {
        match self {
            Self::All => true,
            Self::IntraNodeOnly => pair.sensor_a_node == pair.sensor_b_node,
        }
    }
}

/// Envelope score of a cell plus the widest TDOA interval it spanned.
#[derive(Clone, Copy, Debug)]
pub struct CellScore {
    pub score: f64,
    pub widest_interval_half_width_s: f64,
}

/// Scores positions (exact) and cells (interval envelope) for one clip.
#[derive(Clone, Debug)]
pub struct RobustPairScorer {
    pub pairs: Vec<WeightedPair>,
    pub total_weight: f64,
    pub huber_knee: f64,
    pub sound_speed_mps: f64,
    pub likelihood_temperature: f64,
}

impl RobustPairScorer {
    pub fn new(pairs: Vec<WeightedPair>, config: &RobustScoringConfig, sound_speed_mps: f64) -> Self {
        let total_weight = pairs.iter().map(|pair| pair.weight).sum();
        Self {
            pairs,
            total_weight,
            huber_knee: config.huber_knee,
            sound_speed_mps,
            likelihood_temperature: config.likelihood_temperature,
        }
    }

    /// Score units per likelihood e-fold: temperature × Σw.
    pub fn likelihood_scale(&self) -> f64 {
        (self.likelihood_temperature * self.total_weight).max(f64::MIN_POSITIVE)
    }

    /// Smallest lag step among the curves (data resolution).
    pub fn finest_lag_step_s(&self) -> f64 {
        self.pairs.iter().map(|pair| pair.curve.lag_step_s).fold(f64::INFINITY, f64::min)
    }

    /// Exact score at a point source position.
    pub fn exact_score_at(&self, position_m: [f64; 3]) -> f64 {
        self.exact_point_score(position_m, PairSubset::All)
    }

    fn exact_point_score(&self, position_m: [f64; 3], subset: PairSubset) -> f64 {
        self.score_from_lags(subset, |pair| {
            (norm(sub(position_m, pair.sensor_a_position_m)) - norm(sub(position_m, pair.sensor_b_position_m)))
                / self.sound_speed_mps
        })
    }

    /// Exact score of a plane wave arriving from unit `direction`.
    fn exact_far_field_score(&self, direction: [f64; 3], subset: PairSubset) -> f64 {
        self.score_from_lags(subset, |pair| {
            let baseline = sub(pair.sensor_b_position_m, pair.sensor_a_position_m);
            (direction[0] * baseline[0] + direction[1] * baseline[1] + direction[2] * baseline[2]) / self.sound_speed_mps
        })
    }

    fn score_from_lags(&self, subset: PairSubset, lag_of_pair: impl Fn(&WeightedPair) -> f64) -> f64 {
        -self
            .pairs
            .iter()
            .filter(|pair| subset.includes(pair))
            .map(|pair| {
                let coherence = pair.curve.coherence_at(lag_of_pair(pair));
                pair.weight * huber_loss(pair.normalized_residual(coherence), self.huber_knee)
            })
            .sum::<f64>()
    }

    /// Envelope (upper-bound) score over a cell from explicit intervals.
    pub fn envelope_score_from_intervals(&self, intervals: impl Iterator<Item = (f64, f64)>) -> CellScore {
        let mut loss = 0.0;
        let mut widest: f64 = 0.0;
        for (pair, (center_lag_s, half_width_s)) in self.pairs.iter().zip(intervals) {
            let coherence = pair.curve.max_coherence_over_interval(center_lag_s, half_width_s);
            loss += pair.weight * huber_loss(pair.normalized_residual(coherence), self.huber_knee);
            widest = widest.max(half_width_s);
        }
        CellScore {
            score: -loss,
            widest_interval_half_width_s: widest,
        }
    }

    pub fn envelope_score(&self, cell: &SearchCell, frame: &LayoutFrame) -> CellScore {
        let geometry = cell.geometry(frame);
        self.envelope_score_from_intervals(self.pairs.iter().map(|pair| {
            geometry.tdoa_interval_s(pair.sensor_a_position_m, pair.sensor_b_position_m, self.sound_speed_mps)
        }))
    }

    /// Exact score at a cell's representative point (plane wave for far cells).
    pub fn exact_cell_score(&self, cell: &SearchCell, frame: &LayoutFrame) -> f64 {
        self.exact_cell_score_for(cell, frame, PairSubset::All)
    }

    pub fn exact_cell_score_for(&self, cell: &SearchCell, frame: &LayoutFrame, subset: PairSubset) -> f64 {
        match *cell {
            SearchCell::Near { .. } => self.exact_point_score(cell.representative_position_m(frame, 0.0), subset),
            SearchCell::FarField { azimuth_rad, elevation_rad } => self.exact_far_field_score(
                frame.direction_world(0.5 * (azimuth_rad[0] + azimuth_rad[1]), 0.5 * (elevation_rad[0] + elevation_rad[1])),
                subset,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber_is_continuous_and_linear_beyond_knee() {
        let knee = 0.3;
        assert!((huber_loss(knee, knee) - 0.5 * knee * knee).abs() < 1e-12);
        let slope = huber_loss(0.9, knee) - huber_loss(0.8, knee);
        assert!((slope - knee * 0.1).abs() < 1e-12);
        assert_eq!(huber_loss(0.0, knee), 0.0);
    }
}
