//! Small statistics helpers: percentiles, binomial intervals, χ² quantiles
//! and the empirical coverage curve a calibration fits against.

/// Two-dimensional χ² quantile: P(χ²₂ ≤ q) = p  ⇒  q = −2 ln(1 − p).
pub fn chi_square_two_dof_quantile(probability: f64) -> f64 {
    -2.0 * (1.0 - probability).ln()
}

/// Two-sided standard normal tail P(|Z| > z) via erfc (Abramowitz–Stegun
/// 7.1.26, absolute error < 1.5e-7, ample for gate expectations).
pub fn two_sided_normal_tail(z: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [0.254_829_592, -0.284_496_736, 1.421_413_741, -1.453_152_027, 1.061_405_429];
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + P * x);
    let polynomial = A.iter().rev().fold(0.0, |acc, a| acc * t + a) * t;
    polynomial * (-x * x).exp()
}

/// Nearest-rank percentile (`fraction` in [0, 1]); NaN for an empty set.
pub fn percentile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

pub fn median(values: &[f64]) -> f64 {
    percentile(values, 0.5)
}

/// Wilson score interval at z standard errors.
pub fn wilson_interval(successes: usize, trials: usize, z: f64) -> [f64; 2] {
    if trials == 0 {
        return [0.0, 1.0];
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let denominator = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denominator;
    let half_width = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
    [(centre - half_width).max(0.0), (centre + half_width).min(1.0)]
}

/// Binomial standard error of a rate `p` over `trials`.
pub fn binomial_standard_error(p: f64, trials: usize) -> f64 {
    (p * (1.0 - p) / trials.max(1) as f64).sqrt()
}

/// Each trial's *minimum covering parameter*: the smallest parameter value
/// (temperature, σ multiplier) at which its region contains the truth
/// (∞ when no tested value covers). Coverage at θ is the fraction ≤ θ.
pub fn coverage_at(minimum_covering_parameters: &[f64], parameter: f64) -> f64 {
    if minimum_covering_parameters.is_empty() {
        return f64::NAN;
    }
    minimum_covering_parameters.iter().filter(|value| **value <= parameter).count() as f64
        / minimum_covering_parameters.len() as f64
}

/// Smallest parameter reaching `target` coverage: the ⌈target·n⌉-th order
/// statistic of the minimum covering parameters.
pub fn fit_parameter_for_coverage(minimum_covering_parameters: &[f64], target: f64) -> f64 {
    let mut sorted = minimum_covering_parameters.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((target * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len().max(1));
    sorted.get(rank - 1).copied().unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_square_95_percent_two_dof() {
        assert!((chi_square_two_dof_quantile(0.95) - 5.991).abs() < 1e-3);
    }

    #[test]
    fn normal_tail_at_three_sigma() {
        assert!((two_sided_normal_tail(3.0) - 0.002_699_8).abs() < 1e-6);
    }

    #[test]
    fn fit_hits_target_rank() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(fit_parameter_for_coverage(&values, 0.95), 95.0);
        assert!((coverage_at(&values, 95.0) - 0.95).abs() < 1e-12);
    }

    #[test]
    fn wilson_brackets_rate() {
        let [low, high] = wilson_interval(95, 100, 2.0);
        assert!(low < 0.95 && high > 0.95 && low > 0.85 && high < 0.99);
    }
}
