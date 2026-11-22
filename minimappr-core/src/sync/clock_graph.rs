//! Island clock-offset graph: pairwise two-way offsets → one offset per
//! phone relative to a reference phone, by weighted robust least squares.
//!
//! Each constraint (i, j) is weighted by its variance, which grows with the
//! time between the two phones' slots and with the motion energy in that
//! interval. A still birder still turns or raises binoculars; moving 15 cm
//! between slots changes a flight time by ~440 µs and puts ~220 µs into the
//! two-way offset. Adjacent, quiet slots therefore carry the sync.
//!
//! Variance model (offset error from an unobserved displacement δx is
//! δx / (2c)):
//!   σ² = σ_pick² + (v_unobserved · gap / 2c)² + (k_motion · energy / 2c)²

use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug, PartialEq)]
pub struct ClockGraphConfig {
    /// Timing noise of one two-way offset from pick jitter alone, s.
    pub pick_offset_sigma_s: f64,
    /// Drift speed of a phone that took no steps (turning, shifting), m/s.
    pub unobserved_motion_speed_mps: f64,
    /// Metres of possible displacement per unit of linear-acceleration
    /// energy between the slots (platform-normalised energy).
    pub displacement_per_motion_energy_m: f64,
    pub speed_of_sound_mps: f64,
    /// Huber threshold in σ for IRLS down-weighting.
    pub huber_threshold_sigma: f64,
    /// Constraints still beyond this many σ after convergence are rejected.
    pub rejection_threshold_sigma: f64,
    pub maximum_reweighting_iterations: usize,
}

impl Default for ClockGraphConfig {
    fn default() -> Self {
        Self {
            pick_offset_sigma_s: 15e-6,
            unobserved_motion_speed_mps: 0.02,
            displacement_per_motion_energy_m: 0.1,
            speed_of_sound_mps: 343.0,
            huber_threshold_sigma: 2.5,
            rejection_threshold_sigma: 6.0,
            maximum_reweighting_iterations: 12,
        }
    }
}

/// local_j − local_i = `offset_j_minus_i_s` at the round.
#[derive(Clone, Debug, PartialEq)]
pub struct ClockOffsetConstraint {
    pub node_i: usize,
    pub node_j: usize,
    pub offset_j_minus_i_s: f64,
    pub slot_gap_s: f64,
    /// Linear-acceleration energy of both phones between their slots.
    pub motion_energy: f64,
}

impl ClockGraphConfig {
    pub fn constraint_sigma_s(&self, constraint: &ClockOffsetConstraint) -> f64 {
        let two_c = 2.0 * self.speed_of_sound_mps;
        let gap_term = self.unobserved_motion_speed_mps * constraint.slot_gap_s.abs() / two_c;
        let motion_term =
            self.displacement_per_motion_energy_m * constraint.motion_energy.max(0.0) / two_c;
        (self.pick_offset_sigma_s.powi(2) + gap_term.powi(2) + motion_term.powi(2)).sqrt()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClockGraphSolution {
    /// local_node − local_reference; `None` when not connected to the reference.
    pub offsets_s: Vec<Option<f64>>,
    pub offset_sigmas_s: Vec<Option<f64>>,
    /// Final residual / σ per constraint (input order).
    pub normalized_residuals: Vec<f64>,
    /// Indices of constraints rejected as outliers.
    pub rejected_constraints: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClockGraphError {
    ReferenceOutOfRange,
    SingularSystem,
}

fn reachable_from(
    reference: usize,
    node_count: usize,
    constraints: &[ClockOffsetConstraint],
    active: &[bool],
) -> Vec<bool> {
    let mut reached = vec![false; node_count];
    reached[reference] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for (constraint, &is_active) in constraints.iter().zip(active) {
            if !is_active {
                continue;
            }
            let (i, j) = (constraint.node_i, constraint.node_j);
            if reached[i] != reached[j] {
                reached[i] = true;
                reached[j] = true;
                changed = true;
            }
        }
    }
    reached
}

/// Solve offsets relative to `reference_node` (its offset is 0).
pub fn solve_clock_graph(
    node_count: usize,
    reference_node: usize,
    constraints: &[ClockOffsetConstraint],
    config: &ClockGraphConfig,
) -> Result<ClockGraphSolution, ClockGraphError> {
    if reference_node >= node_count {
        return Err(ClockGraphError::ReferenceOutOfRange);
    }
    let sigmas: Vec<f64> = constraints
        .iter()
        .map(|c| config.constraint_sigma_s(c))
        .collect();
    let mut active: Vec<bool> = constraints
        .iter()
        .map(|c| c.node_i < node_count && c.node_j < node_count && c.node_i != c.node_j)
        .collect();
    let mut robust_weights = vec![1.0; constraints.len()];
    let mut offsets = vec![0.0; node_count];
    let mut covariance_diagonal = vec![0.0; node_count];
    let mut reached = vec![false; node_count];
    let mut rejected = Vec::new();

    for _rejection_pass in 0..constraints.len().max(1) {
        reached = reachable_from(reference_node, node_count, constraints, &active);
        let unknowns: Vec<usize> = (0..node_count)
            .filter(|n| *n != reference_node && reached[*n])
            .collect();
        let mut column_of = vec![usize::MAX; node_count];
        for (column, node) in unknowns.iter().enumerate() {
            column_of[*node] = column;
        }
        for _iteration in 0..config.maximum_reweighting_iterations {
            let size = unknowns.len();
            let mut normal = DMatrix::<f64>::zeros(size, size);
            let mut right = DVector::<f64>::zeros(size);
            for (index, constraint) in constraints.iter().enumerate() {
                if !active[index] || !reached[constraint.node_i] {
                    continue;
                }
                let weight = robust_weights[index] / sigmas[index].powi(2);
                let (ci, cj) = (column_of[constraint.node_i], column_of[constraint.node_j]);
                // Residual model: θ_j − θ_i − z.
                for (column, sign) in [(cj, 1.0), (ci, -1.0)] {
                    if column == usize::MAX {
                        continue;
                    }
                    right[column] += sign * weight * constraint.offset_j_minus_i_s;
                    for (other, other_sign) in [(cj, 1.0), (ci, -1.0)] {
                        if other != usize::MAX {
                            normal[(column, other)] += sign * other_sign * weight;
                        }
                    }
                }
            }
            let cholesky = normal
                .clone()
                .cholesky()
                .ok_or(ClockGraphError::SingularSystem)?;
            let solution = cholesky.solve(&right);
            let inverse = cholesky.inverse();
            offsets = vec![0.0; node_count];
            covariance_diagonal = vec![0.0; node_count];
            for (column, node) in unknowns.iter().enumerate() {
                offsets[*node] = solution[column];
                covariance_diagonal[*node] = inverse[(column, column)];
            }
            let mut changed = false;
            for (index, constraint) in constraints.iter().enumerate() {
                if !active[index] {
                    continue;
                }
                let residual = offsets[constraint.node_j]
                    - offsets[constraint.node_i]
                    - constraint.offset_j_minus_i_s;
                let normalized = (residual / sigmas[index]).abs();
                let new_weight = if normalized <= config.huber_threshold_sigma {
                    1.0
                } else {
                    config.huber_threshold_sigma / normalized
                };
                changed |= (new_weight - robust_weights[index]).abs() > 1e-6;
                robust_weights[index] = new_weight;
            }
            if !changed {
                break;
            }
        }
        // Reject the single worst constraint beyond the threshold, if any, and re-solve.
        let worst = constraints
            .iter()
            .enumerate()
            .filter(|(index, _)| active[*index])
            .map(|(index, c)| {
                let residual = offsets[c.node_j] - offsets[c.node_i] - c.offset_j_minus_i_s;
                (index, (residual / sigmas[index]).abs())
            })
            .filter(|(_, normalized)| *normalized > config.rejection_threshold_sigma)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match worst {
            Some((index, _)) => {
                active[index] = false;
                rejected.push(index);
                robust_weights.iter_mut().for_each(|w| *w = 1.0);
            }
            None => break,
        }
    }

    let normalized_residuals = constraints
        .iter()
        .enumerate()
        .map(|(index, c)| {
            if c.node_i >= node_count || c.node_j >= node_count {
                return f64::INFINITY;
            }
            (offsets[c.node_j] - offsets[c.node_i] - c.offset_j_minus_i_s) / sigmas[index]
        })
        .collect();
    Ok(ClockGraphSolution {
        offsets_s: (0..node_count)
            .map(|n| reached[n].then_some(offsets[n]))
            .collect(),
        offset_sigmas_s: (0..node_count)
            .map(|n| reached[n].then_some(covariance_diagonal[n].max(0.0).sqrt()))
            .collect(),
        normalized_residuals,
        rejected_constraints: rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRUE_OFFSETS_S: [f64; 5] = [0.0, 0.0031, -0.0127, 0.0084, 0.0002];

    fn constraint(
        i: usize,
        j: usize,
        error_s: f64,
        slot_gap_s: f64,
        motion_energy: f64,
    ) -> ClockOffsetConstraint {
        ClockOffsetConstraint {
            node_i: i,
            node_j: j,
            offset_j_minus_i_s: TRUE_OFFSETS_S[j] - TRUE_OFFSETS_S[i] + error_s,
            slot_gap_s,
            motion_energy,
        }
    }

    fn all_pairs() -> Vec<ClockOffsetConstraint> {
        let mut list = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                let error = if (i + j) % 2 == 0 { 5e-6 } else { -5e-6 };
                list.push(constraint(i, j, error, 0.25 * (j - i) as f64, 0.0));
            }
        }
        list
    }

    #[test]
    fn recovers_offsets_from_consistent_pairs() {
        let solution = solve_clock_graph(5, 0, &all_pairs(), &ClockGraphConfig::default()).unwrap();
        for (node, truth) in TRUE_OFFSETS_S.iter().enumerate() {
            assert!((solution.offsets_s[node].unwrap() - truth).abs() < 10e-6);
        }
        assert!(solution.rejected_constraints.is_empty());
        assert_eq!(solution.offsets_s[0], Some(0.0));
    }

    #[test]
    fn gross_outlier_is_rejected() {
        let mut constraints = all_pairs();
        constraints[4].offset_j_minus_i_s += 220e-6; // node moved 15 cm between slots
        let solution = solve_clock_graph(5, 0, &constraints, &ClockGraphConfig::default()).unwrap();
        assert_eq!(solution.rejected_constraints, vec![4]);
        for (node, truth) in TRUE_OFFSETS_S.iter().enumerate() {
            assert!((solution.offsets_s[node].unwrap() - truth).abs() < 12e-6);
        }
    }

    #[test]
    fn long_gap_and_motion_constraints_carry_less_weight() {
        let config = ClockGraphConfig::default();
        let near = config.constraint_sigma_s(&constraint(0, 1, 0.0, 0.2, 0.0));
        let far = config.constraint_sigma_s(&constraint(0, 1, 0.0, 1.4, 0.0));
        let moving = config.constraint_sigma_s(&constraint(0, 1, 0.0, 0.2, 2.0));
        assert!(far > near && moving > far);

        // A biased long-gap edge competes with a clean adjacent chain.
        let constraints = vec![
            constraint(0, 1, 0.0, 0.2, 0.0),
            constraint(1, 2, 0.0, 0.2, 0.0),
            constraint(0, 2, 60e-6, 1.4, 3.0),
        ];
        let solution = solve_clock_graph(3, 0, &constraints, &config).unwrap();
        assert!((solution.offsets_s[2].unwrap() - TRUE_OFFSETS_S[2]).abs() < 10e-6);
    }

    #[test]
    fn disconnected_nodes_have_no_offset() {
        let constraints = vec![
            constraint(0, 1, 0.0, 0.2, 0.0),
            constraint(2, 3, 0.0, 0.2, 0.0),
        ];
        let solution = solve_clock_graph(4, 0, &constraints, &ClockGraphConfig::default()).unwrap();
        assert!(solution.offsets_s[1].is_some());
        assert!(solution.offsets_s[2].is_none() && solution.offsets_s[3].is_none());
    }
}
