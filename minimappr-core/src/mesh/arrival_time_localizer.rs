//! Localize a source from per-node arrival times on one island clock: a
//! coarse polar grid over a robust (Cauchy) TDOA misfit finds the basin, then
//! [`JointLmSeedRefiner`] polishes it through the same `LmSeed` handoff the
//! layout SRP uses.
//!
//! Inputs are arrival times, never waveforms, so a clap test or an
//! impulsive event uses this directly; the Layer 2 mesh simulator uses it
//! as its stand-in for clip SRP (its clips carry arrival times instead of
//! audio). Pairs are formed against the best-timed node only, so the
//! covariance is not overconfident from reusing one arrival n − 1 times.

use crate::srp_layout::lm_handoff::{LmSeed, PairTdoaObservation, TdoaSeedRefiner};
use crate::tdoa_solver::TdoaSolverError;

use super::srp_seed_refiner::{JointLmSeedRefiner, RefinedTdoaFix, SeedRefinerConfig};

/// One node's arrival on the island reference clock.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeArrival {
    pub position_m: [f64; 3],
    pub position_std_m: f64,
    pub reference_arrival_s: f64,
    pub timing_std_s: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrivalGridConfig {
    pub minimum_range_m: f64,
    pub maximum_range_m: f64,
    pub range_ring_count: usize,
    pub azimuth_step_count: usize,
    /// Candidate heights above the inferred ground.
    pub candidate_heights_above_ground_m: Vec<f64>,
    /// Canopy band for the refiner's height prior (above ground).
    pub source_height_band_above_ground_m: [f64; 2],
    /// Phones are held this far above the ground they stand on.
    pub assumed_node_height_above_ground_m: f64,
}

impl Default for ArrivalGridConfig {
    fn default() -> Self {
        Self {
            minimum_range_m: 0.5,
            maximum_range_m: 250.0,
            range_ring_count: 48,
            azimuth_step_count: 72,
            candidate_heights_above_ground_m: vec![2.0, 8.0, 16.0],
            source_height_band_above_ground_m: [0.0, 20.0],
            assumed_node_height_above_ground_m: 1.4,
        }
    }
}

/// A fix plus which arrivals it kept (indices into the input).
#[derive(Debug, Clone, PartialEq)]
pub struct ArrivalFix {
    pub fix: RefinedTdoaFix,
    pub used_arrival_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArrivalLocalizationError {
    TooFewArrivals { available: usize, needed: usize },
    Solver(TdoaSolverError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrivalTimeLocalizer {
    pub grid: ArrivalGridConfig,
    pub refiner: SeedRefinerConfig,
    pub sound_speed_mps: f64,
    pub minimum_arrivals: usize,
    /// χ² must fall by more than this per node a solve does without: the
    /// 99.9 % point of χ²₁, so with 7 clean phones a good one is dropped in
    /// < 1 % of calls.
    pub node_rejection_chi_square_drop: f64,
}

impl Default for ArrivalTimeLocalizer {
    fn default() -> Self {
        Self {
            grid: ArrivalGridConfig::default(),
            refiner: SeedRefinerConfig::default(),
            sound_speed_mps: 343.0,
            minimum_arrivals: 3,
            node_rejection_chi_square_drop: 10.83,
        }
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl ArrivalTimeLocalizer {
    fn pair_observations(&self, arrivals: &[NodeArrival]) -> Vec<PairTdoaObservation> {
        // Every pair shares the reference's error, so pick the node whose
        // timing *and* position are best known.
        let total_std = |arrival: &NodeArrival| arrival.timing_std_s.hypot(arrival.position_std_m / self.sound_speed_mps);
        let reference = (0..arrivals.len())
            .min_by(|&a, &b| total_std(&arrivals[a]).total_cmp(&total_std(&arrivals[b])))
            .unwrap_or(0);
        (0..arrivals.len())
            .filter(|&index| index != reference)
            .map(|index| PairTdoaObservation {
                sensor_a_position_m: arrivals[reference].position_m,
                sensor_b_position_m: arrivals[index].position_m,
                sensor_a_node: reference,
                sensor_b_node: index,
                measured_tdoa_s: arrivals[reference].reference_arrival_s
                    - arrivals[index].reference_arrival_s,
                relative_peak_coherence: 1.0,
                weight: 1.0,
                lag_resolution_s: (arrivals[reference].timing_std_s.powi(2)
                    + arrivals[index].timing_std_s.powi(2))
                .sqrt(),
            })
            .collect()
    }

    /// Cauchy misfit, so one bad arrival cannot pull the coarse search.
    fn robust_misfit(&self, observations: &[PairTdoaObservation], position: [f64; 3]) -> f64 {
        observations
            .iter()
            .map(|observation| {
                let predicted = (distance(position, observation.sensor_a_position_m)
                    - distance(position, observation.sensor_b_position_m))
                    / self.sound_speed_mps;
                let normalized = (predicted - observation.measured_tdoa_s)
                    / observation.lag_resolution_s.max(self.refiner.minimum_timing_std_s);
                (1.0 + normalized * normalized).ln()
            })
            .sum()
    }

    /// Principal horizontal axis of the nodes (radians). The polar grid is
    /// laid out from it so the seed, and with it the LM basin, does not
    /// depend on how the layout happens to be oriented in the world
    /// (W4-sim-gates invariance gate: an east-aligned grid sent rotated
    /// twins of one scene into different basins, metres apart).
    fn principal_axis_rad(arrivals: &[NodeArrival], centroid: [f64; 3]) -> f64 {
        let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
        for arrival in arrivals {
            let (dx, dy) = (arrival.position_m[0] - centroid[0], arrival.position_m[1] - centroid[1]);
            xx += dx * dx;
            xy += dx * dy;
            yy += dy * dy;
        }
        0.5 * (2.0 * xy).atan2(xx - yy)
    }

    fn grid_seed(
        &self,
        observations: &[PairTdoaObservation],
        centroid: [f64; 3],
        ground_z: f64,
        azimuth_origin_rad: f64,
    ) -> [f64; 3] {
        let grid = &self.grid;
        let ring_ratio = (grid.maximum_range_m / grid.minimum_range_m)
            .powf(1.0 / grid.range_ring_count.saturating_sub(1).max(1) as f64);
        let mut best = ([centroid[0], centroid[1], ground_z + grid.candidate_heights_above_ground_m[0]], f64::INFINITY);
        for &height in &grid.candidate_heights_above_ground_m {
            let z = ground_z + height;
            let centre = [centroid[0], centroid[1], z];
            let misfit = self.robust_misfit(observations, centre);
            if misfit < best.1 {
                best = (centre, misfit);
            }
            let mut range = grid.minimum_range_m;
            for _ring in 0..grid.range_ring_count {
                for step in 0..grid.azimuth_step_count {
                    let azimuth = azimuth_origin_rad + std::f64::consts::TAU * step as f64 / grid.azimuth_step_count as f64;
                    let candidate = [
                        centroid[0] + range * azimuth.cos(),
                        centroid[1] + range * azimuth.sin(),
                        z,
                    ];
                    let misfit = self.robust_misfit(observations, candidate);
                    if misfit < best.1 {
                        best = (candidate, misfit);
                    }
                }
                range *= ring_ratio;
            }
        }
        best.0
    }

    fn solve_subset(&self, arrivals: &[NodeArrival], subset: &[usize]) -> Result<ArrivalFix, ArrivalLocalizationError> {
        let chosen: Vec<NodeArrival> = subset.iter().map(|&index| arrivals[index].clone()).collect();
        let count = chosen.len() as f64;
        let centroid = std::array::from_fn(|axis| chosen.iter().map(|arrival| arrival.position_m[axis]).sum::<f64>() / count);
        let ground_z = centroid[2] - self.grid.assumed_node_height_above_ground_m;
        let observations = self.pair_observations(&chosen);
        let reference = observations.first().map_or(0, |observation| observation.sensor_a_node);
        let seed_position = self.grid_seed(&observations, centroid, ground_z, Self::principal_axis_rad(&chosen, centroid));
        let [band_low, band_high] = self.grid.source_height_band_above_ground_m;
        let seed = LmSeed {
            initial_position_m: seed_position,
            far_field_direction: None,
            observations,
            sound_speed_mps: self.sound_speed_mps,
            source_height_bounds_m: [ground_z + band_low, ground_z + band_high],
            huber_knee: 1.0,
            srp_score: 0.0,
        };
        let refiner = JointLmSeedRefiner {
            config: self.refiner.clone(),
            node_position_std_m: chosen.iter().map(|arrival| arrival.position_std_m).collect(),
        };
        let fix = refiner.refine(&seed).map_err(ArrivalLocalizationError::Solver)?;
        // Pair k is (reference, k-th other node); a rejected pair drops that node.
        let rejected: Vec<usize> = fix
            .rejected_observation_indices
            .iter()
            .map(|&pair| if pair < reference { pair } else { pair + 1 })
            .collect();
        let used_arrival_indices = (0..chosen.len()).filter(|k| !rejected.contains(k)).map(|k| subset[k]).collect();
        Ok(ArrivalFix { fix, used_arrival_indices })
    }

    /// Solve with every arrival and with each node left out in turn, and
    /// keep the solve with the lowest χ² + `node_rejection_chi_square_drop`
    /// × (nodes it does without, whether left out here or rejected by the
    /// refiner). A node is therefore dropped only when χ² falls by more than
    /// that per node (the one-outlier likelihood-ratio test; for a clean node
    /// the fall is ≈ χ²₁), and a subset that kept the bad node by rejecting
    /// two good ones pays for both.
    ///
    /// Why not the old χ²/dof > 9 trigger on the full fit, or a residual
    /// test on its pairs: pairs share one reference, so a late reference
    /// shifts every pair alike (and the refiner's worst-pair rejection then
    /// drops good nodes), and over a flat phone layout a late node is largely
    /// absorbed by moving the source up (closeout C4: a 4 ms pocket raised a
    /// fix 8 m and left χ²/dof under 9). Leaving the node out removes both.
    /// A good node whose removal lets the solve slip into another height
    /// basin is kept: the full fit was already consistent, χ² has little to
    /// lose.
    pub fn localize(&self, arrivals: &[NodeArrival]) -> Result<ArrivalFix, ArrivalLocalizationError> {
        if arrivals.len() < self.minimum_arrivals {
            return Err(ArrivalLocalizationError::TooFewArrivals {
                available: arrivals.len(),
                needed: self.minimum_arrivals,
            });
        }
        let all: Vec<usize> = (0..arrivals.len()).collect();
        let full = self.solve_subset(arrivals, &all);
        if arrivals.len() <= self.minimum_arrivals {
            return full;
        }
        let penalized = |fix: &ArrivalFix| {
            fix.fix.chi_square + self.node_rejection_chi_square_drop * (arrivals.len() - fix.used_arrival_indices.len()) as f64
        };
        let best_without_one = all
            .iter()
            .filter_map(|&left_out| {
                let others: Vec<usize> = all.iter().copied().filter(|index| *index != left_out).collect();
                self.solve_subset(arrivals, &others).ok()
            })
            .min_by(|a, b| penalized(a).total_cmp(&penalized(b)));
        match (full, best_without_one) {
            (Ok(full), Some(without)) if penalized(&without) < penalized(&full) => Ok(without),
            (Ok(full), _) => Ok(full),
            (Err(_), Some(without)) => Ok(without),
            (Err(error), None) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrivals_for(source: [f64; 3], nodes: &[[f64; 3]], offset_s: f64) -> Vec<NodeArrival> {
        nodes
            .iter()
            .map(|&node| NodeArrival {
                position_m: node,
                position_std_m: 0.05,
                reference_arrival_s: offset_s + distance(source, node) / 343.0,
                timing_std_s: 2e-5,
            })
            .collect()
    }

    #[test]
    fn recovers_inside_and_outside_sources() {
        let nodes = [
            [0.0, 0.0, 1.4],
            [14.0, 2.0, 1.5],
            [6.0, 12.0, 1.3],
            [-5.0, 9.0, 1.4],
            [3.0, -8.0, 1.6],
        ];
        let localizer = ArrivalTimeLocalizer::default();
        for source in [[4.0, 3.0, 6.0], [40.0, 25.0, 9.0], [-30.0, -60.0, 12.0]] {
            let fix = localizer.localize(&arrivals_for(source, &nodes, 12.5)).expect("solves").fix;
            let horizontal_error =
                ((fix.position_m[0] - source[0]).powi(2) + (fix.position_m[1] - source[1]).powi(2)).sqrt();
            let range = (source[0].powi(2) + source[1].powi(2)).sqrt();
            assert!(horizontal_error < 0.02 * range.max(10.0), "{source:?} -> {:?}", fix.position_m);
            assert!(fix.horizontal_major_sigma_m() > 0.0);
        }
    }

    #[test]
    fn one_corrupted_arrival_is_rejected() {
        let nodes = [
            [0.0, 0.0, 1.4],
            [14.0, 2.0, 1.5],
            [6.0, 12.0, 1.3],
            [-5.0, 9.0, 1.4],
            [3.0, -8.0, 1.6],
            [10.0, -6.0, 1.4],
        ];
        let source = [5.0, 4.0, 5.0];
        let mut arrivals = arrivals_for(source, &nodes, 3.0);
        arrivals[3].reference_arrival_s += 0.004;
        let solved = ArrivalTimeLocalizer::default().localize(&arrivals).expect("solves");
        assert_eq!(solved.used_arrival_indices, vec![0, 1, 2, 4, 5]);
        let fix = solved.fix;
        let error = ((fix.position_m[0] - source[0]).powi(2) + (fix.position_m[1] - source[1]).powi(2)).sqrt();
        assert!(error < 0.5, "{:?}", fix.position_m);
        // A stale clock on the best-timed node (the pair reference) is caught too.
        let mut arrivals = arrivals_for(source, &nodes, 3.0);
        arrivals[2].timing_std_s = 1e-6;
        arrivals[2].reference_arrival_s += 0.08;
        let solved = ArrivalTimeLocalizer::default().localize(&arrivals).expect("solves");
        assert!(!solved.used_arrival_indices.contains(&2), "{:?}", solved.used_arrival_indices);
    }

    /// The best-timed node is the pair reference, so a 3 ms pocket on it
    /// shifts every pair alike (the old χ²/dof > 9 trigger kept it). It sits
    /// at the layout's edge: a late node near the centre is absorbed exactly
    /// by a higher source and no TDOA test can see it.
    #[test]
    fn a_pocketed_pair_reference_is_rejected() {
        let nodes = [
            [0.0, 0.0, 1.4],
            [14.0, 2.0, 1.5],
            [6.0, 12.0, 1.3],
            [-5.0, 9.0, 1.4],
            [3.0, -8.0, 1.6],
            [10.0, -6.0, 1.4],
            [-7.0, -3.0, 1.5],
        ];
        let source = [4.0, 2.0, 7.0];
        let mut arrivals = arrivals_for(source, &nodes, 3.0);
        for arrival in &mut arrivals {
            arrival.position_std_m = 0.1;
        }
        arrivals[1].timing_std_s = 1.9e-5;
        arrivals[1].reference_arrival_s += 0.003;
        let solved = ArrivalTimeLocalizer::default().localize(&arrivals).expect("solves");
        assert!(!solved.used_arrival_indices.contains(&1), "{:?}", solved.used_arrival_indices);
        let error = (solved.fix.position_m[0] - source[0]).hypot(solved.fix.position_m[1] - source[1]);
        assert!(error < 0.5, "{:?}", solved.fix.position_m);
    }

    #[test]
    fn three_nodes_fix_lies_inside_its_reported_region() {
        let nodes = [[8.0, 0.5, 1.4], [3.5, 7.2, 1.3], [-4.0, 6.5, 1.5]];
        let source = [2.0, 3.0, 5.0];
        let mut arrivals = arrivals_for(source, &nodes, 7.0);
        for arrival in &mut arrivals {
            arrival.position_std_m = 0.25;
        }
        let fix = ArrivalTimeLocalizer::default().localize(&arrivals).expect("3 nodes solve").fix;
        let error = ((fix.position_m[0] - source[0]).powi(2) + (fix.position_m[1] - source[1]).powi(2)).sqrt();
        // Two TDOAs and unknown height: the height prior picks a point on the
        // intersection curve, and the reported region must cover the truth.
        assert!(error < 3.0 * fix.horizontal_major_sigma_m() + 0.5, "{:?} σ {}", fix.position_m, fix.horizontal_major_sigma_m());
    }
}
