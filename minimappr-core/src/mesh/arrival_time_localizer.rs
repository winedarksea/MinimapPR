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
}

impl Default for ArrivalTimeLocalizer {
    fn default() -> Self {
        Self {
            grid: ArrivalGridConfig::default(),
            refiner: SeedRefinerConfig::default(),
            sound_speed_mps: 343.0,
            minimum_arrivals: 3,
        }
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl ArrivalTimeLocalizer {
    fn pair_observations(&self, arrivals: &[NodeArrival]) -> Vec<PairTdoaObservation> {
        let reference = (0..arrivals.len())
            .min_by(|&a, &b| arrivals[a].timing_std_s.total_cmp(&arrivals[b].timing_std_s))
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

    fn grid_seed(
        &self,
        observations: &[PairTdoaObservation],
        centroid: [f64; 3],
        ground_z: f64,
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
                    let azimuth = std::f64::consts::TAU * step as f64 / grid.azimuth_step_count as f64;
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

    pub fn localize(&self, arrivals: &[NodeArrival]) -> Result<RefinedTdoaFix, ArrivalLocalizationError> {
        if arrivals.len() < self.minimum_arrivals {
            return Err(ArrivalLocalizationError::TooFewArrivals {
                available: arrivals.len(),
                needed: self.minimum_arrivals,
            });
        }
        let count = arrivals.len() as f64;
        let centroid = std::array::from_fn(|axis| {
            arrivals.iter().map(|arrival| arrival.position_m[axis]).sum::<f64>() / count
        });
        let ground_z = centroid[2] - self.grid.assumed_node_height_above_ground_m;
        let observations = self.pair_observations(arrivals);
        let seed_position = self.grid_seed(&observations, centroid, ground_z);
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
            node_position_std_m: arrivals.iter().map(|arrival| arrival.position_std_m).collect(),
        };
        refiner.refine(&seed).map_err(ArrivalLocalizationError::Solver)
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
            let fix = localizer.localize(&arrivals_for(source, &nodes, 12.5)).expect("solves");
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
        let fix = ArrivalTimeLocalizer::default().localize(&arrivals).expect("solves");
        assert_eq!(fix.rejected_observation_indices.len(), 1);
        let error = ((fix.position_m[0] - source[0]).powi(2) + (fix.position_m[1] - source[1]).powi(2)).sqrt();
        assert!(error < 0.5, "{:?}", fix.position_m);
    }
}
