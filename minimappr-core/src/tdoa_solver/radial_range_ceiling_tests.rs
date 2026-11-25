//! Mirror of `tests/test_radial_range_refinement_ceiling.py`: a plane-wave
//! source's radial cost falls forever; bounded by the envelope the search is
//! asymptotic there instead of diverging to float saturation.

use nalgebra::{DMatrix, DVector, Vector3};

use super::levenberg_marquardt_driver::PositionResidualModel;
use super::radial_range_refinement::{adaptive_radial_refinement, RadialRefinementRequest};
use crate::range_projection::RANGE_ASYMPTOTIC;

const SOUND_SPEED_MPS: f64 = 343.0;
const ENVELOPE_M: f64 = 5_000.0;
const TIME_STD_S: f64 = 1.0 / (48_000.0 * 4.0);
const TETRA_M: [[f64; 3]; 4] = [
    [-0.016238, 0.025, -0.010205],
    [0.027063, 0.0, -0.010205],
    [-0.016238, -0.025, -0.010205],
    [0.005413, 0.0, 0.030615],
];

struct PlaneWave {
    measured: DVector<f64>,
}

impl PlaneWave {
    fn new(direction: Vector3<f64>) -> Self {
        let reference = Vector3::from(TETRA_M[0]);
        let measured = DVector::from_iterator(
            3,
            TETRA_M[1..]
                .iter()
                .map(|sensor| -(Vector3::from(*sensor) - reference).dot(&direction) / SOUND_SPEED_MPS),
        );
        Self { measured }
    }
}

impl PositionResidualModel for PlaneWave {
    fn residuals(&self, position_m: &Vector3<f64>) -> DVector<f64> {
        let distance = |index: usize| (Vector3::from(TETRA_M[index]) - position_m).norm();
        DVector::from_iterator(
            3,
            (1..4).map(|index| (distance(index) - distance(0)) / SOUND_SPEED_MPS - self.measured[index - 1]),
        )
    }

    fn jacobian(&self, _position_m: &Vector3<f64>) -> DMatrix<f64> {
        DMatrix::zeros(3, 3)
    }
}

fn refine(maximum_radius_m: Option<f64>) -> (f64, &'static str, f64) {
    let direction = Vector3::new(0.8, 0.5, 0.2).normalize();
    let centroid = TETRA_M
        .iter()
        .fold(Vector3::zeros(), |sum, sensor| sum + Vector3::from(*sensor))
        / 4.0;
    let model = PlaneWave::new(direction);
    let refinement = adaptive_radial_refinement(&RadialRefinementRequest {
        centroid_m: centroid,
        direction,
        initial_radius_m: 50.0,
        minimum_radius_m: 0.05,
        residual_model: &model,
        effective_time_std_s: TIME_STD_S,
        asymptotic_radial_std_fraction: None,
        maximum_radius_m,
    })
    .expect("refinement");
    (
        (refinement.position_m - centroid).norm(),
        refinement.projection_mode,
        refinement.radial_std_m,
    )
}

#[test]
fn unbounded_search_diverges_on_a_plane_wave() {
    assert!(refine(None).0 > 1.0e6);
}

#[test]
fn bounded_search_is_asymptotic_at_the_envelope() {
    let (radius_m, mode, radial_std_m) = refine(Some(ENVELOPE_M));
    assert_eq!(mode, RANGE_ASYMPTOTIC);
    assert!(radius_m <= ENVELOPE_M * (1.0 + 1e-9), "{radius_m}");
    assert!(radial_std_m >= radius_m);
}
