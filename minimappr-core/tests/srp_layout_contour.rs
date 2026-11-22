//! The reported likelihood region (z-marginalized contour) covers the true
//! source inside and outside the layout, stays compact inside it, and turns
//! into a bearing fan only when range is unobservable.

mod srp_layout_support;

use minimappr_core::srp_layout::{LayoutSrpResult, UncertaintyDisplay};
use srp_layout_support::*;

/// Fixed seeds: every case must be covered (coverage gate 90–98 % is for
/// the nightly Monte Carlo once T is fitted on golden bundles).
const MINIMUM_FIXED_SEED_COVERAGE: f64 = 1.0;
/// Inside the layout the region should be a "tight dot", not a smear.
const INSIDE_MAXIMUM_REGION_RADIAL_SPAN_M: f64 = 4.0;

fn region_area_m2(result: &LayoutSrpResult) -> f64 {
    result
        .region
        .columns
        .iter()
        .filter(|column| !column.is_far_field)
        .map(|column| 0.5 * (column.world_azimuth_rad[1] - column.world_azimuth_rad[0]) * (column.radial_m[1].powi(2) - column.radial_m[0].powi(2)))
        .sum()
}

fn describe(label: &str, result: &LayoutSrpResult, source: [f64; 3]) -> bool {
    let covered = result.region.contains_xy([source[0], source[1]]);
    eprintln!(
        "{label}: covered {covered}, columns {}, area {:.1} m², radial span {:.1}–{:.1} m, far mass {:.2}, fan {}, evals {} complete {}",
        result.region.columns.len(),
        region_area_m2(result),
        result.region.radial_span_m[0],
        result.region.radial_span_m[1],
        result.region.far_field_mass_fraction,
        matches!(result.display, UncertaintyDisplay::BearingFan(_)),
        result.region.refinement_cells_evaluated,
        result.region.is_refinement_complete,
    );
    covered
}

#[test]
fn region_covers_truth_and_stays_compact_inside_layout() {
    let sources = [[7.0, 9.0, 6.0], [12.0, 12.0, 12.0], [-2.0, 6.0, 15.0], [15.0, 6.0, 2.0], [4.0, 16.0, 9.0]];
    let mut covered = 0;
    for (index, source) in sources.iter().enumerate() {
        let mut scenario = scene(near_phone_layout(index as u64), *source, index, 400 + index as u64);
        scenario.reflectors.random_trunks = Some(Default::default());
        let result = localize(&render(&scenario));
        covered += describe(&format!("inside {source:?}"), &result, *source) as usize;
        let span = result.region.radial_span_m[1] - result.region.radial_span_m[0];
        assert!(span < INSIDE_MAXIMUM_REGION_RADIAL_SPAN_M, "inside region radial span {span}");
        assert!(matches!(result.display, UncertaintyDisplay::Contour), "inside source drew a fan");
    }
    assert!(covered as f64 / sources.len() as f64 >= MINIMUM_FIXED_SEED_COVERAGE);
}

#[test]
fn region_covers_truth_outside_layout() {
    let centroid = [6.6, 10.0];
    let mut covered = 0;
    let mut total = 0;
    for (index, (range, azimuth_deg)) in [(40.0, 30.0), (60.0, 160.0), (120.0, 280.0), (180.0, 75.0)].iter().enumerate() {
        let azimuth = f64::to_radians(*azimuth_deg);
        let source = [centroid[0] + range * azimuth.cos(), centroid[1] + range * azimuth.sin(), 10.0];
        let mut scenario = scene(near_phone_layout(index as u64), source, index, 500 + index as u64);
        scenario.reflectors.random_trunks = Some(Default::default());
        let result = localize(&render(&scenario));
        covered += describe(&format!("outside r={range} az={azimuth_deg}"), &result, source) as usize;
        total += 1;
    }
    assert!(covered as f64 / total as f64 >= MINIMUM_FIXED_SEED_COVERAGE);
}
