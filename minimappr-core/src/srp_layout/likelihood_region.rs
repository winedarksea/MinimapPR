//! Height-marginalized likelihood region and the bearing-fan fallback.
//!
//! Raw SRP power is not a probability. Each (refined, see
//! `region_refinement`) cell's envelope score becomes
//! L = exp((S − S_best) / (T · Σw)); a column's horizontal density is
//! Σ_z L · Δz (height marginalized), and the reported region is the
//! highest-density set of columns holding `contour_coverage_fraction` of the
//! mass. T is fitted on golden bundles so that region really covers truth at
//! that rate. Inside the layout this is a tight blob; outside it becomes the
//! honest wedge (tight bearing, wide range) instead of an LM ellipse.

use std::f64::consts::{PI, TAU};

use super::config::UncertaintyConfig;
use super::polar_grid::{CoarsePolarGrid, HorizontalColumn, LayoutFrame};
use super::region_refinement::RegionRefinement;

/// One selected column of the region, in world azimuth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionColumn {
    pub radial_m: [f64; 2],
    /// World azimuth bounds (CCW from +x), may exceed 2π at the wrap.
    pub world_azimuth_rad: [f64; 2],
    pub is_far_field: bool,
    pub mass_fraction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LikelihoodRegion {
    pub columns: Vec<RegionColumn>,
    pub coverage_fraction: f64,
    /// Region mass sitting on the far-field ring.
    pub far_field_mass_fraction: f64,
    /// Radial extent of the region from the layout centroid (∞ if far-field).
    pub radial_span_m: [f64; 2],
    /// True when the region touches the grid's outer radius (range clipped).
    pub reaches_grid_edge: bool,
    /// Region refinement cost and whether it finished within budget.
    pub refinement_cells_evaluated: usize,
    pub is_refinement_complete: bool,
    frame: LayoutFrame,
    selected_layout_columns: Vec<HorizontalColumn>,
    outer_radius_m: f64,
}

/// A wedge from the layout centroid for range-unobservable sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingFan {
    pub origin_xy_m: [f64; 2],
    pub world_azimuth_rad: f64,
    pub half_width_rad: f64,
    pub minimum_range_m: f64,
}

/// What the map should draw for this fix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UncertaintyDisplay {
    /// Draw `LikelihoodRegion::columns`.
    Contour,
    BearingFan(BearingFan),
}

impl LikelihoodRegion {
    /// True when the horizontal point lies in a selected column (points
    /// beyond the outer ring count when their far-field bearing is selected).
    pub fn contains_xy(&self, point_xy_m: [f64; 2]) -> bool {
        let (radius, azimuth, _) = self.frame.world_to_layout([point_xy_m[0], point_xy_m[1], self.frame.ground_height_m]);
        self.selected_layout_columns.iter().any(|column| {
            azimuth >= column.layout_azimuth_rad[0]
                && azimuth < column.layout_azimuth_rad[1]
                && if column.is_far_field {
                    radius >= self.outer_radius_m
                } else {
                    radius >= column.radial_m[0] && radius < column.radial_m[1]
                }
        })
    }
}

/// Highest-density set of refined columns holding the coverage fraction.
pub fn build_likelihood_region(
    refinement: &RegionRefinement,
    grid: &CoarsePolarGrid,
    frame: &LayoutFrame,
    config: &UncertaintyConfig,
) -> LikelihoodRegion {
    let columns = &refinement.columns;
    let outermost_near_area_per_radian = grid.far_field_area_per_radian_m2();
    let column_area = |column: &HorizontalColumn| {
        if column.is_far_field {
            outermost_near_area_per_radian * (column.layout_azimuth_rad[1] - column.layout_azimuth_rad[0])
        } else {
            column.area_m2
        }
    };
    let column_mass: Vec<f64> =
        columns.iter().map(|refined| refined.likelihood_height_integral_m * column_area(&refined.column)).collect();
    let total_mass: f64 = column_mass.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let outer_radius_m = grid
        .columns
        .iter()
        .filter(|column| column.is_far_field)
        .map(|column| column.radial_m[0])
        .fold(f64::INFINITY, f64::min);

    // Density = Σ_z L·Δz (per unit horizontal area).
    let mut order: Vec<usize> = (0..columns.len()).collect();
    order.sort_by(|&left, &right| {
        columns[right].likelihood_height_integral_m.total_cmp(&columns[left].likelihood_height_integral_m)
    });
    let mut selected = Vec::new();
    let mut selected_layout_columns = Vec::new();
    let mut accumulated = 0.0;
    let mut far_field_mass = 0.0;
    let mut radial_span = [f64::INFINITY, 0.0_f64];
    for column_index in order {
        if accumulated >= config.contour_coverage_fraction * total_mass {
            break;
        }
        let column = columns[column_index].column;
        accumulated += column_mass[column_index];
        if column.is_far_field {
            far_field_mass += column_mass[column_index];
        }
        selected_layout_columns.push(column);
        radial_span = [radial_span[0].min(column.radial_m[0]), radial_span[1].max(column.radial_m[1])];
        selected.push(RegionColumn {
            radial_m: column.radial_m,
            world_azimuth_rad: column.layout_azimuth_rad.map(|azimuth| frame.world_azimuth_rad(azimuth)),
            is_far_field: column.is_far_field,
            mass_fraction: column_mass[column_index] / total_mass,
        });
    }
    LikelihoodRegion {
        columns: selected,
        coverage_fraction: accumulated / total_mass,
        far_field_mass_fraction: far_field_mass / accumulated.max(f64::MIN_POSITIVE),
        radial_span_m: radial_span,
        reaches_grid_edge: radial_span[1] >= outer_radius_m,
        refinement_cells_evaluated: refinement.cells_evaluated,
        is_refinement_complete: refinement.is_complete,
        frame: *frame,
        selected_layout_columns,
        outer_radius_m,
    }
}

fn wrapped_angle_difference(angle: f64, reference: f64) -> f64 {
    (angle - reference + PI).rem_euclid(TAU) - PI
}

/// Contour or bearing fan. A fan is drawn when the estimate is far-field,
/// when the region reaches the grid edge or holds significant far-field
/// mass, or when its radial
/// span is large relative to max(range, layout aperture).
pub fn choose_uncertainty_display(
    region: &LikelihoodRegion,
    estimate_is_far_field: bool,
    estimate_world_azimuth_rad: f64,
    estimate_horizontal_range_m: f64,
    layout_aperture_m: f64,
    config: &UncertaintyConfig,
) -> UncertaintyDisplay {
    let range_reference = estimate_horizontal_range_m.max(layout_aperture_m).max(f64::MIN_POSITIVE);
    let relative_span = (region.radial_span_m[1] - region.radial_span_m[0]) / range_reference;
    let range_unobservable = estimate_is_far_field
        || region.reaches_grid_edge
        || region.far_field_mass_fraction > config.bearing_fan_far_field_mass_fraction
        || relative_span > config.bearing_fan_relative_range_span;
    if !range_unobservable {
        return UncertaintyDisplay::Contour;
    }
    let half_width_rad = region
        .columns
        .iter()
        .flat_map(|column| column.world_azimuth_rad)
        .map(|azimuth| wrapped_angle_difference(azimuth, estimate_world_azimuth_rad).abs())
        .fold(0.0_f64, f64::max);
    UncertaintyDisplay::BearingFan(BearingFan {
        origin_xy_m: region.frame.origin_xy_m,
        world_azimuth_rad: estimate_world_azimuth_rad.rem_euclid(TAU),
        half_width_rad,
        minimum_range_m: region.radial_span_m[0],
    })
}
