//! Polar search cells in the layout frame and their TDOA interval bounds.
//!
//! The frame is centred on the node centroid and rotated onto the layout's
//! principal horizontal axis, so the grid moves with the layout (geometric
//! invariance) and θ = 0 / π lie along a collinear trail (mirror split).
//! Heights are above local ground. Dense uniform rings inside ~30 m, log
//! rings out to ~200 m (cell size grows like TDOA range uncertainty) and a
//! far-field bearing ring (plane-wave directions) beyond.

use std::f64::consts::TAU;

use super::config::PolarGridConfig;

/// Layout-frame origin, principal-axis azimuth (world, CCW from +x) and ground z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutFrame {
    pub origin_xy_m: [f64; 2],
    pub principal_axis_azimuth_rad: f64,
    pub ground_height_m: f64,
}

impl LayoutFrame {
    pub fn world_azimuth_rad(&self, layout_azimuth_rad: f64) -> f64 {
        layout_azimuth_rad + self.principal_axis_azimuth_rad
    }

    pub fn layout_to_world(&self, radius_m: f64, layout_azimuth_rad: f64, height_above_ground_m: f64) -> [f64; 3] {
        let azimuth = self.world_azimuth_rad(layout_azimuth_rad);
        [
            self.origin_xy_m[0] + radius_m * azimuth.cos(),
            self.origin_xy_m[1] + radius_m * azimuth.sin(),
            self.ground_height_m + height_above_ground_m,
        ]
    }

    /// (radius, layout azimuth in [0, 2π), height above ground).
    pub fn world_to_layout(&self, position_m: [f64; 3]) -> (f64, f64, f64) {
        let dx = position_m[0] - self.origin_xy_m[0];
        let dy = position_m[1] - self.origin_xy_m[1];
        let azimuth = (dy.atan2(dx) - self.principal_axis_azimuth_rad).rem_euclid(TAU);
        (dx.hypot(dy), azimuth, position_m[2] - self.ground_height_m)
    }

    pub fn direction_world(&self, layout_azimuth_rad: f64, elevation_rad: f64) -> [f64; 3] {
        let azimuth = self.world_azimuth_rad(layout_azimuth_rad);
        [
            elevation_rad.cos() * azimuth.cos(),
            elevation_rad.cos() * azimuth.sin(),
            elevation_rad.sin(),
        ]
    }
}

/// One search region. Near cells are annular sectors × height slabs; far
/// cells are bearing × elevation patches of plane-wave directions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SearchCell {
    Near {
        radial_m: [f64; 2],
        azimuth_rad: [f64; 2],
        height_m: [f64; 2],
    },
    FarField {
        azimuth_rad: [f64; 2],
        elevation_rad: [f64; 2],
    },
}

fn midpoint(bounds: [f64; 2]) -> f64 {
    0.5 * (bounds[0] + bounds[1])
}

fn halves(bounds: [f64; 2]) -> [[f64; 2]; 2] {
    let middle = midpoint(bounds);
    [[bounds[0], middle], [middle, bounds[1]]]
}

impl SearchCell {
    pub fn is_far_field(&self) -> bool {
        matches!(self, Self::FarField { .. })
    }

    pub fn layout_azimuth_bounds_rad(&self) -> [f64; 2] {
        match *self {
            Self::Near { azimuth_rad, .. } | Self::FarField { azimuth_rad, .. } => azimuth_rad,
        }
    }

    /// Near: centre point. Far: centroid + direction × `far_field_range_m`.
    pub fn representative_position_m(&self, frame: &LayoutFrame, far_field_range_m: f64) -> [f64; 3] {
        match *self {
            Self::Near { radial_m, azimuth_rad, height_m } => {
                frame.layout_to_world(midpoint(radial_m), midpoint(azimuth_rad), midpoint(height_m))
            }
            Self::FarField { azimuth_rad, elevation_rad } => {
                let direction = frame.direction_world(midpoint(azimuth_rad), midpoint(elevation_rad));
                [
                    frame.origin_xy_m[0] + direction[0] * far_field_range_m,
                    frame.origin_xy_m[1] + direction[1] * far_field_range_m,
                    frame.ground_height_m + direction[2] * far_field_range_m,
                ]
            }
        }
    }

    /// Radius of a ball around the centre containing the whole near cell.
    /// For sectors spanning ≤ π the farthest points are the corners.
    pub fn bounding_radius_m(&self, frame: &LayoutFrame) -> f64 {
        match *self {
            Self::Near { radial_m, azimuth_rad, height_m } => {
                let center = frame.layout_to_world(midpoint(radial_m), midpoint(azimuth_rad), midpoint(height_m));
                let mut horizontal: f64 = 0.0;
                for radius in radial_m {
                    for azimuth in azimuth_rad {
                        let corner = frame.layout_to_world(radius, azimuth, midpoint(height_m));
                        horizontal = horizontal.max((corner[0] - center[0]).hypot(corner[1] - center[1]));
                    }
                }
                horizontal.hypot(0.5 * (height_m[1] - height_m[0]))
            }
            Self::FarField { .. } => f64::INFINITY,
        }
    }

    /// Largest angle between the centre direction and any corner direction.
    fn angular_radius_rad(&self, frame: &LayoutFrame) -> f64 {
        let Self::FarField { azimuth_rad, elevation_rad } = *self else {
            return 0.0;
        };
        let center = frame.direction_world(midpoint(azimuth_rad), midpoint(elevation_rad));
        let mut largest: f64 = 0.0;
        for azimuth in azimuth_rad {
            for elevation in elevation_rad {
                let corner = frame.direction_world(azimuth, elevation);
                let cosine = (center[0] * corner[0] + center[1] * corner[1] + center[2] * corner[2]).clamp(-1.0, 1.0);
                largest = largest.max(cosine.acos());
            }
        }
        largest
    }

    /// Centre and bounding size, computed once per cell so per-pair
    /// interval bounds cost a few flops (this is the search's inner loop).
    pub fn geometry(&self, frame: &LayoutFrame) -> CellGeometry {
        match *self {
            Self::Near { .. } => CellGeometry::Near {
                center_m: self.representative_position_m(frame, 0.0),
                bounding_radius_m: self.bounding_radius_m(frame),
            },
            Self::FarField { azimuth_rad, elevation_rad } => CellGeometry::FarField {
                direction: frame.direction_world(midpoint(azimuth_rad), midpoint(elevation_rad)),
                angular_radius_rad: self.angular_radius_rad(frame),
            },
        }
    }

    /// Conservative TDOA interval (centre, half-width) of pair (a, b) over
    /// the cell; see `CellGeometry::tdoa_interval_s`.
    pub fn tdoa_interval_s(
        &self,
        sensor_a_m: [f64; 3],
        sensor_b_m: [f64; 3],
        frame: &LayoutFrame,
        sound_speed_mps: f64,
    ) -> (f64, f64) {
        self.geometry(frame).tdoa_interval_s(sensor_a_m, sensor_b_m, sound_speed_mps)
    }

    /// Halve every axis: 8 children (near) or 4 (far).
    pub fn subdivide(&self) -> Vec<SearchCell> {
        match *self {
            Self::Near { radial_m, azimuth_rad, height_m } => {
                let mut children = Vec::with_capacity(8);
                for radial in halves(radial_m) {
                    for azimuth in halves(azimuth_rad) {
                        for height in halves(height_m) {
                            children.push(Self::Near { radial_m: radial, azimuth_rad: azimuth, height_m: height });
                        }
                    }
                }
                children
            }
            Self::FarField { azimuth_rad, elevation_rad } => {
                let mut children = Vec::with_capacity(4);
                for azimuth in halves(azimuth_rad) {
                    for elevation in halves(elevation_rad) {
                        children.push(Self::FarField { azimuth_rad: azimuth, elevation_rad: elevation });
                    }
                }
                children
            }
        }
    }
}

/// Precomputed cell centre and size for interval bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellGeometry {
    Near { center_m: [f64; 3], bounding_radius_m: f64 },
    FarField { direction: [f64; 3], angular_radius_rad: f64 },
}

impl CellGeometry {
    /// Conservative TDOA interval (centre, half-width) of pair (a, b) over
    /// the cell, τ_ab = (|x − a| − |x − b|) / c.
    ///
    /// Near: second-order Taylor bound around the centre. |∇τ| = |u_a − u_b|/c
    /// and each range's Hessian norm is ≤ 1/(d − R) inside the ball, capped by
    /// the global Lipschitz bound 2R/c. Far: |Δ(u·(b − a))| ≤ |b − a|·Δangle.
    pub fn tdoa_interval_s(&self, sensor_a_m: [f64; 3], sensor_b_m: [f64; 3], sound_speed_mps: f64) -> (f64, f64) {
        match *self {
            Self::Near { center_m, bounding_radius_m: radius } => {
                let offset_a = sub(center_m, sensor_a_m);
                let offset_b = sub(center_m, sensor_b_m);
                let range_a = norm(offset_a);
                let range_b = norm(offset_b);
                let center_lag = (range_a - range_b) / sound_speed_mps;
                let lipschitz_bound = 2.0 * radius / sound_speed_mps;
                if range_a <= radius || range_b <= radius {
                    return (center_lag, lipschitz_bound);
                }
                let unit_a = scale(offset_a, 1.0 / range_a);
                let unit_b = scale(offset_b, 1.0 / range_b);
                let gradient_norm = norm(sub(unit_a, unit_b));
                let curvature = 1.0 / (range_a - radius) + 1.0 / (range_b - radius);
                let taylor_bound = (gradient_norm * radius + 0.5 * radius * radius * curvature) / sound_speed_mps;
                (center_lag, taylor_bound.min(lipschitz_bound))
            }
            Self::FarField { direction, angular_radius_rad } => {
                let baseline = sub(sensor_b_m, sensor_a_m);
                (dot(direction, baseline) / sound_speed_mps, norm(baseline) * angular_radius_rad / sound_speed_mps)
            }
        }
    }
}

/// Horizontal footprint shared by a stack of near cells (or one far bearing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizontalColumn {
    pub radial_m: [f64; 2],
    pub layout_azimuth_rad: [f64; 2],
    /// Near columns: annular-sector area. Far columns: 0 (see likelihood_region).
    pub area_m2: f64,
    pub is_far_field: bool,
}

/// The coarse search grid: every cell and the column it belongs to.
#[derive(Clone, Debug)]
pub struct CoarsePolarGrid {
    pub cells: Vec<SearchCell>,
    pub cell_column_index: Vec<usize>,
    pub columns: Vec<HorizontalColumn>,
    /// Height slab thickness of near cells (z-marginalization weight).
    pub height_layer_thickness_m: f64,
}

fn even_at_least(count: usize, minimum: usize) -> usize {
    let count = count.max(minimum).max(2);
    count + count % 2
}

/// Ring edges: uniform to `dense_radius_m`, then ratio (1 + 2π/n) to outer.
fn ring_edges_and_sector_counts(config: &PolarGridConfig) -> Vec<([f64; 2], usize)> {
    let mut rings = Vec::new();
    let dense_ring_count = (config.dense_radius_m / config.dense_cell_size_m).ceil().max(1.0) as usize;
    let dense_step = config.dense_radius_m / dense_ring_count as f64;
    for ring in 0..dense_ring_count {
        let inner = ring as f64 * dense_step;
        let outer = inner + dense_step;
        let sectors = (TAU * midpoint([inner, outer]) / config.dense_cell_size_m).ceil() as usize;
        rings.push(([inner, outer], even_at_least(sectors, config.minimum_ring_azimuth_count)));
    }
    let log_sectors = even_at_least(config.log_ring_azimuth_count, config.minimum_ring_azimuth_count);
    let ratio = 1.0 + TAU / log_sectors as f64;
    let mut inner = config.dense_radius_m;
    while inner < config.outer_radius_m {
        let outer = (inner * ratio).min(config.outer_radius_m);
        rings.push(([inner, outer], log_sectors));
        inner = outer;
    }
    rings
}

fn sector_bounds(sector: usize, sector_count: usize) -> [f64; 2] {
    let width = TAU / sector_count as f64;
    [sector as f64 * width, (sector + 1) as f64 * width]
}

impl CoarsePolarGrid {
    /// Far-field bearings are given the area of the outermost ring sector of
    /// equal width: "beyond the grid" is as likely as the last ring, no more.
    pub fn far_field_area_per_radian_m2(&self) -> f64 {
        self.columns
            .iter()
            .filter(|column| !column.is_far_field)
            .max_by(|left, right| left.radial_m[1].total_cmp(&right.radial_m[1]))
            .map(|column| column.area_m2 / (column.layout_azimuth_rad[1] - column.layout_azimuth_rad[0]))
            .unwrap_or(0.0)
    }

    pub fn build(config: &PolarGridConfig) -> Self {
        let [lowest_height, highest_height] = config.source_height_above_ground_range_m;
        let layer_count = config.height_layer_count.max(1);
        let layer_thickness = (highest_height - lowest_height) / layer_count as f64;
        let mut cells = Vec::new();
        let mut cell_column_index = Vec::new();
        let mut columns = Vec::new();

        for (radial, sector_count) in ring_edges_and_sector_counts(config) {
            for sector in 0..sector_count {
                let azimuth = sector_bounds(sector, sector_count);
                let column_index = columns.len();
                columns.push(HorizontalColumn {
                    radial_m: radial,
                    layout_azimuth_rad: azimuth,
                    area_m2: 0.5 * (azimuth[1] - azimuth[0]) * (radial[1].powi(2) - radial[0].powi(2)),
                    is_far_field: false,
                });
                for layer in 0..layer_count {
                    let bottom = lowest_height + layer as f64 * layer_thickness;
                    cells.push(SearchCell::Near {
                        radial_m: radial,
                        azimuth_rad: azimuth,
                        height_m: [bottom, bottom + layer_thickness],
                    });
                    cell_column_index.push(column_index);
                }
            }
        }

        let far_sectors = even_at_least(config.far_field_azimuth_count, config.minimum_ring_azimuth_count);
        let [lowest_elevation, highest_elevation] = config.far_field_elevation_range_deg.map(f64::to_radians);
        let elevation_layers = config.far_field_elevation_layer_count.max(1);
        let elevation_step = (highest_elevation - lowest_elevation) / elevation_layers as f64;
        for sector in 0..far_sectors {
            let azimuth = sector_bounds(sector, far_sectors);
            let column_index = columns.len();
            columns.push(HorizontalColumn {
                radial_m: [config.outer_radius_m, f64::INFINITY],
                layout_azimuth_rad: azimuth,
                area_m2: 0.0,
                is_far_field: true,
            });
            for layer in 0..elevation_layers {
                let bottom = lowest_elevation + layer as f64 * elevation_step;
                cells.push(SearchCell::FarField {
                    azimuth_rad: azimuth,
                    elevation_rad: [bottom, bottom + elevation_step],
                });
                cell_column_index.push(column_index);
            }
        }

        Self {
            cells,
            cell_column_index,
            columns,
            height_layer_thickness_m: layer_thickness,
        }
    }
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn scale(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOUND_SPEED_MPS: f64 = 343.0;

    fn frame() -> LayoutFrame {
        LayoutFrame { origin_xy_m: [3.0, -2.0], principal_axis_azimuth_rad: 0.4, ground_height_m: -1.5 }
    }

    #[test]
    fn coarse_grid_size_matches_plan_budget() {
        let grid = CoarsePolarGrid::build(&PolarGridConfig::default());
        let near_columns = grid.columns.iter().filter(|column| !column.is_far_field).count();
        assert!((1_500..=3_500).contains(&near_columns), "near columns {near_columns}");
    }

    #[test]
    fn tdoa_interval_contains_every_sampled_point_of_the_cell() {
        let frame = frame();
        let sensors = [[0.0, 0.0, 0.0], [14.0, 3.0, 0.5], [3.02, -1.98, 0.02], [2.0, 8.0, 1.0]];
        let cells = [
            SearchCell::Near { radial_m: [0.0, 2.0], azimuth_rad: [0.0, TAU / 4.0], height_m: [0.0, 4.4] },
            SearchCell::Near { radial_m: [10.0, 12.0], azimuth_rad: [1.0, 1.2], height_m: [4.4, 8.8] },
            SearchCell::Near { radial_m: [120.0, 131.0], azimuth_rad: [2.0, 2.087], height_m: [0.0, 4.4] },
        ];
        for cell in cells {
            let SearchCell::Near { radial_m, azimuth_rad, height_m } = cell else { unreachable!() };
            for a in 0..sensors.len() {
                for b in 0..sensors.len() {
                    let (center, half_width) = cell.tdoa_interval_s(sensors[a], sensors[b], &frame, SOUND_SPEED_MPS);
                    for i in 0..=6 {
                        for j in 0..=6 {
                            for k in 0..=2 {
                                let point = frame.layout_to_world(
                                    radial_m[0] + (radial_m[1] - radial_m[0]) * i as f64 / 6.0,
                                    azimuth_rad[0] + (azimuth_rad[1] - azimuth_rad[0]) * j as f64 / 6.0,
                                    height_m[0] + (height_m[1] - height_m[0]) * k as f64 / 2.0,
                                );
                                let lag = (norm(sub(point, sensors[a])) - norm(sub(point, sensors[b]))) / SOUND_SPEED_MPS;
                                assert!((lag - center).abs() <= half_width + 1e-12, "{cell:?} pair {a}-{b}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn frame_round_trip() {
        let frame = frame();
        let world = frame.layout_to_world(17.0, 5.5, 9.0);
        let (radius, azimuth, height) = frame.world_to_layout(world);
        assert!((radius - 17.0).abs() < 1e-9 && (azimuth - 5.5).abs() < 1e-9 && (height - 9.0).abs() < 1e-9);
    }
}
