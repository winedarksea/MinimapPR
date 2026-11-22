//! End-to-end Layer 2 sessions for the mesh contour calibration: random
//! phone rings with 6 m GPS, calls inside and outside the group, the full
//! chirp-sync → layout → anchor → readiness → clip → LM chain of
//! `minimappr-mesh-sim`. Each fix is scored against the *published polygon*
//! (`mesh::fix_contour`), as its minimum covering σ multiplier.
//!
//! Scoring frame: the host's node positions are aligned to the true ones
//! with a proper rigid fit (as `ScenarioOutcome::scored_fixes` does), so the
//! map anchor's GPS offset is not charged to the contour (it moves phones
//! and birds together on the map). World-frame multipliers are kept too, as
//! a diagnostic of what the anchor's uncertainty would add.

use minimappr_core::layout::gps_anchor_fit::weighted_kabsch_2d;
use minimappr_core::mesh::config::EntityContourConfig;
use minimappr_mesh_sim::invariants::FLAGGED_CHIRALITY_CONFIDENCE;
use minimappr_mesh_sim::scenario::{ring_of_phones, ScenarioSpec};
use minimappr_mesh_sim::{ScenarioOutcome, ScenarioRun};

use crate::trial_geometry::{count_in, uniform_in};

#[derive(Debug, Clone, PartialEq)]
pub struct MeshSessionConfig {
    pub phone_count_range: [usize; 2],
    pub ring_radius_range_m: [f64; 2],
    pub gps_error_std_m: f64,
    pub calls_per_session: usize,
    pub first_call_s: f64,
    pub call_spacing_s: f64,
    pub tail_s: f64,
    /// Even calls (0, 2, …) are inside the ring (within this fraction of its radius),
    /// odd calls outside at a range drawn from `outside_range_m`.
    pub inside_radius_fraction: f64,
    pub outside_range_m: [f64; 2],
    pub source_height_range_m: [f64; 2],
}

impl Default for MeshSessionConfig {
    fn default() -> Self {
        Self {
            phone_count_range: [4, 7],
            ring_radius_range_m: [6.0, 15.0],
            gps_error_std_m: 6.0,
            calls_per_session: 4,
            first_call_s: 20.0,
            call_spacing_s: 12.0,
            tail_s: 15.0,
            inside_radius_fraction: 0.7,
            outside_range_m: [20.0, 60.0],
            source_height_range_m: [2.0, 15.0],
        }
    }
}

/// One fix's contour score.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshFixScore {
    pub seed: u64,
    pub call_index: usize,
    pub is_inside_call: bool,
    pub relative_frame_error_m: f64,
    pub reported_major_sigma_m: f64,
    pub used_node_count: usize,
    pub true_range_from_centroid_m: f64,
    /// Smallest `contour_sigma_multiplier` whose polygon holds the truth.
    pub relative_frame_minimum_multiplier: f64,
    pub world_frame_minimum_multiplier: f64,
    pub chirality_confidence: Option<f64>,
    /// Layout came out mirrored (proper fit to truth fails).
    pub layout_mirrored: bool,
    /// Mirrored but reported with low chirality confidence (the mesh
    /// invariants' "flagged" rule): exempt from the coverage count.
    pub is_flagged_mirror: bool,
}

pub fn session_spec(seed: u64, config: &MeshSessionConfig) -> ScenarioSpec {
    let phone_count = count_in(seed, "mesh_phone_count", config.phone_count_range);
    let radius = uniform_in(seed, "mesh_ring_radius", 0, config.ring_radius_range_m);
    let mut devices = ring_of_phones(phone_count, [0.0, 0.0], radius, seed);
    for device in &mut devices {
        device.gps_error_sigma_m = config.gps_error_std_m;
    }
    let duration = config.first_call_s + config.call_spacing_s * config.calls_per_session as f64 + config.tail_s;
    let mut spec = ScenarioSpec::new("sim_gates_contour_session", duration, devices);
    for call in 0..config.calls_per_session {
        let key = call as u64;
        let angle = std::f64::consts::TAU * uniform_in(seed, "mesh_call_angle", key, [0.0, 1.0]);
        let range = if call % 2 == 0 {
            radius * config.inside_radius_fraction * uniform_in(seed, "mesh_call_radius", key, [0.0, 1.0]).sqrt()
        } else {
            uniform_in(seed, "mesh_call_range", key, config.outside_range_m)
        };
        let height = uniform_in(seed, "mesh_call_height", key, config.source_height_range_m);
        spec = spec.call(config.first_call_s + config.call_spacing_s * call as f64, [range * angle.cos(), range * angle.sin(), height]);
    }
    spec
}

/// Gauge of the published polygon at `point` (host frame): the smallest
/// multiplier k whose k-scaled contour contains it. In whitened coordinates
/// the polygon is a regular n-gon inscribed in the unit circle.
pub fn minimum_contour_multiplier(
    point_xy_m: [f64; 2],
    position_m: [f64; 3],
    covariance_m2: &[[f64; 3]; 3],
    contour: &EntityContourConfig,
) -> f64 {
    let [[a, b, _], [_, d, _], _] = *covariance_m2;
    let half_trace = 0.5 * (a + d);
    let spread = (0.25 * (a - d).powi(2) + b * b).sqrt();
    let (major, minor) = ((half_trace + spread).max(0.0).sqrt(), (half_trace - spread).max(0.0).sqrt());
    if minor.is_nan() || minor <= 0.0 {
        return f64::INFINITY;
    }
    let angle = 0.5 * (2.0 * b).atan2(a - d);
    let (dx, dy) = (point_xy_m[0] - position_m[0], point_xy_m[1] - position_m[1]);
    let (u, v) = ((dx * angle.cos() + dy * angle.sin()) / major, (-dx * angle.sin() + dy * angle.cos()) / minor);
    let sector = std::f64::consts::TAU / contour.contour_point_count as f64;
    let phi = v.atan2(u).rem_euclid(std::f64::consts::TAU);
    let edge_normal = ((phi / sector).floor() + 0.5) * sector;
    u.hypot(v) * (phi - edge_normal).cos() / (0.5 * sector).cos()
}

/// Score every fix of a finished session.
pub fn score_session(outcome: &ScenarioOutcome, contour: &EntityContourConfig) -> Vec<MeshFixScore> {
    // Same traces, same order: `scored_fixes` also keeps traces with a fix.
    let mesh_scored = outcome.scored_fixes();
    outcome
        .traces
        .iter()
        .filter(|trace| trace.fix.is_some())
        .zip(mesh_scored)
        .filter_map(|(trace, mesh_fix)| {
            let fix = trace.fix.as_ref()?;
            let call_index = trace.true_call_index as usize;
            let call = &outcome.world.calls[call_index];
            let estimated: Vec<[f64; 2]> = trace.estimated_node_positions_m.iter().map(|p| [p[0], p[1]]).collect();
            let truth: Vec<[f64; 2]> = trace
                .node_ids
                .iter()
                .map(|node| {
                    let p = outcome.world.position(node.0 as usize, call.time_s);
                    [p[0], p[1]]
                })
                .collect();
            let to_truth = weighted_kabsch_2d(&estimated, &truth, &vec![1.0; estimated.len()], false);
            // Truth carried into the host frame: Rᵀ (q − t).
            let (sin, cos) = to_truth.rotation_rad.sin_cos();
            let (qx, qy) = (call.position_m[0] - to_truth.translation_m[0], call.position_m[1] - to_truth.translation_m[1]);
            let truth_in_host_frame = [cos * qx + sin * qy, -sin * qx + cos * qy];
            let error = (truth_in_host_frame[0] - fix.position_m[0]).hypot(truth_in_host_frame[1] - fix.position_m[1]);
            Some(MeshFixScore {
                seed: outcome.seed,
                call_index,
                is_inside_call: call_index.is_multiple_of(2),
                relative_frame_error_m: error,
                reported_major_sigma_m: fix.horizontal_major_sigma_m,
                used_node_count: fix.used_nodes.len(),
                true_range_from_centroid_m: {
                    let count = truth.len().max(1) as f64;
                    let centre = truth.iter().fold([0.0; 2], |acc, p| [acc[0] + p[0] / count, acc[1] + p[1] / count]);
                    (call.position_m[0] - centre[0]).hypot(call.position_m[1] - centre[1])
                },
                relative_frame_minimum_multiplier: minimum_contour_multiplier(truth_in_host_frame, fix.position_m, &fix.covariance_m2, contour),
                world_frame_minimum_multiplier: minimum_contour_multiplier(
                    [call.position_m[0], call.position_m[1]],
                    fix.position_m,
                    &fix.covariance_m2,
                    contour,
                ),
                chirality_confidence: trace.chirality_confidence,
                layout_mirrored: mesh_fix.layout_mirrored == Some(true),
                is_flagged_mirror: mesh_fix.layout_mirrored == Some(true)
                    && !mesh_fix.chirality_locked
                    && mesh_fix.chirality_confidence.is_none_or(|c| c < FLAGGED_CHIRALITY_CONFIDENCE),
            })
        })
        .collect()
}

/// Run one session and score its fixes.
pub fn run_contour_session(seed: u64, contour: &EntityContourConfig, config: &MeshSessionConfig) -> Vec<MeshFixScore> {
    let mut spec = session_spec(seed, config);
    spec.mesh.entity_contour = contour.clone();
    let outcome = ScenarioRun::new(spec, seed).run();
    score_session(&outcome, contour)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minimappr_core::mesh::fix_contour::horizontal_covariance_contour;

    fn inside_polygon(point: [f64; 2], polygon: &[[f32; 2]]) -> bool {
        let mut inside = false;
        let mut j = polygon.len() - 1;
        for i in 0..polygon.len() {
            let (pi, pj) = (polygon[i].map(f64::from), polygon[j].map(f64::from));
            if (pi[1] > point[1]) != (pj[1] > point[1]) && point[0] < (pj[0] - pi[0]) * (point[1] - pi[1]) / (pj[1] - pi[1]) + pi[0] {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// The gauge agrees with a point-in-polygon test on the shipped contour.
    #[test]
    fn gauge_matches_published_polygon() {
        let covariance = [[4.0, 1.5, 0.0], [1.5, 1.0, 0.0], [0.0, 0.0, 9.0]];
        let position = [10.0, -3.0, 5.0];
        for k in 0..200 {
            let angle = k as f64 * 0.731;
            let radius = 0.2 + (k % 17) as f64 * 0.35;
            let point = [position[0] + radius * angle.cos(), position[1] + radius * angle.sin()];
            let base = EntityContourConfig::default();
            let gauge = minimum_contour_multiplier(point, position, &covariance, &base);
            for (scale, expect_inside) in [(gauge * 1.01, true), (gauge * 0.99, false)] {
                let config = EntityContourConfig { contour_sigma_multiplier: scale, ..base.clone() };
                let polygon = horizontal_covariance_contour(position, &covariance, &config);
                assert_eq!(inside_polygon(point, &polygon), expect_inside, "point {point:?} gauge {gauge}");
            }
        }
    }
}
