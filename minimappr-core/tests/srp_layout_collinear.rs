//! Single-file trail: five phones nearly in a line cannot tell left of the
//! trail from right. The solver must flag collinearity and report both
//! mirror candidates (never a confident wrong side); an array node in the
//! line breaks the tie through its DOA.

mod srp_layout_support;

use srp_layout_support::*;

/// Trail along a 30° heading, phones 6 m apart, small cross-trail jitter.
fn trail_devices() -> Vec<minimappr_sim::scenario::DeviceConfig> {
    let heading = 30.0_f64.to_radians();
    let cross_trail_jitter_m = [0.0, 0.15, -0.1, 0.1, -0.15];
    (0..5)
        .map(|index| {
            let along = index as f64 * 6.0;
            let cross = cross_trail_jitter_m[index];
            let x = along * heading.cos() - cross * heading.sin();
            let y = along * heading.sin() + cross * heading.cos();
            phone(&format!("trail-{index}"), [x, y, 1.4], fixed_clock(0.002 * index as f64, 0.0))
        })
        .collect()
}

/// Bird 15 m to the left of the trail midpoint, in the canopy.
fn bird_beside_trail(side_sign: f64, along_m: f64) -> [f64; 3] {
    let heading = 30.0_f64.to_radians();
    let cross = 15.0 * side_sign;
    [along_m * heading.cos() - cross * heading.sin(), along_m * heading.sin() + cross * heading.cos(), 8.0]
}

/// A candidate within this of truth counts as "the true side".
const CANDIDATE_MATCH_RADIUS_M: f64 = 2.0;

#[test]
fn collinear_trail_reports_both_mirror_candidates_unresolved() {
    for (variant, (side, along)) in [(1.0, 12.0), (-1.0, 5.0), (1.0, 20.0)].into_iter().enumerate() {
        let source = bird_beside_trail(side, along);
        let result = localize(&render(&scene(trail_devices(), source, variant, 600 + variant as u64)));
        assert!(result.geometry.is_collinear, "aspect {}", result.geometry.horizontal_aspect_ratio);
        let mirror = result.mirror.expect("collinear layouts report mirror candidates");
        let errors = [
            horizontal_error_m(mirror.primary.position_m, source),
            horizontal_error_m(mirror.mirror.position_m, source),
        ];
        eprintln!(
            "trail source {source:?}: candidate errors {:.2} / {:.2} m, gap {:.2} T (array {:.2}), resolved {}, aspect {:.3}",
            errors[0], errors[1], mirror.score_gap_temperatures, mirror.array_score_gap_temperatures, mirror.is_resolved, result.geometry.horizontal_aspect_ratio
        );
        assert!(errors.iter().any(|error| *error < CANDIDATE_MATCH_RADIUS_M), "neither candidate near truth");
        // Never a confident wrong side.
        assert!(!mirror.is_resolved || errors[0] < CANDIDATE_MATCH_RADIUS_M, "confidently mirrored");
        assert!(!mirror.is_resolved, "a mono-phone line cannot resolve the mirror");
    }
}

#[test]
fn tetra_node_on_the_trail_resolves_the_mirror_to_the_true_side() {
    for (variant, (side, along)) in [(1.0, 12.0), (-1.0, 9.0)].into_iter().enumerate() {
        let mut devices = trail_devices();
        let middle = devices[2].position_m;
        devices[2] = tetra("trail-tetra", [middle[0], middle[1], 1.5]);
        let source = bird_beside_trail(side, along);
        let result = localize(&render(&scene(devices, source, variant, 700 + variant as u64)));
        let mirror = result.mirror.expect("still a collinear layout");
        let primary_error = horizontal_error_m(mirror.primary.position_m, source);
        eprintln!(
            "trail+tetra source {source:?}: primary error {primary_error:.2} m, gap {:.2} T (array {:.2}), resolved {}",
            mirror.score_gap_temperatures, mirror.array_score_gap_temperatures, mirror.is_resolved
        );
        assert!(primary_error < CANDIDATE_MATCH_RADIUS_M, "primary candidate on the wrong side");
        assert!(mirror.is_resolved, "the tetra DOA should resolve left/right");
    }
}
