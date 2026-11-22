//! Layout-SRP localization accuracy against simulated ground truth:
//! sources inside a phone group, a mixed tetra + phone group, and distant
//! sources scored on bearing (test strategy: inside median < 1.5 m,
//! P95 < 3 m; outside at 50 m bearing error < 5°).

mod srp_layout_support;

use minimappr_core::srp_layout::UncertaintyDisplay;
use srp_layout_support::*;

const INSIDE_MEDIAN_ERROR_GATE_M: f64 = 1.5;
const INSIDE_P95_ERROR_GATE_M: f64 = 3.0;
const OUTSIDE_BEARING_ERROR_GATE_DEG: f64 = 5.0;

/// Sources inside / at the edge of the near layout, canopy heights.
const INSIDE_SOURCES_M: [[f64; 3]; 8] = [
    [7.0, 9.0, 6.0],
    [2.0, 4.0, 3.0],
    [12.0, 12.0, 12.0],
    [15.0, 6.0, 2.0],
    [4.0, 16.0, 9.0],
    [-2.0, 6.0, 15.0],
    [10.0, 19.0, 4.0],
    [9.0, 1.0, 20.0],
];

fn inside_errors(devices_for_seed: impl Fn(u64) -> Vec<minimappr_sim::scenario::DeviceConfig>) -> Vec<f64> {
    INSIDE_SOURCES_M
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let rendered = render(&scene(devices_for_seed(index as u64), *source, index, 100 + index as u64));
            let result = localize(&rendered);
            let error = horizontal_error_m(result.estimate.position_m, *source);
            eprintln!(
                "source {source:?}: estimate {:?} error {error:.2} m, z err {:.1} m, display {:?}",
                result.estimate.position_m.map(|v| (v * 100.0).round() / 100.0),
                result.estimate.position_m[2] - source[2],
                matches!(result.display, UncertaintyDisplay::Contour)
            );
            error
        })
        .collect()
}

#[test]
fn sources_inside_phone_group_localize_within_gates() {
    let errors = inside_errors(near_phone_layout);
    let (median, p95) = (percentile(&errors, 0.5), percentile(&errors, 0.95));
    eprintln!("inside phone group: median {median:.2} m, p95 {p95:.2} m");
    assert!(median < INSIDE_MEDIAN_ERROR_GATE_M, "median {median}");
    assert!(p95 < INSIDE_P95_ERROR_GATE_M, "p95 {p95}");
}

#[test]
fn mixed_tetra_and_phone_group_localizes_within_gates() {
    let errors = inside_errors(|seed| {
        let mut devices = near_phone_layout(seed);
        devices.truncate(3);
        devices.push(tetra("tetra-0", [6.0, 21.0, 1.5]));
        devices.push(tetra("tetra-1", [-5.0, 11.0, 1.5]));
        devices
    });
    let (median, p95) = (percentile(&errors, 0.5), percentile(&errors, 0.95));
    eprintln!("mixed group: median {median:.2} m, p95 {p95:.2} m");
    assert!(median < INSIDE_MEDIAN_ERROR_GATE_M, "median {median}");
    assert!(p95 < INSIDE_P95_ERROR_GATE_M, "p95 {p95}");
}

#[test]
fn distant_sources_have_accurate_bearings() {
    let ranges_m = [50.0, 100.0, 150.0];
    let azimuths_deg = [20.0, 135.0, 250.0];
    let mut worst: f64 = 0.0;
    for (range_index, range) in ranges_m.iter().enumerate() {
        for (azimuth_index, azimuth_deg) in azimuths_deg.iter().enumerate() {
            let variant = range_index * azimuths_deg.len() + azimuth_index;
            let centroid = [6.6, 10.0];
            let azimuth = f64::to_radians(*azimuth_deg);
            let source = [centroid[0] + range * azimuth.cos(), centroid[1] + range * azimuth.sin(), 8.0];
            let rendered = render(&scene(near_phone_layout(variant as u64), source, variant, 200 + variant as u64));
            let result = localize(&rendered);
            let error = bearing_error_deg(result.estimate.world_azimuth_rad, result.frame.origin_xy_m, source);
            eprintln!(
                "range {range} az {azimuth_deg}: bearing err {error:.2}°, est range {:.1} m, far {}, fan {}",
                result.estimate.horizontal_range_m,
                result.estimate.is_far_field,
                matches!(result.display, UncertaintyDisplay::BearingFan(_))
            );
            worst = worst.max(error);
        }
    }
    assert!(worst < OUTSIDE_BEARING_ERROR_GATE_DEG, "worst bearing error {worst}");
}

#[test]
fn far_field_source_reports_bearing_fan_with_accurate_bearing() {
    for (variant, azimuth_deg) in [(0_usize, 60.0_f64), (1, 200.0), (2, 310.0)] {
        let centroid = [6.6, 10.0];
        let azimuth = azimuth_deg.to_radians();
        let range = 400.0;
        let source = [centroid[0] + range * azimuth.cos(), centroid[1] + range * azimuth.sin(), 15.0];
        let rendered = render(&scene(near_phone_layout(variant as u64), source, variant, 300 + variant as u64));
        let result = localize(&rendered);
        let error = bearing_error_deg(result.estimate.world_azimuth_rad, result.frame.origin_xy_m, source);
        eprintln!("far field az {azimuth_deg}: bearing err {error:.2}°, display {:?}", result.display);
        assert!(error < OUTSIDE_BEARING_ERROR_GATE_DEG, "bearing error {error}");
        assert!(matches!(result.display, UncertaintyDisplay::BearingFan(_)), "expected a bearing fan");
    }
}
