use super::*;
use crate::sync::chirp_codes::SplitMix64;

const TRUE_ROTATION_RAD: f64 = 0.7;
const TRUE_TRANSLATION_M: [f64; 2] = [120.0, -45.0];
const LAYOUT: [[f64; 2]; 6] = [
    [0.0, 0.0],
    [14.0, 2.0],
    [6.0, 11.0],
    [-5.0, 8.0],
    [9.0, -9.0],
    [20.0, 12.0],
];

fn gaussian(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn truth() -> RigidTransform2d {
    RigidTransform2d {
        rotation_rad: TRUE_ROTATION_RAD,
        translation_m: TRUE_TRANSLATION_M,
        mirrored: false,
    }
}

fn noisy_fixes(scale: f64, sigma_m: f64, seed: u64) -> Vec<Option<GpsFix>> {
    let mut rng = SplitMix64::new(seed);
    LAYOUT
        .iter()
        .map(|p| {
            let world = truth().apply([p[0] * scale, p[1] * scale]);
            Some(GpsFix {
                east_m: world[0] + sigma_m * gaussian(&mut rng),
                north_m: world[1] + sigma_m * gaussian(&mut rng),
                sigma_m,
            })
        })
        .collect()
}

fn mirrored_layout(scale: f64) -> Vec<[f64; 2]> {
    LAYOUT
        .iter()
        .map(|p| [p[0] * scale, -p[1] * scale])
        .collect()
}

#[test]
fn kabsch_recovers_exact_transform() {
    let target: Vec<[f64; 2]> = LAYOUT.iter().map(|p| truth().apply(*p)).collect();
    let fit = weighted_kabsch_2d(&LAYOUT, &target, &[1.0; 6], false);
    assert!((fit.rotation_rad - TRUE_ROTATION_RAD).abs() < 1e-12);
    assert!((fit.translation_m[0] - TRUE_TRANSLATION_M[0]).abs() < 1e-9);
}

#[test]
fn mirrored_mds_layout_is_detected_with_high_confidence() {
    let fit = fit_layout_to_gps(
        &mirrored_layout(1.0),
        &noisy_fixes(1.0, 2.0, 1),
        ChiralityCue::Unknown,
        &AnchorFitConfig::default(),
    )
    .unwrap();
    assert!(fit.transform.mirrored);
    assert!(
        fit.chirality_confidence > 0.99,
        "{}",
        fit.chirality_confidence
    );
    // The un-mirrored layout point 2 lands within GPS noise of the truth.
    let mapped = fit.transform.apply(mirrored_layout(1.0)[2]);
    let expected = truth().apply(LAYOUT[2]);
    assert!(((mapped[0] - expected[0]).powi(2) + (mapped[1] - expected[1]).powi(2)).sqrt() < 3.0);
}

#[test]
fn small_layout_under_canopy_reports_low_chirality_confidence() {
    // 1/10 scale (~2 m spread) with 8 m GPS error: chirality is a coin flip.
    let fit = fit_layout_to_gps(
        &mirrored_layout(0.1),
        &noisy_fixes(0.1, 8.0, 2),
        ChiralityCue::Unknown,
        &AnchorFitConfig::default(),
    )
    .unwrap();
    assert!(
        fit.chirality_confidence < 0.8,
        "{}",
        fit.chirality_confidence
    );
}

#[test]
fn bearing_cue_locks_chirality() {
    let layout = mirrored_layout(0.1);
    let fixes = noisy_fixes(0.1, 8.0, 3);
    let config = AnchorFitConfig::default();
    let proper = fit_layout_to_gps(
        &layout,
        &fixes,
        ChiralityCue::Locked { mirrored: false },
        &config,
    )
    .unwrap();
    let mirror = fit_layout_to_gps(
        &layout,
        &fixes,
        ChiralityCue::Locked { mirrored: true },
        &config,
    )
    .unwrap();
    let true_from = truth().apply(LAYOUT[0]);
    let true_to = truth().apply(LAYOUT[2]);
    let bearing = (true_to[1] - true_from[1]).atan2(true_to[0] - true_from[0]);
    let cue = chirality_from_bearing(&layout, 0, 2, bearing, &proper.transform, &mirror.transform);
    assert_eq!(cue, ChiralityCue::Locked { mirrored: true });
    let locked = fit_layout_to_gps(&layout, &fixes, cue, &config).unwrap();
    assert!(locked.transform.mirrored);
    assert_eq!(locked.chirality_confidence, 1.0);
}

#[test]
fn canopy_outlier_is_rejected_not_averaged() {
    let mut fixes = noisy_fixes(1.0, 2.0, 4);
    if let Some(fix) = fixes[3].as_mut() {
        fix.east_m += 14.0;
        fix.north_m -= 9.0;
    }
    let fit = fit_layout_to_gps(
        &LAYOUT,
        &fixes,
        ChiralityCue::Unknown,
        &AnchorFitConfig::default(),
    )
    .unwrap();
    assert!(fit.outliers[3]);
    assert_eq!(fit.outliers.iter().filter(|o| **o).count(), 1);
    assert!((fit.transform.rotation_rad - TRUE_ROTATION_RAD).abs() < 0.1);
    assert!((fit.transform.translation_m[0] - TRUE_TRANSLATION_M[0]).abs() < 3.0);
}

#[test]
fn quality_score_inflates_sigma_and_missing_fixes_are_skipped() {
    let config = AnchorFitConfig::default();
    assert_eq!(effective_gps_sigma_m(4.0, 1.0, &config), 4.0);
    assert_eq!(effective_gps_sigma_m(4.0, 0.0, &config), 12.0);
    let mut fixes = noisy_fixes(1.0, 2.0, 5);
    fixes[0] = None;
    fixes[1] = None;
    let fit = fit_layout_to_gps(&LAYOUT, &fixes, ChiralityCue::Unknown, &config).unwrap();
    assert!(fit.normalized_residuals[0].is_none());
    let one_fix: Vec<Option<GpsFix>> = fixes
        .iter()
        .enumerate()
        .map(|(i, f)| if i == 2 { *f } else { None })
        .collect();
    assert_eq!(
        fit_layout_to_gps(&LAYOUT, &one_fix, ChiralityCue::Unknown, &config),
        Err(AnchorFitError::TooFewFixes)
    );
}

#[test]
fn compose_and_inverse_agree_with_point_application_including_mirrors() {
    let maps = [
        RigidTransform2d { rotation_rad: 0.7, translation_m: [3.0, -2.0], mirrored: false },
        RigidTransform2d { rotation_rad: -2.1, translation_m: [-11.0, 5.5], mirrored: true },
        RigidTransform2d { rotation_rad: 3.0, translation_m: [0.4, 9.0], mirrored: true },
    ];
    let point = [4.0, -7.5];
    for outer in &maps {
        let back = outer.inverse().apply(outer.apply(point));
        assert!((back[0] - point[0]).abs() < 1e-9 && (back[1] - point[1]).abs() < 1e-9, "{outer:?}");
        for inner in &maps {
            let composed = outer.compose(inner).apply(point);
            let stepwise = outer.apply(inner.apply(point));
            assert!((composed[0] - stepwise[0]).abs() < 1e-9 && (composed[1] - stepwise[1]).abs() < 1e-9);
        }
    }
}
