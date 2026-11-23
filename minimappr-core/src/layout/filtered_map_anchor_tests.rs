use super::*;
use crate::sync::chirp_codes::SplitMix64;

const LAYOUT: [[f64; 2]; 5] = [[0.0, 0.0], [14.0, 2.0], [6.0, 11.0], [-5.0, 8.0], [9.0, -9.0]];
const GPS_SIGMA_M: f64 = 6.0;
const BIAS_STD_M: f64 = 4.8;
const JITTER_STD_M: f64 = 3.6;
const FIX_INTERVAL_S: f64 = 5.0;

fn gaussian(rng: &mut SplitMix64) -> f64 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn world() -> RigidTransform2d {
    RigidTransform2d { rotation_rad: 0.4, translation_m: [200.0, -80.0], mirrored: false }
}

/// Still group, per-phone multipath bias + jitter; phone 0 jumps 20 m east
/// from `jump_epoch` on (persistently) when given.
fn run(epochs: u64, jump_epoch: Option<u64>) -> FilteredMapAnchor<u32> {
    let mut bias_rng = SplitMix64::new(11);
    let biases: Vec<[f64; 2]> = LAYOUT.iter().map(|_| [BIAS_STD_M * gaussian(&mut bias_rng), BIAS_STD_M * gaussian(&mut bias_rng)]).collect();
    let mut jitter_rng = SplitMix64::new(12);
    let nodes: Vec<u32> = (0..LAYOUT.len() as u32).collect();
    let mut anchor = FilteredMapAnchor::new(FilteredMapAnchorConfig::default());
    for epoch in 0..epochs {
        let fixes: Vec<Option<GpsFix>> = LAYOUT
            .iter()
            .enumerate()
            .map(|(phone, p)| {
                let truth = world().apply(*p);
                let jump = if phone == 0 && jump_epoch.is_some_and(|start| epoch >= start) { 20.0 } else { 0.0 };
                Some(GpsFix {
                    east_m: truth[0] + biases[phone][0] + jump + JITTER_STD_M * gaussian(&mut jitter_rng),
                    north_m: truth[1] + biases[phone][1] + JITTER_STD_M * gaussian(&mut jitter_rng),
                    sigma_m: GPS_SIGMA_M,
                })
            })
            .collect();
        let round = AnchorRound {
            time_s: epoch as f64 * FIX_INTERVAL_S,
            nodes: &nodes,
            layout_xy: &LAYOUT,
            fixes: &fixes,
            cue: ChiralityCue::Locked { mirrored: false },
            group_is_still: true,
        };
        anchor.update(&round, &AnchorFitConfig::default()).unwrap();
    }
    anchor
}

fn largest_gap(a: &AnchorEstimate, b: &AnchorEstimate) -> f64 {
    LAYOUT
        .iter()
        .map(|p| {
            let (x, y) = (a.transform.apply(*p), b.transform.apply(*p));
            (x[0] - y[0]).hypot(x[1] - y[1])
        })
        .fold(0.0, f64::max)
}

#[test]
fn persistent_jump_on_a_still_phone_is_held_at_its_history() {
    let clean = run(120, None).estimate().unwrap();
    let jumped_anchor = run(120, Some(60));
    let jumped = jumped_anchor.estimate().unwrap();
    assert_eq!(jumped_anchor.held_last_update(), &[0]);
    let gap = largest_gap(&clean, &jumped);
    assert!(gap < 0.5, "persistent jump moved the anchor {gap:.2} m");
}

#[test]
fn reframing_keeps_the_anchored_positions() {
    let mut anchor = run(20, None);
    let before = anchor.estimate().unwrap();
    let frame_change = RigidTransform2d { rotation_rad: 2.2, translation_m: [7.0, -3.0], mirrored: true };
    anchor.reframe(Some(&frame_change));
    let after = anchor.estimate().unwrap();
    for p in LAYOUT {
        let (old, new) = (before.transform.apply(p), after.transform.apply(frame_change.apply(p)));
        assert!((old[0] - new[0]).abs() < 1e-9 && (old[1] - new[1]).abs() < 1e-9);
    }
    anchor.reframe(None);
    assert!(anchor.estimate().is_none());
}

#[test]
fn motion_clears_the_histories_so_nothing_is_held() {
    let mut anchor = run(30, None);
    let nodes: Vec<u32> = (0..LAYOUT.len() as u32).collect();
    let fixes: Vec<Option<GpsFix>> = LAYOUT
        .iter()
        .enumerate()
        .map(|(phone, p)| {
            let truth = world().apply(*p);
            Some(GpsFix { east_m: truth[0] + if phone == 0 { 25.0 } else { 0.0 }, north_m: truth[1], sigma_m: GPS_SIGMA_M })
        })
        .collect();
    let round = AnchorRound { time_s: 200.0, nodes: &nodes, layout_xy: &LAYOUT, fixes: &fixes, cue: ChiralityCue::Locked { mirrored: false }, group_is_still: false };
    anchor.update(&round, &AnchorFitConfig::default()).unwrap();
    assert!(anchor.held_last_update().is_empty());
    let filtered = anchor.filtered_fit().unwrap();
    assert_eq!(filtered.transform, anchor.estimate().unwrap().transform);
}
