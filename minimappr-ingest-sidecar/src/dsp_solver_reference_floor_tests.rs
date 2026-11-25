use std::collections::BTreeMap;

use super::*;
use crate::dsp_worker::solve_layout_test_fixture::tetra_nodes_layout;

const RELATIVE_DB: f32 = 3.0;

fn states_with_rms(per_channel_rms: &[f32]) -> Vec<LocalizationChannelState> {
    per_channel_rms
        .iter()
        .map(|rms| LocalizationChannelState {
            coverage: None,
            window: vec![*rms; 64],
        })
        .collect()
}

fn floor_rms(floor: SolverReferenceChannelFloor) -> f32 {
    let relative = floor
        .noise_floor_relative
        .expect("an active channel always yields a noise floor");
    assert_eq!(relative.relative_db, RELATIVE_DB);
    relative.noise_floor_rms
}

#[test]
fn quietest_active_node_floor_sets_the_bar() {
    let layout = tetra_nodes_layout(&[("noisy", [0.0, 0.0, 0.0]), ("quiet", [30.0, 0.0, 0.0])]);
    let states = states_with_rms(&[1e-2; 8]);
    let tracked = BTreeMap::from([("noisy".to_string(), 1e-2), ("quiet".to_string(), 2e-5)]);
    let all_channels: Vec<usize> = (0..8).collect();
    let floor =
        solver_reference_channel_floor(&layout, &all_channels, &states, &tracked, RELATIVE_DB);
    assert_eq!(floor_rms(floor), 2e-5);
    // The quiet node not active: only the noisy floor applies.
    let noisy_only =
        solver_reference_channel_floor(&layout, &[0, 1, 2, 3], &states, &tracked, RELATIVE_DB);
    assert_eq!(floor_rms(noisy_only), 1e-2);
}

#[test]
fn node_without_a_floor_bootstraps_from_its_own_level_like_the_trigger() {
    let layout = tetra_nodes_layout(&[("seen", [0.0, 0.0, 0.0]), ("new", [30.0, 0.0, 0.0])]);
    let states = states_with_rms(&[3e-4, 3e-4, 3e-4, 3e-4, 1e-4, 2e-4, 1e-4, 1e-4]);
    let all_channels: Vec<usize> = (0..8).collect();

    let tracked = BTreeMap::from([("seen".to_string(), 1e-3)]);
    let floor =
        solver_reference_channel_floor(&layout, &all_channels, &states, &tracked, RELATIVE_DB);
    // Seeded at its own current level, it can never pull the bar below its
    // own window (the seen node's 1e-3 floor is higher).
    assert!((floor_rms(floor) - 2e-4).abs() < 1e-9);

    // ...and digital silence is clamped like the trigger's floor, so it still fails.
    let silent = states_with_rms(&[0.0; 4]);
    let solo = tetra_nodes_layout(&[("muted", [0.0, 0.0, 0.0])]);
    let muted = solver_reference_channel_floor(
        &solo,
        &[0, 1, 2, 3],
        &silent,
        &BTreeMap::new(),
        RELATIVE_DB,
    );
    assert!(floor_rms(muted) > 0.0);
    assert!(muted.rejects(0.0));
}

#[test]
fn no_active_channel_leaves_only_the_absolute_level() {
    let layout = tetra_nodes_layout(&[("a", [0.0, 0.0, 0.0])]);
    let floor = solver_reference_channel_floor(
        &layout,
        &[],
        &states_with_rms(&[1.0; 4]),
        &BTreeMap::new(),
        RELATIVE_DB,
    );
    assert_eq!(floor, SolverReferenceChannelFloor::default());
}
