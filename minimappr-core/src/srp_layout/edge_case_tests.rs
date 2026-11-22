//! Degenerate inputs must return errors or finite results, never panic or
//! spin: one sensor, coincident nodes, a lone tetra, empty and NaN audio,
//! NaN geometry.

use super::*;

const SAMPLE_RATE_HZ: u32 = 48_000;
const WINDOW_SAMPLE_COUNT: usize = 4_800;

fn phone_at(position_m: [f64; 3]) -> LayoutNode {
    LayoutNode {
        position_m,
        mic_offsets_m: vec![[0.0, 0.0, 0.0]],
        clock: NodeClockModel::disciplined(),
        pair_weight: 1.0,
    }
}

fn deterministic_noise(length: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn window() -> IslandAnalysisWindow {
    IslandAnalysisWindow { start_island_time_s: 0.0, sample_count: WINDOW_SAMPLE_COUNT }
}

fn config() -> LayoutSrpConfig {
    LayoutSrpConfig { sample_rate_hz: SAMPLE_RATE_HZ, ground_height_m: Some(0.0), ..LayoutSrpConfig::default() }
}

/// Clips of the same noise burst, one per channel (perfectly coherent at lag 0).
fn coherent_clips<'a>(nodes: &[LayoutNode], noise: &'a [f32]) -> Vec<NodeClip<'a>> {
    nodes
        .iter()
        .map(|node| NodeClip { channels: vec![noise; node.mic_offsets_m.len()], first_sample_index: 0 })
        .collect()
}

#[test]
fn single_phone_is_too_few_sensors() {
    let error = LayoutSrpSearcher::new(&[phone_at([0.0, 0.0, 1.5])], &config()).err();
    assert_eq!(error, Some(LayoutSrpError::TooFewSensors));
    assert_eq!(LayoutSrpSearcher::new(&[], &config()).err(), Some(LayoutSrpError::TooFewSensors));
}

#[test]
fn non_finite_geometry_and_config_are_rejected() {
    let nodes = [phone_at([0.0, 0.0, 1.5]), phone_at([f64::NAN, 3.0, 1.5])];
    assert_eq!(LayoutSrpSearcher::new(&nodes, &config()).err(), Some(LayoutSrpError::InvalidInput { node_index: Some(1) }));
    let nodes = [phone_at([0.0, 0.0, 1.5]), phone_at([5.0, 3.0, 1.5])];
    let zero_rate = LayoutSrpConfig { sample_rate_hz: 0, ..config() };
    assert_eq!(LayoutSrpSearcher::new(&nodes, &zero_rate).err(), Some(LayoutSrpError::InvalidInput { node_index: None }));
}

#[test]
fn empty_and_nan_audio_return_errors_without_panicking() {
    let nodes = [phone_at([0.0, 0.0, 1.5]), phone_at([8.0, 0.0, 1.5]), phone_at([0.0, 8.0, 1.5])];
    let searcher = LayoutSrpSearcher::new(&nodes, &config()).expect("searcher");
    let empty: Vec<f32> = Vec::new();
    let empty_clips: Vec<NodeClip<'_>> =
        nodes.iter().map(|_| NodeClip { channels: vec![empty.as_slice()], first_sample_index: 0 }).collect();
    assert_eq!(searcher.localize(&empty_clips, window()).err(), Some(LayoutSrpError::NoCoherentPairs));
    let nan_audio = vec![f32::NAN; WINDOW_SAMPLE_COUNT];
    assert!(searcher.localize(&coherent_clips(&nodes, &nan_audio), window()).is_err());
    let zero_window = IslandAnalysisWindow { sample_count: 0, ..window() };
    let noise = deterministic_noise(WINDOW_SAMPLE_COUNT, 1);
    assert_eq!(
        searcher.localize(&coherent_clips(&nodes, &noise), zero_window).err(),
        Some(LayoutSrpError::EmptyAnalysisWindow)
    );
}

#[test]
fn coincident_nodes_and_lone_tetra_give_finite_results() {
    let noise = deterministic_noise(WINDOW_SAMPLE_COUNT, 2);
    let coincident = [phone_at([2.0, 2.0, 1.5]), phone_at([2.0, 2.0, 1.5])];
    let result = localize_layout(&coincident, &coherent_clips(&coincident, &noise), window(), &config()).expect("coincident");
    assert!(result.estimate.position_m.iter().all(|value| value.is_finite()));
    assert!(result.geometry.is_collinear);

    let tetra = [LayoutNode {
        mic_offsets_m: crate::sirith_array_geometry::SIRITH_MIC_POSITIONS_M
            .iter()
            .map(|mic| mic.map(f64::from))
            .collect(),
        ..phone_at([0.0, 0.0, 1.5])
    }];
    let result = localize_layout(&tetra, &coherent_clips(&tetra, &noise), window(), &config()).expect("lone tetra");
    assert!(result.estimate.position_m.iter().all(|value| value.is_finite()));
    assert!(result.mirror.is_some(), "a single node is a degenerate (collinear) layout");
}
