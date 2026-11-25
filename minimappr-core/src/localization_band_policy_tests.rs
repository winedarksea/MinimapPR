//! Parity values shared with `tests/test_localization_band_policy.py`: both
//! sides assert the same ceilings and taper gains.

use super::*;
use crate::sirith_array_geometry::{SIRITH_MIC_POSITIONS_M, SIRITH_PLANAR_MIC_OFFSETS_M};

const SOUND_SPEED_MPS: f32 = 343.2;

#[test]
fn sirith_tetra_ceiling_comes_from_its_widest_pair() {
    // MK1/MK3 to MK4 is 52.5 mm, a little over the 50 mm base edge, so the
    // derived ceiling sits ~5 % under the old global 3430 Hz.
    let ceiling_hz = array_alias_ceiling_hz(SIRITH_MIC_POSITIONS_M.iter(), SOUND_SPEED_MPS);
    assert!((ceiling_hz - 3266.0).abs() < 2.0, "tetra ceiling {ceiling_hz}");
}

#[test]
fn sirith_planar_ceiling_comes_from_its_diagonal() {
    let ceiling_hz = array_alias_ceiling_hz(SIRITH_PLANAR_MIC_OFFSETS_M.iter(), SOUND_SPEED_MPS);
    assert!((ceiling_hz - 3432.0).abs() < 1.0, "planar ceiling {ceiling_hz}");
}

#[test]
fn single_or_coincident_mics_have_no_alias_ceiling() {
    assert_eq!(array_alias_ceiling_hz([[1.0, 2.0, 3.0]].iter(), SOUND_SPEED_MPS), 0.0);
    assert_eq!(
        array_alias_ceiling_hz([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]].iter(), SOUND_SPEED_MPS),
        0.0
    );
}

fn tetra_plus_points() -> Vec<[f32; 3]> {
    let mut positions: Vec<[f32; 3]> = SIRITH_MIC_POSITIONS_M.to_vec();
    positions.extend([[20.0, 0.0, 1.5], [0.0, 30.0, 1.5]]);
    positions
}

#[test]
fn intra_array_pairs_use_their_array_ceiling_and_cross_node_pairs_the_wide_band() {
    let positions = tetra_plus_points();
    let fronts = ChannelPairFrontEnds::resolve(
        LocalizationBandPolicy::default(),
        &[0, 1, 2, 3, 4, 5],
        &positions,
        Some(&[0, 0, 0, 0, 1, 2]),
        SOUND_SPEED_MPS,
    );
    let intra = fronts.pair_front_end(0, 3);
    assert_eq!(intra.taper_fraction, None);
    assert_eq!(intra.band_hz[0], 50.0);
    assert!((intra.band_hz[1] - 3266.0).abs() < 2.0);
    let cross = fronts.pair_front_end(2, 4);
    assert_eq!(cross.band_hz, [50.0, 11_000.0]);
    assert_eq!(cross.taper_fraction, Some(0.2));
    assert_eq!(cross.phat_beta, Some(0.7));
    assert!(fronts.is_cross_node_pair(4, 5));
}

#[test]
fn only_solving_mics_set_an_arrays_ceiling() {
    // MK4 inactive: the widest correlated pair is the 50 mm base edge.
    let positions = tetra_plus_points();
    let fronts = ChannelPairFrontEnds::resolve(
        LocalizationBandPolicy::default(),
        &[0, 1, 2, 4, 5],
        &positions,
        Some(&[0, 0, 0, 0, 1, 2]),
        SOUND_SPEED_MPS,
    );
    assert!((fronts.pair_front_end(0, 1).band_hz[1] - 3432.0).abs() < 2.0);
}

#[test]
fn an_explicit_intra_array_ceiling_overrides_the_geometry() {
    let positions = tetra_plus_points();
    let fronts = ChannelPairFrontEnds::resolve(
        LocalizationBandPolicy::with_intra_array_band([300.0, 3500.0]),
        &[0, 1, 2, 3],
        &positions,
        None,
        SOUND_SPEED_MPS,
    );
    assert_eq!(
        fronts.pair_front_end(1, 2),
        PairFrontEnd { band_hz: [300.0, 3500.0], taper_fraction: None, phat_beta: None }
    );
}

#[test]
fn no_partition_puts_every_channel_on_one_array() {
    let positions: Vec<[f32; 3]> = SIRITH_MIC_POSITIONS_M.to_vec();
    let fronts = ChannelPairFrontEnds::resolve(
        LocalizationBandPolicy::default(),
        &[0, 1, 2, 3],
        &positions,
        None,
        SOUND_SPEED_MPS,
    );
    assert!(!fronts.is_cross_node_pair(0, 3));
    assert_eq!(fronts.pair_front_end(0, 3).taper_fraction, None);
}

#[test]
fn zero_taper_keeps_the_rectangular_window_on_cross_node_pairs() {
    let positions = tetra_plus_points();
    let policy = LocalizationBandPolicy { cross_node_window_taper_fraction: 0.0, ..LocalizationBandPolicy::default() };
    let fronts = ChannelPairFrontEnds::resolve(policy, &[0, 4], &positions, Some(&[0, 0, 0, 0, 1, 2]), SOUND_SPEED_MPS);
    assert_eq!(fronts.pair_front_end(0, 4).taper_fraction, None);
}

#[test]
fn beta_of_one_is_plain_phat_on_cross_node_pairs() {
    let positions = tetra_plus_points();
    let policy = LocalizationBandPolicy { cross_node_phat_beta: 1.0, ..LocalizationBandPolicy::default() };
    let fronts = ChannelPairFrontEnds::resolve(policy, &[0, 4], &positions, Some(&[0, 0, 0, 0, 1, 2]), SOUND_SPEED_MPS);
    assert_eq!(fronts.pair_front_end(0, 4).phat_beta, None);
}

#[test]
fn tukey_taper_ramps_both_ends_and_leaves_the_middle() {
    let mut window = vec![1.0_f32; 20];
    apply_tukey_taper(&mut window, 0.4);
    // 4-sample ramps: 0, 0.1464, 0.5, 0.8536, then flat.
    let expected_ramp = [0.0, 0.146_446_6, 0.5, 0.853_553_4];
    for (index, expected) in expected_ramp.iter().enumerate() {
        assert!((window[index] - expected).abs() < 1e-6, "head {index}: {}", window[index]);
        assert!((window[19 - index] - expected).abs() < 1e-6, "tail {index}: {}", window[19 - index]);
    }
    assert!(window[4..16].iter().all(|gain| *gain == 1.0));
}

#[test]
fn tukey_taper_of_zero_fraction_or_tiny_window_is_a_no_op() {
    let mut window = vec![1.0_f32; 3];
    apply_tukey_taper(&mut window, 0.0);
    assert_eq!(window, vec![1.0; 3]);
    let mut single = vec![1.0_f32];
    apply_tukey_taper(&mut single, 1.0);
    assert_eq!(single, vec![1.0]);
}
