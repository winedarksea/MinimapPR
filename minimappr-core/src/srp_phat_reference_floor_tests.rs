//! The solver's reference-channel floor: relative to a tracked noise floor it
//! admits a faint event the old absolute floor rejected, and silence still fails.

use super::*;

const SAMPLE_RATE_HZ: u32 = 16_000;
const WINDOW_SAMPLES: usize = 512;
const SOUND_SPEED_MPS: f32 = 343.2;
const TETRA_MICS_M: [[f32; 3]; 4] = [
    [0.0, 0.050, 0.0],
    [0.0433, 0.025, 0.0],
    [0.0, 0.0, 0.0],
    [0.02165, 0.025, 0.04082],
];
/// A quiet site: well under the old 1e-4 absolute floor.
const QUIET_SITE_NOISE_FLOOR_RMS: f32 = 2.0e-5;
const FAINT_EVENT_DB_OVER_FLOOR: f32 = 6.0;
const SOLVER_FLOOR_RELATIVE_DB: f32 = 3.0;

fn white_noise(mut seed: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

/// A far source's broadband call across the tetra, scaled so the loudest
/// channel has `target_rms`.
fn faint_point_source_windows(target_rms: f32) -> Vec<Vec<f32>> {
    let source_position_m = [40.0, 25.0, 3.0];
    let source = white_noise(0x5eed_0001, WINDOW_SAMPLES + 64);
    let delays_samples: Vec<f32> = TETRA_MICS_M
        .iter()
        .map(|mic| {
            let distance_m = (0..3)
                .map(|axis| (source_position_m[axis] - mic[axis]).powi(2))
                .sum::<f32>()
                .sqrt();
            distance_m / SOUND_SPEED_MPS * SAMPLE_RATE_HZ as f32
        })
        .collect();
    let earliest = delays_samples.iter().copied().fold(f32::INFINITY, f32::min);
    let windows: Vec<Vec<f32>> = delays_samples
        .iter()
        .map(|delay| {
            let relative_delay = delay - earliest;
            (0..WINDOW_SAMPLES)
                .map(|index| {
                    let position = index as f32 + 32.0 - relative_delay;
                    let lower = position.floor() as usize;
                    let fraction = position - lower as f32;
                    source[lower] * (1.0 - fraction) + source[lower + 1] * fraction
                })
                .collect()
        })
        .collect();
    let loudest_rms = windows
        .iter()
        .map(|window| window_rms(window))
        .fold(0.0, f32::max);
    windows
        .into_iter()
        .map(|window| {
            window
                .into_iter()
                .map(|sample| sample * target_rms / loudest_rms)
                .collect()
        })
        .collect()
}

fn solve_with_floor(windows: &[Vec<f32>], floor: SolverReferenceChannelFloor) -> String {
    estimate_tetrahedral_steering(
        windows,
        &[0, 1, 2, 3],
        &TETRA_MICS_M,
        SAMPLE_RATE_HZ,
        SOUND_SPEED_MPS,
        SrpPhatConfig {
            reference_channel_floor: floor,
            ..SrpPhatConfig::default()
        },
    )
    .localization
    .resolved_algorithm
}

fn relative_floor_over(noise_floor_rms: f32) -> SolverReferenceChannelFloor {
    SolverReferenceChannelFloor {
        noise_floor_relative: Some(NoiseFloorRelativeReference {
            noise_floor_rms,
            relative_db: SOLVER_FLOOR_RELATIVE_DB,
        }),
        ..SolverReferenceChannelFloor::default()
    }
}

fn relative_floor() -> SolverReferenceChannelFloor {
    relative_floor_over(QUIET_SITE_NOISE_FLOOR_RMS)
}

#[test]
fn faint_event_over_a_low_floor_passes_the_relative_floor_but_not_the_old_absolute_one() {
    let event_rms = QUIET_SITE_NOISE_FLOOR_RMS * 10f32.powf(FAINT_EVENT_DB_OVER_FLOOR / 20.0);
    assert!(event_rms < DEFAULT_ABSOLUTE_REFERENCE_CHANNEL_RMS);
    let windows = faint_point_source_windows(event_rms);

    assert_eq!(
        solve_with_floor(&windows, SrpPhatConfig::default().reference_channel_floor),
        "srp_phat_degraded_low_energy"
    );
    assert_eq!(solve_with_floor(&windows, relative_floor()), "srp_phat");
}

#[test]
fn relative_floor_still_rejects_silence_and_a_window_at_the_floor() {
    let silence = vec![vec![0.0; WINDOW_SAMPLES]; 4];
    assert_eq!(
        solve_with_floor(&silence, relative_floor()),
        "srp_phat_degraded_low_energy"
    );
    let at_floor = faint_point_source_windows(QUIET_SITE_NOISE_FLOOR_RMS);
    assert_eq!(
        solve_with_floor(&at_floor, relative_floor()),
        "srp_phat_degraded_low_energy"
    );
    let zero_floor = relative_floor_over(0.0);
    assert!(zero_floor.rejects(0.0));
    assert!(zero_floor.rejects(f32::NAN));
}

#[test]
fn a_loud_sustained_source_the_floor_has_risen_to_still_solves() {
    // The tracker follows a hovering drone up to its own level; the absolute
    // level still admits it, as before this change.
    let drone_rms = 1.0e-2;
    let windows = faint_point_source_windows(drone_rms);
    assert_eq!(
        solve_with_floor(&windows, relative_floor_over(drone_rms)),
        "srp_phat"
    );
}
