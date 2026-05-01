use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{
    dsp_worker::PairTdoa,
    gcc_phat::{pair_max_tau_s, phat_correlation, GccPhatCorrelation},
};

const EPSILON: f32 = 1e-9;

#[derive(Clone, Copy, Debug)]
pub struct SrpPhatConfig {
    pub localization_band_hz: [f32; 2],
    pub grid_resolution_m: f32,
    pub search_padding_m: f32,
    pub max_grid_points: usize,
    pub min_channel_rms: f32,
}

impl Default for SrpPhatConfig {
    fn default() -> Self {
        Self {
            localization_band_hz: [300.0, 3500.0],
            grid_resolution_m: 0.5,
            search_padding_m: 2.0,
            max_grid_points: 60_000,
            min_channel_rms: 1.0e-4,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SrpPhatLocalization {
    pub attempted_algorithm: String,
    pub resolved_algorithm: String,
    pub steering_direction: [f32; 3],
    pub position_m: Option<[f32; 3]>,
    pub confidence: f32,
    pub residual_rms_seconds: f32,
    pub sound_speed_mps: f32,
}

pub struct SrpPhatEvaluation {
    pub localization: SrpPhatLocalization,
    pub pair_tdoas: Vec<PairTdoa>,
}

#[derive(Clone, Debug)]
struct PairObservation {
    ch_a: usize,
    ch_b: usize,
    correlation: GccPhatCorrelation,
}

pub fn estimate_tetrahedral_steering(
    windows: &[Vec<f32>; 4],
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]; 4],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    config: SrpPhatConfig,
) -> SrpPhatEvaluation {
    let mut ordered_channels = active_channels.to_vec();
    ordered_channels.sort_unstable();

    let pair_observations = build_pair_observations(
        windows,
        &ordered_channels,
        mic_positions_m,
        sample_rate_hz,
        sound_speed_mps,
        config.localization_band_hz,
    );
    let pair_tdoas = pair_observations
        .iter()
        .map(|pair| PairTdoa {
            ch_a: pair.ch_a,
            ch_b: pair.ch_b,
            tdoa: pair.correlation.tdoa.clone(),
        })
        .collect::<Vec<_>>();

    if ordered_channels.len() < 4 {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_partial_array", sound_speed_mps),
            pair_tdoas,
        };
    }

    let reference_channel = ordered_channels
        .iter()
        .copied()
        .max_by(|left, right| {
            window_rms(&windows[*left])
                .partial_cmp(&window_rms(&windows[*right]))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(ordered_channels[0]);
    let reference_rms = window_rms(&windows[reference_channel]);
    if reference_rms < config.min_channel_rms {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_low_energy", sound_speed_mps),
            pair_tdoas,
        };
    }

    let measured_tdoa = build_reference_measurements(reference_channel, &pair_observations);
    if measured_tdoa.len() < 3 {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_no_reference_pairs", sound_speed_mps),
            pair_tdoas,
        };
    }

    let grid = grid_from_bounds(
        &ordered_channels,
        mic_positions_m,
        config.search_padding_m,
        config.grid_resolution_m,
        config.max_grid_points,
    );
    if grid.is_empty() {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_empty_grid", sound_speed_mps),
            pair_tdoas,
        };
    }

    let mut scores = vec![0.0_f32; grid.len()];
    for pair in &pair_observations {
        for (point_index, point_m) in grid.iter().enumerate() {
            let predicted_tau_s = predicted_pair_tau_s(
                *point_m,
                mic_positions_m[pair.ch_a],
                mic_positions_m[pair.ch_b],
                sound_speed_mps,
            );
            scores[point_index] += sample_pair_correlation(&pair.correlation, predicted_tau_s);
        }
    }

    let (best_index, best_score) = scores
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .unwrap_or((0, 0.0));
    let peak_score = measured_tdoa
        .iter()
        .map(|measurement| measurement.peak_value)
        .sum::<f32>()
        / measured_tdoa.len() as f32;
    if best_score <= EPSILON && peak_score <= EPSILON {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_no_srp_peak", sound_speed_mps),
            pair_tdoas,
        };
    }

    let best_position_m = grid[best_index];
    let residual_rms_seconds = residual_rms(
        best_position_m,
        reference_channel,
        &measured_tdoa,
        mic_positions_m,
        sound_speed_mps,
    );
    let tau_scale = measured_tdoa
        .iter()
        .map(|measurement| measurement.tdoa_seconds.abs())
        .fold(1.0e-5_f32, f32::max);
    let fit_score = (1.0 - (residual_rms_seconds / tau_scale)).clamp(0.0, 1.0);
    let median_score = median(&scores);
    let contrast = (best_score - median_score) / (best_score.abs() + EPSILON);
    let confidence = ((0.6 * fit_score)
        + (0.25 * peak_score.clamp(0.0, 1.0))
        + (0.15 * contrast.clamp(0.0, 1.0)))
        .clamp(0.0, 1.0);

    let array_centroid_m = centroid(&ordered_channels, mic_positions_m);
    let steering_direction = normalize_or_zero(sub(best_position_m, array_centroid_m));

    SrpPhatEvaluation {
        localization: SrpPhatLocalization {
            attempted_algorithm: "srp_phat".to_string(),
            resolved_algorithm: "srp_phat".to_string(),
            steering_direction,
            position_m: Some(best_position_m),
            confidence,
            residual_rms_seconds,
            sound_speed_mps,
        },
        pair_tdoas,
    }
}

#[derive(Clone, Copy, Debug)]
struct ReferenceMeasurement {
    channel_index: usize,
    tdoa_seconds: f32,
    peak_value: f32,
}

fn degraded_localization(resolved_algorithm: &str, sound_speed_mps: f32) -> SrpPhatLocalization {
    SrpPhatLocalization {
        attempted_algorithm: "srp_phat".to_string(),
        resolved_algorithm: resolved_algorithm.to_string(),
        steering_direction: [0.0, 0.0, 0.0],
        position_m: None,
        confidence: 0.0,
        residual_rms_seconds: f32::INFINITY,
        sound_speed_mps,
    }
}

fn build_pair_observations(
    windows: &[Vec<f32>; 4],
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]; 4],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    localization_band_hz: [f32; 2],
) -> Vec<PairObservation> {
    let mut pairs = Vec::new();
    for (left_index, &ch_a) in active_channels.iter().enumerate() {
        for &ch_b in &active_channels[left_index + 1..] {
            let max_tau_s = pair_max_tau_s(
                mic_positions_m[ch_a],
                mic_positions_m[ch_b],
                sample_rate_hz,
                sound_speed_mps,
            );
            let correlation = phat_correlation(
                &windows[ch_a],
                &windows[ch_b],
                sample_rate_hz,
                max_tau_s,
                Some(localization_band_hz),
            );
            pairs.push(PairObservation {
                ch_a,
                ch_b,
                correlation,
            });
        }
    }
    pairs
}

fn build_reference_measurements(
    reference_channel: usize,
    pair_observations: &[PairObservation],
) -> Vec<ReferenceMeasurement> {
    let mut measurements = Vec::new();
    for pair in pair_observations {
        if pair.ch_a == reference_channel {
            measurements.push(ReferenceMeasurement {
                channel_index: pair.ch_b,
                tdoa_seconds: -pair.correlation.tdoa.lag_seconds,
                peak_value: pair.correlation.tdoa.peak_value,
            });
        } else if pair.ch_b == reference_channel {
            measurements.push(ReferenceMeasurement {
                channel_index: pair.ch_a,
                tdoa_seconds: pair.correlation.tdoa.lag_seconds,
                peak_value: pair.correlation.tdoa.peak_value,
            });
        }
    }
    measurements.sort_by_key(|measurement| measurement.channel_index);
    measurements
}

fn residual_rms(
    position_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]; 4],
    sound_speed_mps: f32,
) -> f32 {
    if measurements.is_empty() {
        return f32::INFINITY;
    }
    let mut squared_error = 0.0_f32;
    for measurement in measurements {
        let predicted_tdoa_s = predicted_reference_tau_s(
            position_m,
            mic_positions_m[measurement.channel_index],
            mic_positions_m[reference_channel],
            sound_speed_mps,
        );
        let error_seconds = predicted_tdoa_s - measurement.tdoa_seconds;
        squared_error += error_seconds * error_seconds;
    }
    (squared_error / measurements.len() as f32).sqrt()
}

fn grid_from_bounds(
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]; 4],
    search_padding_m: f32,
    grid_resolution_m: f32,
    max_grid_points: usize,
) -> Vec<[f32; 3]> {
    let padding_m = search_padding_m.max(0.05);
    let resolution_m = grid_resolution_m.max(0.05);

    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for &channel_index in active_channels {
        let position_m = mic_positions_m[channel_index];
        for axis in 0..3 {
            mins[axis] = mins[axis].min(position_m[axis] - padding_m);
            maxs[axis] = maxs[axis].max(position_m[axis] + padding_m);
        }
    }

    let xs = axis_points(mins[0], maxs[0], resolution_m);
    let ys = axis_points(mins[1], maxs[1], resolution_m);
    let zs = axis_points(mins[2], maxs[2], resolution_m);
    let total_points = xs.len() * ys.len() * zs.len();
    if total_points == 0 {
        return Vec::new();
    }

    let step = ((total_points as f32 / max_grid_points.max(1) as f32).ceil() as usize).max(1);
    let mut grid = Vec::with_capacity(total_points.min(max_grid_points.max(1)));
    let mut point_index = 0_usize;
    for &x_m in &xs {
        for &y_m in &ys {
            for &z_m in &zs {
                if point_index % step == 0 {
                    grid.push([x_m, y_m, z_m]);
                }
                point_index += 1;
            }
        }
    }
    grid
}

fn axis_points(min_m: f32, max_m: f32, resolution_m: f32) -> Vec<f32> {
    let mut values = Vec::new();
    let mut current_m = min_m;
    while current_m <= max_m + (resolution_m * 0.5) {
        values.push(current_m);
        current_m += resolution_m;
    }
    values
}

fn sample_pair_correlation(correlation: &GccPhatCorrelation, tau_seconds: f32) -> f32 {
    if correlation.lags_seconds.len() < 2 || correlation.magnitudes.is_empty() {
        return 0.0;
    }
    let step_seconds = correlation.lags_seconds[1] - correlation.lags_seconds[0];
    if step_seconds <= 0.0 {
        return 0.0;
    }
    let position = (tau_seconds - correlation.lags_seconds[0]) / step_seconds;
    if !(0.0..=(correlation.magnitudes.len().saturating_sub(1) as f32)).contains(&position) {
        return 0.0;
    }
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    if lower_index == upper_index {
        return correlation.magnitudes[lower_index];
    }
    let weight = position - lower_index as f32;
    let lower_value = correlation.magnitudes[lower_index];
    let upper_value = correlation.magnitudes[upper_index];
    lower_value + ((upper_value - lower_value) * weight)
}

fn predicted_pair_tau_s(
    position_m: [f32; 3],
    position_a_m: [f32; 3],
    position_b_m: [f32; 3],
    sound_speed_mps: f32,
) -> f32 {
    let distance_a_m = euclidean_distance(position_m, position_a_m) + EPSILON;
    let distance_b_m = euclidean_distance(position_m, position_b_m) + EPSILON;
    (distance_a_m - distance_b_m) / sound_speed_mps.max(1.0)
}

fn predicted_reference_tau_s(
    position_m: [f32; 3],
    sensor_position_m: [f32; 3],
    reference_position_m: [f32; 3],
    sound_speed_mps: f32,
) -> f32 {
    let reference_distance_m = euclidean_distance(position_m, reference_position_m) + EPSILON;
    let sensor_distance_m = euclidean_distance(position_m, sensor_position_m) + EPSILON;
    (sensor_distance_m - reference_distance_m) / sound_speed_mps.max(1.0)
}

fn centroid(active_channels: &[usize], mic_positions_m: &[[f32; 3]; 4]) -> [f32; 3] {
    let mut mean = [0.0_f32; 3];
    if active_channels.is_empty() {
        return mean;
    }
    for &channel_index in active_channels {
        let position_m = mic_positions_m[channel_index];
        for axis in 0..3 {
            mean[axis] += position_m[axis];
        }
    }
    for axis in 0..3 {
        mean[axis] /= active_channels.len() as f32;
    }
    mean
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let middle = ordered.len() / 2;
    if ordered.len() % 2 == 0 {
        (ordered[middle - 1] + ordered[middle]) * 0.5
    } else {
        ordered[middle]
    }
}

fn window_rms(window: &[f32]) -> f32 {
    if window.is_empty() {
        return 0.0;
    }
    let energy = window.iter().map(|sample| sample * sample).sum::<f32>();
    (energy / window.len() as f32).sqrt()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn magnitude(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let norm = magnitude(v);
    if norm <= EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

fn euclidean_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    magnitude(sub(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetrahedral_solver_recovers_near_field_position_on_sirith_fixture() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let source_position_m = [0.20, 0.10, 0.15];
        let channels = synthesize_point_source_channels(source_position_m, &mics, 16_000, 1_024);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            16_000,
            343.2,
            SrpPhatConfig {
                grid_resolution_m: 0.05,
                search_padding_m: 0.3,
                ..SrpPhatConfig::default()
            },
        );

        assert_eq!(evaluation.pair_tdoas.len(), 6);
        assert_eq!(evaluation.localization.resolved_algorithm, "srp_phat");
        let recovered_position_m = evaluation.localization.position_m.expect("SRP position");
        assert!(euclidean_distance(recovered_position_m, source_position_m) <= 0.08);

        let expected_direction = normalize_or_zero(sub(source_position_m, centroid(&[0, 1, 2, 3], &mics)));
        assert!(dot(evaluation.localization.steering_direction, expected_direction) > 0.95);
        assert!(evaluation.localization.confidence >= 0.25);
    }

    #[test]
    fn sirith_fixture_pair_tdoas_match_python_oracle_signs_and_order() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let source_position_m = [0.20, 0.10, 0.15];
        let channels = synthesize_point_source_channels(source_position_m, &mics, 16_000, 1_024);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            16_000,
            343.2,
            SrpPhatConfig {
                grid_resolution_m: 0.05,
                search_padding_m: 0.3,
                ..SrpPhatConfig::default()
            },
        );

        let strongest_channel = [0, 1, 2, 3]
            .into_iter()
            .max_by(|left, right| {
                window_rms(&channels[*left])
                    .partial_cmp(&window_rms(&channels[*right]))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("fixture should have a reference channel");
        assert_eq!(strongest_channel, 3);

        let expected_pair_delay_samples = [
            ((0, 1), 1.5_f32),
            ((0, 2), -0.5_f32),
            ((0, 3), 1.5_f32),
            ((1, 2), -2.0_f32),
            ((1, 3), 0.25_f32),
            ((2, 3), 2.0_f32),
        ];

        for ((ch_a, ch_b), expected_delay_samples) in expected_pair_delay_samples {
            let pair = evaluation
                .pair_tdoas
                .iter()
                .find(|pair| pair.ch_a == ch_a && pair.ch_b == ch_b)
                .expect("expected pair TDOA for oracle comparison");
            assert!(
                (pair.tdoa.delay_samples - expected_delay_samples).abs() < 0.6,
                "expected pair ({}, {}) near Python oracle {} samples, got {}",
                ch_a,
                ch_b,
                expected_delay_samples,
                pair.tdoa.delay_samples
            );
            assert_eq!(
                pair.tdoa.delay_samples.is_sign_positive(),
                expected_delay_samples.is_sign_positive(),
                "expected pair ({}, {}) to preserve Python oracle sign",
                ch_a,
                ch_b
            );
        }
    }

    #[test]
    fn degraded_partial_array_stays_explicit() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let source_position_m = [0.20, 0.10, 0.15];
        let channels = synthesize_point_source_channels(source_position_m, &mics, 16_000, 1_024);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2],
            &mics,
            16_000,
            343.2,
            SrpPhatConfig::default(),
        );

        assert_eq!(evaluation.localization.resolved_algorithm, "srp_phat_degraded_partial_array");
        assert_eq!(evaluation.localization.confidence, 0.0);
        assert!(evaluation.localization.position_m.is_none());
        assert_eq!(evaluation.pair_tdoas.len(), 3);
    }

    #[test]
    fn low_energy_windows_degrade_cleanly() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let channels = [vec![0.0; 512], vec![0.0; 512], vec![0.0; 512], vec![0.0; 512]];
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            16_000,
            343.2,
            SrpPhatConfig::default(),
        );

        assert_eq!(evaluation.localization.resolved_algorithm, "srp_phat_degraded_low_energy");
        assert!(evaluation.localization.position_m.is_none());
        assert_eq!(evaluation.localization.confidence, 0.0);
    }

    #[test]
    fn uncorrelated_windows_return_low_confidence() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let channels = [
            pseudo_random_with_seed(0x1111_1111, 1_024),
            pseudo_random_with_seed(0x2222_2222, 1_024),
            pseudo_random_with_seed(0x3333_3333, 1_024),
            pseudo_random_with_seed(0x4444_4444, 1_024),
        ];
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            16_000,
            343.2,
            SrpPhatConfig {
                grid_resolution_m: 0.05,
                search_padding_m: 0.3,
                ..SrpPhatConfig::default()
            },
        );

        assert_eq!(evaluation.localization.resolved_algorithm, "srp_phat");
        assert!(evaluation.localization.confidence < 0.2);
    }

    fn synthesize_point_source_channels(
        source_position_m: [f32; 3],
        mic_positions_m: &[[f32; 3]; 4],
        sample_rate_hz: u32,
        output_len: usize,
    ) -> [Vec<f32>; 4] {
        let source = pseudo_random_with_seed(0x1234_5678, output_len + 64);
        let delays_samples = mic_positions_m.map(|position_m| {
            euclidean_distance(source_position_m, position_m) / 343.2 * sample_rate_hz as f32
        });
        let min_delay_samples = delays_samples.iter().copied().fold(f32::INFINITY, f32::min);
        delays_samples.map(|delay_samples| fractional_delay(&source, delay_samples - min_delay_samples, output_len))
    }

    fn fractional_delay(source: &[f32], delay_samples: f32, output_len: usize) -> Vec<f32> {
        (0..output_len)
            .map(|sample_index| {
                let source_index = sample_index as f32 - delay_samples;
                if source_index < 0.0 || source_index + 1.0 >= source.len() as f32 {
                    return 0.0;
                }
                let lower_index = source_index.floor() as usize;
                let upper_index = lower_index + 1;
                let fraction = source_index - lower_index as f32;
                (source[lower_index] * (1.0 - fraction)) + (source[upper_index] * fraction)
            })
            .collect()
    }

    fn pseudo_random_with_seed(mut seed: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as i32 as f32) / (i32::MAX as f32)
            })
            .collect()
    }
}
