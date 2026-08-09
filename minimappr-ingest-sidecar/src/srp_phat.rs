use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{
    dsp_worker::PairTdoa,
    gcc_phat::{dominant_frequency_hz, pair_max_tau_s, phat_correlation, GccPhatCorrelation},
    range_projection::{
        confidence_cap_for_mode, range_observability_cap_for_mode, RANGE_ASYMPTOTIC,
        RANGE_BEARING_PROJECTED, RANGE_BOUNDARY, RANGE_REFINED,
    },
};

const EPSILON: f32 = 1e-9;
const FAR_FIELD_RANGE_CONVERGENCE_TOLERANCE_M: f32 = 0.05;

/// Coplanar-array up/down constraint (D7). A planar array's TDOA geometry is
/// mirror-symmetric across its own plane, so without an external constraint
/// the near-field grid search wastes half its points (and the solver is
/// ambiguous) on the side of the array nothing can physically occupy. Only
/// the z-axis case (a horizontal array plane) is implemented; a pitched or
/// rolled planar node needs the array's rotated normal, not the raw z-axis —
/// tracked as a follow-up (plan risk #5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HalfSpace {
    Upper,
    Lower,
    #[default]
    None,
}

impl HalfSpace {
    pub fn from_wire_str(value: &str) -> Self {
        match value {
            "upper" => HalfSpace::Upper,
            "lower" => HalfSpace::Lower,
            _ => HalfSpace::None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SrpPhatConfig {
    pub localization_band_hz: [f32; 2],
    pub grid_resolution_m: f32,
    pub search_padding_m: f32,
    pub interp_factor: usize,
    pub max_grid_points: usize,
    pub min_channel_rms: f32,
    pub far_field_default_range_m: f32,
    pub far_field_max_range_m: f32,
    /// Amplitude/SNR-informed range prior (Phase 1c). When `Some`, overrides
    /// `far_field_default_range_m` as the projection distance for unobservable-range
    /// solves. `range_prior_std_m` (if `Some`) sets the cone radial std floor
    /// (`std_factor × prior_range`); otherwise the projection distance itself is used.
    /// Both `None` (the default) reproduces the fixed-prior behaviour exactly.
    pub range_prior_m: Option<f32>,
    pub range_prior_std_m: Option<f32>,
    /// Coplanar array up/down constraint (D7); `HalfSpace::None` (default)
    /// reproduces unconstrained tetra behaviour exactly.
    pub half_space: HalfSpace,
}

impl Default for SrpPhatConfig {
    fn default() -> Self {
        Self {
            localization_band_hz: [50.0, 0.0],
            grid_resolution_m: 0.5,
            search_padding_m: 2.0,
            interp_factor: 4,
            max_grid_points: 60_000,
            min_channel_rms: 1.0e-4,
            far_field_default_range_m: 50.0,
            far_field_max_range_m: 250.0,
            range_prior_m: None,
            range_prior_std_m: None,
            half_space: HalfSpace::None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SrpPhatLocalization {
    pub attempted_algorithm: String,
    pub resolved_algorithm: String,
    pub steering_direction: [f32; 3],
    pub position_m: Option<[f32; 3]>,
    pub position_covariance_m2: Option<[[f32; 3]; 3]>,
    pub range_observability: Option<f32>,
    pub range_projection_mode: Option<String>,
    pub confidence: f32,
    pub residual_rms_seconds: f32,
    pub sound_speed_mps: f32,
    /// Power-weighted mean frequency (Hz) of the reference channel. Consumed by
    /// the Python fusion node to scale lateral covariance for low-frequency
    /// signals (frequency-dependent angular resolution). Mirrors the Python
    /// `dominant_frequency_hz` in `minimappr/core/localization.py`.
    #[serde(default)]
    pub dominant_frequency_hz: Option<f32>,
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

#[derive(Clone, Copy, Debug)]
struct LocalizationCandidate {
    position_m: [f32; 3],
    steering_direction: [f32; 3],
    position_covariance_m2: [[f32; 3]; 3],
    range_observability: Option<f32>,
    range_projection_mode: Option<&'static str>,
    is_boundary_clamped: bool,
    timing_resolution_seconds: f32,
    residual_rms_seconds: f32,
    contrast_score: f32,
}

#[derive(Clone, Copy, Debug)]
struct FarFieldRangeEstimate {
    position_m: [f32; 3],
    radius_m: f32,
    radial_std_m: f32,
    range_observability: f32,
    residual_rms_seconds: f32,
    range_projection_mode: &'static str,
}

pub fn estimate_tetrahedral_steering(
    windows: &[Vec<f32>],
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]],
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
        config.interp_factor,
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
    let reference_dominant_frequency_hz =
        dominant_frequency_hz(&windows[reference_channel], sample_rate_hz);

    let measured_tdoa = build_reference_measurements(reference_channel, &pair_observations);
    if measured_tdoa.len() < 3 {
        return SrpPhatEvaluation {
            localization: degraded_localization(
                "srp_phat_degraded_no_reference_pairs",
                sound_speed_mps,
            ),
            pair_tdoas,
        };
    }

    let peak_score = measured_tdoa
        .iter()
        .map(|measurement| measurement.peak_value)
        .sum::<f32>()
        / measured_tdoa.len() as f32;
    let initial_near_field_candidate = near_field_candidate(
        &ordered_channels,
        &pair_observations,
        reference_channel,
        &measured_tdoa,
        mic_positions_m,
        sample_rate_hz,
        sound_speed_mps,
        config,
        false,
    );
    let far_field_candidate = far_field_candidate(
        &ordered_channels,
        reference_channel,
        &measured_tdoa,
        mic_positions_m,
        sample_rate_hz,
        sound_speed_mps,
        config,
    );
    let near_field_candidate = match (initial_near_field_candidate, far_field_candidate) {
        (Some(near), Some(far))
            if near.is_boundary_clamped
                && near.residual_rms_seconds.is_finite()
                && far.residual_rms_seconds.is_finite()
                && far.residual_rms_seconds <= (near.residual_rms_seconds * 0.8) =>
        {
            Some(
                near_field_candidate(
                    &ordered_channels,
                    &pair_observations,
                    reference_channel,
                    &measured_tdoa,
                    mic_positions_m,
                    sample_rate_hz,
                    sound_speed_mps,
                    config,
                    true,
                )
                .unwrap_or(near),
            )
        }
        (candidate, _) => candidate,
    };
    let selected_candidate = select_candidate(near_field_candidate, far_field_candidate);
    let Some(candidate) = selected_candidate else {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_empty_grid", sound_speed_mps),
            pair_tdoas,
        };
    };
    if candidate.contrast_score <= EPSILON && peak_score <= EPSILON {
        return SrpPhatEvaluation {
            localization: degraded_localization("srp_phat_degraded_no_srp_peak", sound_speed_mps),
            pair_tdoas,
        };
    }

    let tau_scale = measured_tdoa
        .iter()
        .map(|measurement| measurement.tdoa_seconds.abs())
        .fold(1.0e-5_f32, f32::max);
    let (confidence, range_observability) =
        candidate_confidence_and_observability(&candidate, peak_score, tau_scale);

    SrpPhatEvaluation {
        localization: SrpPhatLocalization {
            attempted_algorithm: "srp_phat".to_string(),
            resolved_algorithm: "srp_phat".to_string(),
            steering_direction: candidate.steering_direction,
            position_m: Some(candidate.position_m),
            position_covariance_m2: Some(candidate.position_covariance_m2),
            range_observability,
            range_projection_mode: candidate.range_projection_mode.map(str::to_string),
            confidence,
            residual_rms_seconds: candidate.residual_rms_seconds,
            sound_speed_mps,
            dominant_frequency_hz: Some(reference_dominant_frequency_hz),
        },
        pair_tdoas,
    }
}

/// Fold a solved candidate's fit/peak/contrast scores into a final confidence and
/// a (possibly capped) range observability, applying the canonical per-mode caps so
/// a Rust-native result carries the same ceilings as the identical Python solve.
/// See RANGE_PROJECTION_CONTRACT.md and `range_projection.rs`.
fn candidate_confidence_and_observability(
    candidate: &LocalizationCandidate,
    peak_score: f32,
    tau_scale: f32,
) -> (f32, Option<f32>) {
    let fit_score = (1.0 - (candidate.residual_rms_seconds / tau_scale)).clamp(0.0, 1.0);
    let mode = candidate.range_projection_mode;
    let base_confidence = (0.6 * fit_score)
        + (0.25 * peak_score.clamp(0.0, 1.0))
        + (0.15 * candidate.contrast_score.clamp(0.0, 1.0));
    // Mirror Python's `obs_factor = 1.0` branch (cartesian_tdoa.py:522-528): a
    // bearing-projected result has an intentionally huge radial eigenvalue, so its
    // range_observability is artificially tiny and must NOT penalise the
    // well-observed bearing. Every other mode still attenuates by observability.
    let mut confidence = if mode == Some(RANGE_BEARING_PROJECTED) {
        base_confidence
    } else {
        base_confidence
            * candidate
                .range_observability
                .unwrap_or(1.0)
                .clamp(0.35, 1.0)
    };
    // Apply the canonical per-mode confidence cap (idempotent with the Python
    // ingest-seam haircut).
    if let Some(cap) = confidence_cap_for_mode(mode) {
        confidence = confidence.min(cap);
    }
    let range_observability = match (
        candidate.range_observability,
        range_observability_cap_for_mode(mode),
    ) {
        (Some(obs), Some(cap)) => Some(obs.min(cap)),
        (obs, _) => obs,
    };
    (confidence, range_observability)
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
        position_covariance_m2: None,
        range_observability: None,
        range_projection_mode: None,
        confidence: 0.0,
        residual_rms_seconds: f32::INFINITY,
        sound_speed_mps,
        dominant_frequency_hz: None,
    }
}

fn build_pair_observations(
    windows: &[Vec<f32>],
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    localization_band_hz: [f32; 2],
    interp_factor: usize,
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
                interp_factor,
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
    mic_positions_m: &[[f32; 3]],
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

fn refine_reference_position(
    initial_position_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sound_speed_mps: f32,
) -> [f32; 3] {
    let mut position_m = initial_position_m;
    let mut best_residual_rms_seconds = residual_rms(
        position_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    );
    let mut damping_lambda = 1.0e-4_f32;

    for _ in 0..24 {
        let reciprocal_sound_speed = 1.0 / sound_speed_mps.max(1.0);
        let reference_position_m = mic_positions_m[reference_channel];
        let reference_offset_m = sub(position_m, reference_position_m);
        let reference_distance_m = magnitude(reference_offset_m) + EPSILON;
        let reference_jacobian_scale = reciprocal_sound_speed / reference_distance_m;
        let mut jacobian_rows = Vec::with_capacity(measurements.len());
        let mut residuals = Vec::with_capacity(measurements.len());
        for measurement in measurements {
            let sensor_position_m = mic_positions_m[measurement.channel_index];
            let sensor_offset_m = sub(position_m, sensor_position_m);
            let sensor_distance_m = magnitude(sensor_offset_m) + EPSILON;
            let sensor_jacobian_scale = reciprocal_sound_speed / sensor_distance_m;
            let predicted_tdoa_s =
                (sensor_distance_m - reference_distance_m) * reciprocal_sound_speed;
            residuals.push(predicted_tdoa_s - measurement.tdoa_seconds);
            jacobian_rows.push([
                (sensor_offset_m[0] * sensor_jacobian_scale)
                    - (reference_offset_m[0] * reference_jacobian_scale),
                (sensor_offset_m[1] * sensor_jacobian_scale)
                    - (reference_offset_m[1] * reference_jacobian_scale),
                (sensor_offset_m[2] * sensor_jacobian_scale)
                    - (reference_offset_m[2] * reference_jacobian_scale),
            ]);
        }

        let mut normal_matrix = [[0.0_f32; 3]; 3];
        let mut normal_rhs = [0.0_f32; 3];
        for (row, residual) in jacobian_rows.iter().zip(residuals.iter()) {
            for out_axis in 0..3 {
                normal_rhs[out_axis] -= row[out_axis] * *residual;
                for in_axis in 0..3 {
                    normal_matrix[out_axis][in_axis] += row[out_axis] * row[in_axis];
                }
            }
        }
        for axis in 0..3 {
            normal_matrix[axis][axis] += damping_lambda;
        }
        let Some(delta_m) = solve_3x3(normal_matrix, normal_rhs) else {
            break;
        };

        let mut candidate_delta_m = delta_m;
        let mut accepted = false;
        for _ in 0..8 {
            let candidate_position_m = add(position_m, candidate_delta_m);
            let candidate_residual_rms_seconds = residual_rms(
                candidate_position_m,
                reference_channel,
                measurements,
                mic_positions_m,
                sound_speed_mps,
            );
            if candidate_residual_rms_seconds < best_residual_rms_seconds {
                position_m = candidate_position_m;
                best_residual_rms_seconds = candidate_residual_rms_seconds;
                damping_lambda = (damping_lambda * 0.5).max(1.0e-6);
                accepted = true;
                break;
            }
            candidate_delta_m = scale(candidate_delta_m, 0.5);
        }

        if !accepted {
            damping_lambda = (damping_lambda * 4.0).min(1.0e3);
            if magnitude(delta_m) <= 1.0e-4 {
                break;
            }
            continue;
        }

        if magnitude(candidate_delta_m) <= 1.0e-4 {
            break;
        }
    }
    position_m
}

fn near_field_candidate(
    active_channels: &[usize],
    pair_observations: &[PairObservation],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    config: SrpPhatConfig,
    allow_boundary_escape: bool,
) -> Option<LocalizationCandidate> {
    let grid = grid_from_bounds(
        active_channels,
        mic_positions_m,
        config.search_padding_m,
        config.grid_resolution_m,
        config.max_grid_points,
        config.half_space,
    );
    if grid.is_empty() {
        return None;
    }

    let mut scores = vec![0.0_f32; grid.len()];
    for pair in pair_observations {
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
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))?;
    let centroid_m = centroid(active_channels, mic_positions_m);
    let mut ranked_grid_points: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked_grid_points
        .sort_by(|(_, left), (_, right)| right.partial_cmp(left).unwrap_or(Ordering::Equal));
    const REFINED_SRP_SEED_COUNT: usize = 8;
    ranked_grid_points.truncate(REFINED_SRP_SEED_COUNT.min(ranked_grid_points.len()));

    let mut best_position_m = grid[best_index];
    let mut refined_position_m = best_position_m;
    let mut residual_rms_seconds = f32::INFINITY;
    for (candidate_index, _) in ranked_grid_points {
        let candidate_seed_m = grid[candidate_index];
        let candidate_position_m = refine_reference_position(
            candidate_seed_m,
            reference_channel,
            measurements,
            mic_positions_m,
            sound_speed_mps,
        );
        let candidate_residual_rms_seconds = residual_rms(
            candidate_position_m,
            reference_channel,
            measurements,
            mic_positions_m,
            sound_speed_mps,
        );
        if candidate_residual_rms_seconds < residual_rms_seconds {
            best_position_m = candidate_seed_m;
            refined_position_m = candidate_position_m;
            residual_rms_seconds = candidate_residual_rms_seconds;
        }
    }
    let minimum_std_m = (config.grid_resolution_m * 0.5).max(0.05);
    let timing_resolution_seconds =
        1.0 / (sample_rate_hz.max(1) as f32 * config.interp_factor.max(1) as f32);
    let (mins, maxs) = padded_bounds(active_channels, mic_positions_m, config.search_padding_m);
    let boundary_clamped =
        is_boundary_clamped(best_position_m, mins, maxs, config.grid_resolution_m)
            && !is_well_inside_bounds(refined_position_m, mins, maxs, config.grid_resolution_m);
    let boundary_radius_m =
        magnitude(sub(best_position_m, centroid_m)).max(config.grid_resolution_m.max(0.05));

    if boundary_clamped && allow_boundary_escape {
        const BOUNDARY_ESCAPE_SEED_SCALES: [f32; 4] = [2.0, 4.0, 8.0, 16.0];
        const MIN_BOUNDARY_ESCAPE_RATIO_FOR_NEAR_UPGRADE: f32 = 4.0;
        let steering_direction = normalize_or_zero(sub(best_position_m, centroid_m));
        if magnitude(steering_direction) > EPSILON {
            for scale_factor in BOUNDARY_ESCAPE_SEED_SCALES {
                let seed_position_m = add(
                    centroid_m,
                    scale(steering_direction, boundary_radius_m * scale_factor),
                );
                let candidate_position_m = refine_reference_position(
                    seed_position_m,
                    reference_channel,
                    measurements,
                    mic_positions_m,
                    sound_speed_mps,
                );
                let candidate_residual_rms_seconds = residual_rms(
                    candidate_position_m,
                    reference_channel,
                    measurements,
                    mic_positions_m,
                    sound_speed_mps,
                );
                let candidate_boundary_escape_ratio =
                    magnitude(sub(candidate_position_m, centroid_m)) / boundary_radius_m;
                if candidate_boundary_escape_ratio >= MIN_BOUNDARY_ESCAPE_RATIO_FOR_NEAR_UPGRADE
                    && candidate_residual_rms_seconds < residual_rms_seconds
                {
                    refined_position_m = candidate_position_m;
                    residual_rms_seconds = candidate_residual_rms_seconds;
                }
            }
        }
    }

    let (position_covariance_m2, range_observability) = position_covariance_and_observability(
        refined_position_m,
        centroid_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sample_rate_hz,
        sound_speed_mps,
        minimum_std_m,
    );
    Some(LocalizationCandidate {
        position_m: refined_position_m,
        steering_direction: normalize_or_zero(sub(refined_position_m, centroid_m)),
        position_covariance_m2,
        range_observability: Some(if boundary_clamped {
            range_observability.min(0.25)
        } else {
            range_observability
        }),
        range_projection_mode: Some(if boundary_clamped {
            RANGE_BOUNDARY
        } else {
            RANGE_REFINED
        }),
        is_boundary_clamped: boundary_clamped,
        timing_resolution_seconds,
        residual_rms_seconds,
        contrast_score: ((best_score - median(&scores)) / (best_score.abs() + EPSILON))
            .clamp(0.0, 1.0),
    })
}

fn far_field_candidate(
    active_channels: &[usize],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    config: SrpPhatConfig,
) -> Option<LocalizationCandidate> {
    let (direction, direction_fit_residual) = pairwise_far_field_direction(
        active_channels,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    )?;
    let range_estimate = fit_far_field_range(
        direction,
        centroid(active_channels, mic_positions_m),
        reference_channel,
        measurements,
        mic_positions_m,
        sample_rate_hz,
        sound_speed_mps,
        config,
    );
    let aperture_m = aperture(active_channels, mic_positions_m);
    let lateral_std_m = (config.grid_resolution_m * 0.5)
        .max(range_estimate.radius_m * direction_fit_residual.clamp(0.05, 0.6))
        .max(aperture_m * 0.5)
        .max(sound_speed_mps / sample_rate_hz.max(1) as f32);
    let radial_std_m = range_estimate.radial_std_m.max(lateral_std_m);
    // Replace the hard aperture gate (aperture <= 0.35 → RANGE_ASYMPTOTIC) with a
    // bearing-quality check. When the range axis is unobservable but the direction
    // fit is tight (residual < 0.5), emit RANGE_BEARING_PROJECTED so the Python
    // fusion node can keep high confidence for the bearing while honestly marking
    // range as uncertain. Cone radial (range) std = loosest of 4× solved range, the
    // amplitude-prior radial std (std_factor × prior_range, when supplied), and a
    // 200 m floor. Mirrors cartesian_tdoa.py's cone construction (Phase 1c).
    let prior_radial_std_m = config
        .range_prior_std_m
        .unwrap_or(config.range_prior_m.unwrap_or(0.0));
    let (range_projection_mode, cone_radial_std_m) =
        if range_estimate.range_projection_mode == RANGE_ASYMPTOTIC {
            if direction_fit_residual < 0.5 {
                let r = (range_estimate.radius_m * 4.0)
                    .max(prior_radial_std_m)
                    .max(200.0);
                (RANGE_BEARING_PROJECTED, r)
            } else {
                (RANGE_ASYMPTOTIC, radial_std_m)
            }
        } else {
            (range_estimate.range_projection_mode, radial_std_m)
        };
    let timing_resolution_seconds =
        1.0 / (sample_rate_hz.max(1) as f32 * config.interp_factor.max(1) as f32);
    Some(LocalizationCandidate {
        position_m: range_estimate.position_m,
        steering_direction: direction,
        position_covariance_m2: directional_covariance(direction, lateral_std_m, cone_radial_std_m),
        range_observability: Some(range_estimate.range_observability),
        range_projection_mode: Some(range_projection_mode),
        is_boundary_clamped: false,
        timing_resolution_seconds,
        residual_rms_seconds: range_estimate.residual_rms_seconds,
        contrast_score: (1.0 - direction_fit_residual).clamp(0.0, 1.0),
    })
}

fn select_candidate(
    near_field_candidate: Option<LocalizationCandidate>,
    far_field_candidate: Option<LocalizationCandidate>,
) -> Option<LocalizationCandidate> {
    match (near_field_candidate, far_field_candidate) {
        (Some(near), Some(far)) => {
            let minimum_residual_improvement_seconds = near
                .timing_resolution_seconds
                .max(far.timing_resolution_seconds);
            let far_improves_enough = if matches!(
                far.range_projection_mode,
                Some(RANGE_ASYMPTOTIC) | Some(RANGE_BEARING_PROJECTED)
            ) && near.residual_rms_seconds.is_finite()
            {
                let near_range_m = magnitude(near.position_m);
                far.residual_rms_seconds <= near.residual_rms_seconds * 0.12
                    || (near.is_boundary_clamped
                        && near_range_m > 20.0
                        && magnitude(far.position_m) > 100.0
                        && far.residual_rms_seconds < near.residual_rms_seconds)
                    // When the near-field range is unobservable, the residual gap
                    // between near and far narrows because phase-slope-refined TDOAs
                    // let both models fit the near-planar wavefront well. A genuine
                    // near-field source still leaves the near fit clearly better
                    // (far/near ratio ≈ 0.85), while a truly distant source lets the
                    // far model fit appreciably better (ratio ≈ 0.5). Require far to
                    // fit at least ~1.7× better (≤ 0.6) — relaxed from the old 0.25
                    // which the cleaner TDOAs no longer reach — so distant sources
                    // surface as a high-contrast far-field bearing_projected estimate
                    // (known direction, honest range uncertainty) instead of an
                    // overconfident near-field range guess.
                    || (near.range_observability.unwrap_or(1.0) < 0.15
                        && far.contrast_score >= near.contrast_score + 0.2
                        && magnitude(far.position_m) > 100.0
                        && far.residual_rms_seconds <= near.residual_rms_seconds * 0.6)
            } else {
                far.residual_rms_seconds + minimum_residual_improvement_seconds
                    < near.residual_rms_seconds
            };
            if far.residual_rms_seconds.is_finite()
                && (!near.residual_rms_seconds.is_finite() || far_improves_enough)
            {
                Some(far)
            } else {
                Some(near)
            }
        }
        (Some(near), None) => Some(near),
        (None, Some(far)) => Some(far),
        (None, None) => None,
    }
}

fn pairwise_far_field_direction(
    active_channels: &[usize],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sound_speed_mps: f32,
) -> Option<([f32; 3], f32)> {
    let mut relative_tdoa_seconds = vec![None; mic_positions_m.len()];
    relative_tdoa_seconds[reference_channel] = Some(0.0);
    for measurement in measurements {
        relative_tdoa_seconds[measurement.channel_index] = Some(measurement.tdoa_seconds);
    }

    let mut rows = Vec::new();
    let mut targets = Vec::new();
    for (left_index, &channel_a) in active_channels.iter().enumerate() {
        let tau_a_seconds = relative_tdoa_seconds[channel_a]?;
        for &channel_b in &active_channels[left_index + 1..] {
            let tau_b_seconds = relative_tdoa_seconds[channel_b]?;
            rows.push(sub(mic_positions_m[channel_b], mic_positions_m[channel_a]));
            targets.push(sound_speed_mps * (tau_a_seconds - tau_b_seconds));
        }
    }
    let direction = solve_normal_equations(&rows, &targets)?;
    let direction = normalize_or_zero(direction);
    if magnitude(direction) <= EPSILON {
        return None;
    }
    let target_norm = targets
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(EPSILON);
    let fit_error = rows
        .iter()
        .zip(targets.iter())
        .map(|(row, target)| {
            let error = dot(*row, direction) - *target;
            error * error
        })
        .sum::<f32>()
        .sqrt()
        / target_norm;
    Some((direction, fit_error))
}

fn fit_far_field_range(
    direction: [f32; 3],
    centroid_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    config: SrpPhatConfig,
) -> FarFieldRangeEstimate {
    let max_range_m = config
        .far_field_max_range_m
        .max(config.far_field_default_range_m)
        .max(1.0);
    // Amplitude/SNR range prior (Phase 1c) overrides the fixed default projection
    // distance when supplied; otherwise the config default is used. Clamped into the
    // valid search band exactly like the fixed default.
    let prior_range_m = config
        .range_prior_m
        .unwrap_or(config.far_field_default_range_m);
    let default_range_m = prior_range_m.max(0.5).min(max_range_m);
    let mut left_m = 0.05_f32;
    let mut right_m = max_range_m;
    let inverse_phi = 0.618_033_95_f32;
    let mut c_m = right_m - ((right_m - left_m) * inverse_phi);
    let mut d_m = left_m + ((right_m - left_m) * inverse_phi);
    let mut fc = ray_mean_squared_error(
        c_m,
        direction,
        centroid_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    );
    let mut fd = ray_mean_squared_error(
        d_m,
        direction,
        centroid_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    );
    for _ in 0..48 {
        if (right_m - left_m) <= FAR_FIELD_RANGE_CONVERGENCE_TOLERANCE_M {
            break;
        }
        if fc < fd {
            right_m = d_m;
            d_m = c_m;
            fd = fc;
            c_m = right_m - ((right_m - left_m) * inverse_phi);
            fc = ray_mean_squared_error(
                c_m,
                direction,
                centroid_m,
                reference_channel,
                measurements,
                mic_positions_m,
                sound_speed_mps,
            );
        } else {
            left_m = c_m;
            c_m = d_m;
            fc = fd;
            d_m = left_m + ((right_m - left_m) * inverse_phi);
            fd = ray_mean_squared_error(
                d_m,
                direction,
                centroid_m,
                reference_channel,
                measurements,
                mic_positions_m,
                sound_speed_mps,
            );
        }
    }
    let radius_m = if fc.is_finite() || fd.is_finite() {
        if fc <= fd {
            c_m
        } else {
            d_m
        }
    } else {
        default_range_m
    };
    let residual_rms_seconds = ray_mean_squared_error(
        radius_m,
        direction,
        centroid_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    )
    .sqrt();

    let delta_m = (radius_m * 0.05).abs().max(0.5).min(max_range_m * 0.25);
    let left_radius_m = (radius_m - delta_m).max(0.05);
    let right_radius_m = (radius_m + delta_m).min(max_range_m);
    let step_m = (right_radius_m - left_radius_m).max(EPSILON);
    let mut derivative_info = 0.0_f32;
    for measurement in measurements {
        let left_tau_s = predicted_reference_tau_s(
            add(centroid_m, scale(direction, left_radius_m)),
            mic_positions_m[measurement.channel_index],
            mic_positions_m[reference_channel],
            sound_speed_mps,
        );
        let right_tau_s = predicted_reference_tau_s(
            add(centroid_m, scale(direction, right_radius_m)),
            mic_positions_m[measurement.channel_index],
            mic_positions_m[reference_channel],
            sound_speed_mps,
        );
        let derivative = (right_tau_s - left_tau_s) / step_m;
        derivative_info += derivative * derivative;
    }
    let time_std_seconds = residual_rms_seconds.max(1.0 / sample_rate_hz.max(1) as f32);
    let radial_std_m = if derivative_info > EPSILON {
        time_std_seconds / derivative_info.sqrt()
    } else {
        max_range_m * 0.5
    };
    let range_scale_m = radius_m.abs().max(1.0);
    let aperture_m = aperture(
        &(0..mic_positions_m.len()).collect::<Vec<_>>(),
        mic_positions_m,
    );
    let geometric_range_observability = aperture_m / radius_m.abs().max(aperture_m).max(1.0);
    let range_observability = (range_scale_m / (range_scale_m + radial_std_m))
        .min(geometric_range_observability)
        .clamp(0.0, 1.0);
    let prior_radius_tolerance_m = (default_range_m * 0.02).max(0.5);
    let range_projection_mode = if (radius_m - default_range_m).abs() <= prior_radius_tolerance_m
        || range_observability < 0.10
        || radial_std_m >= radius_m.abs().max(1.0)
    {
        RANGE_ASYMPTOTIC
    } else {
        RANGE_REFINED
    };

    FarFieldRangeEstimate {
        position_m: add(centroid_m, scale(direction, radius_m)),
        radius_m,
        radial_std_m,
        range_observability,
        residual_rms_seconds,
        range_projection_mode,
    }
}

fn ray_mean_squared_error(
    radius_m: f32,
    direction: [f32; 3],
    centroid_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sound_speed_mps: f32,
) -> f32 {
    if measurements.is_empty() {
        return f32::INFINITY;
    }
    let position_m = add(centroid_m, scale(direction, radius_m.max(0.05)));
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
    squared_error / measurements.len() as f32
}

fn padded_bounds(
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]],
    search_padding_m: f32,
) -> ([f32; 3], [f32; 3]) {
    let padding_m = search_padding_m.max(0.05);
    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for &channel_index in active_channels {
        let position_m = mic_positions_m[channel_index];
        for axis in 0..3 {
            mins[axis] = mins[axis].min(position_m[axis] - padding_m);
            maxs[axis] = maxs[axis].max(position_m[axis] + padding_m);
        }
    }
    (mins, maxs)
}

fn is_boundary_clamped(
    position_m: [f32; 3],
    mins: [f32; 3],
    maxs: [f32; 3],
    grid_resolution_m: f32,
) -> bool {
    let margin_m = grid_resolution_m.max(0.05);
    (0..3).any(|axis| {
        (position_m[axis] - mins[axis]).abs() <= margin_m
            || (maxs[axis] - position_m[axis]).abs() <= margin_m
    })
}

fn is_well_inside_bounds(
    position_m: [f32; 3],
    mins: [f32; 3],
    maxs: [f32; 3],
    grid_resolution_m: f32,
) -> bool {
    let margin_m = grid_resolution_m.max(0.05);
    (0..3).all(|axis| {
        position_m[axis] > (mins[axis] + margin_m) && position_m[axis] < (maxs[axis] - margin_m)
    })
}

fn aperture(active_channels: &[usize], mic_positions_m: &[[f32; 3]]) -> f32 {
    let mut max_distance_m = 0.0_f32;
    for (left_index, &channel_a) in active_channels.iter().enumerate() {
        for &channel_b in &active_channels[left_index + 1..] {
            max_distance_m = max_distance_m.max(euclidean_distance(
                mic_positions_m[channel_a],
                mic_positions_m[channel_b],
            ));
        }
    }
    max_distance_m
}

fn diagonal_covariance(x_std_m: f32, y_std_m: f32, z_std_m: f32) -> [[f32; 3]; 3] {
    [
        [x_std_m * x_std_m, 0.0, 0.0],
        [0.0, y_std_m * y_std_m, 0.0],
        [0.0, 0.0, z_std_m * z_std_m],
    ]
}

fn directional_covariance(
    direction: [f32; 3],
    lateral_std_m: f32,
    radial_std_m: f32,
) -> [[f32; 3]; 3] {
    let radial_axis = normalize_or_zero(direction);
    if magnitude(radial_axis) <= EPSILON {
        return diagonal_covariance(lateral_std_m, lateral_std_m, lateral_std_m);
    }
    let mut helper_axis = [1.0_f32, 0.0, 0.0];
    if dot(radial_axis, helper_axis).abs() > 0.9 {
        helper_axis = [0.0, 1.0, 0.0];
    }
    let lateral_axis_a = normalize_or_zero(cross(radial_axis, helper_axis));
    let lateral_axis_b = normalize_or_zero(cross(radial_axis, lateral_axis_a));
    let radial_variance_m2 = radial_std_m.max(lateral_std_m).powi(2);
    let lateral_variance_m2 = lateral_std_m.powi(2);
    let mut covariance = [[0.0_f32; 3]; 3];
    accumulate_outer_product(&mut covariance, radial_axis, radial_variance_m2);
    accumulate_outer_product(&mut covariance, lateral_axis_a, lateral_variance_m2);
    accumulate_outer_product(&mut covariance, lateral_axis_b, lateral_variance_m2);
    covariance
}

fn accumulate_outer_product(covariance_m2: &mut [[f32; 3]; 3], axis: [f32; 3], variance_m2: f32) {
    for row in 0..3 {
        for col in 0..3 {
            covariance_m2[row][col] += axis[row] * axis[col] * variance_m2;
        }
    }
}

fn solve_normal_equations(rows: &[[f32; 3]], targets: &[f32]) -> Option<[f32; 3]> {
    if rows.is_empty() || rows.len() != targets.len() {
        return None;
    }
    let mut ata = [[0.0_f32; 3]; 3];
    let mut atb = [0.0_f32; 3];
    for (row, target) in rows.iter().zip(targets.iter()) {
        for out_axis in 0..3 {
            atb[out_axis] += row[out_axis] * *target;
            for in_axis in 0..3 {
                ata[out_axis][in_axis] += row[out_axis] * row[in_axis];
            }
        }
    }
    solve_3x3(ata, atb)
}

fn solve_3x3(mut matrix: [[f32; 3]; 3], mut rhs: [f32; 3]) -> Option<[f32; 3]> {
    for pivot in 0..3 {
        let mut best_row = pivot;
        let mut best_value = matrix[pivot][pivot].abs();
        for row in (pivot + 1)..3 {
            let candidate = matrix[row][pivot].abs();
            if candidate > best_value {
                best_value = candidate;
                best_row = row;
            }
        }
        if best_value <= EPSILON {
            return None;
        }
        if best_row != pivot {
            matrix.swap(pivot, best_row);
            rhs.swap(pivot, best_row);
        }
        let pivot_value = matrix[pivot][pivot];
        for col in pivot..3 {
            matrix[pivot][col] /= pivot_value;
        }
        rhs[pivot] /= pivot_value;
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor.abs() <= EPSILON {
                continue;
            }
            for col in pivot..3 {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn invert_3x3(matrix: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let col0 = solve_3x3(matrix, [1.0, 0.0, 0.0])?;
    let col1 = solve_3x3(matrix, [0.0, 1.0, 0.0])?;
    let col2 = solve_3x3(matrix, [0.0, 0.0, 1.0])?;
    Some([
        [col0[0], col1[0], col2[0]],
        [col0[1], col1[1], col2[1]],
        [col0[2], col1[2], col2[2]],
    ])
}

fn position_covariance_and_observability(
    position_m: [f32; 3],
    centroid_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
    minimum_std_m: f32,
) -> ([[f32; 3]; 3], f32) {
    let reciprocal_sound_speed = 1.0 / sound_speed_mps.max(1.0);
    let reference_position_m = mic_positions_m[reference_channel];
    let reference_offset_m = sub(position_m, reference_position_m);
    let reference_distance_m = magnitude(reference_offset_m) + EPSILON;
    let reference_jacobian_scale = reciprocal_sound_speed / reference_distance_m;
    let mut normal_matrix = [[0.0_f32; 3]; 3];
    for measurement in measurements {
        let sensor_position_m = mic_positions_m[measurement.channel_index];
        let sensor_offset_m = sub(position_m, sensor_position_m);
        let sensor_distance_m = magnitude(sensor_offset_m) + EPSILON;
        let sensor_jacobian_scale = reciprocal_sound_speed / sensor_distance_m;
        let jacobian_row = [
            (sensor_offset_m[0] * sensor_jacobian_scale)
                - (reference_offset_m[0] * reference_jacobian_scale),
            (sensor_offset_m[1] * sensor_jacobian_scale)
                - (reference_offset_m[1] * reference_jacobian_scale),
            (sensor_offset_m[2] * sensor_jacobian_scale)
                - (reference_offset_m[2] * reference_jacobian_scale),
        ];
        for out_axis in 0..3 {
            for in_axis in 0..3 {
                normal_matrix[out_axis][in_axis] += jacobian_row[out_axis] * jacobian_row[in_axis];
            }
        }
    }
    for axis in 0..3 {
        normal_matrix[axis][axis] += 1.0e-9;
    }
    let time_std_seconds = residual_rms(
        position_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    )
    .max(1.0 / sample_rate_hz.max(1) as f32);
    let variance_floor_m2 = minimum_std_m * minimum_std_m;
    let mut covariance_m2 = invert_3x3(normal_matrix)
        .map(|inverse| scale_matrix(inverse, time_std_seconds * time_std_seconds))
        .unwrap_or_else(|| diagonal_covariance(minimum_std_m, minimum_std_m, minimum_std_m));
    for axis in 0..3 {
        covariance_m2[axis][axis] = covariance_m2[axis][axis].max(variance_floor_m2);
    }
    let direction = normalize_or_zero(sub(position_m, centroid_m));
    let radial_variance_m2 = quadratic_form(covariance_m2, direction).max(EPSILON);
    let trace = covariance_m2[0][0] + covariance_m2[1][1] + covariance_m2[2][2];
    let lateral_variance_m2 = ((trace - radial_variance_m2) * 0.5).max(EPSILON);
    let range_observability = (lateral_variance_m2
        / radial_variance_m2.max(lateral_variance_m2).max(EPSILON))
    .sqrt()
    .clamp(0.0, 1.0);
    (covariance_m2, range_observability)
}

fn scale_matrix(matrix: [[f32; 3]; 3], scalar: f32) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = matrix[row][col] * scalar;
        }
    }
    out
}

fn quadratic_form(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> f32 {
    let product = [
        (matrix[0][0] * vector[0]) + (matrix[0][1] * vector[1]) + (matrix[0][2] * vector[2]),
        (matrix[1][0] * vector[0]) + (matrix[1][1] * vector[1]) + (matrix[1][2] * vector[2]),
        (matrix[2][0] * vector[0]) + (matrix[2][1] * vector[1]) + (matrix[2][2] * vector[2]),
    ];
    dot(vector, product)
}

fn grid_from_bounds(
    active_channels: &[usize],
    mic_positions_m: &[[f32; 3]],
    search_padding_m: f32,
    grid_resolution_m: f32,
    max_grid_points: usize,
    half_space: HalfSpace,
) -> Vec<[f32; 3]> {
    let padding_m = search_padding_m.max(0.05);
    let resolution_m = grid_resolution_m.max(0.05);

    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    let mut plane_z_sum = 0.0_f32;
    for &channel_index in active_channels {
        let position_m = mic_positions_m[channel_index];
        for axis in 0..3 {
            mins[axis] = mins[axis].min(position_m[axis] - padding_m);
            maxs[axis] = maxs[axis].max(position_m[axis] + padding_m);
        }
        plane_z_sum += position_m[2];
    }

    // D7: a coplanar array's TDOA solution is mirror-symmetric across its own
    // plane. Clamping the z-search bound to the array's own mean-z plane
    // halves the grid to the physically-constrained side instead of wasting
    // points (and solver ambiguity) on the unreachable side.
    if !active_channels.is_empty() {
        let plane_z_m = plane_z_sum / active_channels.len() as f32;
        match half_space {
            HalfSpace::Upper => mins[2] = mins[2].max(plane_z_m),
            HalfSpace::Lower => maxs[2] = maxs[2].min(plane_z_m),
            HalfSpace::None => {}
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
                if point_index.is_multiple_of(step) {
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
    let fraction = position - lower_index as f32;
    let lower_value = correlation.magnitudes[lower_index];
    let upper_value = correlation.magnitudes[upper_index];
    lower_value + ((upper_value - lower_value) * fraction)
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

fn centroid(active_channels: &[usize], mic_positions_m: &[[f32; 3]]) -> [f32; 3] {
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
    for item in &mut mean {
        *item /= active_channels.len() as f32;
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
    if ordered.len().is_multiple_of(2) {
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

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    ]
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
        assert!(
            euclidean_distance(recovered_position_m, source_position_m) <= 0.08,
            "expected <= 0.08 m error, got recovered={:?} source={:?} mode={:?}",
            recovered_position_m,
            source_position_m,
            evaluation.localization.range_projection_mode,
        );
        assert!(evaluation.localization.position_covariance_m2.is_some());
        assert!(evaluation.localization.range_observability.is_some());

        let expected_direction =
            normalize_or_zero(sub(source_position_m, centroid(&[0, 1, 2, 3], &mics)));
        assert!(
            dot(
                evaluation.localization.steering_direction,
                expected_direction
            ) > 0.95
        );
        assert!(
            evaluation.localization.confidence >= 0.20,
            "expected confidence >= 0.20, got {} mode={:?}",
            evaluation.localization.confidence,
            evaluation.localization.range_projection_mode,
        );
    }

    #[test]
    fn tetrahedral_solver_far_field_candidate_escapes_near_field_grid_bounds() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let source_position_m = [80.0, 34.0, 11.0];
        let channels = synthesize_point_source_channels(source_position_m, &mics, 48_000, 8_192);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            48_000,
            343.2,
            SrpPhatConfig {
                grid_resolution_m: 0.05,
                search_padding_m: 0.3,
                far_field_default_range_m: 60.0,
                far_field_max_range_m: 250.0,
                ..SrpPhatConfig::default()
            },
        );
        let config = SrpPhatConfig {
            grid_resolution_m: 0.05,
            search_padding_m: 0.3,
            far_field_default_range_m: 60.0,
            far_field_max_range_m: 250.0,
            ..SrpPhatConfig::default()
        };
        let pair_observations = build_pair_observations(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            48_000,
            343.2,
            config.localization_band_hz,
            config.interp_factor,
        );
        let reference_channel = [0, 1, 2, 3]
            .into_iter()
            .max_by(|left, right| {
                window_rms(&channels[*left])
                    .partial_cmp(&window_rms(&channels[*right]))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("reference channel");
        let measured_tdoa = build_reference_measurements(reference_channel, &pair_observations);
        let near_candidate = near_field_candidate(
            &[0, 1, 2, 3],
            &pair_observations,
            reference_channel,
            &measured_tdoa,
            &mics,
            48_000,
            343.2,
            config,
            false,
        );
        let far_candidate = far_field_candidate(
            &[0, 1, 2, 3],
            reference_channel,
            &measured_tdoa,
            &mics,
            48_000,
            343.2,
            config,
        );
        let recovered_position_m = evaluation
            .localization
            .position_m
            .expect("far-field position");
        let array_centroid_m = centroid(&[0, 1, 2, 3], &mics);
        let recovered_offset_m = sub(recovered_position_m, array_centroid_m);
        let expected_direction = normalize_or_zero(sub(source_position_m, array_centroid_m));
        assert!(
            magnitude(recovered_offset_m) > 20.0,
            "expected far-field escape, got recovered={:?} near={:?} far={:?}",
            recovered_position_m,
            near_candidate,
            far_candidate,
        );
        assert!(dot(normalize_or_zero(recovered_offset_m), expected_direction) > 0.8);
        assert!(evaluation.localization.position_covariance_m2.is_some());
        assert!(evaluation.localization.range_observability.is_some());
        assert_eq!(
            evaluation.localization.range_projection_mode.as_deref(),
            Some("range_bearing_projected")
        );
    }

    #[test]
    fn tetrahedral_solver_matches_python_near_field_regression_at_mid_range() {
        let mics = [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.05],
        ];
        let source_position_m = [1.5, 0.6, 0.4];
        let channel_count = 8_640_usize;
        let channels =
            synthesize_point_source_channels(source_position_m, &mics, 48_000, channel_count);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            48_000,
            343.2,
            SrpPhatConfig {
                grid_resolution_m: 0.05,
                search_padding_m: 0.3,
                far_field_default_range_m: 60.0,
                far_field_max_range_m: 250.0,
                ..SrpPhatConfig::default()
            },
        );
        let config = SrpPhatConfig {
            grid_resolution_m: 0.05,
            search_padding_m: 0.3,
            far_field_default_range_m: 60.0,
            far_field_max_range_m: 250.0,
            ..SrpPhatConfig::default()
        };
        let pair_observations = build_pair_observations(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            48_000,
            343.2,
            config.localization_band_hz,
            config.interp_factor,
        );
        let reference_channel = [0, 1, 2, 3]
            .into_iter()
            .max_by(|left, right| {
                window_rms(&channels[*left])
                    .partial_cmp(&window_rms(&channels[*right]))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("reference channel");
        let measured_tdoa = build_reference_measurements(reference_channel, &pair_observations);
        let near_candidate = near_field_candidate(
            &[0, 1, 2, 3],
            &pair_observations,
            reference_channel,
            &measured_tdoa,
            &mics,
            48_000,
            343.2,
            config,
            false,
        );
        let far_candidate = far_field_candidate(
            &[0, 1, 2, 3],
            reference_channel,
            &measured_tdoa,
            &mics,
            48_000,
            343.2,
            config,
        );

        let recovered_position_m = evaluation
            .localization
            .position_m
            .expect("mid-range position");
        let array_centroid_m = centroid(&[0, 1, 2, 3], &mics);
        let expected_direction = normalize_or_zero(sub(source_position_m, array_centroid_m));
        let recovered_direction = normalize_or_zero(sub(recovered_position_m, array_centroid_m));

        assert!(dot(recovered_direction, expected_direction) > 0.95);
        assert!(
            euclidean_distance(recovered_position_m, source_position_m) <= 1.5,
            "expected <= 1.5 m error, got recovered={:?} near={:?} far={:?}",
            recovered_position_m,
            near_candidate,
            far_candidate,
        );
        assert_ne!(
            evaluation.localization.range_projection_mode.as_deref(),
            Some("range_asymptotic")
        );
    }

    #[test]
    fn tetrahedral_solver_far_edge_source_reports_bearing_projected() {
        let mics = [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.05],
        ];
        let source_position_m = [4.0, -2.0, 1.0];
        let channel_count = 8_640_usize;
        let channels =
            synthesize_point_source_channels(source_position_m, &mics, 48_000, channel_count);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            48_000,
            343.2,
            SrpPhatConfig {
                grid_resolution_m: 0.05,
                search_padding_m: 0.3,
                far_field_default_range_m: 60.0,
                far_field_max_range_m: 250.0,
                ..SrpPhatConfig::default()
            },
        );

        let config = SrpPhatConfig {
            grid_resolution_m: 0.05,
            search_padding_m: 0.3,
            far_field_default_range_m: 60.0,
            far_field_max_range_m: 250.0,
            ..SrpPhatConfig::default()
        };
        let pair_observations = build_pair_observations(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            48_000,
            343.2,
            config.localization_band_hz,
            config.interp_factor,
        );
        let reference_channel = [0, 1, 2, 3]
            .into_iter()
            .max_by(|left, right| {
                window_rms(&channels[*left])
                    .partial_cmp(&window_rms(&channels[*right]))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("reference channel");
        let measured_tdoa = build_reference_measurements(reference_channel, &pair_observations);
        let near_candidate = near_field_candidate(
            &[0, 1, 2, 3],
            &pair_observations,
            reference_channel,
            &measured_tdoa,
            &mics,
            48_000,
            343.2,
            config,
            false,
        );
        let far_candidate = far_field_candidate(
            &[0, 1, 2, 3],
            reference_channel,
            &measured_tdoa,
            &mics,
            48_000,
            343.2,
            config,
        );

        let recovered_position_m = evaluation
            .localization
            .position_m
            .expect("far-edge position");
        let array_centroid_m = centroid(&[0, 1, 2, 3], &mics);
        let expected_direction = normalize_or_zero(sub(source_position_m, array_centroid_m));
        let recovered_direction = normalize_or_zero(sub(recovered_position_m, array_centroid_m));

        // The bearing must stay accurate even though the range is unobservable.
        assert!(dot(recovered_direction, expected_direction) > 0.95);
        // At ~4.6 m a 5 cm array cannot resolve range — wavefront curvature across
        // the aperture is sub-sample. The honest, high-quality result is therefore
        // a bearing projected along the ray with explicit range uncertainty
        // (range_bearing_projected), never range_asymptotic and never a falsely
        // precise near-field range.
        assert_eq!(
            evaluation.localization.range_projection_mode.as_deref(),
            Some("range_bearing_projected"),
            "expected range_bearing_projected, got mode={:?} recovered={:?} near={:?} far={:?}",
            evaluation.localization.range_projection_mode,
            recovered_position_m,
            near_candidate,
            far_candidate,
        );
        assert!(
            evaluation.localization.range_observability.unwrap_or(1.0) < 0.15,
            "expected low range observability, got {:?}",
            evaluation.localization.range_observability,
        );
        // Confidence-parity contract: a bearing-projected result must NOT be
        // penalised by the ×range_observability.clamp(0.35, 1.0) multiplier (that
        // would floor it near 0.35 despite a well-observed bearing). Instead it is
        // capped at BEARING_PROJECTED_CONFIDENCE_CAP (0.85), mirroring the Python
        // `obs_factor = 1.0` branch. With a tight bearing the confidence should
        // comfortably exceed the 0.35 floor the old multiplier would have imposed.
        assert!(
            evaluation.localization.confidence > 0.35,
            "bearing-projected confidence should not be floored by observability, got {}",
            evaluation.localization.confidence,
        );
        assert!(
            evaluation.localization.confidence <= 0.85 + 1e-6,
            "bearing-projected confidence must be capped at 0.85, got {}",
            evaluation.localization.confidence,
        );
    }

    #[test]
    fn confidence_caps_per_mode_match_contract() {
        // Perfect fit/peak/contrast → base_confidence = 1.0. The per-mode caps then
        // decide the final ceiling.
        let base = LocalizationCandidate {
            position_m: [60.0, 0.0, 0.0],
            steering_direction: [1.0, 0.0, 0.0],
            position_covariance_m2: [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]],
            range_observability: Some(0.9),
            range_projection_mode: Some("range_refined"),
            is_boundary_clamped: false,
            timing_resolution_seconds: 1.0e-5,
            residual_rms_seconds: 0.0,
            contrast_score: 1.0,
        };
        let tau_scale = 1.0e-4;
        let peak_score = 1.0;

        // Asymptotic: range AND bearing uncertain → harsh 0.20 cap.
        let asymptotic = LocalizationCandidate {
            range_projection_mode: Some("range_asymptotic"),
            ..base
        };
        let (conf, obs) =
            candidate_confidence_and_observability(&asymptotic, peak_score, tau_scale);
        assert!(
            conf <= 0.20 + 1e-6,
            "asymptotic confidence must be capped at 0.20, got {conf}"
        );
        assert!(
            obs.unwrap() <= 0.05 + 1e-6,
            "asymptotic obs cap 0.05, got {obs:?}"
        );

        // Bearing-projected with a tiny range_observability: the ×clamp(0.35,1.0)
        // multiplier is skipped, so confidence rides the 0.85 cap, NOT ~0.35.
        let bearing = LocalizationCandidate {
            range_projection_mode: Some("range_bearing_projected"),
            range_observability: Some(0.05),
            ..base
        };
        let (conf, obs) = candidate_confidence_and_observability(&bearing, peak_score, tau_scale);
        assert!(
            (conf - 0.85).abs() < 1e-6,
            "bearing-projected confidence should hit the 0.85 cap, got {conf}"
        );
        assert!(
            obs.unwrap() <= 0.05 + 1e-6,
            "bearing obs cap 0.05, got {obs:?}"
        );

        // Refined: no cap; low observability still attenuates via the multiplier.
        let refined = LocalizationCandidate {
            range_observability: Some(0.5),
            ..base
        };
        let (conf, _obs) = candidate_confidence_and_observability(&refined, peak_score, tau_scale);
        assert!(
            (conf - 0.5).abs() < 1e-6,
            "refined confidence should attenuate by observability (0.5), got {conf}"
        );
    }

    #[test]
    fn tight_array_boundary_clamp_requires_stronger_prior_projected_improvement() {
        let near_boundary = LocalizationCandidate {
            position_m: [0.0, 0.0, 0.0],
            steering_direction: [1.0, 0.0, 0.0],
            position_covariance_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            range_observability: Some(0.25),
            range_projection_mode: Some("range_boundary"),
            is_boundary_clamped: true,
            timing_resolution_seconds: 1.0e-5,
            residual_rms_seconds: 1.0,
            contrast_score: 0.8,
        };
        let near_interior = LocalizationCandidate {
            is_boundary_clamped: false,
            range_projection_mode: Some("range_refined"),
            ..near_boundary
        };
        let marginal_far = LocalizationCandidate {
            position_m: [50.0, 0.0, 0.0],
            steering_direction: [1.0, 0.0, 0.0],
            position_covariance_m2: [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            range_observability: Some(0.05),
            range_projection_mode: Some("range_asymptotic"),
            is_boundary_clamped: false,
            timing_resolution_seconds: 1.0e-5,
            residual_rms_seconds: 0.4,
            contrast_score: 0.6,
        };

        let selected_boundary = select_candidate(Some(near_boundary), Some(marginal_far))
            .expect("boundary-clamped near candidate");
        assert_eq!(
            selected_boundary.range_projection_mode,
            Some("range_boundary")
        );

        let selected_interior = select_candidate(Some(near_interior), Some(marginal_far))
            .expect("interior near candidate");
        assert_eq!(
            selected_interior.range_projection_mode,
            Some("range_refined")
        );
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

        assert_eq!(
            evaluation.localization.resolved_algorithm,
            "srp_phat_degraded_partial_array"
        );
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
        let channels = [
            vec![0.0; 512],
            vec![0.0; 512],
            vec![0.0; 512],
            vec![0.0; 512],
        ];
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &[0, 1, 2, 3],
            &mics,
            16_000,
            343.2,
            SrpPhatConfig::default(),
        );

        assert_eq!(
            evaluation.localization.resolved_algorithm,
            "srp_phat_degraded_low_energy"
        );
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

    #[test]
    fn sample_pair_correlation_interpolates_between_adjacent_bins() {
        let correlation = GccPhatCorrelation {
            lags_seconds: vec![-0.001, 0.0, 0.001],
            magnitudes: vec![0.2, 1.0, 0.4],
            tdoa: crate::gcc_phat::TdoaResult {
                delay_samples: 0.0,
                lag_seconds: 0.0,
                confidence: 1.0,
                peak_value: 1.0,
            },
        };

        assert_eq!(sample_pair_correlation(&correlation, -0.001), 0.2);
        assert_eq!(sample_pair_correlation(&correlation, 0.0), 1.0);
        assert!((sample_pair_correlation(&correlation, -0.0005) - 0.6).abs() < 1.0e-6);
        assert!((sample_pair_correlation(&correlation, 0.0005) - 0.7).abs() < 1.0e-6);
        assert_eq!(sample_pair_correlation(&correlation, -0.002), 0.0);
        assert_eq!(sample_pair_correlation(&correlation, 0.002), 0.0);
    }

    #[test]
    fn far_field_range_fit_converges_with_acoustic_tolerance() {
        let mics = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ];
        let active_channels = [0, 1, 2, 3];
        let centroid_m = centroid(&active_channels, &mics);
        let direction = normalize_or_zero([0.84, 0.42, 0.34]);
        let expected_radius_m = 80.0;
        let source_position_m = add(centroid_m, scale(direction, expected_radius_m));
        let reference_channel = 0;
        let measurements: Vec<ReferenceMeasurement> = [1, 2, 3]
            .into_iter()
            .map(|channel_index| ReferenceMeasurement {
                channel_index,
                tdoa_seconds: predicted_reference_tau_s(
                    source_position_m,
                    mics[channel_index],
                    mics[reference_channel],
                    343.2,
                ),
                peak_value: 1.0,
            })
            .collect();

        let range_estimate = fit_far_field_range(
            direction,
            centroid_m,
            reference_channel,
            &measurements,
            &mics,
            48_000,
            343.2,
            SrpPhatConfig {
                far_field_default_range_m: 60.0,
                far_field_max_range_m: 250.0,
                ..SrpPhatConfig::default()
            },
        );

        assert!(
            (range_estimate.radius_m - expected_radius_m).abs() <= 0.10,
            "expected radius near {expected_radius_m}, got {}",
            range_estimate.radius_m,
        );
        assert!(
            dot(
                normalize_or_zero(sub(range_estimate.position_m, centroid_m)),
                direction,
            ) > 0.999
        );
        assert!(range_estimate.residual_rms_seconds.is_finite());
    }

    #[test]
    fn far_field_range_fit_converges_at_one_km_within_iteration_bound() {
        // Phase 2 compute audit: at the 1 km envelope the golden-section search still
        // converges to the 0.05 m tolerance well within the hard 48-iteration cap.
        // Theoretical bound: iterations = ceil(ln(tol / range) / ln(phi)).
        let max_range_m: f64 = 1000.0;
        let tol_m: f64 = FAR_FIELD_RANGE_CONVERGENCE_TOLERANCE_M as f64;
        let inverse_phi: f64 = 0.618_033_988_75;
        let iterations = ((tol_m / max_range_m).ln() / inverse_phi.ln()).ceil() as i32;
        assert!(
            iterations <= 48,
            "golden-section needs {iterations} iters at 1 km, exceeds the 48 cap"
        );

        // And it actually resolves a 900 m source with a wide (10 m) array.
        let mics = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ];
        let active_channels = [0, 1, 2, 3];
        let centroid_m = centroid(&active_channels, &mics);
        let direction = normalize_or_zero([0.6, 0.5, 0.62]);
        let expected_radius_m = 900.0;
        let source_position_m = add(centroid_m, scale(direction, expected_radius_m));
        let reference_channel = 0;
        let measurements: Vec<ReferenceMeasurement> = [1, 2, 3]
            .into_iter()
            .map(|channel_index| ReferenceMeasurement {
                channel_index,
                tdoa_seconds: predicted_reference_tau_s(
                    source_position_m,
                    mics[channel_index],
                    mics[reference_channel],
                    343.2,
                ),
                peak_value: 1.0,
            })
            .collect();
        let range_estimate = fit_far_field_range(
            direction,
            centroid_m,
            reference_channel,
            &measurements,
            &mics,
            48_000,
            343.2,
            SrpPhatConfig {
                far_field_default_range_m: 50.0,
                far_field_max_range_m: 1000.0,
                ..SrpPhatConfig::default()
            },
        );
        // A 10 m array resolves range at 900 m to within a few percent.
        assert!(
            (range_estimate.radius_m - expected_radius_m).abs() <= 0.05 * expected_radius_m,
            "expected radius near {expected_radius_m}, got {}",
            range_estimate.radius_m,
        );
        assert!(range_estimate.residual_rms_seconds.is_finite());
    }

    #[test]
    fn fit_far_field_range_is_fast_at_one_km() {
        // Phase 6 compute budget: the 1 km golden-section fit must stay well under
        // 1 ms so it never threatens the localization cadence.
        let mics = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ];
        let active_channels = [0, 1, 2, 3];
        let centroid_m = centroid(&active_channels, &mics);
        let direction = normalize_or_zero([0.6, 0.5, 0.62]);
        let source_position_m = add(centroid_m, scale(direction, 900.0));
        let reference_channel = 0;
        let measurements: Vec<ReferenceMeasurement> = [1, 2, 3]
            .into_iter()
            .map(|channel_index| ReferenceMeasurement {
                channel_index,
                tdoa_seconds: predicted_reference_tau_s(
                    source_position_m,
                    mics[channel_index],
                    mics[reference_channel],
                    343.2,
                ),
                peak_value: 1.0,
            })
            .collect();
        let config = SrpPhatConfig {
            far_field_default_range_m: 50.0,
            far_field_max_range_m: 1000.0,
            ..SrpPhatConfig::default()
        };
        // Warm up, then time a batch to amortise timer overhead.
        let iterations = 200;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = fit_far_field_range(
                direction,
                centroid_m,
                reference_channel,
                &measurements,
                &mics,
                48_000,
                343.2,
                config,
            );
        }
        let per_call = start.elapsed() / iterations;
        assert!(
            per_call < std::time::Duration::from_millis(1),
            "fit_far_field_range at 1 km took {per_call:?}/call (budget <1 ms)"
        );
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
        delays_samples.map(|delay_samples| {
            fractional_delay(&source, delay_samples - min_delay_samples, output_len)
        })
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

    // --- Distance-matrix coverage (single tetra + synchronized multi-node) ----
    //
    // The N-sensor synthesizers below generalize `synthesize_point_source_channels`
    // (hard-typed to a 4-mic array) to any number of microphones, so the
    // geometry-agnostic solver can be exercised with combined positions from
    // several nodes. Like the Python `synthesize_multinode_windows` helper, a
    // single *global* minimum delay is subtracted so inter-node arrival timing
    // (the parallax that makes range observable) is preserved.

    const SIRITH_TETRA_OFFSETS_M: [[f32; 3]; 4] = [
        [-0.016238, 0.025000, -0.010205],
        [0.027063, 0.000000, -0.010205],
        [-0.016238, -0.025000, -0.010205],
        [0.005413, 0.000000, 0.030615],
    ];

    fn relative_delays_samples(
        source_position_m: [f32; 3],
        mic_positions_m: &[[f32; 3]],
        sample_rate_hz: u32,
    ) -> Vec<f32> {
        let delays: Vec<f32> = mic_positions_m
            .iter()
            .map(|position_m| {
                euclidean_distance(source_position_m, *position_m) / 343.2 * sample_rate_hz as f32
            })
            .collect();
        let min_delay = delays.iter().copied().fold(f32::INFINITY, f32::min);
        delays.into_iter().map(|delay| delay - min_delay).collect()
    }

    fn synthesize_point_source_channels_n(
        source_position_m: [f32; 3],
        mic_positions_m: &[[f32; 3]],
        sample_rate_hz: u32,
        output_len: usize,
    ) -> Vec<Vec<f32>> {
        let relative = relative_delays_samples(source_position_m, mic_positions_m, sample_rate_hz);
        let max_delay = relative.iter().copied().fold(0.0_f32, f32::max).ceil() as usize;
        let source = pseudo_random_with_seed(0x1234_5678, output_len + max_delay + 64);
        relative
            .iter()
            .map(|delay| fractional_delay(&source, *delay, output_len))
            .collect()
    }

    fn synthesize_tone_source_channels_n(
        source_position_m: [f32; 3],
        mic_positions_m: &[[f32; 3]],
        freq_hz: f32,
        sample_rate_hz: u32,
        output_len: usize,
    ) -> Vec<Vec<f32>> {
        let relative = relative_delays_samples(source_position_m, mic_positions_m, sample_rate_hz);
        relative
            .iter()
            .map(|delay| {
                (0..output_len)
                    .map(|sample_index| {
                        let t = (sample_index as f32 - delay) / sample_rate_hz as f32;
                        if t < 0.0 {
                            0.0
                        } else {
                            (2.0 * std::f32::consts::PI * freq_hz * t).sin()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Flat microphone list for a tetra node at `tetra_origin_m` plus single-mic
    /// "point" nodes at `point_origins_m`.
    fn multinode_mics(tetra_origin_m: [f32; 3], point_origins_m: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut mics: Vec<[f32; 3]> = SIRITH_TETRA_OFFSETS_M
            .iter()
            .map(|offset| {
                [
                    tetra_origin_m[0] + offset[0],
                    tetra_origin_m[1] + offset[1],
                    tetra_origin_m[2] + offset[2],
                ]
            })
            .collect();
        mics.extend_from_slice(point_origins_m);
        mics
    }

    fn array_centroid(mics: &[[f32; 3]]) -> [f32; 3] {
        let channels: Vec<usize> = (0..mics.len()).collect();
        centroid(&channels, mics)
    }

    fn bearing_dot(localization: &SrpPhatLocalization, expected_direction: [f32; 3]) -> f32 {
        dot(localization.steering_direction, expected_direction)
    }

    const SINGLE_TETRA_DISTANCES_M: [f32; 4] = [2.0, 10.0, 100.0, 1000.0];

    fn look_direction() -> [f32; 3] {
        normalize_or_zero([0.6, 0.7, 0.39])
    }

    // Single tetra: a bearing instrument at every range. Direction is recovered
    // accurately from 2 m to 1 km; range is not observable for a ~7 cm array
    // (mirrors `test_single_tetra_distance_matrix` on the Python side).
    #[test]
    fn distance_matrix_single_tetra_reports_accurate_bearing() {
        let mics: Vec<[f32; 3]> = SIRITH_TETRA_OFFSETS_M.to_vec();
        let centroid_m = array_centroid(&mics);
        let direction = look_direction();
        for distance_m in SINGLE_TETRA_DISTANCES_M {
            let source_position_m = [
                centroid_m[0] + direction[0] * distance_m,
                centroid_m[1] + direction[1] * distance_m,
                centroid_m[2] + direction[2] * distance_m,
            ];
            let channels =
                synthesize_point_source_channels_n(source_position_m, &mics, 16_000, 8_192);
            let evaluation = estimate_tetrahedral_steering(
                &channels,
                &[0, 1, 2, 3],
                &mics,
                16_000,
                343.2,
                SrpPhatConfig {
                    grid_resolution_m: 0.05,
                    search_padding_m: 0.3,
                    far_field_default_range_m: 60.0,
                    far_field_max_range_m: 250.0,
                    ..SrpPhatConfig::default()
                },
            );
            let observed = bearing_dot(&evaluation.localization, direction);
            assert!(
                observed > 0.9,
                "distance {distance_m} m: bearing dot {observed} too low (dir={:?})",
                evaluation.localization.steering_direction,
            );
        }
    }

    // Synchronized multi-node fusion. The geometry-agnostic SRP-PHAT solver
    // accepts the combined microphone positions of several nodes — here a 4-mic
    // tetra plus three single-mic "point" nodes (7 sensors) — and fuses them
    // into a single estimate.
    //
    // SCOPE: this solver is a *near-field grid* method. It fixes position for a
    // source that falls within (roughly) the array aperture and reports bearing
    // for anything well beyond it. Wide-baseline, long-range *position* fusion
    // (e.g. a 1 km source) is the job of the Python cartesian-TDOA engine and is
    // covered by tests/test_localization_distance_matrix.py. Here we verify (a)
    // multi-node fusion produces an accurate position for an in-range source for
    // both a tight and a wider cluster, and (b) a far source degrades gracefully
    // to a bearing-only estimate with honestly collapsed range observability.
    fn multinode_config() -> SrpPhatConfig {
        SrpPhatConfig {
            grid_resolution_m: 1.0,
            search_padding_m: 10.0,
            far_field_default_range_m: 250.0,
            far_field_max_range_m: 2_000.0,
            max_grid_points: 200_000,
            ..SrpPhatConfig::default()
        }
    }

    const MULTINODE_TIGHT_POINTS_M: [[f32; 3]; 3] =
        [[8.0, 0.0, 3.0], [0.0, 8.0, 1.0], [8.0, 8.0, 4.0]];
    const MULTINODE_WIDE_POINTS_M: [[f32; 3]; 3] =
        [[50.0, 0.0, 6.0], [0.0, 50.0, 1.0], [50.0, 50.0, 9.0]];

    fn localize_multinode(
        point_origins_m: &[[f32; 3]],
        distance_m: f32,
    ) -> (SrpPhatLocalization, [f32; 3], [f32; 3]) {
        let direction = look_direction();
        let mics = multinode_mics([0.0, 0.0, 1.5], point_origins_m);
        let active: Vec<usize> = (0..mics.len()).collect();
        let centroid_m = array_centroid(&mics);
        let source_position_m = [
            centroid_m[0] + direction[0] * distance_m,
            centroid_m[1] + direction[1] * distance_m,
            centroid_m[2] + direction[2] * distance_m,
        ];
        let channels = synthesize_point_source_channels_n(source_position_m, &mics, 16_000, 16_384);
        let evaluation = estimate_tetrahedral_steering(
            &channels,
            &active,
            &mics,
            16_000,
            343.2,
            multinode_config(),
        );
        (evaluation.localization, source_position_m, direction)
    }

    #[test]
    fn distance_matrix_multinode_near_field_position_fix() {
        // (points, distance_m, max_error_m, min_bearing_dot)
        let cases: [(&[[f32; 3]], f32, f32, f32); 3] = [
            (&MULTINODE_TIGHT_POINTS_M, 2.0, 1.0, 0.95),
            (&MULTINODE_TIGHT_POINTS_M, 10.0, 1.0, 0.95),
            (&MULTINODE_WIDE_POINTS_M, 10.0, 3.0, 0.95),
        ];
        for (points, distance_m, max_error_m, min_bearing_dot) in cases {
            let (localization, source_position_m, direction) =
                localize_multinode(points, distance_m);
            let position_m = localization
                .position_m
                .unwrap_or_else(|| panic!("{distance_m} m: expected a fused position"));
            let error_m = euclidean_distance(position_m, source_position_m);
            assert!(
                error_m <= max_error_m,
                "{distance_m} m: fused position error {error_m:.2} m exceeded {max_error_m:.2} m (pos={position_m:?})",
            );
            let observed_bearing = bearing_dot(&localization, direction);
            assert!(
                observed_bearing > min_bearing_dot,
                "{distance_m} m: bearing dot {observed_bearing} below {min_bearing_dot}",
            );
        }
    }

    #[test]
    fn distance_matrix_multinode_far_source_is_bearing_only() {
        // A 1 km source is far beyond the cluster aperture: the grid solver can
        // no longer fix range, but the steering direction stays accurate and the
        // reported range observability collapses.
        let (localization, _source_position_m, direction) =
            localize_multinode(&MULTINODE_TIGHT_POINTS_M, 1000.0);
        let observed_bearing = bearing_dot(&localization, direction);
        assert!(
            observed_bearing > 0.99,
            "far source bearing dot {observed_bearing} too low",
        );
        if let Some(observability) = localization.range_observability {
            assert!(
                observability < 0.10,
                "expected collapsed range observability for a far source, got {observability}",
            );
        }
    }

    // Frequency sensitivity, scoped to the tight tetra where a pure tone is
    // spatially unambiguous. The ~5 cm tetra baseline aliases above
    // c / (2 * d_max) ≈ 3.2 kHz, so the band is sampled below that limit.
    #[test]
    fn distance_matrix_single_tetra_tone_band_bearing() {
        let mics: Vec<[f32; 3]> = SIRITH_TETRA_OFFSETS_M.to_vec();
        let centroid_m = array_centroid(&mics);
        let direction = look_direction();
        for freq_hz in [500.0_f32, 1500.0, 2500.0] {
            for distance_m in [2.0_f32, 100.0] {
                let source_position_m = [
                    centroid_m[0] + direction[0] * distance_m,
                    centroid_m[1] + direction[1] * distance_m,
                    centroid_m[2] + direction[2] * distance_m,
                ];
                let channels = synthesize_tone_source_channels_n(
                    source_position_m,
                    &mics,
                    freq_hz,
                    16_000,
                    8_192,
                );
                let evaluation = estimate_tetrahedral_steering(
                    &channels,
                    &[0, 1, 2, 3],
                    &mics,
                    16_000,
                    343.2,
                    SrpPhatConfig {
                        grid_resolution_m: 0.05,
                        search_padding_m: 0.3,
                        far_field_default_range_m: 60.0,
                        far_field_max_range_m: 250.0,
                        ..SrpPhatConfig::default()
                    },
                );
                let observed = bearing_dot(&evaluation.localization, direction);
                assert!(
                    observed > 0.85,
                    "{freq_hz} Hz @ {distance_m} m: bearing dot {observed} too low",
                );
            }
        }
    }

    fn planar_mics() -> [[f32; 3]; 5] {
        let r = 0.025 * std::f32::consts::FRAC_1_SQRT_2;
        [
            [r, r, 0.0],
            [-r, r, 0.0],
            [-r, -r, 0.0],
            [r, -r, 0.0],
            [0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn half_space_none_produces_symmetric_grid_around_array_plane() {
        let mics = planar_mics();
        let grid = grid_from_bounds(&[0, 1, 2, 3, 4], &mics, 0.1, 0.05, 60_000, HalfSpace::None);
        assert!(grid.iter().any(|p| p[2] < 0.0));
        assert!(grid.iter().any(|p| p[2] > 0.0));
    }

    #[test]
    fn half_space_upper_clamps_grid_to_non_negative_z() {
        let mics = planar_mics();
        let grid = grid_from_bounds(&[0, 1, 2, 3, 4], &mics, 0.1, 0.05, 60_000, HalfSpace::Upper);
        assert!(!grid.is_empty());
        assert!(grid.iter().all(|p| p[2] >= -1e-6));
        assert!(grid.iter().any(|p| p[2] > 0.0));
    }

    #[test]
    fn half_space_lower_clamps_grid_to_non_positive_z() {
        let mics = planar_mics();
        let grid = grid_from_bounds(&[0, 1, 2, 3, 4], &mics, 0.1, 0.05, 60_000, HalfSpace::Lower);
        assert!(!grid.is_empty());
        assert!(grid.iter().all(|p| p[2] <= 1e-6));
        assert!(grid.iter().any(|p| p[2] < 0.0));
    }

    #[test]
    fn half_space_clamps_relative_to_pitched_array_plane_not_world_origin() {
        // A "pitched" node: the array plane itself sits off the world-frame
        // z=0 origin (e.g. mounted at a 2m height offset carried through in
        // mic_positions_m). The clamp must key off the array's own mean-z
        // plane, not an assumed world origin.
        let mut mics = planar_mics();
        for m in &mut mics {
            m[2] += 2.0;
        }
        let grid = grid_from_bounds(&[0, 1, 2, 3, 4], &mics, 0.1, 0.05, 60_000, HalfSpace::Upper);
        assert!(!grid.is_empty());
        assert!(grid.iter().all(|p| p[2] >= 2.0 - 1e-6));
        assert!(grid.iter().any(|p| p[2] > 2.0));
    }

    #[test]
    fn half_space_from_wire_str_matches_node_spec_values() {
        assert_eq!(HalfSpace::from_wire_str("upper"), HalfSpace::Upper);
        assert_eq!(HalfSpace::from_wire_str("lower"), HalfSpace::Lower);
        assert_eq!(HalfSpace::from_wire_str("none"), HalfSpace::None);
        assert_eq!(HalfSpace::from_wire_str("garbage"), HalfSpace::None);
    }
}
