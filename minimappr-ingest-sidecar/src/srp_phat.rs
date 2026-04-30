use serde::{Deserialize, Serialize};

use crate::dsp_worker::PairTdoa;

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

pub fn estimate_tetrahedral_steering(
    pair_tdoas: &[PairTdoa],
    mic_positions_m: &[[f32; 3]; 4],
    sound_speed_mps: f32,
) -> Option<SrpPhatLocalization> {
    let mut normal = [[0.0_f32; 3]; 3];
    let mut rhs = [0.0_f32; 3];
    let mut usable_pairs = 0_usize;
    let mut mean_pair_confidence = 0.0_f32;

    for pair in pair_tdoas {
        if pair.ch_a >= mic_positions_m.len() || pair.ch_b >= mic_positions_m.len() {
            continue;
        }
        let baseline = sub(mic_positions_m[pair.ch_b], mic_positions_m[pair.ch_a]);
        let observed_path_delta_m = pair.tdoa.lag_seconds * sound_speed_mps;
        for row in 0..3 {
            rhs[row] += baseline[row] * observed_path_delta_m;
            for col in 0..3 {
                normal[row][col] += baseline[row] * baseline[col];
            }
        }
        usable_pairs += 1;
        mean_pair_confidence += pair.tdoa.confidence;
    }

    if usable_pairs < 3 {
        return None;
    }

    let raw_direction = solve_3x3(normal, rhs)?;
    let norm = magnitude(raw_direction);
    if norm < 1e-6 {
        return None;
    }
    let steering_direction = [
        raw_direction[0] / norm,
        raw_direction[1] / norm,
        raw_direction[2] / norm,
    ];
    mean_pair_confidence /= usable_pairs as f32;

    let residual_rms_seconds = residual_rms(
        pair_tdoas,
        mic_positions_m,
        steering_direction,
        sound_speed_mps,
    );
    let aperture_m = max_aperture(mic_positions_m);
    let residual_limit_s = (aperture_m / sound_speed_mps).max(1e-6);
    let residual_confidence = (1.0 - (residual_rms_seconds / residual_limit_s)).clamp(0.0, 1.0);
    let confidence = (mean_pair_confidence * residual_confidence).clamp(0.0, 1.0);

    Some(SrpPhatLocalization {
        attempted_algorithm: "srp_phat".to_string(),
        resolved_algorithm: "srp_phat".to_string(),
        steering_direction,
        position_m: Some(steering_direction),
        confidence,
        residual_rms_seconds,
        sound_speed_mps,
    })
}

fn residual_rms(
    pair_tdoas: &[PairTdoa],
    mic_positions_m: &[[f32; 3]; 4],
    direction: [f32; 3],
    sound_speed_mps: f32,
) -> f32 {
    let mut squared = 0.0_f32;
    let mut count = 0_usize;
    for pair in pair_tdoas {
        if pair.ch_a >= mic_positions_m.len() || pair.ch_b >= mic_positions_m.len() {
            continue;
        }
        let baseline = sub(mic_positions_m[pair.ch_b], mic_positions_m[pair.ch_a]);
        let predicted = dot(baseline, direction) / sound_speed_mps;
        let error = pair.tdoa.lag_seconds - predicted;
        squared += error * error;
        count += 1;
    }
    if count == 0 {
        return f32::INFINITY;
    }
    (squared / count as f32).sqrt()
}

fn max_aperture(mic_positions_m: &[[f32; 3]; 4]) -> f32 {
    let mut max_distance = 0.0_f32;
    for a in 0..mic_positions_m.len() {
        for b in a + 1..mic_positions_m.len() {
            max_distance = max_distance.max(magnitude(sub(mic_positions_m[a], mic_positions_m[b])));
        }
    }
    max_distance
}

fn solve_3x3(mut matrix: [[f32; 3]; 3], mut rhs: [f32; 3]) -> Option<[f32; 3]> {
    for pivot in 0..3 {
        let mut best_row = pivot;
        let mut best_abs = matrix[pivot][pivot].abs();
        for row in pivot + 1..3 {
            let candidate_abs = matrix[row][pivot].abs();
            if candidate_abs > best_abs {
                best_abs = candidate_abs;
                best_row = row;
            }
        }
        if best_abs < 1e-9 {
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
            for col in pivot..3 {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dsp_worker::PairTdoa, gcc_phat::TdoaResult};

    #[test]
    fn tetrahedral_solver_recovers_far_field_direction() {
        let mics = [
            [0.0, 0.050, 0.0],
            [0.0433, 0.025, 0.0],
            [0.0, 0.0, 0.0],
            [0.02165, 0.025, 0.04082],
        ];
        let direction = normalize([0.4, -0.2, 0.7]);
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            .into_iter()
            .map(|(a, b)| {
                let baseline = sub(mics[b], mics[a]);
                PairTdoa {
                    ch_a: a,
                    ch_b: b,
                    tdoa: TdoaResult {
                        delay_samples: 0.0,
                        lag_seconds: dot(baseline, direction) / 343.2,
                        confidence: 0.95,
                    },
                }
            })
            .collect::<Vec<_>>();

        let result = estimate_tetrahedral_steering(&pairs, &mics, 343.2).unwrap();
        assert!(dot(result.steering_direction, direction) > 0.99);
        assert_eq!(result.resolved_algorithm, "srp_phat");
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let mag = magnitude(v);
        [v[0] / mag, v[1] / mag, v[2] / mag]
    }
}
