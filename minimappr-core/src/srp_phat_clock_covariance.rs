//! Clock-offset term of the SRP position covariance.
//!
//! The solver's own covariance treats every reference TDOA as independent
//! noise at the fit residual. A node's clock offset is different: it shifts
//! every TDOA between that node and the reference node by the same amount,
//! and its effect on position grows with geometric dilution (roughly r/b
//! laterally and r^2/b in range for a spread b at range r). Adding a flat
//! (c * u)^2 per axis understates it by orders of magnitude at range, so the
//! offset is propagated through the solve's Jacobian instead.
//!
//! The Gauss-Newton refinement is unweighted, so the propagated covariance
//! is the least-squares sandwich `G R G^T` with `G = (J^T J)^-1 J^T` and
//! `R` the per-clock correlated timing noise, evaluated at the solved
//! position (the same linearisation the Python re-solve uses).

use super::{magnitude, sub, ReferenceMeasurement, EPSILON};
use crate::tdoa_node_position_covariance::{invert_symmetric_3x3, multiply_3x3_vector};

/// Which clock stamps each buffer channel, and every clock's 1-sigma offset.
/// Channels sharing a clock index (one node) have no offset between them.
#[derive(Clone, Copy, Debug)]
pub struct ChannelClockOffsets<'a> {
    /// Indexed by buffer channel: index into `clock_offset_std_seconds`.
    pub clock_index_by_channel: &'a [usize],
    pub clock_offset_std_seconds: &'a [f64],
}

/// Ridge on `J^T J`, relative to its trace: keeps a direction the
/// measurements cannot see at all (a planar array's normal) from blowing
/// up the inverse, while staying far below any direction they can see.
const RELATIVE_NORMAL_MATRIX_RIDGE: f64 = 1.0e-12;

/// Position covariance (m^2) from the clocks' offsets, or `None` when every
/// measured channel shares the reference channel's clock (no offset term).
pub(super) fn clock_offset_position_covariance_m2(
    position_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sound_speed_mps: f32,
    clocks: ChannelClockOffsets<'_>,
) -> Option<[[f32; 3]; 3]> {
    let reference_clock = *clocks.clock_index_by_channel.get(reference_channel)?;
    let measurement_clocks: Vec<usize> = measurements
        .iter()
        .map(|measurement| {
            clocks
                .clock_index_by_channel
                .get(measurement.channel_index)
                .copied()
        })
        .collect::<Option<_>>()?;
    if measurement_clocks
        .iter()
        .all(|clock| *clock == reference_clock)
    {
        return None;
    }

    let jacobian_rows = reference_tdoa_jacobian_rows(
        position_m,
        reference_channel,
        measurements,
        mic_positions_m,
        sound_speed_mps,
    );
    let mut normal_matrix = [[0.0_f64; 3]; 3];
    for row in &jacobian_rows {
        for out_axis in 0..3 {
            for in_axis in 0..3 {
                normal_matrix[out_axis][in_axis] += row[out_axis] * row[in_axis];
            }
        }
    }
    let trace = normal_matrix[0][0] + normal_matrix[1][1] + normal_matrix[2][2];
    if trace <= 0.0 || !trace.is_finite() {
        return None;
    }
    for (axis, row) in normal_matrix.iter_mut().enumerate() {
        row[axis] += trace * RELATIVE_NORMAL_MATRIX_RIDGE;
    }
    let normal_inverse = invert_symmetric_3x3(normal_matrix)?;

    // TDOA_i = t_i - t_ref, so clock k's offset d_k moves measurement i by
    // ([clock(i) == k] - [clock(ref) == k]) * d_k. Clocks are independent,
    // so each contributes sigma_k^2 * g_k g_k^T with g_k = G e_k.
    let mut covariance_m2 = [[0.0_f64; 3]; 3];
    for (clock_index, &offset_std_seconds) in clocks.clock_offset_std_seconds.iter().enumerate() {
        let reference_on_clock = f64::from(u8::from(reference_clock == clock_index));
        let mut projected = [0.0_f64; 3];
        let mut clock_moves_any_measurement = false;
        for (row, &measurement_clock) in jacobian_rows.iter().zip(&measurement_clocks) {
            let shift = f64::from(u8::from(measurement_clock == clock_index)) - reference_on_clock;
            if shift == 0.0 {
                continue;
            }
            clock_moves_any_measurement = true;
            for axis in 0..3 {
                projected[axis] += row[axis] * shift;
            }
        }
        if !clock_moves_any_measurement {
            continue;
        }
        let sensitivity_m_per_s = multiply_3x3_vector(normal_inverse, projected);
        let offset_variance_s2 = offset_std_seconds * offset_std_seconds;
        for row in 0..3 {
            for col in 0..3 {
                covariance_m2[row][col] +=
                    offset_variance_s2 * sensitivity_m_per_s[row] * sensitivity_m_per_s[col];
            }
        }
    }
    Some(covariance_m2.map(|row| row.map(|value| value as f32)))
}

/// d(TDOA_i)/d(position) in s/m, for each reference measurement.
fn reference_tdoa_jacobian_rows(
    position_m: [f32; 3],
    reference_channel: usize,
    measurements: &[ReferenceMeasurement],
    mic_positions_m: &[[f32; 3]],
    sound_speed_mps: f32,
) -> Vec<[f64; 3]> {
    let reciprocal_sound_speed = 1.0 / f64::from(sound_speed_mps.max(1.0));
    let unit_toward_position = |sensor_position_m: [f32; 3]| {
        let offset_m = sub(position_m, sensor_position_m);
        let distance_m = f64::from(magnitude(offset_m)) + f64::from(EPSILON);
        offset_m.map(|axis| f64::from(axis) / distance_m)
    };
    let reference_unit = unit_toward_position(mic_positions_m[reference_channel]);
    measurements
        .iter()
        .map(|measurement| {
            let sensor_unit = unit_toward_position(mic_positions_m[measurement.channel_index]);
            [0, 1, 2]
                .map(|axis| (sensor_unit[axis] - reference_unit[axis]) * reciprocal_sound_speed)
        })
        .collect()
}

#[cfg(test)]
#[path = "srp_phat_clock_covariance_tests.rs"]
mod srp_phat_clock_covariance_tests;
