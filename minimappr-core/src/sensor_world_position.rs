//! Sensor world position = node position + R(body -> world) x body offset.
//!
//! R is the matrix of Python `orientation_rotation_matrix`
//! (`minimappr/spatial_audio/geometry.py`: Rz(yaw) Ry(pitch) Rx(roll), world =
//! R x body), which `minimappr/core/cross_node_beamform.py` inverts as
//! `local = R.T @ (world - node)`. On the wire R travels as a unit quaternion
//! (w, x, y, z); a pure yaw of psi is `[cos(psi/2), 0, 0, sin(psi/2)]`.

/// World position of one sensor. `None` rotation is identity. The quaternion
/// need not be exactly unit length (it is normalised), but must be finite and
/// non-zero; pushes are validated for that before they reach a solve.
pub fn sensor_world_position_m(
    node_position_m: [f64; 3],
    body_to_world_quaternion_wxyz: Option<[f64; 4]>,
    body_offset_m: [f64; 3],
) -> [f64; 3] {
    let rotated_offset_m = match body_to_world_quaternion_wxyz {
        None => body_offset_m,
        Some(quaternion_wxyz) => {
            let rotation = body_to_world_rotation_matrix(quaternion_wxyz);
            rotation.map(|row| {
                row[0] * body_offset_m[0] + row[1] * body_offset_m[1] + row[2] * body_offset_m[2]
            })
        }
    };
    [
        node_position_m[0] + rotated_offset_m[0],
        node_position_m[1] + rotated_offset_m[1],
        node_position_m[2] + rotated_offset_m[2],
    ]
}

/// Rotation matrix of a (normalised) body -> world quaternion.
pub fn body_to_world_rotation_matrix(quaternion_wxyz: [f64; 4]) -> [[f64; 3]; 3] {
    let norm = quaternion_wxyz
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let [w, x, y, z] = quaternion_wxyz.map(|component| component / norm);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Quaternion of Python `orientation_rotation_matrix(yaw, pitch, roll)`:
/// q = q_z(yaw) * q_y(pitch) * q_x(roll), the same intrinsic ZYX order.
pub fn body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg(
    yaw_deg: f64,
    pitch_deg: f64,
    roll_deg: f64,
) -> [f64; 4] {
    let (sin_half_yaw, cos_half_yaw) = (yaw_deg.to_radians() / 2.0).sin_cos();
    let (sin_half_pitch, cos_half_pitch) = (pitch_deg.to_radians() / 2.0).sin_cos();
    let (sin_half_roll, cos_half_roll) = (roll_deg.to_radians() / 2.0).sin_cos();
    [
        cos_half_yaw * cos_half_pitch * cos_half_roll
            + sin_half_yaw * sin_half_pitch * sin_half_roll,
        cos_half_yaw * cos_half_pitch * sin_half_roll
            - sin_half_yaw * sin_half_pitch * cos_half_roll,
        cos_half_yaw * sin_half_pitch * cos_half_roll
            + sin_half_yaw * cos_half_pitch * sin_half_roll,
        sin_half_yaw * cos_half_pitch * cos_half_roll
            - cos_half_yaw * sin_half_pitch * sin_half_roll,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python `orientation_rotation_matrix`, transcribed.
    fn python_rotation_matrix(yaw_deg: f64, pitch_deg: f64, roll_deg: f64) -> [[f64; 3]; 3] {
        let (sy, cy) = yaw_deg.to_radians().sin_cos();
        let (sp, cp) = pitch_deg.to_radians().sin_cos();
        let (sr, cr) = roll_deg.to_radians().sin_cos();
        let rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]];
        let ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]];
        let rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]];
        let multiply = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| -> [[f64; 3]; 3] {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
            })
        };
        multiply(multiply(rz, ry), rx)
    }

    #[test]
    fn identity_when_no_orientation() {
        assert_eq!(
            sensor_world_position_m([1.0, 2.0, 3.0], None, [0.1, -0.2, 0.3]),
            [1.1, 1.8, 3.3]
        );
    }

    #[test]
    fn pure_yaw_matches_the_contract_quaternion() {
        let quaternion = body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg(90.0, 0.0, 0.0);
        let half = std::f64::consts::FRAC_PI_4;
        let expected = [half.cos(), 0.0, 0.0, half.sin()];
        for (got, want) in quaternion.iter().zip(expected) {
            assert!((got - want).abs() < 1e-15);
        }
        // +x body offset points north (+y world) after a +90 deg yaw.
        let world = sensor_world_position_m([10.0, 0.0, 0.0], Some(quaternion), [1.0, 0.0, 0.0]);
        assert!((world[0] - 10.0).abs() < 1e-12 && (world[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn quaternion_matrix_equals_python_zyx_matrix() {
        for (yaw, pitch, roll) in [(30.0, 0.0, 0.0), (-75.0, 12.0, -8.0), (170.0, -40.0, 65.0)] {
            let quaternion =
                body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg(yaw, pitch, roll);
            let from_quaternion = body_to_world_rotation_matrix(quaternion);
            let python = python_rotation_matrix(yaw, pitch, roll);
            for row in 0..3 {
                for column in 0..3 {
                    assert!(
                        (from_quaternion[row][column] - python[row][column]).abs() < 1e-12,
                        "({yaw},{pitch},{roll}) [{row}][{column}]"
                    );
                }
            }
        }
    }

    #[test]
    fn non_unit_quaternion_is_normalised() {
        let unit = body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg(40.0, 5.0, 0.0);
        let scaled = unit.map(|component| component * 3.0);
        let offset = [0.2, 0.1, -0.05];
        let from_unit = sensor_world_position_m([0.0; 3], Some(unit), offset);
        let from_scaled = sensor_world_position_m([0.0; 3], Some(scaled), offset);
        for axis in 0..3 {
            assert!((from_unit[axis] - from_scaled[axis]).abs() < 1e-15);
        }
    }
}
