//! `NodeDescriptor`: how one session node (tetra array, mono phone, stereo
//! phone, generic sub-array) describes its geometry, orientation and clock to
//! the joint solver (plan §Joint TDOA + DOA over mixed nodes).
//!
//! Plain `[f64; N]` arrays (not nalgebra types) so the descriptor serialises
//! with serde unchanged on the wire and in the PWA.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use super::TdoaSolverError;

/// Unit-quaternion norm tolerance before we refuse an orientation.
const QUATERNION_NORM_TOLERANCE: f64 = 1.0e-6;
const TETRAHEDRAL_MIC_COUNT: usize = 4;
const STEREO_PHONE_MIC_COUNT: usize = 2;
const MONO_PHONE_MIC_COUNT: usize = 1;
/// Smallest sub-array that yields a full 3D bearing on its own.
const MINIMUM_SUB_ARRAY_MIC_COUNT: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Sirith 4-mic tetrahedral node: full 3D DOA without inter-node sync.
    TetrahedralArray,
    /// Single raw mic: contributes only inter-node TDOA.
    PhoneMono,
    /// Two raw mics (~15 cm): an intra-node TDOA cone about the mic axis.
    PhoneStereo,
    /// Any other rigid sub-array with known mic offsets (e.g. XVF3800).
    GenericSubArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrientationSource {
    Fixed,
    Surveyed,
    Imu,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockSource {
    Pps,
    Ntp,
    AcousticChirp,
    Unsynced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSource {
    Fixed,
    Surveyed,
    AcousticMesh,
    Gps,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeOrientation {
    /// Body → world rotation as a unit quaternion `[w, x, y, z]`.
    pub body_to_world_quaternion_wxyz: [f64; 4],
    pub source: OrientationSource,
    /// One-sigma attitude error (radians), inflates DOA and cone residuals.
    pub orientation_std_rad: f64,
}

impl NodeOrientation {
    pub fn identity(source: OrientationSource) -> Self {
        Self {
            body_to_world_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
            source,
            orientation_std_rad: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeClock {
    pub source: ClockSource,
    /// Nodes may form TDOA residuals only within one domain (an acoustic
    /// island, or the PPS/UTC domain that bridges islands).
    pub clock_domain_id: u32,
    /// One-sigma residual clock offset after sync (seconds).
    pub clock_offset_std_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeDescriptor {
    pub node_id: String,
    pub kind: NodeKind,
    /// World position of the body-frame origin (metres, solver ENU frame).
    pub position_m: [f64; 3],
    pub position_source: PositionSource,
    /// One-sigma isotropic position error (metres).
    pub position_std_m: f64,
    /// Mic offsets in the body frame (metres).
    pub mic_offsets_m: Vec<[f64; 3]>,
    pub orientation: NodeOrientation,
    pub clock: NodeClock,
    pub has_speaker: bool,
}

impl NodeDescriptor {
    pub fn body_to_world_rotation(&self) -> Result<UnitQuaternion<f64>, TdoaSolverError> {
        let [w, x, y, z] = self.orientation.body_to_world_quaternion_wxyz;
        let quaternion = Quaternion::new(w, x, y, z);
        let norm = quaternion.norm();
        if !norm.is_finite() || (norm - 1.0).abs() > QUATERNION_NORM_TOLERANCE {
            return Err(TdoaSolverError::InvalidNodeDescriptor(format!(
                "{}: orientation quaternion norm {norm} is not 1",
                self.node_id
            )));
        }
        Ok(UnitQuaternion::new_normalize(quaternion))
    }

    pub fn position_vector(&self) -> Vector3<f64> {
        Vector3::new(self.position_m[0], self.position_m[1], self.position_m[2])
    }

    /// World positions of every mic, in `mic_offsets_m` order.
    pub fn mic_world_positions(&self) -> Result<Vec<Vector3<f64>>, TdoaSolverError> {
        let rotation = self.body_to_world_rotation()?;
        let origin = self.position_vector();
        Ok(self
            .mic_offsets_m
            .iter()
            .map(|offset| origin + rotation * Vector3::new(offset[0], offset[1], offset[2]))
            .collect())
    }

    pub fn mic_world_position(&self, mic_index: usize) -> Result<Vector3<f64>, TdoaSolverError> {
        let offset = self.mic_offsets_m.get(mic_index).ok_or_else(|| {
            TdoaSolverError::MicIndexOutOfRange {
                node_id: self.node_id.clone(),
                mic_index,
            }
        })?;
        Ok(self.position_vector()
            + self.body_to_world_rotation()? * Vector3::new(offset[0], offset[1], offset[2]))
    }

    /// Reject descriptors whose mic count contradicts their kind or whose
    /// numbers are non-finite / negative.
    pub fn validate(&self) -> Result<(), TdoaSolverError> {
        let invalid = |reason: String| {
            Err(TdoaSolverError::InvalidNodeDescriptor(format!(
                "{}: {reason}",
                self.node_id
            )))
        };
        let mic_count = self.mic_offsets_m.len();
        let count_ok = match self.kind {
            NodeKind::TetrahedralArray => mic_count == TETRAHEDRAL_MIC_COUNT,
            NodeKind::PhoneStereo => mic_count == STEREO_PHONE_MIC_COUNT,
            NodeKind::PhoneMono => mic_count == MONO_PHONE_MIC_COUNT,
            NodeKind::GenericSubArray => mic_count >= MINIMUM_SUB_ARRAY_MIC_COUNT,
        };
        if !count_ok {
            return invalid(format!("{:?} with {mic_count} mics", self.kind));
        }
        let scalars = [
            self.position_std_m,
            self.orientation.orientation_std_rad,
            self.clock.clock_offset_std_s,
        ];
        if scalars
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return invalid("negative or non-finite uncertainty".into());
        }
        let coordinates = self
            .position_m
            .iter()
            .chain(self.mic_offsets_m.iter().flatten());
        if coordinates.into_iter().any(|value| !value.is_finite()) {
            return invalid("non-finite position or mic offset".into());
        }
        self.body_to_world_rotation().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn stereo_phone(
        node_id: &str,
        position_m: [f64; 3],
        quaternion_wxyz: [f64; 4],
    ) -> NodeDescriptor {
        NodeDescriptor {
            node_id: node_id.into(),
            kind: NodeKind::PhoneStereo,
            position_m,
            position_source: PositionSource::AcousticMesh,
            position_std_m: 0.0,
            mic_offsets_m: vec![[0.0, 0.0, 0.075], [0.0, 0.0, -0.075]],
            orientation: NodeOrientation {
                body_to_world_quaternion_wxyz: quaternion_wxyz,
                source: OrientationSource::Imu,
                orientation_std_rad: 0.0,
            },
            clock: NodeClock {
                source: ClockSource::AcousticChirp,
                clock_domain_id: 1,
                clock_offset_std_s: 0.0,
            },
            has_speaker: true,
        }
    }

    #[test]
    fn rotation_moves_mic_axis_into_world_frame() {
        // 90° about +x maps body +z onto world −y.
        let half = std::f64::consts::FRAC_PI_4;
        let phone = stereo_phone("p", [1.0, 2.0, 1.5], [half.cos(), half.sin(), 0.0, 0.0]);
        phone.validate().unwrap();
        let mics = phone.mic_world_positions().unwrap();
        assert!((mics[0] - Vector3::new(1.0, 2.0 - 0.075, 1.5)).norm() < 1e-12);
    }

    #[test]
    fn mic_count_must_match_kind() {
        let mut phone = stereo_phone("p", [0.0; 3], [1.0, 0.0, 0.0, 0.0]);
        phone.kind = NodeKind::TetrahedralArray;
        assert!(phone.validate().is_err());
        let skewed = stereo_phone("q", [0.0; 3], [2.0, 0.0, 0.0, 0.0]);
        assert!(skewed.validate().is_err());
    }
}
