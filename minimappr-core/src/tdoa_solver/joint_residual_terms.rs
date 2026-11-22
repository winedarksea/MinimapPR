//! Observation types and the whitened residual model of the joint
//! TDOA + DOA objective (a small factor graph over one source position).
//!
//! Every residual is divided by its own one-sigma, so the cost is a χ² and
//! `(JᵀJ)⁻¹` is the position covariance directly.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use serde::{Deserialize, Serialize};

use super::bearing_fusion::BearingObservation;
use super::canopy_height_prior::CanopyHeightPrior;
use super::levenberg_marquardt_driver::PositionResidualModel;
use super::node_descriptor::NodeDescriptor;
use super::spatial_constraint_terms::{stack_jacobian_rows, EPSILON};
use super::TdoaSolverError;

/// Arrival(mic A) − arrival(mic B). Same node → a stereo/sub-array cone
/// (exact hyperboloid, which is the cone asymptotically); different nodes →
/// an inter-node hyperboloid, allowed only within one clock domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicPairTdoaObservation {
    pub node_a: String,
    pub mic_a: usize,
    pub node_b: String,
    pub mic_b: usize,
    pub tdoa_seconds: f64,
    /// Measurement-only one-sigma (clock / geometry terms are added from the descriptors).
    pub timing_std_s: f64,
}

/// Direction of arrival. Body-frame angles: azimuth from body +x toward +y,
/// elevation above the body x–y plane (radians).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DoaDirection {
    BodyFrameAzimuthElevation {
        azimuth_rad: f64,
        elevation_rad: f64,
    },
    WorldUnitVector([f64; 3]),
}

/// A DOA measured by an array node, about that node's mic centroid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoaObservation {
    pub node_id: String,
    pub direction: DoaDirection,
    pub angular_std_rad: f64,
}

/// A world-frame bearing ray not tied to a descriptor (e.g. a server
/// single-node cone from `multi_node_bearing_fusion`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldBearingRay {
    pub origin_m: [f64; 3],
    pub direction: [f64; 3],
    pub angular_std_rad: f64,
}

impl WorldBearingRay {
    /// Convert a fusion cone: angular σ ≈ lateral σ at the cone's range prior.
    pub fn from_bearing_observation(observation: &BearingObservation, range_prior_m: f64) -> Self {
        Self {
            origin_m: observation.origin_m,
            direction: observation.direction,
            angular_std_rad: (observation.lateral_std_m / range_prior_m.max(EPSILON)).atan(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JointObservations {
    pub mic_pair_tdoas: Vec<MicPairTdoaObservation>,
    pub doas: Vec<DoaObservation>,
    pub world_bearing_rays: Vec<WorldBearingRay>,
}

/// A range-difference row with mic positions resolved to world and σ composed.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedRangeDifferenceTerm {
    pub mic_a_m: Vector3<f64>,
    pub mic_b_m: Vector3<f64>,
    pub tdoa_seconds: f64,
    pub total_std_s: f64,
    pub is_intra_node_cone: bool,
}

/// A bearing term with world-frame unit direction and composed σ.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedBearingTerm {
    pub origin_m: Vector3<f64>,
    pub direction: Vector3<f64>,
    pub total_std_rad: f64,
}

/// A direction on the sphere has two angular degrees of freedom.
pub const ANGULAR_DEGREES_OF_FREEDOM_PER_BEARING: usize = 2;

pub struct JointResidualModel {
    pub range_difference_terms: Vec<ResolvedRangeDifferenceTerm>,
    pub bearing_terms: Vec<ResolvedBearingTerm>,
    pub canopy_height_prior: Option<CanopyHeightPrior>,
    pub sound_speed_mps: f64,
}

fn find_node<'nodes>(
    nodes: &'nodes [NodeDescriptor],
    node_id: &str,
) -> Result<&'nodes NodeDescriptor, TdoaSolverError> {
    nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .ok_or_else(|| TdoaSolverError::UnknownNode(node_id.to_string()))
}

fn mic_centroid(node: &NodeDescriptor) -> Result<Vector3<f64>, TdoaSolverError> {
    let mics = node.mic_world_positions()?;
    if mics.is_empty() {
        return Ok(node.position_vector());
    }
    Ok(mics.iter().fold(Vector3::zeros(), |sum, mic| sum + mic) / mics.len() as f64)
}

fn validated_std(value: f64, context: &'static str) -> Result<f64, TdoaSolverError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(TdoaSolverError::NonFiniteInput(context))
    }
}

pub fn resolve_range_difference_term(
    nodes: &[NodeDescriptor],
    observation: &MicPairTdoaObservation,
    sound_speed_mps: f64,
) -> Result<ResolvedRangeDifferenceTerm, TdoaSolverError> {
    let node_a = find_node(nodes, &observation.node_a)?;
    let node_b = find_node(nodes, &observation.node_b)?;
    let mic_a_m = node_a.mic_world_position(observation.mic_a)?;
    let mic_b_m = node_b.mic_world_position(observation.mic_b)?;
    let is_intra_node_cone = node_a.node_id == node_b.node_id;
    let timing_variance = observation.timing_std_s.powi(2);
    let total_variance = if is_intra_node_cone {
        // Rotating a baseline L by δ moves the path difference by up to L·δ.
        let baseline_m = (mic_a_m - mic_b_m).norm();
        timing_variance
            + (node_a.orientation.orientation_std_rad * baseline_m / sound_speed_mps).powi(2)
    } else {
        if node_a.clock.clock_domain_id != node_b.clock.clock_domain_id {
            return Err(TdoaSolverError::ClockDomainMismatch {
                node_a: node_a.node_id.clone(),
                node_b: node_b.node_id.clone(),
            });
        }
        // Each node's position error moves its range by up to σ_pos.
        timing_variance
            + node_a.clock.clock_offset_std_s.powi(2)
            + node_b.clock.clock_offset_std_s.powi(2)
            + (node_a.position_std_m.powi(2) + node_b.position_std_m.powi(2))
                / sound_speed_mps.powi(2)
    };
    if !observation.tdoa_seconds.is_finite() {
        return Err(TdoaSolverError::NonFiniteInput("mic pair tdoa"));
    }
    Ok(ResolvedRangeDifferenceTerm {
        mic_a_m,
        mic_b_m,
        tdoa_seconds: observation.tdoa_seconds,
        total_std_s: validated_std(total_variance.sqrt(), "mic pair tdoa std")?,
        is_intra_node_cone,
    })
}

pub fn resolve_doa_term(
    nodes: &[NodeDescriptor],
    observation: &DoaObservation,
) -> Result<ResolvedBearingTerm, TdoaSolverError> {
    let node = find_node(nodes, &observation.node_id)?;
    let (direction, total_variance) = match &observation.direction {
        DoaDirection::BodyFrameAzimuthElevation {
            azimuth_rad,
            elevation_rad,
        } => {
            let body = Vector3::new(
                elevation_rad.cos() * azimuth_rad.cos(),
                elevation_rad.cos() * azimuth_rad.sin(),
                elevation_rad.sin(),
            );
            let world = node.body_to_world_rotation()? * body;
            (
                world,
                observation.angular_std_rad.powi(2) + node.orientation.orientation_std_rad.powi(2),
            )
        }
        DoaDirection::WorldUnitVector(vector) => (
            Vector3::new(vector[0], vector[1], vector[2]),
            observation.angular_std_rad.powi(2),
        ),
    };
    let norm = direction.norm();
    if !norm.is_finite() || norm < EPSILON {
        return Err(TdoaSolverError::DegenerateDirection);
    }
    Ok(ResolvedBearingTerm {
        origin_m: mic_centroid(node)?,
        direction: direction / norm,
        total_std_rad: validated_std(total_variance.sqrt(), "doa std")?,
    })
}

pub fn resolve_world_ray(ray: &WorldBearingRay) -> Result<ResolvedBearingTerm, TdoaSolverError> {
    let direction = Vector3::new(ray.direction[0], ray.direction[1], ray.direction[2]);
    let norm = direction.norm();
    if !norm.is_finite() || norm < EPSILON {
        return Err(TdoaSolverError::DegenerateDirection);
    }
    Ok(ResolvedBearingTerm {
        origin_m: Vector3::new(ray.origin_m[0], ray.origin_m[1], ray.origin_m[2]),
        direction: direction / norm,
        total_std_rad: validated_std(ray.angular_std_rad, "world ray std")?,
    })
}

impl JointResidualModel {
    pub fn residual_count(&self) -> usize {
        self.range_difference_terms.len()
            + 3 * self.bearing_terms.len()
            + usize::from(self.canopy_height_prior.is_some())
    }

    /// Independent measurement count for χ² degrees of freedom: a bearing's
    /// three unit-vector rows carry only two angular dof (the radial row is
    /// second order), and the canopy hinge is a prior, not a measurement.
    pub fn independent_measurement_count(&self) -> usize {
        self.range_difference_terms.len()
            + ANGULAR_DEGREES_OF_FREEDOM_PER_BEARING * self.bearing_terms.len()
    }
}

impl PositionResidualModel for JointResidualModel {
    fn residuals(&self, position_m: &Vector3<f64>) -> DVector<f64> {
        let mut rows = Vec::with_capacity(self.residual_count());
        for term in &self.range_difference_terms {
            let predicted_s = ((position_m - term.mic_a_m).norm()
                - (position_m - term.mic_b_m).norm())
                / self.sound_speed_mps;
            rows.push((predicted_s - term.tdoa_seconds) / term.total_std_s);
        }
        for term in &self.bearing_terms {
            let offset = position_m - term.origin_m;
            let distance = offset.norm();
            // At the apex the bearing is undefined; contribute the worst case
            // (orthogonal) rather than zero so the apex is never a free minimum.
            let unit = if distance < EPSILON {
                Vector3::zeros()
            } else {
                offset / distance
            };
            rows.extend(
                ((unit - term.direction) / term.total_std_rad)
                    .iter()
                    .copied(),
            );
        }
        if let Some(canopy) = &self.canopy_height_prior {
            rows.push(canopy.residual(position_m, 1.0));
        }
        DVector::from_vec(rows)
    }

    fn jacobian(&self, position_m: &Vector3<f64>) -> DMatrix<f64> {
        let mut rows = Vec::with_capacity(self.residual_count());
        for term in &self.range_difference_terms {
            let offset_a = position_m - term.mic_a_m;
            let offset_b = position_m - term.mic_b_m;
            let gradient =
                offset_a / offset_a.norm().max(EPSILON) - offset_b / offset_b.norm().max(EPSILON);
            rows.push(gradient / (self.sound_speed_mps * term.total_std_s));
        }
        for term in &self.bearing_terms {
            let offset = position_m - term.origin_m;
            let distance = offset.norm().max(EPSILON);
            let unit = offset / distance;
            let block =
                (Matrix3::identity() - unit * unit.transpose()) / (distance * term.total_std_rad);
            for row_index in 0..3 {
                rows.push(block.row(row_index).transpose());
            }
        }
        if let Some(canopy) = &self.canopy_height_prior {
            rows.push(canopy.jacobian(position_m, 1.0));
        }
        stack_jacobian_rows(&rows)
    }
}
