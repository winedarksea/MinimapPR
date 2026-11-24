//! Site geometry the sidecar solves in: Python's pushed snapshot
//! (`POST /api/v1/dsp/geometry`, `MINIMAPPR_SITE_GEOMETRY_JSON`) or, before
//! any push, an automatic ENU frame anchored at the first trusted GPS fix.
//!
//! Placement precedence per node: pushed placement > trusted GPS > untrusted.
//! Pushed placements carry Python's filtered stationary estimate or an
//! operator entry, so raw per-packet GPS jitter only reaches a solve for nodes
//! Python has not placed. Both frames use the projection Python's
//! `LocalCoordinateFrame` uses (`minimappr_core::site_frame_projection`).

use std::sync::Arc;

use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, CoordinateProjection, GeodeticOrigin, SensorPositionSource,
    SiteGeometryPush, SiteGeometryPushAcknowledgement, SolveCoordinateFrame, SolveExclusionReason,
};
use minimappr_core::sensor_world_position::body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg;
use minimappr_core::site_frame_projection::LocalEnuProjector;

/// Python `cross_node_beamform._UNTRUSTED_POSITION_SOURCES`: firmware or
/// config pseudo-positions that cannot form a cross-node baseline.
const UNTRUSTED_REPORTED_POSITION_SOURCES: [&str; 3] =
    ["gps_fallback", "static", "fallback_static"];
/// A quaternion this short has no defined rotation axis.
const MINIMUM_QUATERNION_NORM: f64 = 1e-9;

/// A node's own position report from its packet's `node_context.node`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportedNodeFix {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    /// `ReportedGps` or `FallbackStatic`; any other value is treated as untrusted.
    pub position_source: SensorPositionSource,
    pub reported_accuracy_m: Option<f64>,
    /// The node's reported orientation, used only when no placement is pushed
    /// (a pushed placement's own rotation always wins, absent = identity).
    pub reported_body_to_world_quaternion_wxyz: Option<[f64; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodePlacementResolution {
    Placed {
        /// Node origin in the frame of [`SiteGeometryState::coordinate_frame`].
        position_m: [f64; 3],
        /// 1-sigma; `None` when nothing states one (GPS without accuracy).
        position_std_m: Option<f64>,
        position_source: SensorPositionSource,
        body_to_world_quaternion_wxyz: Option<[f64; 4]>,
    },
    Untrusted {
        reason: SolveExclusionReason,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SiteGeometryError {
    EmptySiteFrameId,
    InvalidSiteOrigin {
        detail: String,
    },
    InvalidNodePlacement {
        node_id: String,
        detail: String,
    },
    StaleGeometryPushRevision {
        pushed_revision: u64,
        applied_revision: u64,
    },
}

impl std::fmt::Display for SiteGeometryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySiteFrameId => write!(formatter, "site_frame_id must not be empty"),
            Self::InvalidSiteOrigin { detail } => write!(formatter, "site_origin: {detail}"),
            Self::InvalidNodePlacement { node_id, detail } => {
                write!(formatter, "node_placements[{node_id:?}]: {detail}")
            }
            Self::StaleGeometryPushRevision {
                pushed_revision,
                applied_revision,
            } => write!(
                formatter,
                "geometry_push_revision {pushed_revision} is older than the applied revision {applied_revision}"
            ),
        }
    }
}

impl std::error::Error for SiteGeometryError {}

#[derive(Clone, Debug)]
pub struct SiteGeometryState {
    frame: SiteGeometryFrame,
}

#[derive(Clone, Debug)]
enum SiteGeometryFrame {
    Automatic {
        projection: CoordinateProjection,
        /// Set by the first trusted GPS fix and kept for the process lifetime,
        /// so positions already handed out never shift underneath a buffer.
        adopted_origin: Option<LocalEnuProjector>,
    },
    Pushed {
        push: Arc<SiteGeometryPush>,
        projector: LocalEnuProjector,
    },
}

impl Default for SiteGeometryState {
    /// Python's `LocalCoordinateFrame` defaults to `mode="flat"`.
    fn default() -> Self {
        Self::automatic(CoordinateProjection::FlatWgs84Series)
    }
}

impl SiteGeometryState {
    pub fn automatic(projection: CoordinateProjection) -> Self {
        Self {
            frame: SiteGeometryFrame::Automatic {
                projection,
                adopted_origin: None,
            },
        }
    }

    pub fn from_push(push: SiteGeometryPush) -> Result<Self, SiteGeometryError> {
        validate_site_geometry_push(&push)?;
        let projector = LocalEnuProjector::new(push.site_origin, push.projection);
        Ok(Self {
            frame: SiteGeometryFrame::Pushed {
                push: Arc::new(push),
                projector,
            },
        })
    }

    /// Full-snapshot replace. An equal revision is accepted (idempotent
    /// re-push); a lower one is refused so a delayed request cannot roll the
    /// geometry back.
    pub fn apply_push(
        &mut self,
        push: SiteGeometryPush,
    ) -> Result<SiteGeometryPushAcknowledgement, SiteGeometryError> {
        if let Some(applied_revision) = self.geometry_push_revision() {
            if push.geometry_push_revision < applied_revision {
                return Err(SiteGeometryError::StaleGeometryPushRevision {
                    pushed_revision: push.geometry_push_revision,
                    applied_revision,
                });
            }
        }
        *self = Self::from_push(push)?;
        Ok(self
            .acknowledgement()
            .expect("a pushed frame always has an acknowledgement"))
    }

    /// The revision now in effect, as a push reply; `None` before any push.
    pub fn acknowledgement(&self) -> Option<SiteGeometryPushAcknowledgement> {
        match &self.frame {
            SiteGeometryFrame::Automatic { .. } => None,
            SiteGeometryFrame::Pushed { push, .. } => Some(SiteGeometryPushAcknowledgement {
                site_frame_id: push.site_frame_id.clone(),
                applied_geometry_push_revision: push.geometry_push_revision,
                node_placement_count: push.node_placements.len(),
            }),
        }
    }

    pub fn coordinate_frame(&self) -> SolveCoordinateFrame {
        match &self.frame {
            SiteGeometryFrame::Automatic { adopted_origin, .. } => SolveCoordinateFrame {
                frame_id: None,
                kind: CoordinateFrameKind::SidecarAutomaticEnu,
                origin: adopted_origin.map(|projector| projector.origin()),
                projection: adopted_origin.map(|projector| projector.projection()),
            },
            SiteGeometryFrame::Pushed { push, projector } => SolveCoordinateFrame {
                frame_id: Some(push.site_frame_id.clone()),
                kind: CoordinateFrameKind::SiteEnu,
                origin: Some(projector.origin()),
                projection: Some(projector.projection()),
            },
        }
    }

    pub fn geometry_push_revision(&self) -> Option<u64> {
        match &self.frame {
            SiteGeometryFrame::Automatic { .. } => None,
            SiteGeometryFrame::Pushed { push, .. } => Some(push.geometry_push_revision),
        }
    }

    /// `&mut` because automatic mode adopts the first trusted GPS fix as origin.
    pub fn resolve_node_placement(
        &mut self,
        node_id: &str,
        reported_fix: Option<&ReportedNodeFix>,
    ) -> NodePlacementResolution {
        if let SiteGeometryFrame::Pushed { push, .. } = &self.frame {
            if let Some(placement) = push.node_placements.get(node_id) {
                return NodePlacementResolution::Placed {
                    position_m: placement.position_m,
                    position_std_m: Some(placement.position_std_m),
                    position_source: placement.placement_source.sensor_position_source(),
                    body_to_world_quaternion_wxyz: placement.body_to_world_quaternion_wxyz,
                };
            }
        }
        // Two nodes at one built-in point cannot form a baseline, so a node
        // without a trusted fix stays node-local until Python places it.
        let Some(fix) = reported_fix.filter(|fix| is_trusted_gps_fix(fix)) else {
            return NodePlacementResolution::Untrusted {
                reason: SolveExclusionReason::NodePositionUntrustedNoOverride,
            };
        };
        let projector = match &mut self.frame {
            SiteGeometryFrame::Pushed { projector, .. } => *projector,
            SiteGeometryFrame::Automatic {
                projection,
                adopted_origin,
            } => *adopted_origin.get_or_insert_with(|| {
                LocalEnuProjector::new(
                    GeodeticOrigin {
                        lat: fix.lat_deg,
                        lon: fix.lon_deg,
                        alt_m: fix.alt_m,
                    },
                    *projection,
                )
            }),
        };
        NodePlacementResolution::Placed {
            position_m: projector.geodetic_to_local_enu_m(fix.lat_deg, fix.lon_deg, fix.alt_m),
            position_std_m: fix.reported_accuracy_m,
            position_source: SensorPositionSource::ReportedGps,
            body_to_world_quaternion_wxyz: fix.reported_body_to_world_quaternion_wxyz,
        }
    }
}

fn is_trusted_gps_fix(fix: &ReportedNodeFix) -> bool {
    fix.position_source == SensorPositionSource::ReportedGps
        && is_valid_geodetic_point(fix.lat_deg, fix.lon_deg, fix.alt_m)
}

fn is_valid_geodetic_point(lat_deg: f64, lon_deg: f64, alt_m: f64) -> bool {
    lat_deg.is_finite()
        && lon_deg.is_finite()
        && alt_m.is_finite()
        && (-90.0..=90.0).contains(&lat_deg)
        && (-180.0..=180.0).contains(&lon_deg)
}

/// Shape checks a push must pass before it can replace the solve geometry.
pub fn validate_site_geometry_push(push: &SiteGeometryPush) -> Result<(), SiteGeometryError> {
    if push.site_frame_id.trim().is_empty() {
        return Err(SiteGeometryError::EmptySiteFrameId);
    }
    let origin = push.site_origin;
    if !is_valid_geodetic_point(origin.lat, origin.lon, origin.alt_m) {
        return Err(SiteGeometryError::InvalidSiteOrigin {
            detail: format!(
                "lat/lon must be finite within [-90, 90]/[-180, 180] and alt_m finite (got {}, {}, {})",
                origin.lat, origin.lon, origin.alt_m
            ),
        });
    }
    for (node_id, placement) in &push.node_placements {
        let invalid = |detail: String| SiteGeometryError::InvalidNodePlacement {
            node_id: node_id.clone(),
            detail,
        };
        if node_id.trim().is_empty() {
            return Err(invalid("node_id must not be empty".to_string()));
        }
        if !placement.position_m.iter().all(|axis| axis.is_finite()) {
            return Err(invalid(format!(
                "position_m must be finite (got {:?})",
                placement.position_m
            )));
        }
        if !placement.position_std_m.is_finite() || placement.position_std_m < 0.0 {
            return Err(invalid(format!(
                "position_std_m must be finite and >= 0 (got {})",
                placement.position_std_m
            )));
        }
        if let Some(quaternion) = placement.body_to_world_quaternion_wxyz {
            let norm = quaternion.iter().map(|c| c * c).sum::<f64>().sqrt();
            if !norm.is_finite() || norm < MINIMUM_QUATERNION_NORM {
                return Err(invalid(format!(
                    "body_to_world_quaternion_wxyz must be finite and non-zero (got {quaternion:?})"
                )));
            }
        }
    }
    Ok(())
}

/// `node_context.node` as the ingest backend embeds it (`position_geo`,
/// `metadata.gps.position_source`, `orientation`). `None` without a usable
/// `position_geo`. A missing `position_source` counts as GPS: legacy payloads
/// omit it and the sidecar has always solved them on their GPS.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "called by R1's sensor-registry adapter")
)]
pub fn reported_node_fix_from_node_context(node: &serde_json::Value) -> Option<ReportedNodeFix> {
    let position = node.get("position_geo")?;
    let lat_deg = position.get("lat")?.as_f64()?;
    let lon_deg = position.get("lon")?.as_f64()?;
    let alt_m = position
        .get("alt_m")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let reported_source = node
        .pointer("/metadata/gps/position_source")
        .and_then(serde_json::Value::as_str)
        .map(|source| source.trim().to_ascii_lowercase());
    let position_source = match reported_source.as_deref() {
        Some(source) if UNTRUSTED_REPORTED_POSITION_SOURCES.contains(&source) => {
            SensorPositionSource::FallbackStatic
        }
        _ => SensorPositionSource::ReportedGps,
    };
    let reported_body_to_world_quaternion_wxyz = node
        .get("orientation")
        .filter(|orientation| orientation.is_object())
        .map(|orientation| {
            let angle_deg = |key: &str| {
                orientation
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0)
            };
            body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg(
                angle_deg("yaw_deg"),
                angle_deg("pitch_deg"),
                angle_deg("roll_deg"),
            )
        });
    Some(ReportedNodeFix {
        lat_deg,
        lon_deg,
        alt_m,
        position_source,
        reported_accuracy_m: None,
        reported_body_to_world_quaternion_wxyz,
    })
}

#[cfg(test)]
#[path = "dsp_geometry_config_tests.rs"]
mod dsp_geometry_config_tests;
