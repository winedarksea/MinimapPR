//! Hidden `site-geometry-oracle` subcommand: runs the R2 geometry code
//! (`minimappr_core::site_frame_projection`, `sensor_world_position`, and the
//! sidecar's `SiteGeometryState`) on JSON cases from stdin and prints one
//! result per case. Driven by
//! `tests/test_site_geometry_cross_language_parity.py`, which computes the
//! expected values live from Python `LocalCoordinateFrame` and
//! `orientation_rotation_matrix`.
//!
//! Case kinds:
//! - `projection`: `{origin, projection, points: [[lat, lon, alt_m]]}` -> `{enu_m}`
//! - `sensor_world_position`: `{node_position_m, body_offsets_m, orientation?:
//!   {yaw_deg, pitch_deg, roll_deg}}` -> `{body_to_world_quaternion_wxyz, world_positions_m}`
//! - `resolve_placements`: `{push?: SiteGeometryPush, nodes: [{node_id, fix?:
//!   {lat, lon, alt_m, position_source}}]}` -> `{coordinate_frame, resolutions}`

use std::io::Read;

use minimappr_core::localization_sensor_entry::{
    CoordinateProjection, GeodeticOrigin, SensorPositionSource, SiteGeometryPush,
};
use minimappr_core::sensor_world_position::{
    body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg, sensor_world_position_m,
};
use minimappr_core::site_frame_projection::LocalEnuProjector;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::dsp_geometry_config::{NodePlacementResolution, ReportedNodeFix, SiteGeometryState};

type OracleResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Deserialize)]
struct ProjectionCase {
    origin: GeodeticOrigin,
    projection: CoordinateProjection,
    points: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct OrientationDeg {
    yaw_deg: f64,
    pitch_deg: f64,
    roll_deg: f64,
}

#[derive(Deserialize)]
struct SensorWorldPositionCase {
    node_position_m: [f64; 3],
    body_offsets_m: Vec<[f64; 3]>,
    orientation: Option<OrientationDeg>,
}

#[derive(Deserialize)]
struct OracleFix {
    lat: f64,
    lon: f64,
    alt_m: f64,
    position_source: SensorPositionSource,
}

#[derive(Deserialize)]
struct OracleNode {
    node_id: String,
    fix: Option<OracleFix>,
}

#[derive(Deserialize)]
struct ResolvePlacementsCase {
    push: Option<SiteGeometryPush>,
    nodes: Vec<OracleNode>,
}

pub(crate) fn run_site_geometry_oracle() -> OracleResult<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let request: Value = serde_json::from_str(&input)?;
    let cases = request["cases"]
        .as_array()
        .ok_or("expected {\"cases\": [...]}")?;
    let results = cases
        .iter()
        .map(run_case)
        .collect::<OracleResult<Vec<_>>>()?;
    println!("{}", json!({ "results": results }));
    Ok(())
}

fn run_case(case: &Value) -> OracleResult<Value> {
    match case["kind"].as_str() {
        Some("projection") => run_projection_case(serde_json::from_value(case.clone())?),
        Some("sensor_world_position") => {
            run_sensor_world_position_case(serde_json::from_value(case.clone())?)
        }
        Some("resolve_placements") => {
            run_resolve_placements_case(serde_json::from_value(case.clone())?)
        }
        other => Err(format!("unknown site-geometry oracle case kind {other:?}").into()),
    }
}

fn run_projection_case(case: ProjectionCase) -> OracleResult<Value> {
    let projector = LocalEnuProjector::new(case.origin, case.projection);
    let enu_m: Vec<[f64; 3]> = case
        .points
        .iter()
        .map(|[lat, lon, alt_m]| projector.geodetic_to_local_enu_m(*lat, *lon, *alt_m))
        .collect();
    Ok(json!({ "enu_m": enu_m }))
}

fn run_sensor_world_position_case(case: SensorWorldPositionCase) -> OracleResult<Value> {
    let quaternion = case.orientation.map(|orientation| {
        body_to_world_quaternion_wxyz_from_yaw_pitch_roll_deg(
            orientation.yaw_deg,
            orientation.pitch_deg,
            orientation.roll_deg,
        )
    });
    let world_positions_m: Vec<[f64; 3]> = case
        .body_offsets_m
        .iter()
        .map(|offset| sensor_world_position_m(case.node_position_m, quaternion, *offset))
        .collect();
    Ok(json!({
        "body_to_world_quaternion_wxyz": quaternion,
        "world_positions_m": world_positions_m,
    }))
}

fn run_resolve_placements_case(case: ResolvePlacementsCase) -> OracleResult<Value> {
    let mut state = match case.push {
        Some(push) => SiteGeometryState::from_push(push)?,
        None => SiteGeometryState::default(),
    };
    let resolutions: Vec<Value> = case
        .nodes
        .iter()
        .map(|node| {
            let fix = node.fix.as_ref().map(|fix| ReportedNodeFix {
                lat_deg: fix.lat,
                lon_deg: fix.lon,
                alt_m: fix.alt_m,
                position_source: fix.position_source,
                reported_accuracy_m: None,
                reported_body_to_world_quaternion_wxyz: None,
            });
            match state.resolve_node_placement(&node.node_id, fix.as_ref()) {
                NodePlacementResolution::Placed {
                    position_m,
                    position_source,
                    ..
                } => json!({"position_m": position_m, "position_source": position_source}),
                NodePlacementResolution::Untrusted { reason } => {
                    json!({"untrusted_reason": reason})
                }
            }
        })
        .collect();
    Ok(json!({
        "coordinate_frame": state.coordinate_frame(),
        "resolutions": resolutions,
    }))
}
