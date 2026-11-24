//! `POST /api/v1/dsp/geometry` (Python's full-snapshot site geometry push)
//! and `GET` on the same path, which reports the frame and revision in effect
//! so Python can reconcile after either process restarts.
//!
//! Status codes: 200 applied (body `SiteGeometryPushAcknowledgement`); 409
//! stale revision (body carries the applied revision); 422 malformed or
//! invalid body. Error bodies are `{"error": <code>, "detail": <text>, ...}`.

use axum::{
    extract::{rejection::JsonRejection, FromRef, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use minimappr_core::localization_sensor_entry::{SiteGeometryPush, SolveCoordinateFrame};
use serde::Serialize;
use tracing::{info, warn};

use crate::dsp_geometry_config::{SiteGeometryError, SiteGeometryState};
use crate::dsp_worker::SharedDspState;

pub(crate) const SITE_GEOMETRY_ENDPOINT: &str = "/api/v1/dsp/geometry";

impl FromRef<crate::AppState> for SharedDspState {
    fn from_ref(app_state: &crate::AppState) -> Self {
        app_state.dsp_state.clone()
    }
}

/// `GET /api/v1/dsp/geometry` body.
#[derive(Debug, Serialize)]
pub(crate) struct SiteGeometryStatus {
    /// What a solve started now would publish as `coordinate_frame`.
    pub coordinate_frame: SolveCoordinateFrame,
    /// Null until a push (launch env or POST) has been applied.
    pub applied_geometry_push_revision: Option<u64>,
    pub node_placement_count: usize,
}

impl SiteGeometryStatus {
    pub(crate) fn of(state: &SiteGeometryState) -> Self {
        Self {
            coordinate_frame: state.coordinate_frame(),
            applied_geometry_push_revision: state.geometry_push_revision(),
            node_placement_count: state
                .acknowledgement()
                .map_or(0, |acknowledgement| acknowledgement.node_placement_count),
        }
    }
}

pub(crate) async fn get_dsp_geometry(State(dsp_state): State<SharedDspState>) -> Response {
    Json(SiteGeometryStatus::of(
        &dsp_state.read().await.site_geometry,
    ))
    .into_response()
}

pub(crate) async fn post_dsp_geometry(
    State(dsp_state): State<SharedDspState>,
    body: Result<Json<SiteGeometryPush>, JsonRejection>,
) -> Response {
    let push = match body {
        Ok(Json(push)) => push,
        Err(rejection) => {
            warn!(error = %rejection.body_text(), "site geometry push body rejected");
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({
                    "error": "invalid_site_geometry_body",
                    "detail": rejection.body_text(),
                })),
            )
                .into_response();
        }
    };
    let applied = dsp_state.write().await.site_geometry.apply_push(push);
    match applied {
        Ok(acknowledgement) => {
            info!(
                site_frame_id = %acknowledgement.site_frame_id,
                geometry_push_revision = acknowledgement.applied_geometry_push_revision,
                node_placement_count = acknowledgement.node_placement_count,
                "site geometry push applied"
            );
            Json(acknowledgement).into_response()
        }
        Err(error) => {
            warn!(error = %error, "site geometry push refused");
            site_geometry_error_response(&error)
        }
    }
}

fn site_geometry_error_response(error: &SiteGeometryError) -> Response {
    match error {
        SiteGeometryError::StaleGeometryPushRevision {
            applied_revision, ..
        } => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "stale_geometry_push_revision",
                "detail": error.to_string(),
                "applied_geometry_push_revision": applied_revision,
            })),
        )
            .into_response(),
        _ => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "invalid_site_geometry",
                "detail": error.to_string(),
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{body::Body, http::Request, routing::get, Router};
    use http_body_util::BodyExt;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use super::*;
    use crate::dsp_worker::DspWorkerState;

    fn geometry_router(dsp_state: SharedDspState) -> Router {
        Router::new()
            .route(
                SITE_GEOMETRY_ENDPOINT,
                get(get_dsp_geometry).post(post_dsp_geometry),
            )
            .with_state(dsp_state)
    }

    async fn send(
        dsp_state: &SharedDspState,
        method: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(method)
            .uri(SITE_GEOMETRY_ENDPOINT)
            .header("content-type", "application/json")
            .body(body.map_or_else(Body::empty, |json| Body::from(json.to_string())))
            .unwrap();
        let response = geometry_router(dsp_state.clone())
            .oneshot(request)
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn push_body(revision: u64) -> serde_json::Value {
        serde_json::json!({
            "site_frame_id": "site-7",
            "geometry_push_revision": revision,
            "site_origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
            "projection": "flat_wgs84_series",
            "node_placements": {
                "planar-a": {"position_m": [1.0, 2.0, 1.5], "position_std_m": 0.05,
                             "placement_source": "operator_override",
                             "body_to_world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "point-b": {"position_m": [6.0, 0.0, 1.5], "position_std_m": 0.4,
                            "placement_source": "stationary_estimate"}
            }
        })
    }

    #[tokio::test]
    async fn push_is_acknowledged_reported_and_stale_revisions_conflict() {
        let dsp_state: SharedDspState = Arc::new(RwLock::new(DspWorkerState::default()));

        let (status, before) = send(&dsp_state, "GET", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            before["applied_geometry_push_revision"],
            serde_json::Value::Null
        );
        assert_eq!(before["coordinate_frame"]["kind"], "sidecar_automatic_enu");

        let (status, acknowledgement) = send(&dsp_state, "POST", Some(push_body(3))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            acknowledgement,
            serde_json::json!({"site_frame_id": "site-7", "applied_geometry_push_revision": 3,
                               "node_placement_count": 2})
        );

        let (status, conflict) = send(&dsp_state, "POST", Some(push_body(2))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(conflict["error"], "stale_geometry_push_revision");
        assert_eq!(conflict["applied_geometry_push_revision"], 3);

        let (_, after) = send(&dsp_state, "GET", None).await;
        assert_eq!(after["applied_geometry_push_revision"], 3);
        assert_eq!(after["node_placement_count"], 2);
        assert_eq!(after["coordinate_frame"]["kind"], "site_enu");
        assert_eq!(after["coordinate_frame"]["frame_id"], "site-7");
        assert_eq!(after["coordinate_frame"]["projection"], "flat_wgs84_series");
    }

    #[tokio::test]
    async fn malformed_and_invalid_pushes_are_unprocessable() {
        let dsp_state: SharedDspState = Arc::new(RwLock::new(DspWorkerState::default()));
        let (status, body) = send(
            &dsp_state,
            "POST",
            Some(serde_json::json!({"site_frame_id": "x"})),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["error"], "invalid_site_geometry_body");

        let mut invalid = push_body(1);
        invalid["node_placements"]["point-b"]["position_std_m"] = serde_json::json!(-1.0);
        let (status, body) = send(&dsp_state, "POST", Some(invalid)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["error"], "invalid_site_geometry");
        assert_eq!(
            dsp_state
                .read()
                .await
                .site_geometry
                .geometry_push_revision(),
            None
        );
    }
}
