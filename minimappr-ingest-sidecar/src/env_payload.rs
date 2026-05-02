use serde::{Deserialize, Serialize};

/// A single environmental sensor reading sent to `/api/v1/ingest/env`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EnvSampleDto {
    pub node_id: String,
    #[serde(default)]
    pub sensor_id: Option<String>,
    /// Nanosecond timestamp of the reading (wall-clock, same epoch as audio frames).
    pub t_ns: u64,
    pub temp_c: f32,
    /// Relative humidity in percent (0–100).
    pub rh_pct: f32,
    #[serde(default)]
    pub pressure_hpa: Option<f32>,
}

/// Request body for `POST /api/v1/ingest/env`.
#[derive(Clone, Debug, Deserialize)]
pub struct EnvIngestPayload {
    pub samples: Vec<EnvSampleDto>,
}

#[derive(Debug, Serialize)]
pub struct EnvIngestResponse {
    pub accepted: usize,
}
