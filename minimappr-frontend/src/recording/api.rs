/// Recording API harness — all functions call their respective backend endpoints.
///
/// Until the recording backend is implemented, these endpoints return 404 and
/// the UI will surface a "recording backend not yet available" message.
///
/// Expected API surface (to be implemented in main.py):
///
///   POST   /api/v1/recordings                         → RecordingSession
///   PATCH  /api/v1/recordings/{id}/stop               → RecordingSession
///   GET    /api/v1/recordings/{id}                    → RecordingSession
///   GET    /api/v1/recordings                         → Vec<RecordingLibraryEntry>
///   DELETE /api/v1/recordings/{id}                    → 204
///   GET    /api/v1/cameras                            → Vec<CameraDevice>
///   GET    /api/v1/recordings/{id}/download?format=…  → file (ambisonics|iamf|visual|video)
use crate::recording::{
    CalibrationManifest, CameraDevice, GroundTruthEvent, GroundTruthEventIn,
    RecordingLibraryEntry, RecordingSession, StartRecordingRequest,
};
use gloo_net::http::Request;
use js_sys::encode_uri_component;
use serde_json::Value;

fn enc(s: &str) -> String {
    encode_uri_component(s).as_string().unwrap_or_default()
}

async fn expect_json<T: serde::de::DeserializeOwned>(
    resp: gloo_net::http::Response,
) -> Result<T, String> {
    if resp.ok() {
        resp.json::<T>().await.map_err(|e| e.to_string())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status == 404 {
            Err("Recording backend not yet available.".into())
        } else {
            Err(if body.is_empty() {
                format!("HTTP {status}")
            } else {
                extract_error_detail(&body).unwrap_or(body)
            })
        }
    }
}

fn extract_error_detail(body: &str) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(body).ok()?;
    parsed
        .get("detail")
        .and_then(|detail| detail.as_str())
        .map(ToOwned::to_owned)
}

/// Start a new recording session.
///
/// Returns immediately with `status: Starting`; poll via `fetch_recording` to
/// track the Starting → Active transition once capture is confirmed.
pub async fn start_recording(req: StartRecordingRequest) -> Result<RecordingSession, String> {
    let body = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    let resp = Request::post("/api/v1/recordings")
        .header("Content-Type", "application/json")
        .body(body)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    expect_json(resp).await
}

/// Stop an active recording session.
///
/// Transitions status to Stopping; poll via `fetch_recording` to detect
/// Stopping → Completed once all output files have been finalized.
pub async fn stop_recording(session_id: &str) -> Result<RecordingSession, String> {
    let url = format!("/api/v1/recordings/{}/stop", enc(session_id));
    let resp = Request::patch(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    expect_json(resp).await
}

/// Fetch the current state of a single session. Use for status polling.
#[allow(dead_code)]
pub async fn fetch_recording(session_id: &str) -> Result<RecordingSession, String> {
    let url = format!("/api/v1/recordings/{}", enc(session_id));
    let resp = Request::get(&url).send().await.map_err(|e| e.to_string())?;
    expect_json(resp).await
}

/// Fetch all past and in-progress recordings (most recent first).
pub async fn fetch_recordings() -> Result<Vec<RecordingLibraryEntry>, String> {
    let resp = Request::get("/api/v1/recordings")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.status() == 404 {
        return Ok(vec![]);
    }
    expect_json(resp).await
}

/// Delete a recording and all associated output files.
pub async fn delete_recording(session_id: &str) -> Result<(), String> {
    let url = format!("/api/v1/recordings/{}", enc(session_id));
    let resp = Request::delete(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() || resp.status() == 204 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(if body.is_empty() {
            format!("HTTP {status}")
        } else {
            body
        })
    }
}

/// List available video capture devices.
///
/// Backend enumerates AVFoundation (macOS) or v4l2 (Linux/RPi) devices.
/// Returns empty list (not an error) when the endpoint doesn't exist yet.
pub async fn fetch_cameras() -> Result<Vec<CameraDevice>, String> {
    let resp = Request::get("/api/v1/cameras")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.status() == 404 {
        return Ok(vec![]);
    }
    expect_json(resp).await
}

/// List ground-truth events for a calibration session.
pub async fn fetch_ground_truth(session_id: &str) -> Result<Vec<GroundTruthEvent>, String> {
    let url = format!("/api/v1/calibration/{}/ground-truth", enc(session_id));
    let resp = Request::get(&url).send().await.map_err(|e| e.to_string())?;
    expect_json(resp).await
}

/// Add a ground-truth event to a calibration session.
pub async fn add_ground_truth(
    session_id: &str,
    event: GroundTruthEventIn,
) -> Result<GroundTruthEvent, String> {
    let body = serde_json::to_string(&event).map_err(|e| e.to_string())?;
    let url = format!("/api/v1/calibration/{}/ground-truth", enc(session_id));
    let resp = Request::post(&url)
        .header("Content-Type", "application/json")
        .body(body)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    expect_json(resp).await
}

/// Delete a ground-truth event.
pub async fn delete_ground_truth(event_id: &str) -> Result<(), String> {
    let url = format!("/api/v1/calibration/ground-truth/{}", enc(event_id));
    let resp = Request::delete(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() || resp.status() == 204 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(if body.is_empty() {
            format!("HTTP {status}")
        } else {
            extract_error_detail(&body).unwrap_or(body)
        })
    }
}

/// Download URL for a calibration session's replayable bundle zip.
pub fn calibration_bundle_url(session_id: &str) -> String {
    format!("/api/v1/calibration/{}/bundle", enc(session_id))
}

/// Lightweight manifest (node ids + audio window) for the waveform trimmer.
pub async fn fetch_calibration_manifest(session_id: &str) -> Result<CalibrationManifest, String> {
    let url = format!("/api/v1/calibration/{}/manifest", enc(session_id));
    let resp = Request::get(&url).send().await.map_err(|e| e.to_string())?;
    expect_json(resp).await
}

/// URL for one node's raw multichannel WAV (fetched + decoded client-side).
pub fn calibration_audio_url(session_id: &str, node_id: &str) -> String {
    format!(
        "/api/v1/calibration/{}/audio/{}",
        enc(session_id),
        enc(node_id)
    )
}

/// Fetch a node's raw WAV bytes for client-side waveform decode.
pub async fn fetch_calibration_audio_bytes(
    session_id: &str,
    node_id: &str,
) -> Result<Vec<u8>, String> {
    let url = calibration_audio_url(session_id, node_id);
    let resp = Request::get(&url).send().await.map_err(|e| e.to_string())?;
    if !resp.ok() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(if body.is_empty() {
            format!("HTTP {status}")
        } else {
            extract_error_detail(&body).unwrap_or(body)
        });
    }
    resp.binary().await.map_err(|e| e.to_string())
}

/// Build the download URL for a completed recording.
///
/// `format` is one of: `ambisonics`, `iamf`, `visual`, `video`.
pub fn download_url(session_id: &str, format: &str) -> String {
    format!(
        "/api/v1/recordings/{}/download?format={}",
        enc(session_id),
        enc(format)
    )
}
