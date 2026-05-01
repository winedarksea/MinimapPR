use std::{collections::BTreeMap, path::Path, process::Stdio, time::Duration};

use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::timeout,
};
use tracing::warn;

const _CLASSIFIER_HELPER_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Debug)]
pub struct AuthoritativeClassification {
    pub label: String,
    pub label_confidence: f32,
    pub scores: BTreeMap<String, f32>,
}

#[derive(Debug, Serialize)]
struct ClassifierRequest<'a> {
    request_id: u64,
    pcm16le_path: &'a str,
    sample_rate_hz: u32,
}

#[derive(Debug, Deserialize)]
struct ClassifierResponse {
    request_id: u64,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    label_confidence: Option<f32>,
    #[serde(default)]
    scores: Option<BTreeMap<String, f32>>,
    #[serde(default)]
    error: Option<String>,
}

struct ClassifierBridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

pub struct ManifestClassificationAnnotator {
    command_argv: Vec<String>,
    bridge: Option<ClassifierBridge>,
    next_request_id: u64,
}

impl ManifestClassificationAnnotator {
    pub fn from_command_json(command_json: Option<&str>) -> Result<Option<Self>, String> {
        let Some(raw_command_json) = command_json else {
            return Ok(None);
        };
        if raw_command_json.trim().is_empty() {
            return Ok(None);
        }
        let command_argv: Vec<String> = serde_json::from_str(raw_command_json)
            .map_err(|error| format!("invalid sidecar classifier command json: {error}"))?;
        if command_argv.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            command_argv,
            bridge: None,
            next_request_id: 1,
        }))
    }

    pub async fn classify_render(
        &mut self,
        pcm16le_path: &Path,
        sample_rate_hz: u32,
    ) -> Result<Option<AuthoritativeClassification>, String> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);
        let response_result = {
            let bridge = self.ensure_bridge().await?;
            let request = serde_json::to_string(&ClassifierRequest {
                request_id,
                pcm16le_path: &pcm16le_path.display().to_string(),
                sample_rate_hz,
            })
            .map_err(|error| format!("failed to serialize classifier request: {error}"))?;

            timeout(
                _CLASSIFIER_HELPER_TIMEOUT,
                async {
                    bridge
                        .stdin
                        .write_all(request.as_bytes())
                        .await
                        .map_err(|error| format!("failed to write classifier request: {error}"))?;
                    bridge
                        .stdin
                        .write_all(b"\n")
                        .await
                        .map_err(|error| format!("failed to terminate classifier request: {error}"))?;
                    bridge
                        .stdin
                        .flush()
                        .await
                        .map_err(|error| format!("failed to flush classifier request: {error}"))?;

                    let mut response_line = String::new();
                    let read_count = bridge
                        .stdout
                        .read_line(&mut response_line)
                        .await
                        .map_err(|error| format!("failed to read classifier response: {error}"))?;
                    if read_count == 0 {
                        return Err("classifier helper closed stdout unexpectedly".to_string());
                    }
                    Self::parse_response(request_id, &response_line)
                },
            )
            .await
            .map_err(|_| "classifier helper timed out".to_string())?
        };

        if response_result.is_err() {
            self.bridge = None;
        }
        response_result
    }

    async fn ensure_bridge(&mut self) -> Result<&mut ClassifierBridge, String> {
        if let Some(existing_bridge) = self.bridge.take() {
            match Self::bridge_is_running(existing_bridge).await {
                Ok(active_bridge) => {
                    self.bridge = Some(active_bridge);
                }
                Err(error) => {
                    warn!(error = %error, "Classifier helper bridge exited; respawning");
                }
            }
        }
        if self.bridge.is_none() {
            self.bridge = Some(Self::spawn_bridge(&self.command_argv).await?);
        }
        self.bridge
            .as_mut()
            .ok_or_else(|| "classifier helper bridge is unavailable".to_string())
    }

    async fn bridge_is_running(mut bridge: ClassifierBridge) -> Result<ClassifierBridge, String> {
        match bridge
            .child
            .try_wait()
            .map_err(|error| format!("failed to inspect classifier helper process: {error}"))?
        {
            None => Ok(bridge),
            Some(status) => Err(format!("classifier helper exited with status {status}")),
        }
    }

    async fn spawn_bridge(command_argv: &[String]) -> Result<ClassifierBridge, String> {
        let executable = command_argv
            .first()
            .ok_or_else(|| "classifier helper command is empty".to_string())?;
        let mut child = Command::new(executable)
            .args(command_argv.iter().skip(1))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("failed to spawn classifier helper: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "classifier helper did not expose stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "classifier helper did not expose stdout".to_string())?;
        Ok(ClassifierBridge {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn parse_response(
        request_id: u64,
        response_line: &str,
    ) -> Result<Option<AuthoritativeClassification>, String> {
        let response: ClassifierResponse = serde_json::from_str(response_line.trim())
            .map_err(|error| format!("invalid classifier response: {error}"))?;
        if response.request_id != request_id {
            return Err(format!(
                "classifier response id mismatch: expected {request_id}, got {}",
                response.request_id
            ));
        }
        if let Some(error) = response.error {
            return Err(error);
        }
        let Some(label) = response.label.map(|value| value.trim().to_string()) else {
            return Ok(None);
        };
        if label.is_empty() {
            return Ok(None);
        }
        Ok(Some(AuthoritativeClassification {
            label,
            label_confidence: response.label_confidence.unwrap_or(0.0).clamp(0.0, 1.0),
            scores: response.scores.unwrap_or_default(),
        }))
    }
}