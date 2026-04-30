use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::fs;
use uuid::Uuid;

use crate::journal_reader::JournalPayloadHandle;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairTdoaDiagnostic {
    pub ch_a: usize,
    pub ch_b: usize,
    pub delay_samples: f32,
    pub lag_seconds: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalizationManifestPayload {
    pub attempted_algorithm: String,
    pub resolved_algorithm: String,
    pub steering_direction: Option<[f32; 3]>,
    pub position_m: Option<[f32; 3]>,
    pub confidence: f32,
    pub residual_rms_seconds: Option<f32>,
    pub sound_speed_mps: f32,
    pub effective_band_hz: Option<[f32; 2]>,
    pub pair_tdoas: Vec<PairTdoaDiagnostic>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifierRenderManifestPayload {
    pub render_id: String,
    pub render_kind: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_count: usize,
    pub sample_format: String,
    pub effective_spatial_band: Option<[f32; 2]>,
    pub source_channel_count: usize,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BirdnetHybridProvenance {
    pub steering_solution: Option<String>,
    pub classifier_source_node: Option<String>,
    pub spatial_blend_mode: String,
    pub effective_spatial_band: Option<[f32; 2]>,
    pub confidence: Option<f32>,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DspManifest {
    pub manifest_id: String,
    pub manifest_type: String,
    pub created_ns: u128,
    pub source_handles: Vec<JournalPayloadHandle>,
    pub derived_handle: Option<JournalPayloadHandle>,
    #[serde(default)]
    pub localization: Option<LocalizationManifestPayload>,
    #[serde(default)]
    pub classifier_render: Option<ClassifierRenderManifestPayload>,
    pub birdnet: Option<BirdnetHybridProvenance>,
    pub coverage_stats: Option<serde_json::Value>,
    pub promotion_ready: bool,
}

#[derive(Clone, Debug)]
pub struct ManifestStore {
    root: PathBuf,
}

impl ManifestStore {
    pub fn new(journal_root: &Path) -> Self {
        Self {
            root: journal_root.join("manifests"),
        }
    }

    pub async fn ensure_initialized(&self) -> BoxedResult<()> {
        fs::create_dir_all(&self.root).await?;
        Ok(())
    }

    pub async fn publish(&self, mut manifest: DspManifest) -> BoxedResult<PathBuf> {
        fs::create_dir_all(&self.root).await?;
        if manifest.manifest_id.is_empty() {
            manifest.manifest_id = format!("manifest-{}", Uuid::new_v4());
        }
        let path = self.root.join(format!("{}.json", manifest.manifest_id));
        let tmp_path = self
            .root
            .join(format!(".{}.json.tmp", manifest.manifest_id));
        let mut bytes = serde_json::to_vec(&manifest)?;
        bytes.push(b'\n');
        fs::write(&tmp_path, bytes).await?;
        fs::rename(&tmp_path, &path).await?;
        Ok(path)
    }

    /// Return all manifests of the given type that have not been consumed yet.
    pub async fn query_pending(&self, manifest_type: &str) -> BoxedResult<Vec<DspManifest>> {
        let mut entries = match fs::read_dir(&self.root).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(Box::new(error)),
        };
        let mut manifests = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !name.ends_with(".json") || name.ends_with(".consumed") || name.starts_with('.') {
                continue;
            }
            let contents = match fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let manifest: DspManifest = match serde_json::from_str(contents.trim()) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if manifest.manifest_type == manifest_type {
                manifests.push(manifest);
            }
        }
        manifests.sort_by_key(|m| m.created_ns);
        Ok(manifests)
    }

    /// Atomically rename a manifest file to mark it consumed.
    pub async fn mark_consumed(&self, manifest_id: &str) -> BoxedResult<()> {
        let src = self.root.join(format!("{manifest_id}.json"));
        let dst = self.root.join(format!("{manifest_id}.json.consumed"));
        fs::rename(&src, &dst).await?;
        Ok(())
    }
}
