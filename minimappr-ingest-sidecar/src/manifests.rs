use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::SystemTime,
};

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
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub label_confidence: Option<f32>,
    #[serde(default)]
    pub scores: Option<BTreeMap<String, f32>>,
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
    /// Populated for `manifest_type == "env_sample_append"` manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_samples: Option<serde_json::Value>,
    /// Node spec JSON embedded in memory-path localization_result manifests.
    /// Python uses this to reconstruct node context without a persisted raw payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_context: Option<serde_json::Value>,
    /// Raw audio payload bytes carried in-process by the fast-path channel only.
    /// Never serialized to disk; skipped by serde.
    #[serde(skip, default)]
    pub raw_payload: Option<Vec<u8>>,
    /// Base64-encoded PCM16LE mono render bytes for inline SSE delivery.
    /// Set only on manifests sent via the broadcast channel; always cleared to
    /// None before ManifestStore.publish() so disk files stay compact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_render_bytes: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ManifestStore {
    root: PathBuf,
    initialized: Arc<AtomicBool>,
}

impl ManifestStore {
    pub fn new(journal_root: &Path) -> Self {
        Self {
            root: journal_root.join("manifests"),
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn ensure_initialized(&self) -> BoxedResult<()> {
        if self.initialized.load(Ordering::Acquire) {
            return Ok(());
        }
        fs::create_dir_all(&self.root).await?;
        fs::create_dir_all(self.pending_root()).await?;
        fs::create_dir_all(self.raw_pending_root()).await?;
        fs::create_dir_all(self.consumed_root()).await?;
        self.migrate_legacy_pending_manifests().await?;
        self.migrate_raw_pending_manifests().await?;
        self.initialized.store(true, Ordering::Release);
        Ok(())
    }

    pub async fn publish(&self, mut manifest: DspManifest) -> BoxedResult<PathBuf> {
        self.ensure_initialized().await?;
        if manifest.manifest_id.is_empty() {
            let prefix = if manifest.manifest_type == "raw_journal_append" {
                "manifest-jrnl"
            } else {
                "manifest"
            };
            manifest.manifest_id = format!("{}-{}", prefix, Uuid::new_v4());
        }
        let pending_root = if manifest.manifest_type == "raw_journal_append" {
            self.raw_pending_root()
        } else {
            self.pending_root()
        };
        let path = pending_root.join(format!("{}.json", manifest.manifest_id));
        let tmp_path = pending_root.join(format!(".{}.json.tmp", manifest.manifest_id));
        let mut bytes = serde_json::to_vec(&manifest)?;
        bytes.push(b'\n');
        fs::write(&tmp_path, bytes).await?;
        fs::rename(&tmp_path, &path).await?;
        Ok(path)
    }

    /// Return all manifests of the given type that have not been consumed yet.
    #[allow(dead_code)]
    pub async fn query_pending(&self, manifest_type: &str) -> BoxedResult<Vec<DspManifest>> {
        self.query_pending_limited(manifest_type, usize::MAX).await
    }

    /// Return up to `limit` manifests of the given type that have not been consumed yet.
    ///
    /// Two-phase approach to avoid reading all N pending files when backlog is large:
    /// 1. Collect (mtime, path) for all files — metadata only, much cheaper than content.
    /// 2. Sort by mtime, truncate to `limit`, then read only those file contents.
    ///
    /// This changes O(N_files × content_read) to O(N_files × metadata_read + limit × content_read).
    pub async fn query_pending_limited(
        &self,
        manifest_type: &str,
        limit: usize,
    ) -> BoxedResult<Vec<DspManifest>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query_root = if manifest_type == "raw_journal_append" {
            self.raw_pending_root()
        } else {
            self.pending_root()
        };
        let mut entries = match fs::read_dir(&query_root).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(Box::new(error)),
        };

        // Phase 1: collect metadata for every pending file (no content reads).
        let mut file_mtimes: Vec<(SystemTime, PathBuf)> = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !is_pending_manifest_file_name(name) {
                continue;
            }
            let mtime = entry
                .metadata()
                .await
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            file_mtimes.push((mtime, path));
        }

        // Phase 2: sort by mtime (oldest first), keep only the first `limit` entries,
        // then read only those files. This prevents the O(N) content-read death spiral
        // when a large backlog accumulates.
        file_mtimes.sort_unstable_by_key(|(mtime, _)| *mtime);
        file_mtimes.truncate(limit);

        let mut manifests = Vec::with_capacity(file_mtimes.len());
        for (_, path) in file_mtimes {
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
        self.ensure_initialized().await?;
        let dst = self.consumed_root().join(format!("{manifest_id}.json"));
        for src in [
            self.raw_pending_root().join(format!("{manifest_id}.json")),
            self.pending_root().join(format!("{manifest_id}.json")),
            self.root.join(format!("{manifest_id}.json")),
        ] {
            if fs::metadata(&src).await.is_ok() {
                fs::rename(&src, &dst).await?;
                return Ok(());
            }
        }

        // External producers may write pending manifests with a prefixed filename
        // (for example, "1700000000-node1.json") while JSON.manifest_id differs.
        // Fall back to scanning pending files by payload manifest_id.
        let mut entries = match fs::read_dir(self.pending_root()).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("pending manifest {manifest_id} not found"),
                )));
            }
            Err(error) => return Err(Box::new(error)),
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if !is_pending_manifest_file_name(file_name) {
                continue;
            }
            let contents = match fs::read_to_string(&path).await {
                Ok(contents) => contents,
                Err(_) => continue,
            };
            let pending_manifest: DspManifest = match serde_json::from_str(contents.trim()) {
                Ok(manifest) => manifest,
                Err(_) => continue,
            };
            if pending_manifest.manifest_id != manifest_id {
                continue;
            }
            let discovered_dst = self.consumed_root().join(file_name);
            fs::rename(&path, discovered_dst).await?;
            return Ok(());
        }

        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("pending manifest {manifest_id} not found"),
        )))
    }

    /// Keep at most `max_files` consumed manifests, removing oldest first.
    pub async fn prune_consumed_manifests(&self, max_files: usize) -> BoxedResult<()> {
        self.ensure_initialized().await?;
        if max_files == 0 {
            let mut entries = fs::read_dir(self.consumed_root()).await?;
            while let Some(entry) = entries.next_entry().await? {
                let file_type = entry.file_type().await?;
                if !file_type.is_file() {
                    continue;
                }
                let path = entry.path();
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                if is_pending_manifest_file_name(file_name) {
                    let _ = fs::remove_file(path).await;
                }
            }
            return Ok(());
        }

        let mut consumed_files = Vec::new();
        let mut entries = fs::read_dir(self.consumed_root()).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_type = entry.file_type().await?;
            if !file_type.is_file() {
                continue;
            }
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if !is_pending_manifest_file_name(file_name) {
                continue;
            }
            let modified_at = entry
                .metadata()
                .await
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            consumed_files.push((modified_at, path));
        }

        if consumed_files.len() <= max_files {
            return Ok(());
        }

        consumed_files.sort_by_key(|(modified_at, _)| *modified_at);
        let files_to_remove = consumed_files.len().saturating_sub(max_files);
        for (_, path) in consumed_files.into_iter().take(files_to_remove) {
            let _ = fs::remove_file(path).await;
        }
        Ok(())
    }

    fn pending_root(&self) -> PathBuf {
        self.root.join("pending")
    }

    fn raw_pending_root(&self) -> PathBuf {
        self.root.join("raw_pending")
    }

    fn consumed_root(&self) -> PathBuf {
        self.root.join("consumed")
    }

    async fn migrate_legacy_pending_manifests(&self) -> BoxedResult<()> {
        let marker_path = self.root.join(".legacy_pending_migrated");
        if fs::metadata(&marker_path).await.is_ok() {
            return Ok(());
        }
        let mut entries = match fs::read_dir(&self.root).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(Box::new(error)),
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !is_pending_manifest_file_name(name) {
                continue;
            }
            let file_type = entry.file_type().await?;
            if !file_type.is_file() {
                continue;
            }
            let dst = self.pending_root().join(name);
            if fs::metadata(&dst).await.is_ok() {
                continue;
            }
            fs::rename(path, dst).await?;
        }
        let _ = fs::write(marker_path, b"1\n").await;
        Ok(())
    }

    async fn migrate_raw_pending_manifests(&self) -> BoxedResult<()> {
        let marker_path = self.root.join(".raw_pending_migrated");
        if fs::metadata(&marker_path).await.is_ok() {
            return Ok(());
        }
        let mut entries = match fs::read_dir(self.pending_root()).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(Box::new(error)),
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !name.starts_with("manifest-jrnl-") || !is_pending_manifest_file_name(name) {
                continue;
            }
            let file_type = entry.file_type().await?;
            if !file_type.is_file() {
                continue;
            }
            let dst = self.raw_pending_root().join(name);
            if fs::metadata(&dst).await.is_ok() {
                continue;
            }
            fs::rename(path, dst).await?;
        }
        let _ = fs::write(marker_path, b"1\n").await;
        Ok(())
    }
}

fn is_pending_manifest_file_name(file_name: &str) -> bool {
    file_name.ends_with(".json") && !file_name.starts_with('.')
}
