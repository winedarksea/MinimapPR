use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{header, HeaderMap},
};
use http_body_util::BodyExt;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncSeekExt, AsyncWriteExt},
    sync::{mpsc, Mutex},
};
use tracing::warn;
use uuid::Uuid;

use crate::derived_cache::{DerivedCache, DerivedCacheConfig};
use crate::envelope::{parse_capture_envelope, CaptureEnvelope};
use crate::journal_reader::{stable_segment_path, JournalPayloadHandle};
use crate::leases::{
    hard_segment_pin_counts, segment_pin_counts, LeaseStore, PinLeaseRequest, PinLeaseResponse,
};
use crate::manifests::{DspManifest, ManifestStore};
use crate::storage_class::{validate_journal_storage_class, StorageClassReport};

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug)]
pub struct JournalRuntimeConfig {
    pub consumer_name: String,
    pub total_journal_budget_bytes: u64,
    pub admission_reserve_bytes: u64,
    pub enforce_tmpfs: bool,
    pub derived_cache_budget_bytes: u64,
    pub derived_cache_admission_reserve_bytes: u64,
    /// Optional in-process channel for zero-latency delivery of raw_journal_append
    /// manifests to the DspWorker, bypassing the 20ms filesystem polling cycle.
    /// Unbounded so that the ingest path never blocks or drops when the DSP worker
    /// is briefly busy; memory growth is bounded by the DSP worker's buffer window.
    pub raw_manifest_tx: Option<mpsc::UnboundedSender<DspManifest>>,
}

impl Default for JournalRuntimeConfig {
    fn default() -> Self {
        Self {
            consumer_name: "python-ingest".to_string(),
            total_journal_budget_bytes: 268_435_456,
            admission_reserve_bytes: 16_777_216,
            enforce_tmpfs: true,
            derived_cache_budget_bytes: 67_108_864,
            derived_cache_admission_reserve_bytes: 8_388_608,
            raw_manifest_tx: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestStorageMode {
    Journal,
}

#[derive(Clone, Debug)]
pub struct IngestBackend {
    inner: Arc<BackendInner>,
}

#[derive(Debug)]
enum BackendInner {
    Journal(Box<SegmentJournalBackend>),
}

#[derive(Debug)]
struct SegmentJournalBackend {
    journal_root: PathBuf,
    streams_dir: PathBuf,
    max_segment_bytes: u64,
    runtime_config: JournalRuntimeConfig,
    storage_class_report: StorageClassReport,
    lease_store: LeaseStore,
    derived_cache: DerivedCache,
    manifest_store: ManifestStore,
    state: Mutex<SegmentJournalState>,
}

#[derive(Debug)]
struct SegmentJournalState {
    journal_epoch: u64,
    total_journal_bytes: u64,
    streams: HashMap<String, StreamJournalState>,
}

#[derive(Debug)]
struct StreamJournalState {
    next_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentJournalEntry {
    observation_id: String,
    journal_id: String,
    journal_epoch: u64,
    journal_sequence: u64,
    node_id: String,
    stream_id: String,
    stream_key: String,
    sensor_type: String,
    source_type: String,
    transport: String,
    toa_ns: Option<u64>,
    tor_ns: Option<u64>,
    ingest_received_ns: u128,
    time_quality: Option<String>,
    clock_domain: Option<String>,
    sync_source: Option<String>,
    clock_correction_ns: Option<i64>,
    clock_drift_ppm: Option<f64>,
    sample_rate_hz: Option<u32>,
    channel_count: Option<u16>,
    channel_layout: Option<String>,
    sample_index_start: Option<u64>,
    sample_count: Option<u64>,
    geometry_version: Option<String>,
    orientation_version: Option<String>,
    calibration_version: Option<String>,
    retention_hint: Option<String>,
    payload_codec: String,
    integrity_hash: String,
    endpoint: String,
    content_type: Option<String>,
    segment_id: String,
    segment_path: PathBuf,
    payload_offset_bytes: u64,
    payload_length_bytes: u64,
    body_offset_bytes: u64,
    body_length_bytes: u64,
    received_ns: u128,
}

#[derive(Clone, Debug, Serialize)]
pub struct IngestBackendHealth {
    pub storage_mode: String,
    pub journal: Option<JournalHealth>,
}

#[derive(Clone, Debug, Serialize)]
pub struct JournalHealth {
    pub journal_root: PathBuf,
    pub storage_class: StorageClassReport,
    pub total_journal_budget_bytes: u64,
    pub admission_reserve_bytes: u64,
    pub derived_cache_budget_bytes: u64,
    pub active_pin_leases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentJournalHeader {
    segment_id: String,
    journal_epoch: u64,
    stream_key: String,
    first_journal_sequence: Option<u64>,
    last_journal_sequence: Option<u64>,
    first_received_ns: Option<u128>,
    last_received_ns: Option<u128>,
    first_toa_ns: Option<u64>,
    last_toa_ns: Option<u64>,
    first_tor_ns: Option<u64>,
    last_tor_ns: Option<u64>,
    entry_count: u64,
    payload_bytes: u64,
    pin_count: u64,
    #[serde(default)]
    hard_pin_count: u64,
    sealed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct JournalEpochState {
    active_epoch: u64,
    last_assigned_epoch: u64,
}

#[derive(Debug)]
pub struct BodyTooLargeError {
    limit_bytes: usize,
}

impl std::fmt::Display for BodyTooLargeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "request body exceeds maximum of {} bytes",
            self.limit_bytes
        )
    }
}

impl std::error::Error for BodyTooLargeError {}

#[derive(Debug)]
pub struct ClientDisconnectError;

impl std::fmt::Display for ClientDisconnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "client disconnected before request body was fully received"
        )
    }
}

impl std::error::Error for ClientDisconnectError {}

#[derive(Debug)]
pub struct InvalidIngestEnvelopeError {
    detail: String,
}

impl InvalidIngestEnvelopeError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for InvalidIngestEnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for InvalidIngestEnvelopeError {}

#[derive(Debug)]
pub struct JournalCapacityExceededError {
    total_journal_budget_bytes: u64,
    admission_reserve_bytes: u64,
    projected_total_bytes: u64,
}

impl std::fmt::Display for JournalCapacityExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "journal memory budget exhausted: projected {} bytes plus {} byte admission reserve exceeds {} bytes",
            self.projected_total_bytes,
            self.admission_reserve_bytes,
            self.total_journal_budget_bytes,
        )
    }
}

impl std::error::Error for JournalCapacityExceededError {}

impl IngestBackend {
    pub async fn open(
        base_dir: PathBuf,
        storage_mode: IngestStorageMode,
        max_segment_bytes: u64,
        journal_runtime_config: JournalRuntimeConfig,
    ) -> BoxedResult<Self> {
        let inner = match storage_mode {
            IngestStorageMode::Journal => BackendInner::Journal(Box::new(
                SegmentJournalBackend::open(base_dir, max_segment_bytes, journal_runtime_config)
                    .await?,
            )),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub async fn enqueue(
        &self,
        max_body_bytes: usize,
        endpoint: &'static str,
        headers: HeaderMap,
        body: Body,
    ) -> BoxedResult<String> {
        match self.inner.as_ref() {
            BackendInner::Journal(backend) => {
                backend
                    .enqueue(max_body_bytes, endpoint, headers, body)
                    .await
            }
        }
    }

    pub async fn health(&self) -> BoxedResult<IngestBackendHealth> {
        match self.inner.as_ref() {
            BackendInner::Journal(backend) => backend.health().await,
        }
    }

    pub async fn create_pin_lease(
        &self,
        request: PinLeaseRequest,
    ) -> BoxedResult<PinLeaseResponse> {
        match self.inner.as_ref() {
            BackendInner::Journal(backend) => backend.create_pin_lease(request).await,
        }
    }

    pub async fn release_pin_lease(&self, lease_id: &str) -> BoxedResult<()> {
        match self.inner.as_ref() {
            BackendInner::Journal(backend) => backend.release_pin_lease(lease_id).await,
        }
    }

    /// Returns the manifest store if the backend is in journal mode.
    pub fn manifest_store(&self) -> Option<ManifestStore> {
        match self.inner.as_ref() {
            BackendInner::Journal(b) => Some(b.manifest_store.clone()),
        }
    }

    /// Returns the derived cache if the backend is in journal mode.
    pub fn derived_cache(&self) -> Option<DerivedCache> {
        match self.inner.as_ref() {
            BackendInner::Journal(b) => Some(b.derived_cache.clone()),
        }
    }

    /// Returns the journal root path if the backend is in journal mode.
    pub fn journal_root(&self) -> Option<&std::path::Path> {
        match self.inner.as_ref() {
            BackendInner::Journal(b) => Some(&b.journal_root),
        }
    }
}

impl SegmentJournalBackend {
    async fn open(
        base_dir: PathBuf,
        max_segment_bytes: u64,
        runtime_config: JournalRuntimeConfig,
    ) -> BoxedResult<Self> {
        let journal_root = base_dir.join("journal");
        let streams_dir = journal_root.join("streams");
        fs::create_dir_all(&streams_dir).await?;
        let storage_class_report =
            validate_journal_storage_class(&journal_root, runtime_config.enforce_tmpfs)
                .map_err(|error| -> BoxedError { Box::new(error) })?;

        let journal_epoch = open_or_advance_epoch(&journal_root, &streams_dir).await?;
        let (streams, total_journal_bytes) = recover_stream_states(&streams_dir).await?;
        let lease_store = LeaseStore::new(&journal_root);
        let derived_cache = DerivedCache::new(
            &journal_root,
            DerivedCacheConfig {
                budget_bytes: runtime_config.derived_cache_budget_bytes,
                admission_reserve_bytes: runtime_config.derived_cache_admission_reserve_bytes,
            },
        );
        derived_cache.ensure_initialized().await?;
        let manifest_store = ManifestStore::new(&journal_root);
        manifest_store.ensure_initialized().await?;
        Ok(Self {
            journal_root,
            streams_dir,
            max_segment_bytes,
            runtime_config,
            storage_class_report,
            lease_store,
            derived_cache,
            manifest_store,
            state: Mutex::new(SegmentJournalState {
                journal_epoch,
                total_journal_bytes,
                streams,
            }),
        })
    }

    async fn health(&self) -> BoxedResult<IngestBackendHealth> {
        self.derived_cache.reserve_for_write(0).await?;
        let lease_index = self.lease_store.active_index(now_ns()?).await?;
        Ok(IngestBackendHealth {
            storage_mode: "journal".to_string(),
            journal: Some(JournalHealth {
                journal_root: self.journal_root.clone(),
                storage_class: self.storage_class_report.clone(),
                total_journal_budget_bytes: self.runtime_config.total_journal_budget_bytes,
                admission_reserve_bytes: self.runtime_config.admission_reserve_bytes,
                derived_cache_budget_bytes: self.runtime_config.derived_cache_budget_bytes,
                active_pin_leases: lease_index.leases.len(),
            }),
        })
    }

    async fn create_pin_lease(&self, request: PinLeaseRequest) -> BoxedResult<PinLeaseResponse> {
        let now_ns = now_ns()?;
        let (response, active_index) = self.lease_store.upsert(request, now_ns).await?;
        self.refresh_segment_pin_counts(&active_index, now_ns)
            .await?;
        Ok(response)
    }

    async fn release_pin_lease(&self, lease_id: &str) -> BoxedResult<()> {
        let now_ns = now_ns()?;
        let active_index = self.lease_store.release(lease_id, now_ns).await?;
        self.refresh_segment_pin_counts(&active_index, now_ns)
            .await?;
        Ok(())
    }

    async fn enqueue(
        &self,
        max_body_bytes: usize,
        endpoint: &'static str,
        headers: HeaderMap,
        body: Body,
    ) -> BoxedResult<String> {
        validate_declared_content_length(&headers, max_body_bytes)?;
        let received_ns = now_ns()?;
        let content_type = content_type_from_headers(&headers);
        let body_bytes = read_body_bytes(body, max_body_bytes).await?;
        let capture_envelope = parse_capture_envelope(endpoint, &body_bytes)
            .map_err(InvalidIngestEnvelopeError::new)
            .map_err(|error| -> BoxedError { Box::new(error) })?;

        self.enqueue_channel_only(
            endpoint,
            content_type,
            received_ns,
            &capture_envelope,
            &body_bytes,
        )
        .await
    }

    async fn enqueue_channel_only(
        &self,
        endpoint: &'static str,
        content_type: Option<String>,
        received_ns: u128,
        capture_envelope: &CaptureEnvelope,
        body_bytes: &[u8],
    ) -> BoxedResult<String> {
        let (journal_id, entry) = {
            let mut state = self.state.lock().await;
            let journal_epoch = state.journal_epoch;
            let stream_state = state
                .streams
                .entry(capture_envelope.stream_key.clone())
                .or_insert_with(|| StreamJournalState {
                    next_sequence: 1,
                    open_segment: None,
                });
            if stream_state.next_sequence == 0 {
                stream_state.next_sequence = 1;
            }

            let sequence = stream_state.next_sequence;
            stream_state.next_sequence += 1;

            let journal_id = format!(
                "jrnl-{:020}-{:020}-{}",
                journal_epoch, sequence, capture_envelope.stream_key
            );
            let observation_id = format!(
                "obs-{:020}-{:020}-{}",
                journal_epoch, sequence, capture_envelope.stream_key
            );
            let virtual_segment_id = format!("seg-mem-{:020}-{:020}", journal_epoch, sequence);
            let entry = build_segment_journal_entry(
                capture_envelope,
                endpoint,
                content_type,
                received_ns,
                journal_epoch,
                sequence,
                journal_id.clone(),
                observation_id,
                virtual_segment_id.clone(),
                stable_segment_path(
                    &self.journal_root,
                    &capture_envelope.stream_key,
                    &virtual_segment_id,
                ),
                0,
                0,
            );
            (journal_id, entry)
        };

        let _ = self.publish_capture_manifest(&entry, body_bytes).await;
        Ok(journal_id)
    }

    async fn publish_capture_manifest(
        &self,
        entry: &SegmentJournalEntry,
        raw_bytes: &[u8],
    ) -> BoxedResult<()> {
        let handle = entry.payload_handle();
        let manifest = DspManifest {
            manifest_id: format!("manifest-{}", entry.journal_id),
            manifest_type: "raw_journal_append".to_string(),
            // Raw audio processing must be ordered on firmware packet time.
            // Receipt time stays on the journal entry for freshness/backlog diagnostics.
            created_ns: entry.toa_ns.unwrap_or(entry.ingest_received_ns as u64) as u128,
            source_handles: vec![handle],
            derived_handle: None,
            localization: None,
            classifier_render: None,
            birdnet: None,
            coverage_stats: None,
            promotion_ready: false,
            env_samples: None,
            // raw_payload is #[serde(skip)] — not written to disk.
            node_context: None,
            raw_payload: None,
        };
        // Deliver audio to the DSP worker exclusively via the in-process channel.
        // Never fall back to disk — a channel-send failure means the DSP worker has
        // stopped, and writing to the manifest store would not help it recover.
        if let Some(tx) = &self.runtime_config.raw_manifest_tx {
            let mut manifest_with_payload = manifest;
            manifest_with_payload.raw_payload = Some(raw_bytes.to_vec());
            // Embed node spec and frame metadata so Python can reconstruct node context
            // without relying on any persisted raw audio payload.
            let node_value_opt = if raw_bytes.starts_with(b"MMB1") {
                // Binary MMB1: parse node spec from the binary header.
                crate::envelope::extract_mmb1_node_json(raw_bytes)
            } else {
                // StoreForward JSON: extract the top-level "node" object.
                serde_json::from_slice::<serde_json::Value>(raw_bytes)
                    .ok()
                    .and_then(|body_json| body_json.get("node").cloned())
            };
            if let Some(node_value) = node_value_opt {
                manifest_with_payload.node_context = Some(serde_json::json!({
                    "node": node_value,
                    "toa_ns": entry.toa_ns,
                    "time_quality": entry.time_quality,
                    "source_type": entry.source_type,
                }));
            }
            if tx.send(manifest_with_payload).is_err() {
                warn!(
                    manifest_id = %entry.journal_id,
                    "raw manifest channel receiver unavailable; frame dropped (DSP worker stopped)"
                );
            }
            return Ok(());
        }
        // No channel configured — full journal path (capture pipeline only).
        self.manifest_store.publish(manifest).await?;
        Ok(())
    }

    async fn refresh_segment_pin_counts(
        &self,
        active_index: &crate::leases::PinLeaseIndex,
        now_ns: u128,
    ) -> BoxedResult<()> {
        let counts = segment_pin_counts(active_index, now_ns);
        let hard_counts = hard_segment_pin_counts(active_index, now_ns);
        let mut stream_entries = match fs::read_dir(&self.streams_dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(Box::new(error)),
        };
        while let Some(stream_entry) = stream_entries.next_entry().await? {
            if !stream_entry.file_type().await?.is_dir() {
                continue;
            }
            let stream_key = stream_entry.file_name().to_string_lossy().into_owned();
            let segments_dir = stream_entry.path().join("segments");
            let mut segment_entries = match fs::read_dir(&segments_dir).await {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(Box::new(error)),
            };
            while let Some(segment_entry) = segment_entries.next_entry().await? {
                let metadata_path = segment_entry.path();
                if !metadata_path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.ends_with(".segment.json"))
                {
                    continue;
                }
                let mut header: SegmentJournalHeader =
                    serde_json::from_str(&fs::read_to_string(&metadata_path).await?)?;
                let key = (stream_key.clone(), header.segment_id.clone());
                let count = *counts.get(&key).unwrap_or(&0);
                let hard_count = *hard_counts.get(&key).unwrap_or(&0);
                if header.pin_count != count || header.hard_pin_count != hard_count {
                    header.pin_count = count;
                    header.hard_pin_count = hard_count;
                    write_json_file(&metadata_path, &header).await?;
                }
            }
        }
        Ok(())
    }

    async fn ensure_capacity_for_append(
        &self,
        state: &mut SegmentJournalState,
        append_length_bytes: u64,
    ) -> BoxedResult<()> {
        if self.runtime_config.total_journal_budget_bytes == 0 {
            return Ok(());
        }

        if state
            .total_journal_bytes
            .saturating_add(append_length_bytes)
            .saturating_add(self.runtime_config.admission_reserve_bytes)
            > self.runtime_config.total_journal_budget_bytes
        {
            let now_ns = now_ns()?;
            let active_index = self.lease_store.active_index(now_ns).await?;
            self.refresh_segment_pin_counts(&active_index, now_ns)
                .await?;
        }

        while state
            .total_journal_bytes
            .saturating_add(append_length_bytes)
            .saturating_add(self.runtime_config.admission_reserve_bytes)
            > self.runtime_config.total_journal_budget_bytes
        {
            let cursor_by_stream = load_consumer_cursors(
                self.journal_root.join("consumer_state.sqlite3"),
                self.runtime_config.consumer_name.clone(),
            )
            .await?;
            let mut candidates =
                collect_evictable_segments(&self.streams_dir, &cursor_by_stream).await?;
            candidates.sort_by_key(|candidate| {
                (
                    candidate.header.first_received_ns.unwrap_or(u128::MAX),
                    candidate.header.first_journal_sequence.unwrap_or(u64::MAX),
                )
            });

            let candidate = if let Some(c) = candidates.into_iter().next() {
                c
            } else {
                // No unpinned candidates — try soft-pinned segments as last resort.
                let mut soft_candidates =
                    collect_soft_evictable_segments(&self.streams_dir, &cursor_by_stream).await?;
                soft_candidates.sort_by_key(|c| {
                    (
                        c.header.first_received_ns.unwrap_or(u128::MAX),
                        c.header.first_journal_sequence.unwrap_or(u64::MAX),
                    )
                });
                let Some(c) = soft_candidates.into_iter().next() else {
                    return Err(Box::new(JournalCapacityExceededError {
                        total_journal_budget_bytes: self.runtime_config.total_journal_budget_bytes,
                        admission_reserve_bytes: self.runtime_config.admission_reserve_bytes,
                        projected_total_bytes: state
                            .total_journal_bytes
                            .saturating_add(append_length_bytes),
                    }));
                };
                warn!(
                    segment_id = %c.header.segment_id,
                    soft_pin_count = c.header.pin_count,
                    "evicting soft-pinned segment as last resort under journal capacity pressure"
                );
                c
            };

            evict_segment_files(&candidate).await?;
            state.total_journal_bytes = state
                .total_journal_bytes
                .saturating_sub(candidate.header.payload_bytes);
        }
        Ok(())
    }

    async fn ensure_open_segment(
        &self,
        state: &mut SegmentJournalState,
        capture_envelope: &CaptureEnvelope,
        append_length_bytes: u64,
    ) -> BoxedResult<()> {
        let stream_state = state
            .streams
            .entry(capture_envelope.stream_key.clone())
            .or_insert_with(|| StreamJournalState {
                next_sequence: 1,
                open_segment: None,
            });

        let needs_rotation = match stream_state.open_segment.as_ref() {
            Some(open_segment) => {
                open_segment.size_bytes > 0
                    && open_segment.size_bytes.saturating_add(append_length_bytes)
                        > self.max_segment_bytes
            }
            None => true,
        };

        if needs_rotation {
            if let Some(open_segment) = stream_state.open_segment.as_mut() {
                open_segment.header.sealed = true;
                let _ = write_json_file(&open_segment.metadata_path, &open_segment.header).await;
            }
            stream_state.open_segment = Some(
                self.create_segment(&capture_envelope.stream_key, state.journal_epoch)
                    .await?,
            );
        }
        Ok(())
    }

    async fn create_segment(
        &self,
        stream_key: &str,
        journal_epoch: u64,
    ) -> BoxedResult<OpenSegment> {
        let stream_dir = self.streams_dir.join(stream_key);
        let segments_dir = stream_dir.join("segments");
        fs::create_dir_all(&segments_dir).await?;

        let created_ns = now_ns()?;
        let segment_id = format!("seg-{}-{}", created_ns, Uuid::new_v4());
        let segment_path = segments_dir.join(format!("{}.bin", segment_id));
        let index_path = segments_dir.join(format!("{}.index.jsonl", segment_id));
        let metadata_path = segments_dir.join(format!("{}.segment.json", segment_id));
        fs::File::create(&segment_path).await?;
        fs::File::create(&index_path).await?;

        let header = SegmentJournalHeader {
            segment_id: segment_id.clone(),
            journal_epoch,
            stream_key: stream_key.to_string(),
            first_journal_sequence: None,
            last_journal_sequence: None,
            first_received_ns: None,
            last_received_ns: None,
            first_toa_ns: None,
            last_toa_ns: None,
            first_tor_ns: None,
            last_tor_ns: None,
            entry_count: 0,
            payload_bytes: 0,
            pin_count: 0,
            hard_pin_count: 0,
            sealed: false,
        };
        write_json_file(&metadata_path, &header).await?;

        Ok(OpenSegment {
            segment_id,
            segment_path,
            index_path,
            metadata_path,
            size_bytes: 0,
            index_size_bytes: 0,
            header,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn build_segment_journal_entry(
    capture_envelope: &CaptureEnvelope,
    endpoint: &str,
    content_type: Option<String>,
    received_ns: u128,
    journal_epoch: u64,
    journal_sequence: u64,
    journal_id: String,
    observation_id: String,
    segment_id: String,
    segment_path: PathBuf,
    payload_offset_bytes: u64,
    payload_length_bytes: u64,
) -> SegmentJournalEntry {
    SegmentJournalEntry {
        observation_id,
        journal_id,
        journal_epoch,
        journal_sequence,
        node_id: capture_envelope.node_id.clone(),
        stream_id: capture_envelope.stream_id.clone(),
        stream_key: capture_envelope.stream_key.clone(),
        sensor_type: capture_envelope.sensor_type.clone(),
        source_type: capture_envelope.source_type.clone(),
        transport: capture_envelope.transport.clone(),
        toa_ns: capture_envelope.toa_ns,
        tor_ns: capture_envelope.tor_ns,
        ingest_received_ns: received_ns,
        time_quality: capture_envelope.time_quality.clone(),
        clock_domain: capture_envelope.clock_domain.clone(),
        sync_source: capture_envelope.sync_source.clone(),
        clock_correction_ns: capture_envelope.clock_correction_ns,
        clock_drift_ppm: capture_envelope.clock_drift_ppm,
        sample_rate_hz: capture_envelope.sample_rate_hz,
        channel_count: capture_envelope.channel_count,
        channel_layout: capture_envelope.channel_layout.clone(),
        sample_index_start: capture_envelope.sample_index_start,
        sample_count: capture_envelope.sample_count,
        geometry_version: capture_envelope.geometry_version.clone(),
        orientation_version: capture_envelope.orientation_version.clone(),
        calibration_version: capture_envelope.calibration_version.clone(),
        retention_hint: capture_envelope.retention_hint.clone(),
        payload_codec: capture_envelope.payload_codec.clone(),
        integrity_hash: capture_envelope.integrity_hash.clone(),
        endpoint: endpoint.to_string(),
        content_type,
        segment_id,
        segment_path,
        payload_offset_bytes,
        payload_length_bytes,
        body_offset_bytes: payload_offset_bytes,
        body_length_bytes: payload_length_bytes,
        received_ns,
    }
}

impl SegmentJournalEntry {
    fn payload_handle(&self) -> JournalPayloadHandle {
        JournalPayloadHandle {
            journal_epoch: self.journal_epoch,
            segment_id: self.segment_id.clone(),
            stream_key: self.stream_key.clone(),
            payload_offset_bytes: self.payload_offset_bytes,
            payload_length_bytes: self.payload_length_bytes,
            toa_ns: self.toa_ns,
            tor_ns: self.tor_ns,
            sample_index_start: self.sample_index_start,
            sample_count: self.sample_count,
            integrity_hash: self.integrity_hash.clone(),
            segment_path: self.segment_path.clone(),
        }
    }
}

fn update_segment_header(
    header: &mut SegmentJournalHeader,
    entry: &SegmentJournalEntry,
    payload_bytes: u64,
) {
    if header.first_journal_sequence.is_none() {
        header.first_journal_sequence = Some(entry.journal_sequence);
        header.first_received_ns = Some(entry.ingest_received_ns);
        header.first_toa_ns = entry.toa_ns;
        header.first_tor_ns = entry.tor_ns;
    }
    header.last_journal_sequence = Some(entry.journal_sequence);
    header.last_received_ns = Some(entry.ingest_received_ns);
    header.last_toa_ns = entry.toa_ns.or(header.last_toa_ns);
    header.last_tor_ns = entry.tor_ns.or(header.last_tor_ns);
    header.entry_count += 1;
    header.payload_bytes = payload_bytes;
}

async fn read_body_bytes(body: Body, max_body_bytes: usize) -> BoxedResult<Vec<u8>> {
    let mut body = body;
    let mut bytes = Vec::new();
    while let Some(frame_result) = body.frame().await {
        let frame = match frame_result {
            Ok(frame) => frame,
            Err(error) => {
                let boxed: BoxedError = Box::new(error);
                if is_client_disconnect(boxed.as_ref()) {
                    return Err(Box::new(ClientDisconnectError));
                }
                return Err(boxed);
            }
        };
        if let Some(chunk) = frame.data_ref() {
            if bytes.len().saturating_add(chunk.len()) > max_body_bytes {
                return Err(Box::new(BodyTooLargeError {
                    limit_bytes: max_body_bytes,
                }));
            }
            bytes.extend_from_slice(chunk);
        }
    }
    Ok(bytes)
}

async fn open_or_advance_epoch(journal_root: &Path, streams_dir: &Path) -> BoxedResult<u64> {
    let state_path = journal_root.join("epoch.json");
    let existing_state = match fs::read_to_string(&state_path).await {
        Ok(contents) => Some(serde_json::from_str::<JournalEpochState>(&contents)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(Box::new(error)),
    };

    let has_live_segments = streams_have_live_segments(streams_dir).await?;
    let epoch_state = if has_live_segments {
        existing_state.unwrap_or(JournalEpochState {
            active_epoch: 1,
            last_assigned_epoch: 1,
        })
    } else {
        let next_epoch = existing_state
            .as_ref()
            .map(|state| state.last_assigned_epoch.saturating_add(1))
            .unwrap_or(1);
        JournalEpochState {
            active_epoch: next_epoch,
            last_assigned_epoch: next_epoch,
        }
    };
    write_json_file(&state_path, &epoch_state).await?;
    Ok(epoch_state.active_epoch)
}

async fn streams_have_live_segments(streams_dir: &Path) -> std::io::Result<bool> {
    let mut stream_entries = fs::read_dir(streams_dir).await?;
    while let Some(stream_entry) = stream_entries.next_entry().await? {
        if !stream_entry.file_type().await?.is_dir() {
            continue;
        }
        let segments_dir = stream_entry.path().join("segments");
        let mut segment_entries = match fs::read_dir(&segments_dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        while let Some(segment_entry) = segment_entries.next_entry().await? {
            if segment_entry.path().extension().and_then(OsStr::to_str) == Some("bin") {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

async fn recover_stream_states(
    streams_dir: &Path,
) -> BoxedResult<(HashMap<String, StreamJournalState>, u64)> {
    let mut states = HashMap::new();
    let mut total_journal_bytes = 0_u64;
    let mut stream_entries = match fs::read_dir(streams_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((states, total_journal_bytes))
        }
        Err(error) => return Err(Box::new(error)),
    };

    while let Some(stream_entry) = stream_entries.next_entry().await? {
        if !stream_entry.file_type().await?.is_dir() {
            continue;
        }
        let stream_key = stream_entry.file_name().to_string_lossy().into_owned();
        let (stream_state, recovered_stream_bytes) =
            recover_single_stream_state(stream_entry.path(), &stream_key).await?;
        states.insert(stream_key, stream_state);
        total_journal_bytes = total_journal_bytes.saturating_add(recovered_stream_bytes);
    }
    Ok((states, total_journal_bytes))
}

async fn recover_single_stream_state(
    stream_dir: PathBuf,
    stream_key: &str,
) -> BoxedResult<(StreamJournalState, u64)> {
    let segments_dir = stream_dir.join("segments");
    let mut segment_entries = match fs::read_dir(&segments_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((
                StreamJournalState {
                    next_sequence: 1,
                    open_segment: None,
                },
                0,
            ));
        }
        Err(error) => return Err(Box::new(error)),
    };

    let mut next_sequence = 1_u64;
    let mut open_segment: Option<OpenSegment> = None;
    let mut recovered_stream_bytes = 0_u64;
    while let Some(segment_entry) = segment_entries.next_entry().await? {
        let index_path = segment_entry.path();
        if index_path.extension().and_then(OsStr::to_str) != Some("jsonl") {
            continue;
        }
        let recovered_segment = recover_segment(index_path, stream_key).await?;
        recovered_stream_bytes =
            recovered_stream_bytes.saturating_add(recovered_segment.header.payload_bytes);
        next_sequence = next_sequence.max(
            recovered_segment
                .header
                .last_journal_sequence
                .unwrap_or(0)
                .saturating_add(1),
        );

        if !recovered_segment.header.sealed {
            let should_replace = match open_segment.as_ref() {
                Some(existing_open_segment) => {
                    recovered_segment.header.last_journal_sequence.unwrap_or(0)
                        > existing_open_segment
                            .header
                            .last_journal_sequence
                            .unwrap_or(0)
                }
                None => true,
            };
            if should_replace {
                let index_size_bytes = fs::metadata(&recovered_segment.index_path)
                    .await
                    .map(|m| m.len())
                    .unwrap_or(0);
                open_segment = Some(OpenSegment {
                    segment_id: recovered_segment.header.segment_id.clone(),
                    segment_path: recovered_segment.segment_path,
                    index_path: recovered_segment.index_path,
                    metadata_path: recovered_segment.metadata_path,
                    size_bytes: recovered_segment.header.payload_bytes,
                    index_size_bytes,
                    header: recovered_segment.header,
                });
            }
        }
    }

    Ok((
        StreamJournalState {
            next_sequence,
            open_segment,
        },
        recovered_stream_bytes,
    ))
}

async fn load_consumer_cursors(
    cursor_state_path: PathBuf,
    consumer_name: String,
) -> BoxedResult<HashMap<String, ConsumerCursorWatermark>> {
    tokio::task::spawn_blocking(
        move || -> BoxedResult<HashMap<String, ConsumerCursorWatermark>> {
            if !cursor_state_path.exists() {
                return Ok(HashMap::new());
            }
            let connection = Connection::open(cursor_state_path)?;
            let mut statement = connection.prepare(
                r#"
            SELECT stream_key, journal_epoch, last_fully_processed_journal_sequence
            FROM consumer_cursors
            WHERE consumer_name = ?1
            "#,
            )?;
            let mut rows = statement.query([consumer_name])?;
            let mut cursor_by_stream = HashMap::new();
            while let Some(row) = rows.next()? {
                cursor_by_stream.insert(
                    row.get::<_, String>(0)?,
                    ConsumerCursorWatermark {
                        journal_epoch: row.get::<_, u64>(1)?,
                        last_fully_processed_journal_sequence: row.get::<_, u64>(2)?,
                    },
                );
            }
            Ok(cursor_by_stream)
        },
    )
    .await
    .map_err(|error| -> BoxedError { Box::new(error) })?
}

async fn collect_evictable_segments(
    streams_dir: &Path,
    cursor_by_stream: &HashMap<String, ConsumerCursorWatermark>,
) -> BoxedResult<Vec<EvictableSegment>> {
    let mut candidates = Vec::new();
    let mut stream_entries = match fs::read_dir(streams_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(candidates),
        Err(error) => return Err(Box::new(error)),
    };

    while let Some(stream_entry) = stream_entries.next_entry().await? {
        if !stream_entry.file_type().await?.is_dir() {
            continue;
        }
        let stream_key = stream_entry.file_name().to_string_lossy().into_owned();
        let Some(cursor) = cursor_by_stream.get(&stream_key) else {
            continue;
        };

        let segments_dir = stream_entry.path().join("segments");
        let mut segment_entries = match fs::read_dir(&segments_dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(Box::new(error)),
        };
        while let Some(segment_entry) = segment_entries.next_entry().await? {
            let metadata_path = segment_entry.path();
            if !metadata_path
                .file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.ends_with(".segment.json"))
            {
                continue;
            }
            let header: SegmentJournalHeader =
                serde_json::from_str(&fs::read_to_string(&metadata_path).await?)?;
            if !header.sealed || header.pin_count > 0 || !cursor_covers_segment(*cursor, &header) {
                continue;
            }
            candidates.push(EvictableSegment {
                segment_path: metadata_path.with_file_name(format!("{}.bin", header.segment_id)),
                index_path: metadata_path
                    .with_file_name(format!("{}.index.jsonl", header.segment_id)),
                metadata_path,
                header,
            });
        }
    }

    Ok(candidates)
}

/// Like `collect_evictable_segments` but treats soft-pinned segments as
/// evictable.  Used as a last-resort when the budget is exhausted and no
/// unpinned candidates remain.
async fn collect_soft_evictable_segments(
    streams_dir: &Path,
    cursor_by_stream: &HashMap<String, ConsumerCursorWatermark>,
) -> BoxedResult<Vec<EvictableSegment>> {
    let mut candidates = Vec::new();
    let mut stream_entries = match fs::read_dir(streams_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(candidates),
        Err(error) => return Err(Box::new(error)),
    };
    while let Some(stream_entry) = stream_entries.next_entry().await? {
        if !stream_entry.file_type().await?.is_dir() {
            continue;
        }
        let stream_key = stream_entry.file_name().to_string_lossy().into_owned();
        let Some(cursor) = cursor_by_stream.get(&stream_key) else {
            continue;
        };
        let segments_dir = stream_entry.path().join("segments");
        let mut segment_entries = match fs::read_dir(&segments_dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(Box::new(error)),
        };
        while let Some(segment_entry) = segment_entries.next_entry().await? {
            let metadata_path = segment_entry.path();
            if !metadata_path
                .file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.ends_with(".segment.json"))
            {
                continue;
            }
            let header: SegmentJournalHeader =
                serde_json::from_str(&fs::read_to_string(&metadata_path).await?)?;
            // Only hard pins are an absolute block; soft-only pinned segments are eligible here.
            if !header.sealed
                || header.hard_pin_count > 0
                || !cursor_covers_segment(*cursor, &header)
            {
                continue;
            }
            candidates.push(EvictableSegment {
                segment_path: metadata_path.with_file_name(format!("{}.bin", header.segment_id)),
                index_path: metadata_path
                    .with_file_name(format!("{}.index.jsonl", header.segment_id)),
                metadata_path,
                header,
            });
        }
    }
    Ok(candidates)
}

fn cursor_covers_segment(cursor: ConsumerCursorWatermark, header: &SegmentJournalHeader) -> bool {
    let Some(last_journal_sequence) = header.last_journal_sequence else {
        return false;
    };
    if cursor.journal_epoch != header.journal_epoch {
        return cursor.journal_epoch > header.journal_epoch;
    }
    cursor.last_fully_processed_journal_sequence >= last_journal_sequence
}

async fn evict_segment_files(candidate: &EvictableSegment) -> BoxedResult<()> {
    let _ = fs::remove_file(&candidate.segment_path).await;
    let _ = fs::remove_file(&candidate.index_path).await;
    let _ = fs::remove_file(&candidate.metadata_path).await;
    Ok(())
}

async fn recover_segment(index_path: PathBuf, stream_key: &str) -> BoxedResult<RecoveredSegment> {
    let index_name = index_path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| {
            InvalidIngestEnvelopeError::new("segment index filename was not valid utf-8")
        })?;
    let segment_id = index_name
        .strip_suffix(".index.jsonl")
        .ok_or_else(|| {
            InvalidIngestEnvelopeError::new("segment index filename did not end with .index.jsonl")
        })?
        .to_string();
    let segment_path = index_path.with_file_name(format!("{}.bin", segment_id));
    let metadata_path = index_path.with_file_name(format!("{}.segment.json", segment_id));

    let existing_header = match fs::read_to_string(&metadata_path).await {
        Ok(contents) => Some(serde_json::from_str::<SegmentJournalHeader>(&contents)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(Box::new(error)),
    };

    let index_bytes = fs::read(&index_path).await?;
    let mut derived_journal_epoch = existing_header
        .as_ref()
        .map(|header| header.journal_epoch)
        .unwrap_or(0);
    let mut first_sequence = None;
    let mut last_sequence = None;
    let mut first_received_ns = None;
    let mut last_received_ns = None;
    let mut first_toa_ns = None;
    let mut last_toa_ns = None;
    let mut first_tor_ns = None;
    let mut last_tor_ns = None;
    let mut entry_count = 0_u64;
    let mut payload_bytes = 0_u64;
    let mut committed_index_bytes = 0_u64;

    let mut observe_entry = |entry: &SegmentJournalEntry| {
        derived_journal_epoch = derived_journal_epoch.max(entry.journal_epoch);
        if first_sequence.is_none() {
            first_sequence = Some(entry.journal_sequence);
            first_received_ns = Some(entry.ingest_received_ns);
            first_toa_ns = entry.toa_ns;
            first_tor_ns = entry.tor_ns;
        }
        last_sequence = Some(entry.journal_sequence);
        last_received_ns = Some(entry.ingest_received_ns);
        last_toa_ns = entry.toa_ns.or(last_toa_ns);
        last_tor_ns = entry.tor_ns.or(last_tor_ns);
        entry_count += 1;
        payload_bytes = payload_bytes.max(
            entry
                .payload_offset_bytes
                .saturating_add(entry.payload_length_bytes),
        );
    };

    let mut cursor = 0_usize;
    while cursor < index_bytes.len() {
        let Some(relative_newline_index) =
            index_bytes[cursor..].iter().position(|byte| *byte == b'\n')
        else {
            break;
        };
        let line_end = cursor + relative_newline_index;
        let line_bytes = &index_bytes[cursor..line_end];
        cursor = line_end + 1;
        committed_index_bytes = u64::try_from(cursor)?;
        if line_bytes.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let entry: SegmentJournalEntry = serde_json::from_slice(line_bytes)?;
        observe_entry(&entry);
    }

    let trailing_bytes = &index_bytes[cursor..];
    let trailing_bytes_are_whitespace =
        trailing_bytes.iter().all(|byte| byte.is_ascii_whitespace());
    let mut append_trailing_newline = false;
    if !trailing_bytes.is_empty() && !trailing_bytes_are_whitespace {
        match serde_json::from_slice::<SegmentJournalEntry>(trailing_bytes) {
            Ok(entry) => {
                observe_entry(&entry);
                committed_index_bytes = u64::try_from(index_bytes.len())?;
                append_trailing_newline = true;
            }
            Err(_) => {
                // Crash recovery only trusts newline-delimited records plus an
                // otherwise complete final record. Any other tail bytes are
                // treated as a torn append and truncated below.
            }
        }
    } else if trailing_bytes_are_whitespace {
        committed_index_bytes = u64::try_from(cursor)?;
    }

    if append_trailing_newline {
        let mut index_file = OpenOptions::new().write(true).open(&index_path).await?;
        index_file.seek(std::io::SeekFrom::End(0)).await?;
        index_file.write_all(b"\n").await?;
        index_file.flush().await?;
    } else if committed_index_bytes < u64::try_from(index_bytes.len())? {
        let mut index_file = OpenOptions::new().write(true).open(&index_path).await?;
        index_file.set_len(committed_index_bytes).await?;
        index_file.flush().await?;
    }

    let mut header = existing_header.unwrap_or(SegmentJournalHeader {
        segment_id: segment_id.clone(),
        journal_epoch: derived_journal_epoch,
        stream_key: stream_key.to_string(),
        first_journal_sequence: first_sequence,
        last_journal_sequence: last_sequence,
        first_received_ns,
        last_received_ns,
        first_toa_ns,
        last_toa_ns,
        first_tor_ns,
        last_tor_ns,
        entry_count,
        payload_bytes,
        pin_count: 0,
        hard_pin_count: 0,
        sealed: false,
    });

    header.journal_epoch = derived_journal_epoch.max(header.journal_epoch);
    header.stream_key = stream_key.to_string();
    header.first_journal_sequence = first_sequence.or(header.first_journal_sequence);
    header.last_journal_sequence = last_sequence.or(header.last_journal_sequence);
    header.first_received_ns = first_received_ns.or(header.first_received_ns);
    header.last_received_ns = last_received_ns.or(header.last_received_ns);
    header.first_toa_ns = first_toa_ns.or(header.first_toa_ns);
    header.last_toa_ns = last_toa_ns.or(header.last_toa_ns);
    header.first_tor_ns = first_tor_ns.or(header.first_tor_ns);
    header.last_tor_ns = last_tor_ns.or(header.last_tor_ns);
    header.entry_count = entry_count.max(header.entry_count);
    header.payload_bytes = payload_bytes.max(header.payload_bytes);

    let segment_metadata = fs::metadata(&segment_path).await?;
    if segment_metadata.len() > header.payload_bytes {
        let segment_file = OpenOptions::new().write(true).open(&segment_path).await?;
        segment_file.set_len(header.payload_bytes).await?;
    }
    let _ = write_json_file(&metadata_path, &header).await;

    Ok(RecoveredSegment {
        segment_path,
        index_path,
        metadata_path,
        header,
    })
}

async fn rollback_segment_append(
    segment_file: &mut fs::File,
    body_offset_bytes: u64,
) -> BoxedResult<()> {
    segment_file.set_len(body_offset_bytes).await?;
    segment_file.flush().await?;
    Ok(())
}

#[allow(dead_code)]
async fn rollback_index_append(
    index_file: &mut fs::File,
    index_offset_bytes: u64,
) -> BoxedResult<()> {
    index_file.set_len(index_offset_bytes).await?;
    index_file.flush().await?;
    Ok(())
}

fn validate_declared_content_length(headers: &HeaderMap, max_body_bytes: usize) -> BoxedResult<()> {
    if let Some(content_length_bytes) = declared_content_length_bytes(headers) {
        if content_length_bytes > max_body_bytes {
            return Err(Box::new(BodyTooLargeError {
                limit_bytes: max_body_bytes,
            }));
        }
    }
    Ok(())
}

fn declared_content_length_bytes(headers: &HeaderMap) -> Option<usize> {
    headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
}

fn content_type_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

async fn write_json_file<T: Serialize>(path: &Path, value: &T) -> BoxedResult<()> {
    let tmp_path = temporary_path_for(path)?;
    let bytes = serde_json::to_vec(value)?;
    fs::write(&tmp_path, bytes).await?;
    fs::rename(&tmp_path, path).await?;
    Ok(())
}

fn temporary_path_for(path: &Path) -> BoxedResult<PathBuf> {
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| InvalidIngestEnvelopeError::new("path filename was not valid utf-8"))?;
    Ok(path.with_file_name(format!(".{file_name}.tmp")))
}

async fn fsync_dir(path: &Path) -> std::io::Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || std::fs::File::open(path).and_then(|file| file.sync_all()))
        .await
        .map_err(std::io::Error::other)?
}

fn now_ns() -> Result<u128, std::time::SystemTimeError> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())
}

fn is_client_disconnect(err: &dyn std::error::Error) -> bool {
    let mut cursor: Option<&dyn std::error::Error> = Some(err);
    while let Some(error) = cursor {
        let description = error.to_string().to_lowercase();
        if description.contains("connection reset")
            || description.contains("broken pipe")
            || description.contains("connection aborted")
            || description.contains("incomplete message")
        {
            return true;
        }
        cursor = error.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifests::ManifestStore;
    use rusqlite::params;
    use serde_json::json;

    fn store_forward_body(node_id: &str, sequence: u64, start_sample_index: u64) -> String {
        json!({
            "node": {
                "id": node_id,
                "node_type": "point",
                "position_m": [0.0, 0.0, 0.0],
                "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                "capabilities": ["audio"],
                "metadata": {},
                "properties": {}
            },
            "buffered_frames": [
                {
                    "frame": {
                        "start_time_ns": 1000 + sequence,
                        "utc_start_ns": 1000 + sequence,
                        "utc_end_ns": 2000 + sequence,
                        "start_sample_index": start_sample_index,
                        "end_sample_index": start_sample_index + 4,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "pcm16le",
                        "samples_per_channel": 4,
                        "samples_b64": "AA==",
                        "sequence": sequence,
                        "time_quality": "gps_locked",
                        "toa_ns": 1000 + sequence,
                        "tor_ns": 2000 + sequence,
                        "source_type": "raw_sensor"
                    }
                }
            ],
            "sort_by_toa": true
        })
        .to_string()
    }

    async fn read_single_index_entry(stream_dir: &Path) -> SegmentJournalEntry {
        let segments_dir = stream_dir.join("segments");
        let mut entries = fs::read_dir(&segments_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();
            if path.extension().and_then(OsStr::to_str) == Some("jsonl") {
                let contents = fs::read_to_string(path).await.unwrap();
                let line = contents
                    .lines()
                    .find(|line| !line.trim().is_empty())
                    .unwrap();
                return serde_json::from_str(line).unwrap();
            }
        }
        panic!("missing index entry");
    }

    async fn read_segment_headers(stream_dir: &Path) -> Vec<SegmentJournalHeader> {
        let segments_dir = stream_dir.join("segments");
        let mut headers = Vec::new();
        let mut entries = fs::read_dir(&segments_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();
            if !path
                .file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.ends_with(".segment.json"))
            {
                continue;
            }
            headers.push(serde_json::from_str(&fs::read_to_string(path).await.unwrap()).unwrap());
        }
        headers
    }

    fn test_journal_runtime_config() -> JournalRuntimeConfig {
        JournalRuntimeConfig {
            consumer_name: "python-ingest".to_string(),
            total_journal_budget_bytes: 268_435_456,
            admission_reserve_bytes: 16_777_216,
            enforce_tmpfs: false,
            derived_cache_budget_bytes: 67_108_864,
            derived_cache_admission_reserve_bytes: 8_388_608,
            raw_manifest_tx: None,
        }
    }

    fn write_cursor(
        journal_root: &Path,
        consumer_name: &str,
        stream_key: &str,
        journal_epoch: u64,
        journal_sequence: u64,
    ) {
        let db_path = journal_root.join("consumer_state.sqlite3");
        let connection = Connection::open(db_path).unwrap();
        connection
            .execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS consumer_cursors (
                    consumer_name TEXT NOT NULL,
                    stream_key TEXT NOT NULL,
                    journal_epoch INTEGER NOT NULL,
                    last_fully_processed_journal_sequence INTEGER NOT NULL,
                    updated_ns INTEGER NOT NULL,
                    PRIMARY KEY (consumer_name, stream_key)
                );
                "#,
            )
            .unwrap();
        connection
            .execute(
                r#"
                INSERT OR REPLACE INTO consumer_cursors (
                    consumer_name,
                    stream_key,
                    journal_epoch,
                    last_fully_processed_journal_sequence,
                    updated_ns
                )
                VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
                params![
                    consumer_name,
                    stream_key,
                    journal_epoch,
                    journal_sequence,
                    123_i64
                ],
            )
            .unwrap();
    }

    #[tokio::test]
    async fn raw_capture_manifest_prefers_packet_toa_for_created_time() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body("node-a", 7, 0);
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();

        backend
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                HeaderMap::new(),
                Body::from(payload),
            )
            .await
            .unwrap();

        let manifest_store = ManifestStore::new(&tmp.path().join("store").join("journal"));
        let manifests = manifest_store
            .query_pending("raw_journal_append")
            .await
            .unwrap();

        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].created_ns, 1007);
        assert_eq!(manifests[0].source_handles[0].toa_ns, Some(1007));
        assert_eq!(manifests[0].source_handles[0].tor_ns, Some(2007));
    }

    #[tokio::test]
    async fn journal_recovers_next_sequence_after_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");

        let backend = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            65_536,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        backend
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                HeaderMap::new(),
                Body::from(store_forward_body("node-a", 1, 0)),
            )
            .await
            .unwrap();
        drop(backend);

        let reopened = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            65_536,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        let journal_id = reopened
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                HeaderMap::new(),
                Body::from(store_forward_body("node-a", 2, 4)),
            )
            .await
            .unwrap();

        assert!(journal_id.contains("-00000000000000000002-"));
    }

    fn read_all_entries(stream_dir: &std::path::Path) -> Vec<SegmentJournalEntry> {
        let segments_dir = stream_dir.join("segments");
        let mut all = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&segments_dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(std::ffi::OsStr::to_str) != Some("jsonl") {
                    continue;
                }
                if let Ok(contents) = std::fs::read_to_string(&p) {
                    for line in contents.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(je) = serde_json::from_str::<SegmentJournalEntry>(line) {
                            all.push(je);
                        }
                    }
                }
            }
        }
        all.sort_by_key(|e| e.journal_sequence);
        all
    }

    /// A hard-pinned segment is never evicted; the adjacent unpinned sealed segment
    /// is evicted to make room for a new frame.
    ///
    /// Setup: segment_max_bytes = 1 forces every append to seal the previous segment.
    /// After 3 frames: seg1 (seq 1, sealed) + seg2 (seq 2, sealed) + seg3 (seq 3, open).
    /// Cursor covers all 3.  Hard-pin seg1 → only seg2 is an unpinned eviction candidate.
    #[tokio::test]
    async fn manifest_publish_and_query_pending_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            8_388_608,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        let manifest_store = backend.manifest_store().expect("journal mode required");

        let manifest = crate::manifests::DspManifest {
            manifest_id: String::new(),
            manifest_type: "raw_journal_append".to_string(),
            created_ns: 12345,
            source_handles: vec![],
            derived_handle: None,
            localization: None,
            classifier_render: None,
            birdnet: None,
            coverage_stats: None,
            promotion_ready: true,
            env_samples: None,
            node_context: None,
            raw_payload: None,
        };
        let path = manifest_store.publish(manifest.clone()).await.unwrap();
        assert!(path.exists(), "manifest file should exist after publish");

        let pending = manifest_store
            .query_pending("raw_journal_append")
            .await
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].promotion_ready, true);
    }

    /// mark_consumed renames the manifest file so it no longer appears in
    /// query_pending results.
    #[tokio::test]
    async fn mark_consumed_removes_manifest_from_pending() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            8_388_608,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        let manifest_store = backend.manifest_store().expect("journal mode required");

        let path = manifest_store
            .publish(crate::manifests::DspManifest {
                manifest_id: "test-manifest-1".to_string(),
                manifest_type: "raw_journal_append".to_string(),
                created_ns: 1,
                source_handles: vec![],
                derived_handle: None,
                localization: None,
                classifier_render: None,
                birdnet: None,
                coverage_stats: None,
                promotion_ready: false,
                env_samples: None,
                node_context: None,
                raw_payload: None,
            })
            .await
            .unwrap();

        manifest_store
            .mark_consumed("test-manifest-1")
            .await
            .unwrap();
        assert!(!path.exists(), "original manifest file should be renamed");
        let consumed_path = path
            .parent()
            .and_then(|parent| parent.parent())
            .expect("pending manifest path should have root parent")
            .join("consumed")
            .join(path.file_name().expect("manifest file name should exist"));
        assert!(consumed_path.exists(), "consumed file should exist");

        let pending = manifest_store
            .query_pending("raw_journal_append")
            .await
            .unwrap();
        assert_eq!(
            pending.len(),
            0,
            "consumed manifest should not appear in pending"
        );
    }

    #[tokio::test]
    async fn mark_consumed_matches_manifest_id_when_filename_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            8_388_608,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        let manifest_store = backend.manifest_store().expect("journal mode required");

        manifest_store.ensure_initialized().await.unwrap();

        let manifest = crate::manifests::DspManifest {
            manifest_id: "manifest-xyz".to_string(),
            manifest_type: "raw_journal_append".to_string(),
            created_ns: 1,
            source_handles: vec![],
            derived_handle: None,
            localization: None,
            classifier_render: None,
            birdnet: None,
            coverage_stats: None,
            promotion_ready: false,
            env_samples: None,
            node_context: None,
            raw_payload: None,
        };
        let pending_file = tmp
            .path()
            .join("store")
            .join("journal")
            .join("manifests")
            .join("pending")
            .join("1700000000-node1.json");
        tokio::fs::write(&pending_file, serde_json::to_vec(&manifest).unwrap())
            .await
            .unwrap();

        manifest_store.mark_consumed("manifest-xyz").await.unwrap();
        assert!(
            !pending_file.exists(),
            "pending manifest should be moved out of pending directory"
        );
        let consumed_file = tmp
            .path()
            .join("store")
            .join("journal")
            .join("manifests")
            .join("consumed")
            .join("1700000000-node1.json");
        assert!(
            consumed_file.exists(),
            "consumed manifest should preserve file name"
        );

        let pending = manifest_store
            .query_pending("raw_journal_append")
            .await
            .unwrap();
        assert_eq!(
            pending.len(),
            0,
            "consumed manifest should not appear in pending"
        );
    }

    #[tokio::test]
    async fn query_pending_limited_respects_batch_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            8_388_608,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        let manifest_store = backend.manifest_store().expect("journal mode required");

        for index in 0..4_u64 {
            manifest_store
                .publish(crate::manifests::DspManifest {
                    manifest_id: format!("test-manifest-{index}"),
                    manifest_type: "raw_journal_append".to_string(),
                    created_ns: index as u128,
                    source_handles: vec![],
                    derived_handle: None,
                    localization: None,
                    classifier_render: None,
                    birdnet: None,
                    coverage_stats: None,
                    promotion_ready: false,
                    env_samples: None,
                    node_context: None,
                    raw_payload: None,
                })
                .await
                .unwrap();
        }

        let pending = manifest_store
            .query_pending_limited("raw_journal_append", 2)
            .await
            .unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].created_ns, 0);
        assert_eq!(pending[1].created_ns, 1);
    }

    #[tokio::test]
    async fn prune_consumed_manifests_keeps_latest_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            8_388_608,
            test_journal_runtime_config(),
        )
        .await
        .unwrap();
        let manifest_store = backend.manifest_store().expect("journal mode required");

        for index in 0..5_u64 {
            let manifest_id = format!("test-manifest-prune-{index}");
            manifest_store
                .publish(crate::manifests::DspManifest {
                    manifest_id: manifest_id.clone(),
                    manifest_type: "raw_journal_append".to_string(),
                    created_ns: index as u128,
                    source_handles: vec![],
                    derived_handle: None,
                    localization: None,
                    classifier_render: None,
                    birdnet: None,
                    coverage_stats: None,
                    promotion_ready: false,
                    env_samples: None,
                    node_context: None,
                    raw_payload: None,
                })
                .await
                .unwrap();
            manifest_store.mark_consumed(&manifest_id).await.unwrap();
        }

        manifest_store.prune_consumed_manifests(2).await.unwrap();

        let consumed_dir = tmp
            .path()
            .join("store")
            .join("journal")
            .join("manifests")
            .join("consumed");
        let mut consumed_files = Vec::new();
        let mut entries = tokio::fs::read_dir(&consumed_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let file_type = entry.file_type().await.unwrap();
            if file_type.is_file() {
                consumed_files.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        consumed_files.sort();
        assert_eq!(consumed_files.len(), 2);
        assert_eq!(consumed_files[0], "test-manifest-prune-3.json");
        assert_eq!(consumed_files[1], "test-manifest-prune-4.json");
    }
}
