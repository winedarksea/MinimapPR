use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{header, HeaderMap},
};
use http_body_util::BodyExt;
use serde::{Deserialize, Serialize};
use tokio::{
    fs,
    sync::{mpsc, Mutex},
};

use crate::derived_cache::{DerivedCache, DerivedCacheConfig};
use crate::dsp::coverage_stats;
use crate::envelope::{parse_capture_envelope, CaptureEnvelope};
use crate::journal_reader::{stable_segment_path, JournalPayloadHandle};
use crate::leases::{LeaseStore, PinLeaseRequest, PinLeaseResponse};
use crate::manifests::{DspManifest, ManifestStore};
use crate::storage_class::{validate_journal_storage_class, StorageClassReport};

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug)]
pub struct JournalRuntimeConfig {
    pub total_journal_budget_bytes: u64,
    pub admission_reserve_bytes: u64,
    pub enforce_tmpfs: bool,
    pub memory_only_live_path: bool,
    pub derived_cache_budget_bytes: u64,
    pub derived_cache_admission_reserve_bytes: u64,
    pub raw_manifest_channel_capacity: usize,
    pub raw_manifest_queue_backpressure: Option<Arc<RawManifestQueueBackpressure>>,
    /// Optional in-process channel for zero-latency delivery of raw_journal_append
    /// manifests to the DspWorker, bypassing the 20ms filesystem polling cycle.
    /// Bounded so overload is surfaced immediately to the ingest client instead
    /// of accumulating raw audio payloads unbounded in heap memory.
    pub raw_manifest_tx: Option<mpsc::Sender<QueuedRawManifest>>,
}

impl Default for JournalRuntimeConfig {
    fn default() -> Self {
        Self {
            total_journal_budget_bytes: 268_435_456,
            admission_reserve_bytes: 16_777_216,
            enforce_tmpfs: true,
            memory_only_live_path: false,
            derived_cache_budget_bytes: 67_108_864,
            derived_cache_admission_reserve_bytes: 8_388_608,
            raw_manifest_channel_capacity: 2048,
            raw_manifest_queue_backpressure: None,
            raw_manifest_tx: None,
        }
    }
}

#[derive(Debug)]
pub struct RawManifestQueueBackpressure {
    queued_messages: AtomicUsize,
    queued_bytes: AtomicUsize,
    byte_limit: usize,
}

impl RawManifestQueueBackpressure {
    pub fn new(byte_limit: usize) -> Self {
        Self {
            queued_messages: AtomicUsize::new(0),
            queued_bytes: AtomicUsize::new(0),
            byte_limit,
        }
    }

    pub fn byte_limit(&self) -> usize {
        self.byte_limit
    }

    pub fn queued_messages(&self) -> usize {
        self.queued_messages.load(Ordering::Relaxed)
    }

    pub fn queued_bytes(&self) -> usize {
        self.queued_bytes.load(Ordering::Relaxed)
    }

    fn try_reserve(
        self: &Arc<Self>,
        requested_bytes: usize,
    ) -> Result<RawManifestQueuePermit, RawManifestQueueBytesExceededError> {
        loop {
            let current_queued_bytes = self.queued_bytes.load(Ordering::Relaxed);
            let projected_queued_bytes = current_queued_bytes.saturating_add(requested_bytes);
            if projected_queued_bytes > self.byte_limit {
                return Err(RawManifestQueueBytesExceededError {
                    byte_limit: self.byte_limit,
                    current_queued_bytes,
                    requested_bytes,
                });
            }
            if self
                .queued_bytes
                .compare_exchange(
                    current_queued_bytes,
                    projected_queued_bytes,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                self.queued_messages.fetch_add(1, Ordering::AcqRel);
                return Ok(RawManifestQueuePermit {
                    tracker: Arc::clone(self),
                    queued_bytes: requested_bytes,
                    released: false,
                });
            }
        }
    }

    fn release(&self, queued_bytes: usize) {
        self.queued_messages.fetch_sub(1, Ordering::AcqRel);
        self.queued_bytes.fetch_sub(queued_bytes, Ordering::AcqRel);
    }
}

#[derive(Debug)]
struct RawManifestQueuePermit {
    tracker: Arc<RawManifestQueueBackpressure>,
    queued_bytes: usize,
    released: bool,
}

impl RawManifestQueuePermit {
    fn release(&mut self) {
        if self.released {
            return;
        }
        self.tracker.release(self.queued_bytes);
        self.released = true;
    }

    fn try_resize(
        &mut self,
        requested_bytes: usize,
    ) -> Result<(), RawManifestQueueBytesExceededError> {
        if self.released || requested_bytes == self.queued_bytes {
            return Ok(());
        }
        if requested_bytes < self.queued_bytes {
            let released_bytes = self.queued_bytes - requested_bytes;
            self.tracker
                .queued_bytes
                .fetch_sub(released_bytes, Ordering::AcqRel);
            self.queued_bytes = requested_bytes;
            return Ok(());
        }

        let additional_bytes = requested_bytes - self.queued_bytes;
        loop {
            let current_queued_bytes = self.tracker.queued_bytes.load(Ordering::Relaxed);
            let projected_queued_bytes = current_queued_bytes.saturating_add(additional_bytes);
            if projected_queued_bytes > self.tracker.byte_limit {
                return Err(RawManifestQueueBytesExceededError {
                    byte_limit: self.tracker.byte_limit,
                    current_queued_bytes,
                    requested_bytes: additional_bytes,
                });
            }
            if self
                .tracker
                .queued_bytes
                .compare_exchange(
                    current_queued_bytes,
                    projected_queued_bytes,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                self.queued_bytes = requested_bytes;
                return Ok(());
            }
        }
    }
}

impl Drop for RawManifestQueuePermit {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug)]
pub(crate) struct QueuedRawManifest {
    manifest: DspManifest,
    _queue_permit: Option<RawManifestQueuePermit>,
}

struct AdmittedEnqueueRequest {
    received_ns: u128,
    content_type: Option<String>,
    capture_envelope: CaptureEnvelope,
    body_bytes: Vec<u8>,
    preflight_queue_permit: Option<RawManifestQueuePermit>,
}

impl QueuedRawManifest {
    pub(crate) fn into_manifest(mut self) -> DspManifest {
        if let Some(queue_permit) = self._queue_permit.as_mut() {
            queue_permit.release();
        }
        self.manifest
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
    runtime_config: JournalRuntimeConfig,
    storage_class_report: StorageClassReport,
    lease_store: LeaseStore,
    derived_cache: DerivedCache,
    manifest_store: ManifestStore,
    state: Mutex<SegmentJournalState>,
    sequence_tracker: Arc<SequenceTracker>,
}

#[derive(Debug)]
struct SegmentJournalState {
    journal_epoch: u64,
    streams: HashMap<String, StreamJournalState>,
}

#[derive(Debug)]
struct StreamJournalState {
    next_sequence: u64,
}

#[derive(Debug)]
struct MemoryOnlyLivePathLeaseUnsupportedError;

impl std::fmt::Display for MemoryOnlyLivePathLeaseUnsupportedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pin leases are unavailable while the sidecar is running in memory-only live mode"
        )
    }
}

impl std::error::Error for MemoryOnlyLivePathLeaseUnsupportedError {}

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
    /// Per-node firmware sequence-gap counters. Mirrors Python's
    /// `_logger.warning("Detected ingest frame sequence gap", ...)` at
    /// ingest.py:281 — Rust now matches that observability surface.
    #[serde(default)]
    pub sequence_gaps: SequenceTrackerSnapshot,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SequenceTrackerSnapshot {
    /// Cumulative gap-frame count per node_id (summed across boot sessions).
    /// "Gap-frame count" means: number of *missing* sequence numbers detected.
    /// E.g. expected 4, received 7 → contributes 3 to this counter.
    pub per_node_gap_count: HashMap<String, u64>,
    pub total_gap_count: u64,
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

/// Tracks per-(node_id, boot_session) firmware sequence numbers across
/// inbound payloads to surface dropped frames.
///
/// A *gap* means the firmware reported a sequence > expected_next; the
/// difference is added to the per-node and total counters. Reboots (boot_session
/// change) re-anchor the tracker without warning.
#[derive(Debug, Default)]
pub(crate) struct SequenceTracker {
    inner: std::sync::Mutex<SequenceTrackerInner>,
    total_gap_count: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Default)]
struct SequenceTrackerInner {
    last_seen: HashMap<(String, u32), u64>,
    per_node_gap_count: HashMap<String, u64>,
}

impl SequenceTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record the (first_sequence, last_sequence) range of a payload from
    /// `(node_id, boot_session)` and return any newly-detected gap size.
    /// A gap of 0 means contiguous with the prior payload (or first ever).
    ///
    /// Out-of-order / duplicate payloads do *not* rewind `last_seen` — the
    /// high-water mark only advances. This guards against the following
    /// false-gap pattern: prev_last=15 → payload {first=5, last=8} arrives
    /// out-of-order → if we naively rewrote last_seen=8, the next legitimate
    /// payload {first=16, ...} would appear to have a gap of 16-(8+1)=7.
    pub(crate) fn record(
        &self,
        node_id: &str,
        boot_session: u32,
        first_sequence: u64,
        last_sequence: u64,
    ) -> u64 {
        let key = (node_id.to_string(), boot_session);
        let mut inner = self.inner.lock().expect("sequence tracker poisoned");
        let (gap, advance_high_water) = match inner.last_seen.get(&key) {
            Some(&prev_last) => {
                // Use saturating arithmetic to defend against an adversarial
                // prev_last = u64::MAX (not reachable in practice but cheap to guard).
                let expected = prev_last.saturating_add(1);
                let gap = if first_sequence > expected {
                    first_sequence - expected
                } else {
                    0
                };
                // Only advance the high-water mark when this payload extends
                // it. Out-of-order/duplicate payloads observe but do not rewrite.
                (gap, last_sequence > prev_last)
            }
            None => (0, true),
        };
        if advance_high_water {
            inner.last_seen.insert(key, last_sequence);
        }
        if gap > 0 {
            *inner
                .per_node_gap_count
                .entry(node_id.to_string())
                .or_insert(0) += gap;
            self.total_gap_count
                .fetch_add(gap, std::sync::atomic::Ordering::Relaxed);
        }
        gap
    }

    pub(crate) fn snapshot(&self) -> SequenceTrackerSnapshot {
        let inner = self.inner.lock().expect("sequence tracker poisoned");
        SequenceTrackerSnapshot {
            per_node_gap_count: inner.per_node_gap_count.clone(),
            total_gap_count: self
                .total_gap_count
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }
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

#[derive(Debug)]
pub struct RawManifestChannelFullError {
    channel_capacity: usize,
}

impl std::fmt::Display for RawManifestChannelFullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "raw manifest channel is full (capacity {}); ingest backpressure engaged",
            self.channel_capacity,
        )
    }
}

impl std::error::Error for RawManifestChannelFullError {}

#[derive(Debug)]
pub struct RawManifestQueueBytesExceededError {
    byte_limit: usize,
    current_queued_bytes: usize,
    requested_bytes: usize,
}

impl std::fmt::Display for RawManifestQueueBytesExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "raw manifest queue byte budget exhausted: current {} bytes + requested {} bytes exceeds {} bytes",
            self.current_queued_bytes,
            self.requested_bytes,
            self.byte_limit,
        )
    }
}

impl std::error::Error for RawManifestQueueBytesExceededError {}

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

}

impl SegmentJournalBackend {
    async fn open(
        base_dir: PathBuf,
        _max_segment_bytes: u64,
        runtime_config: JournalRuntimeConfig,
    ) -> BoxedResult<Self> {
        let journal_root = base_dir.join("journal");
        let streams_dir = journal_root.join("streams");
        let (storage_class_report, journal_epoch, streams, _total_journal_bytes) =
            if runtime_config.memory_only_live_path {
                (
                    StorageClassReport {
                        path: journal_root.clone(),
                        storage_class: "memory-only-live-path".to_string(),
                        tmpfs_backed: true,
                        enforcement_enabled: false,
                    },
                    0,
                    HashMap::new(),
                    0,
                )
            } else {
                fs::create_dir_all(&streams_dir).await?;
                let storage_class_report =
                    validate_journal_storage_class(&journal_root, runtime_config.enforce_tmpfs)
                        .map_err(|error| -> BoxedError { Box::new(error) })?;
                let journal_epoch = open_or_advance_epoch(&journal_root, &streams_dir).await?;
                let (streams, total_journal_bytes) = recover_stream_states(&streams_dir).await?;
                (
                    storage_class_report,
                    journal_epoch,
                    streams,
                    total_journal_bytes,
                )
            };
        let lease_store = LeaseStore::new(&journal_root);
        let derived_cache = DerivedCache::new(
            &journal_root,
            DerivedCacheConfig {
                budget_bytes: runtime_config.derived_cache_budget_bytes,
                admission_reserve_bytes: runtime_config.derived_cache_admission_reserve_bytes,
            },
        );
        let manifest_store = ManifestStore::new(&journal_root);
        coordinate_runtime_store_initialization(
            &derived_cache,
            &manifest_store,
            runtime_config.memory_only_live_path,
        )
        .await?;
        Ok(Self {
            journal_root,
            runtime_config,
            storage_class_report,
            lease_store,
            derived_cache,
            manifest_store,
            state: Mutex::new(SegmentJournalState {
                journal_epoch,
                streams,
            }),
            sequence_tracker: Arc::new(SequenceTracker::new()),
        })
    }

    async fn health(&self) -> BoxedResult<IngestBackendHealth> {
        let active_pin_leases = if self.runtime_config.memory_only_live_path {
            0
        } else {
            self.derived_cache.reserve_for_write(0).await?;
            self.lease_store.active_index(now_ns()?).await?.leases.len()
        };
        Ok(IngestBackendHealth {
            storage_mode: if self.runtime_config.memory_only_live_path {
                "memory_only_live_path".to_string()
            } else {
                "journal".to_string()
            },
            journal: Some(JournalHealth {
                journal_root: self.journal_root.clone(),
                storage_class: self.storage_class_report.clone(),
                total_journal_budget_bytes: self.runtime_config.total_journal_budget_bytes,
                admission_reserve_bytes: self.runtime_config.admission_reserve_bytes,
                derived_cache_budget_bytes: self.runtime_config.derived_cache_budget_bytes,
                active_pin_leases,
            }),
            sequence_gaps: self.sequence_tracker.snapshot(),
        })
    }

    async fn create_pin_lease(&self, request: PinLeaseRequest) -> BoxedResult<PinLeaseResponse> {
        if self.runtime_config.memory_only_live_path {
            let _ = request;
            return Err(Box::new(MemoryOnlyLivePathLeaseUnsupportedError));
        }
        let now_ns = now_ns()?;
        let (response, active_index) = self.lease_store.upsert(request, now_ns).await?;
        self.refresh_segment_pin_counts(&active_index, now_ns)
            .await?;
        Ok(response)
    }

    async fn release_pin_lease(&self, lease_id: &str) -> BoxedResult<()> {
        if self.runtime_config.memory_only_live_path {
            let _ = lease_id;
            return Err(Box::new(MemoryOnlyLivePathLeaseUnsupportedError));
        }
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
        let admitted = self
            .admit_enqueue_request(max_body_bytes, endpoint, headers, body)
            .await?;
        self.enqueue_channel_only(endpoint, admitted).await
    }

    async fn admit_enqueue_request(
        &self,
        max_body_bytes: usize,
        endpoint: &'static str,
        headers: HeaderMap,
        body: Body,
    ) -> BoxedResult<AdmittedEnqueueRequest> {
        validate_declared_content_length(&headers, max_body_bytes)?;
        let mut preflight_queue_permit = self.preflight_raw_manifest_queue_reserve(&headers)?;
        let received_ns = now_ns()?;
        let content_type = content_type_from_headers(&headers);
        let body_bytes = read_body_bytes(body, max_body_bytes).await?;
        if let Some(queue_permit) = preflight_queue_permit.as_mut() {
            queue_permit.try_resize(body_bytes.len())?;
        }
        let (body_bytes, capture_envelope) =
            parse_capture_envelope_for_enqueue(endpoint, body_bytes).await?;
        Ok(AdmittedEnqueueRequest {
            received_ns,
            content_type,
            capture_envelope,
            body_bytes,
            preflight_queue_permit,
        })
    }

    fn preflight_raw_manifest_queue_reserve(
        &self,
        headers: &HeaderMap,
    ) -> BoxedResult<Option<RawManifestQueuePermit>> {
        let Some(queue_backpressure) = &self.runtime_config.raw_manifest_queue_backpressure else {
            return Ok(None);
        };
        let Some(content_length_bytes) = declared_content_length_bytes(headers) else {
            return Ok(None);
        };
        Ok(Some(queue_backpressure.try_reserve(content_length_bytes)?))
    }

    fn check_sequence_gap(&self, capture_envelope: &CaptureEnvelope) {
        let (Some(first_sequence), Some(last_sequence), Some(boot_session)) = (
            capture_envelope.first_sequence,
            capture_envelope.last_sequence,
            capture_envelope.boot_session,
        ) else {
            return;
        };
        let gap = self.sequence_tracker.record(
            &capture_envelope.node_id,
            boot_session,
            first_sequence,
            last_sequence,
        );
        if gap > 0 {
            let expected_sequence = first_sequence - gap;
            tracing::warn!(
                node_id = %capture_envelope.node_id,
                boot_session,
                expected_sequence,
                received_sequence = first_sequence,
                gap_size = gap,
                "Detected ingest frame sequence gap"
            );
        }
    }

    async fn enqueue_channel_only(
        &self,
        endpoint: &'static str,
        admitted: AdmittedEnqueueRequest,
    ) -> BoxedResult<String> {
        self.check_sequence_gap(&admitted.capture_envelope);
        let (journal_id, entry) = self
            .allocate_live_journal_entry(
                endpoint,
                admitted.content_type,
                admitted.received_ns,
                &admitted.capture_envelope,
            )
            .await;

        self.publish_capture_manifest(
            &entry,
            &admitted.body_bytes,
            admitted.preflight_queue_permit,
        )
            .await?;
        Ok(journal_id)
    }

    async fn allocate_live_journal_entry(
        &self,
        endpoint: &'static str,
        content_type: Option<String>,
        received_ns: u128,
        capture_envelope: &CaptureEnvelope,
    ) -> (String, SegmentJournalEntry) {
        let mut state = self.state.lock().await;
        let journal_epoch = state.journal_epoch;
        let stream_state = state
            .streams
            .entry(capture_envelope.stream_key.clone())
            .or_insert_with(|| StreamJournalState { next_sequence: 1 });
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
    }

    async fn publish_capture_manifest(
        &self,
        entry: &SegmentJournalEntry,
        raw_bytes: &[u8],
        preflight_queue_permit: Option<RawManifestQueuePermit>,
    ) -> BoxedResult<()> {
        let tx = self.raw_manifest_sender()?;
        let manifest = self.build_validated_raw_capture_manifest(entry, raw_bytes);
        let queue_permit =
            self.finalize_raw_manifest_queue_permit(raw_bytes.len(), preflight_queue_permit)?;
        let queued_manifest = QueuedRawManifest {
            manifest,
            _queue_permit: queue_permit,
        };
        match tx.try_send(queued_manifest) {
            Ok(()) => Ok(()),
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                Err(Box::new(RawManifestChannelFullError {
                    channel_capacity: self.runtime_config.raw_manifest_channel_capacity,
                }) as BoxedError)
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                Err(Box::new(InvalidIngestEnvelopeError::new(
                    "raw manifest channel receiver unavailable — DSP worker stopped",
                )) as BoxedError)
            }
        }
    }

    fn raw_manifest_sender(&self) -> BoxedResult<&mpsc::Sender<QueuedRawManifest>> {
        self.runtime_config
            .raw_manifest_tx
            .as_ref()
            .ok_or_else(|| {
                Box::new(InvalidIngestEnvelopeError::new(
                    "raw manifest channel not configured — DSP worker is not connected",
                )) as BoxedError
            })
    }

    fn build_validated_raw_capture_manifest(
        &self,
        entry: &SegmentJournalEntry,
        raw_bytes: &[u8],
    ) -> DspManifest {
        // raw_journal_append is a single contiguous frame at ingest time, so
        // coverage is trivially all-true. Surface the AudioCoverageStats 9-field
        // JSON shape so downstream consumers see consistent shape on every raw
        // path. Matches Python AudioCoverageStats.to_json() at audio_buffer.py:30.
        let coverage_stats_json = match (entry.sample_count, entry.sample_rate_hz) {
            (Some(sample_count), Some(sample_rate_hz))
                if sample_count > 0 && sample_rate_hz > 0 =>
            {
                let coverage = vec![true; sample_count as usize];
                serde_json::to_value(coverage_stats(&coverage, sample_rate_hz)).ok()
            }
            _ => None,
        };
        let mut manifest = DspManifest {
            manifest_id: format!("manifest-{}", entry.journal_id),
            manifest_type: "raw_journal_append".to_string(),
            created_ns: entry.toa_ns.unwrap_or(entry.ingest_received_ns as u64) as u128,
            source_handles: vec![entry.payload_handle()],
            derived_handle: None,
            localization: None,
            classifier_render: None,
            birdnet: None,
            coverage_stats: coverage_stats_json,
            promotion_ready: false,
            env_samples: None,
            node_context: None,
            cluster_id: None,
            cluster_sensor_positions: None,
            raw_payload: Some(raw_bytes.to_vec()),
            raw_render_bytes: None,
            raw_audio_frame: None,
            raw_audio_bytes: None,
        };
        let node_value_opt = if raw_bytes.starts_with(b"MMB1") || raw_bytes.starts_with(b"MMB2") {
            crate::envelope::extract_binary_node_json(raw_bytes)
        } else {
            serde_json::from_slice::<serde_json::Value>(raw_bytes)
                .ok()
                .and_then(|body_json| body_json.get("node").cloned())
        };
        if let Some(node_value) = node_value_opt {
            manifest.cluster_id = Self::extract_cluster_id_from_node_json(&node_value);
            manifest.cluster_sensor_positions =
                Self::extract_cluster_sensor_positions_from_node_json(&node_value);
            manifest.node_context = Some(serde_json::json!({
                "node": node_value,
                "toa_ns": entry.toa_ns,
                "time_quality": entry.time_quality,
                "source_type": entry.source_type,
            }));
        }
        manifest
    }

    fn extract_cluster_id_from_node_json(node_value: &serde_json::Value) -> Option<String> {
        node_value
            .get("cluster_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                node_value
                    .get("properties")
                    .and_then(|value| value.get("cluster_id"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .or_else(|| {
                node_value
                    .get("metadata")
                    .and_then(|value| value.get("cluster_id"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
    }

    fn extract_cluster_sensor_positions_from_node_json(
        node_value: &serde_json::Value,
    ) -> Option<Vec<(String, [f32; 3])>> {
        let encoded_positions = node_value
            .get("cluster_sensor_positions")
            .or_else(|| {
                node_value
                    .get("properties")
                    .and_then(|value| value.get("cluster_sensor_positions"))
            })
            .or_else(|| {
                node_value
                    .get("metadata")
                    .and_then(|value| value.get("cluster_sensor_positions"))
            })?
            .clone();

        if let Ok(positions) =
            serde_json::from_value::<Vec<(String, [f32; 3])>>(encoded_positions.clone())
        {
            if !positions.is_empty() {
                return Some(positions);
            }
        }

        let mapped_positions = encoded_positions
            .as_object()
            .and_then(|positions| {
                positions
                    .iter()
                    .map(|(sensor_id, position)| {
                        parse_cluster_position(position).map(|position_m| (sensor_id.clone(), position_m))
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .filter(|positions| !positions.is_empty());

        mapped_positions
    }

    fn finalize_raw_manifest_queue_permit(
        &self,
        requested_bytes: usize,
        mut preflight_queue_permit: Option<RawManifestQueuePermit>,
    ) -> BoxedResult<Option<RawManifestQueuePermit>> {
        if let Some(queue_permit) = preflight_queue_permit.as_mut() {
            queue_permit.try_resize(requested_bytes)?;
        }
        if let Some(queue_permit) = preflight_queue_permit {
            return Ok(Some(queue_permit));
        }
        if let Some(queue_backpressure) = &self.runtime_config.raw_manifest_queue_backpressure {
            return Ok(Some(queue_backpressure.try_reserve(requested_bytes)?));
        }
        Ok(None)
    }

    async fn refresh_segment_pin_counts(
        &self,
        active_index: &crate::leases::PinLeaseIndex,
        now_ns: u128,
    ) -> BoxedResult<()> {
        let _ = (active_index, now_ns);
        Ok(())
    }
}

fn parse_cluster_position(value: &serde_json::Value) -> Option<[f32; 3]> {
    let coordinates = value.as_array()?;
    if coordinates.len() != 3 {
        return None;
    }
    Some([
        coordinates[0].as_f64()? as f32,
        coordinates[1].as_f64()? as f32,
        coordinates[2].as_f64()? as f32,
    ])
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
            received_ns: Some(self.ingest_received_ns),
            sample_index_start: self.sample_index_start,
            sample_count: self.sample_count,
            integrity_hash: self.integrity_hash.clone(),
            segment_path: self.segment_path.clone(),
        }
    }
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
                    return Err(Box::new(ClientDisconnectError) as BoxedError);
                }
                return Err(boxed);
            }
        };
        if let Some(chunk) = frame.data_ref() {
            if bytes.len().saturating_add(chunk.len()) > max_body_bytes {
                return Err(Box::new(BodyTooLargeError {
                    limit_bytes: max_body_bytes,
                }) as BoxedError);
            }
            bytes.extend_from_slice(chunk);
        }
    }
    Ok(bytes)
}

async fn parse_capture_envelope_for_enqueue(
    endpoint: &'static str,
    body_bytes: Vec<u8>,
) -> BoxedResult<(Vec<u8>, CaptureEnvelope)> {
    if endpoint == "/api/v1/ingest/store-forward" {
        return tokio::task::spawn_blocking(move || {
            let envelope = parse_capture_envelope(endpoint, &body_bytes)
                .map_err(InvalidIngestEnvelopeError::new)
                .map_err(|error| -> BoxedError { Box::new(error) })?;
            Ok((body_bytes, envelope))
        })
        .await
        .map_err(|error| -> BoxedError { Box::new(error) })?;
    }

    let envelope = parse_capture_envelope(endpoint, &body_bytes)
        .map_err(InvalidIngestEnvelopeError::new)
        .map_err(|error| -> BoxedError { Box::new(error) })?;
    Ok((body_bytes, envelope))
}

async fn coordinate_runtime_store_initialization(
    derived_cache: &DerivedCache,
    manifest_store: &ManifestStore,
    memory_only_live_path: bool,
) -> BoxedResult<()> {
    if memory_only_live_path {
        return Ok(());
    }
    derived_cache.ensure_initialized().await?;
    manifest_store.ensure_initialized().await?;
    Ok(())
}

async fn open_or_advance_epoch(journal_root: &Path, _streams_dir: &Path) -> BoxedResult<u64> {
    let state_path = journal_root.join("epoch.json");
    let existing_state = match fs::read_to_string(&state_path).await {
        Ok(contents) => Some(serde_json::from_str::<JournalEpochState>(&contents)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(Box::new(error)),
    };

    let next_epoch = existing_state
        .as_ref()
        .map(|state| state.last_assigned_epoch.saturating_add(1))
        .unwrap_or(1);
    let epoch_state = JournalEpochState {
        active_epoch: next_epoch,
        last_assigned_epoch: next_epoch,
    };
    write_json_file(&state_path, &epoch_state).await?;
    Ok(epoch_state.active_epoch)
}

async fn recover_stream_states(
    _streams_dir: &Path,
) -> BoxedResult<(HashMap<String, StreamJournalState>, u64)> {
    Ok((HashMap::new(), 0))
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
    use serde_json::json;

    #[test]
    fn sequence_tracker_isolates_per_node_counters() {
        let tracker = SequenceTracker::new();

        // node-a streams contiguously: 1, 2, 3 → no gaps.
        assert_eq!(tracker.record("node-a", 0, 1, 1), 0);
        assert_eq!(tracker.record("node-a", 0, 2, 2), 0);
        assert_eq!(tracker.record("node-a", 0, 3, 3), 0);

        // node-b starts fresh: no prior state, no gap.
        assert_eq!(tracker.record("node-b", 0, 100, 100), 0);

        // node-a drops sequences 4..=6, jumps to 7 → gap of 3 frames.
        assert_eq!(tracker.record("node-a", 0, 7, 7), 3);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.per_node_gap_count.get("node-a"), Some(&3));
        // node-b must remain clean — one node's gaps must not pollute another's counter.
        assert_eq!(snapshot.per_node_gap_count.get("node-b"), None);
        assert_eq!(snapshot.total_gap_count, 3);

        // boot-session change re-anchors: a reboot from sequence 7 down to 1
        // under a new boot_session must not register as a gap.
        assert_eq!(tracker.record("node-a", 1, 1, 1), 0);
        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.per_node_gap_count.get("node-a"), Some(&3));
        assert_eq!(snapshot.total_gap_count, 3);
    }

    #[test]
    fn sequence_tracker_handles_multi_frame_payloads() {
        let tracker = SequenceTracker::new();

        // Payload carrying frames 1..=4 in one shot.
        assert_eq!(tracker.record("node-a", 0, 1, 4), 0);
        // Next payload carries frames 5..=8 — contiguous.
        assert_eq!(tracker.record("node-a", 0, 5, 8), 0);
        // Next payload carries frames 12..=14 — gap of 3 (9, 10, 11 missing).
        assert_eq!(tracker.record("node-a", 0, 12, 14), 3);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.per_node_gap_count.get("node-a"), Some(&3));
        assert_eq!(snapshot.total_gap_count, 3);
    }

    #[test]
    fn sequence_tracker_out_of_order_payload_does_not_rewind_high_water() {
        // Guards the false-gap pattern: a late/out-of-order payload arriving
        // *after* a contiguous run must not appear to "reset" the high-water
        // mark — otherwise the next legitimate payload would falsely register
        // a gap relative to the older sequence.
        let tracker = SequenceTracker::new();

        // Establish contiguous run 1..=15.
        assert_eq!(tracker.record("node-a", 0, 1, 15), 0);

        // Late-arriving out-of-order payload covering 5..=8.
        // Already-observed range — no gap to report, no high-water change.
        assert_eq!(tracker.record("node-a", 0, 5, 8), 0);

        // Next legitimate payload picks up where the high-water mark really
        // was (15), not where the out-of-order payload ended (8).
        assert_eq!(tracker.record("node-a", 0, 16, 20), 0);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.per_node_gap_count.get("node-a"), None);
        assert_eq!(snapshot.total_gap_count, 0);
    }

    #[test]
    fn sequence_tracker_does_not_overflow_at_u64_max() {
        // Defensive: a tracker observing prev_last = u64::MAX must not panic
        // on the next record() — `prev_last + 1` would overflow in debug.
        let tracker = SequenceTracker::new();
        assert_eq!(tracker.record("node-a", 0, u64::MAX, u64::MAX), 0);
        // Any subsequent first_sequence is by definition ≤ u64::MAX, so no
        // gap and no overflow. saturating_add(1) clamps to u64::MAX.
        assert_eq!(tracker.record("node-a", 0, u64::MAX, u64::MAX), 0);
    }

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

    fn store_forward_body_with_cluster_metadata(
        node_id: &str,
        sequence: u64,
        start_sample_index: u64,
    ) -> String {
        json!({
            "node": {
                "id": node_id,
                "node_type": "point",
                "position_m": [0.0, 0.0, 0.0],
                "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                "capabilities": ["audio"],
                "metadata": {},
                "properties": {
                    "cluster_sensor_positions": {
                        "node-a:ch0": [0.0, 0.0, 2.0],
                        "node-b:ch0": [2.0, 0.0, 2.0],
                        "node-c:ch0": [2.0, 2.0, 2.0],
                        "node-d:ch0": [0.0, 2.0, 2.0]
                    }
                },
                "cluster_id": "cluster-square"
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

    fn test_journal_runtime_config() -> JournalRuntimeConfig {
        JournalRuntimeConfig {
            total_journal_budget_bytes: 268_435_456,
            admission_reserve_bytes: 16_777_216,
            enforce_tmpfs: false,
            memory_only_live_path: false,
            derived_cache_budget_bytes: 67_108_864,
            derived_cache_admission_reserve_bytes: 8_388_608,
            raw_manifest_channel_capacity: 2048,
            raw_manifest_queue_backpressure: None,
            raw_manifest_tx: None,
        }
    }

    #[tokio::test]
    async fn raw_capture_manifest_preserves_packet_time_and_records_ingest_receipt_time() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body("node-a", 7, 0);

        // Wire a bounded channel so the manifest arrives in-memory (no disk write).
        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let config = JournalRuntimeConfig {
            raw_manifest_channel_capacity: 4,
            raw_manifest_queue_backpressure: Some(queue_backpressure),
            raw_manifest_tx: Some(tx),
            ..test_journal_runtime_config()
        };
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            config,
        )
        .await
        .unwrap();

        let before_enqueue_ns = now_ns().unwrap();
        backend
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                HeaderMap::new(),
                Body::from(payload),
            )
            .await
            .unwrap();
        let after_enqueue_ns = now_ns().unwrap();

        let manifest = rx
            .try_recv()
            .expect("manifest delivered via channel")
            .into_manifest();
        assert_eq!(manifest.created_ns, 1007);
        assert_eq!(manifest.source_handles[0].toa_ns, Some(1007));
        assert_eq!(manifest.source_handles[0].tor_ns, Some(2007));
        assert!(manifest.source_handles[0].received_ns >= Some(before_enqueue_ns));
        assert!(manifest.source_handles[0].received_ns <= Some(after_enqueue_ns));
    }

    #[tokio::test]
    async fn raw_capture_enqueue_does_not_persist_raw_manifest_file() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body("node-a", 7, 0);

        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let config = JournalRuntimeConfig {
            raw_manifest_channel_capacity: 4,
            raw_manifest_queue_backpressure: Some(queue_backpressure),
            raw_manifest_tx: Some(tx),
            ..test_journal_runtime_config()
        };
        let store_dir = tmp.path().join("store");
        let backend = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            config,
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

        let manifest = rx
            .try_recv()
            .expect("manifest delivered via channel")
            .into_manifest();
        assert!(manifest.raw_payload.is_some());

        let raw_pending_dir = store_dir
            .join("journal")
            .join("manifests")
            .join("raw_pending");
        let mut entries = tokio::fs::read_dir(&raw_pending_dir).await.unwrap();
        assert!(
            entries.next_entry().await.unwrap().is_none(),
            "raw ingest must stay on the in-memory channel, not raw_pending JSON files"
        );
    }

    #[tokio::test]
    async fn raw_capture_manifest_preserves_cluster_metadata_without_persisting_raw_audio() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body_with_cluster_metadata("node-a", 7, 0);

        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let config = JournalRuntimeConfig {
            raw_manifest_channel_capacity: 4,
            raw_manifest_queue_backpressure: Some(queue_backpressure),
            raw_manifest_tx: Some(tx),
            ..test_journal_runtime_config()
        };
        let store_dir = tmp.path().join("store");
        let backend = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            config,
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

        let manifest = rx
            .try_recv()
            .expect("manifest delivered via channel")
            .into_manifest();
        assert!(manifest.raw_payload.is_some());
        assert_eq!(manifest.cluster_id.as_deref(), Some("cluster-square"));
        assert_eq!(
            manifest
                .cluster_sensor_positions
                .as_ref()
                .map(Vec::len),
            Some(4)
        );
        assert_eq!(
            manifest
                .cluster_sensor_positions
                .as_ref()
                .and_then(|positions| positions.first())
                .map(|(sensor_id, _)| sensor_id.as_str()),
            Some("node-a:ch0")
        );

        let raw_pending_dir = store_dir
            .join("journal")
            .join("manifests")
            .join("raw_pending");
        let mut entries = tokio::fs::read_dir(&raw_pending_dir).await.unwrap();
        assert!(
            entries.next_entry().await.unwrap().is_none(),
            "cluster metadata plumbing must not cause raw audio manifests to spill to disk"
        );
    }

    #[tokio::test]
    async fn raw_capture_manifest_returns_backpressure_error_when_channel_is_full() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body("node-a", 7, 0);
        let next_payload = store_forward_body("node-a", 8, 4);

        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let config = JournalRuntimeConfig {
            raw_manifest_channel_capacity: 1,
            raw_manifest_queue_backpressure: Some(queue_backpressure),
            raw_manifest_tx: Some(tx),
            ..test_journal_runtime_config()
        };
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            config,
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

        let err = backend
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                HeaderMap::new(),
                Body::from(next_payload),
            )
            .await
            .expect_err("second enqueue should hit channel backpressure");

        assert!(err.downcast_ref::<RawManifestChannelFullError>().is_some());
    }

    #[tokio::test]
    async fn raw_capture_manifest_returns_backpressure_error_when_byte_budget_is_exhausted() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body("node-a", 7, 0);
        let payload_len = payload.len();
        let next_payload = store_forward_body("node-a", 8, 4);

        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(
            payload_len.saturating_mul(2).saturating_sub(1),
        ));
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let config = JournalRuntimeConfig {
            raw_manifest_channel_capacity: 8,
            raw_manifest_queue_backpressure: Some(queue_backpressure),
            raw_manifest_tx: Some(tx),
            ..test_journal_runtime_config()
        };
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            config,
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

        let err = backend
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                HeaderMap::new(),
                Body::from(next_payload),
            )
            .await
            .expect_err("second enqueue should hit byte-budget backpressure");

        assert!(err
            .downcast_ref::<RawManifestQueueBytesExceededError>()
            .is_some());
    }

    #[tokio::test]
    async fn store_forward_fast_fails_on_declared_length_before_parsing_when_queue_budget_full() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = store_forward_body("node-a", 7, 0);
        let payload_len = payload.len();

        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(payload_len));
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let config = JournalRuntimeConfig {
            raw_manifest_channel_capacity: 8,
            raw_manifest_queue_backpressure: Some(queue_backpressure),
            raw_manifest_tx: Some(tx),
            ..test_journal_runtime_config()
        };
        let backend = IngestBackend::open(
            tmp.path().join("store"),
            IngestStorageMode::Journal,
            usize::MAX as u64,
            config,
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

        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_LENGTH,
            header::HeaderValue::from_static("1"),
        );

        let err = backend
            .enqueue(
                usize::MAX,
                "/api/v1/ingest/store-forward",
                headers,
                Body::from("{"),
            )
            .await
            .expect_err("declared length should be rejected before parsing malformed body");

        assert!(err
            .downcast_ref::<RawManifestQueueBytesExceededError>()
            .is_some());
    }

    #[tokio::test]
    async fn journal_recovers_next_sequence_after_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, _rx) = tokio::sync::mpsc::channel(4);

        let backend = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            65_536,
            JournalRuntimeConfig {
                raw_manifest_channel_capacity: 4,
                raw_manifest_queue_backpressure: Some(queue_backpressure),
                raw_manifest_tx: Some(tx),
                ..test_journal_runtime_config()
            },
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

        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, _rx) = tokio::sync::mpsc::channel(4);

        let reopened = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            65_536,
            JournalRuntimeConfig {
                raw_manifest_channel_capacity: 4,
                raw_manifest_queue_backpressure: Some(queue_backpressure),
                raw_manifest_tx: Some(tx),
                ..test_journal_runtime_config()
            },
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

    #[tokio::test]
    async fn memory_only_live_path_does_not_initialize_journal_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let store_dir = tmp.path().join("store");
        let queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(1_048_576));
        let (tx, _rx) = tokio::sync::mpsc::channel(4);

        let backend = IngestBackend::open(
            store_dir.clone(),
            IngestStorageMode::Journal,
            65_536,
            JournalRuntimeConfig {
                memory_only_live_path: true,
                raw_manifest_channel_capacity: 4,
                raw_manifest_queue_backpressure: Some(queue_backpressure),
                raw_manifest_tx: Some(tx),
                ..test_journal_runtime_config()
            },
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

        assert_eq!(
            backend.health().await.unwrap().storage_mode,
            "memory_only_live_path"
        );
        assert!(!store_dir.join("journal").exists());
    }

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
            cluster_id: None,
            cluster_sensor_positions: None,
            raw_payload: None,
            raw_render_bytes: None,
            raw_audio_frame: None,
            raw_audio_bytes: None,
        };
        let path = manifest_store.publish(manifest.clone()).await.unwrap();
        assert!(path.exists(), "manifest file should exist after publish");

        let pending = manifest_store
            .query_pending("raw_journal_append")
            .await
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].promotion_ready);
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
                cluster_id: None,
                cluster_sensor_positions: None,
                raw_payload: None,
                raw_render_bytes: None,
                raw_audio_frame: None,
                raw_audio_bytes: None,
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
            cluster_id: None,
            cluster_sensor_positions: None,
            raw_payload: None,
            raw_render_bytes: None,
            raw_audio_frame: None,
            raw_audio_bytes: None,
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
                    cluster_id: None,
                    cluster_sensor_positions: None,
                    raw_payload: None,
                    raw_render_bytes: None,
                    raw_audio_frame: None,
                    raw_audio_bytes: None,
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
                    cluster_id: None,
                    cluster_sensor_positions: None,
                    raw_payload: None,
                    raw_render_bytes: None,
                    raw_audio_frame: None,
                    raw_audio_bytes: None,
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
