use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    actors::{classification::ClassificationWorker, environment::EnvironmentCache},
    audio_payload::{decode_audio_payload_segments, DecodedAudioPayload},
    classifier_helper::ManifestClassificationAnnotator,
    deferred_localization::{
        localization_only_template, DeferredClassifierRender, DeferredLocalizationScheduler,
    },
    derived_cache::DerivedCache,
    diagnostics::IngestDiagnostics,
    dsp::{coverage_stats, AudioCoverageStats, SensorStreamBuffer},
    dsp_events::DspEventPublisher,
    ingest_backend::QueuedRawManifest,
    journal_reader::JournalPayloadHandle,
    localization_coverage_wait::{
        CoverageWaitDecision, LocalizationCoverageWaitConfig, LocalizationWindowSpan,
    },
    manifests::{DspManifest, ManifestStore},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, sync::RwLock, time};
use tracing::{debug, error, info, warn};

const MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 15.0;
const DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 30.0;
const DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS: f64 = 2.0;
const AUTOMATIC_GPS_CLUSTER_ID: &str = "auto-gps";
/// SRP-PHAT needs at least this many channels in the localization buffer.
pub(crate) const MIN_SRP_CHANNEL_COUNT: usize = 4;
const NANOSECONDS_PER_MILLISECOND_F64: f64 = 1_000_000.0;
/// Shared multi-node buffers are keyed `cluster::<cluster_id>::<stream_id>`.
const CLUSTER_BUFFER_KEY_PREFIX: &str = "cluster::";

pub(crate) use minimappr_core::sirith_array_geometry::SIRITH_MIC_POSITIONS_M;

#[path = "dsp_array_geometry.rs"]
mod dsp_array_geometry;
#[path = "dsp_automatic_cluster_placement.rs"]
mod dsp_automatic_cluster_placement;
#[path = "dsp_buffer_channel_reconcile.rs"]
mod dsp_buffer_channel_reconcile;
#[path = "dsp_buffer_sensor_set.rs"]
mod dsp_buffer_sensor_set;
#[path = "dsp_cluster_sensor_registry.rs"]
mod dsp_cluster_sensor_registry;
#[path = "dsp_compute_payload.rs"]
mod dsp_compute_payload;
#[path = "dsp_coverage_integration.rs"]
mod dsp_coverage_integration;
#[path = "dsp_covered_event_solve.rs"]
mod dsp_covered_event_solve;
#[path = "dsp_dispatch_gate_key.rs"]
mod dsp_dispatch_gate_key;
#[path = "dsp_dispatch_gates.rs"]
mod dsp_dispatch_gates;
#[path = "dsp_localization_trigger.rs"]
mod dsp_localization_trigger;
#[path = "dsp_manifest_buffering.rs"]
mod dsp_manifest_buffering;
#[path = "dsp_manifest_publish.rs"]
mod dsp_manifest_publish;
#[path = "dsp_node_array_geometry_gate.rs"]
mod dsp_node_array_geometry_gate;
#[path = "dsp_node_clock_assessment.rs"]
mod dsp_node_clock_assessment;
#[path = "dsp_node_clock_quality.rs"]
mod dsp_node_clock_quality;
#[path = "dsp_preprocess.rs"]
mod dsp_preprocess;
#[path = "dsp_solve_pair_admission.rs"]
mod dsp_solve_pair_admission;
#[path = "dsp_solve_sensor_layout.rs"]
mod dsp_solve_sensor_layout;
#[path = "dsp_solver_reference_floor.rs"]
mod dsp_solver_reference_floor;
#[path = "dsp_window_geometry.rs"]
mod dsp_window_geometry;
#[path = "explicit_cluster_sensor_list.rs"]
mod explicit_cluster_sensor_list;
#[path = "localization_event_onset.rs"]
mod localization_event_onset;
#[path = "localization_event_span.rs"]
mod localization_event_span;
#[cfg(test)]
#[path = "solve_layout_test_fixture.rs"]
mod solve_layout_test_fixture;

use self::dsp_array_geometry::*;
pub(crate) use self::dsp_automatic_cluster_placement::*;
pub(crate) use self::dsp_buffer_channel_reconcile::*;
pub(crate) use self::dsp_buffer_sensor_set::*;
pub(crate) use self::dsp_cluster_sensor_registry::*;
pub(crate) use self::dsp_compute_payload::*;
use self::dsp_covered_event_solve::*;
use self::dsp_dispatch_gate_key::*;
use self::dsp_localization_trigger::*;
use self::localization_event_onset::*;
use self::localization_event_span::*;
use self::dsp_manifest_buffering::*;
pub(crate) use self::dsp_manifest_publish::*;
pub(crate) use self::dsp_node_array_geometry_gate::*;
pub(crate) use self::dsp_node_clock_assessment::*;
pub(crate) use self::dsp_node_clock_quality::*;
pub(crate) use self::dsp_preprocess::*;
pub(crate) use self::dsp_solve_pair_admission::*;
pub(crate) use self::dsp_solve_sensor_layout::*;
pub(crate) use self::dsp_solver_reference_floor::*;
pub(crate) use self::dsp_window_geometry::*;
pub(crate) use self::explicit_cluster_sensor_list::*;

pub use minimappr_core::gcc_phat::PairTdoa;

/// Shared state exposed via the /api/v1/dsp/* endpoints.
#[derive(Debug, Default)]
pub struct DspWorkerState {
    pub worker_running: bool,
    pub last_heartbeat_ns: Option<u128>,
    pub last_processed_ns: Option<u128>,
    pub total_localization_attempts: u64,
    pub total_classification_attempts: u64,
    pub total_tdoa_results: u64,
    pub total_localization_results: u64,
    pub total_classifier_renders: u64,
    pub total_failures: u64,
    pub total_stale_manifest_skips: u64,
    pub total_classification_drops: u64,
    /// Per-sensor SensorStreamBuffer re-anchor events (NTP/GPS correction or
    /// large frame jumps). Mirrors `SensorStreamBuffer.reanchor_count` on the
    /// Python side. Visible degradation signal for long-uptime timeline drift.
    pub total_buffer_reanchors: u64,
    /// Node timeline step re-anchors / drift slews (origin shifts that keep
    /// the audio, `timeline_discipline.rs`). Mirror the Python per-sensor
    /// `timeline_step_reanchor_count` / `timeline_drift_slew_count`, counted
    /// once per node correction here (once per channel buffer in Python).
    pub total_timeline_step_reanchors: u64,
    pub total_timeline_drift_slews: u64,
    /// `compute_manifest_frame` returned None because the requested audio
    /// window fell outside the buffer's coverage. Mirrors the Python
    /// `localization_drops_by_reason["no_window"]` counter — both are the
    /// canonical "pipeline is silently dropping" signal.
    pub total_window_underrun_drops: u64,
    pub total_stale_streams_evicted: u64,
    /// Coverage-wait counters, same names as the Python `FusionMetrics`
    /// fields: one stop reason per cadence-triggered localization candidate
    /// (>= 4-channel buffer), and the longest wall-clock deferral.
    pub localization_coverage_wait_stop_reasons: HashMap<String, u64>,
    pub localization_coverage_wait_max_ms: f64,
    pub pending_count: usize,
    pub recent_results: Vec<DspManifest>,
    /// Per-node audio overrides applied during ingest (set via POST /api/v1/dsp/config).
    pub node_audio_overrides: HashMap<String, NodeAudioConfig>,
    /// Site frame + node placements (launch env or `POST /api/v1/dsp/geometry`,
    /// else automatic GPS); a solve already parked keeps the layout it
    /// triggered with.
    pub site_geometry: crate::dsp_geometry_config::SiteGeometryState,
}

pub type SharedDspState = Arc<RwLock<DspWorkerState>>;

pub struct DspWorker {
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    config: DspWorkerConfig,
    state: SharedDspState,
    classifier_annotator: Option<ManifestClassificationAnnotator>,
    classification_tx: Option<flume::Sender<ClassificationRequest>>,
    raw_manifest_rx: Option<mpsc::Receiver<QueuedRawManifest>>,
    shutdown_requested: Option<Arc<AtomicBool>>,
    buffers: HashMap<String, Vec<SensorStreamBuffer>>,
    /// Sensor id behind each channel of `buffers[key]`, so a registry
    /// reorder (join / eviction) can move the audio with its sensor.
    buffer_channel_sensor_ids: HashMap<String, Arc<Vec<String>>>,
    /// Which sensors back each buffer, their order, positions and rates.
    sensor_registry: ClusterSensorRegistry,
    /// Per-node clock evidence; each solve snapshots its layout's nodes.
    node_clock_quality: NodeClockQualityTracker,
    /// Per-node preprocessing state (biquad memory etc.). Keyed by `node_id`
    /// extracted from the stream_key prefix, matching how `node_audio_overrides`
    /// is keyed in [`DspWorkerState`]. Lives outside the shared RwLock so frame
    /// preprocessing only contends on the single owning worker's `&mut self`.
    node_audio_state: HashMap<String, NodeAudioState>,
    last_classifier_render_ns_by_stream: HashMap<String, u128>,
    last_localization_ns_by_stream: HashMap<String, u128>,
    /// Absolute-RMS OR noise-floor-relative trigger, node floors and the
    /// per-buffer cooldown.
    localization_trigger: LocalizationEventTrigger,
    /// Time-only cadence gate used for node liveness heartbeats.
    /// Unlike `last_localization_ns_by_stream`, this is never blocked by the RMS
    /// energy gate or trigger cooldown — it fires unconditionally every
    /// `localization_cadence_ms` so nodes appear alive even during quiet periods.
    last_heartbeat_ns_by_stream: HashMap<String, u128>,
    consumed_manifests_since_prune: Arc<AtomicU64>,
    last_consumed_prune_ns: u128,
    last_stream_purge_ns: u128,
    deferred_source_manifest_ids: Vec<String>,
    env_cache: Option<EnvironmentCache>,
    /// Shared publisher for DSP result manifests streamed to Python via SSE.
    dsp_event_publisher: Option<DspEventPublisher>,
    /// Lock-free queue-wait/processing-latency + overload counters, mirrored
    /// against the Python `FusionMetrics` equivalents. See `diagnostics.rs`.
    diagnostics: Option<Arc<IngestDiagnostics>>,
    /// Localizations held until every streaming cluster member covers the
    /// window (coverage race fix). At most one pending per buffer key.
    deferred_localizations: DeferredLocalizationScheduler,
    /// Latest event-triggered solve per buffer, which a covering render repeats.
    recent_event_trigger_solves: RecentEventTriggerSolves,
}

impl DspWorker {
    pub fn new(
        manifest_store: ManifestStore,
        derived_cache: DerivedCache,
        config: DspWorkerConfig,
        state: SharedDspState,
    ) -> Self {
        let classifier_annotator = match ManifestClassificationAnnotator::from_command_json(
            config.classifier_command_json.as_deref(),
        ) {
            Ok(a) => a,
            Err(error) => {
                warn!(error = %error, "DSP worker classifier helper disabled");
                None
            }
        };
        let deferred_localizations =
            DeferredLocalizationScheduler::new(LocalizationCoverageWaitConfig::from_seconds(
                config.localization_coverage_wait_all_streaming_nodes_enabled,
                config.localization_coverage_stall_budget_seconds,
                config.localization_coverage_total_cap_seconds,
            ));
        let sensor_registry = ClusterSensorRegistry::new(&config.localization_solve);
        let localization_trigger = LocalizationEventTrigger::new(&config);
        let node_clock_quality = NodeClockQualityTracker::new(&config.localization_solve);
        Self {
            manifest_store,
            derived_cache,
            config,
            state,
            classifier_annotator,
            classification_tx: None,
            raw_manifest_rx: None,
            shutdown_requested: None,
            buffers: HashMap::new(),
            buffer_channel_sensor_ids: HashMap::new(),
            sensor_registry,
            node_clock_quality,
            node_audio_state: HashMap::new(),
            last_classifier_render_ns_by_stream: HashMap::new(),
            last_localization_ns_by_stream: HashMap::new(),
            localization_trigger,
            last_heartbeat_ns_by_stream: HashMap::new(),
            consumed_manifests_since_prune: Arc::new(AtomicU64::new(0)),
            last_consumed_prune_ns: system_now_ns(),
            last_stream_purge_ns: system_now_ns(),
            deferred_source_manifest_ids: Vec::new(),
            env_cache: None,
            dsp_event_publisher: None,
            diagnostics: None,
            deferred_localizations,
            recent_event_trigger_solves: RecentEventTriggerSolves::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_coverage_wait_clock(
        mut self,
        clock: crate::localization_coverage_wait::CoverageWaitClock,
    ) -> Self {
        self.deferred_localizations.set_clock(clock);
        self
    }

    /// Wires the ManifestClassificationAnnotator to a dedicated `ClassificationWorker`
    /// task via a flume bounded channel (capacity 64, raised from legacy 16).
    /// Returns `(self, Some(worker))` when classification is configured,
    /// `(self, None)` when no classifier command is set.
    pub fn with_classification_worker(
        mut self,
        channel_capacity: usize,
    ) -> (Self, Option<ClassificationWorker>) {
        let Some(annotator) = self.classifier_annotator.take() else {
            return (self, None);
        };
        let (tx, rx) = flume::bounded(channel_capacity);
        self.classification_tx = Some(tx);
        let worker = ClassificationWorker {
            annotator,
            rx,
            dsp_event_publisher: self.dsp_event_publisher.clone(),
        };
        (self, Some(worker))
    }

    /// Injects an in-process receiver for raw_journal_append manifests from the
    /// co-located ingest backend, bypassing the 20ms filesystem poll cycle.
    pub fn with_raw_manifest_receiver(mut self, rx: mpsc::Receiver<QueuedRawManifest>) -> Self {
        self.raw_manifest_rx = Some(rx);
        self
    }

    /// Injects a shared EnvironmentCache for sound-speed interpolation (Phase 6).
    pub fn with_env_cache(mut self, cache: EnvironmentCache) -> Self {
        self.env_cache = Some(cache);
        self
    }

    /// Injects the shared publisher used for SSE delivery and replay.
    pub fn with_dsp_event_publisher(mut self, publisher: DspEventPublisher) -> Self {
        self.dsp_event_publisher = Some(publisher);
        self
    }

    /// Signals the worker to stop after the HTTP server has finished draining
    /// accepted requests and this worker has emptied its manifest queue.
    pub fn with_shutdown_signal(mut self, shutdown_requested: Arc<AtomicBool>) -> Self {
        self.shutdown_requested = Some(shutdown_requested);
        self
    }

    /// Injects the shared diagnostics counters used by `/api/v1/diagnostics/summary`.
    pub fn with_diagnostics(mut self, diagnostics: Arc<IngestDiagnostics>) -> Self {
        self.diagnostics = Some(diagnostics);
        self
    }

    pub async fn run_loop(mut self) {
        info!("DSP worker started");
        let interval = time::Duration::from_millis(self.config.poll_interval_ms);
        const STREAM_PURGE_INTERVAL_NS: u128 = 60 * 1_000_000_000; // once per minute
        loop {
            {
                let mut st = self.state.write().await;
                st.worker_running = true;
                st.last_heartbeat_ns = Some(system_now_ns());
            }
            let processed_manifest_count = self.process_pending().await;
            if self.should_exit_after_drain(processed_manifest_count) {
                break;
            }
            let now_ns = system_now_ns();
            if now_ns.saturating_sub(self.last_stream_purge_ns) >= STREAM_PURGE_INTERVAL_NS {
                self.last_stream_purge_ns = now_ns;
                self.purge_stale_streams().await;
            }
            if should_sleep_after_poll_cycle(processed_manifest_count) {
                time::sleep(interval).await;
            }
        }
        {
            let mut st = self.state.write().await;
            st.worker_running = false;
            st.last_heartbeat_ns = Some(system_now_ns());
        }
        info!("DSP worker drained and exited");
    }

    fn should_exit_after_drain(&self, processed_manifest_count: usize) -> bool {
        let shutdown_requested = self
            .shutdown_requested
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire));
        if !shutdown_requested || processed_manifest_count > 0 {
            return false;
        }
        self.raw_manifest_rx.as_ref().is_none_or(|rx| rx.is_empty())
    }

    async fn process_pending(&mut self) -> usize {
        let batch_limit = self.config.pending_manifest_batch_size.max(1);
        let channel_limit = (batch_limit / 2).max(1);

        // Drain a bounded amount from the in-process channel, then also query disk
        // so fresh spillover manifests are not starved behind channel backlog.
        let channel_manifests: Vec<DspManifest> = if let Some(rx) = &mut self.raw_manifest_rx {
            let mut v = Vec::new();
            while let Ok(m) = rx.try_recv() {
                v.push(m.into_manifest());
                if v.len() >= channel_limit {
                    break;
                }
            }
            v
        } else {
            Vec::new()
        };

        // Only hit the filesystem when the in-process channel did not already
        // fill the batch budget. This keeps steady-state ingest on memory paths.
        let should_query_disk =
            self.config.query_persisted_raw_manifests && channel_manifests.len() < channel_limit;
        let disk_manifests = if should_query_disk {
            match self
                .manifest_store
                .query_pending_limited("raw_journal_append", batch_limit)
                .await
            {
                Ok(m) => m,
                Err(err) => {
                    warn!(error = %err, "DSP worker failed to query pending manifests");
                    return 0;
                }
            }
        } else {
            Vec::new()
        };
        let pending =
            merge_pending_manifests_for_batch(channel_manifests, disk_manifests, batch_limit);

        {
            let mut st = self.state.write().await;
            st.pending_count = pending.len();
        }

        let pending_backlog_depth = pending.len();

        // Capture the Tokio handle once so each rayon closure can call
        // `handle.spawn(run_io(...))` after the sync math phase completes.
        let handle = tokio::runtime::Handle::current();
        // Dispatch to the dedicated DSP rayon pool. run_compute is sync: it
        // runs CPU math on the Rayon thread, then calls handle.spawn(run_io(...))
        // which queues async I/O onto Tokio and returns immediately. The Rayon
        // thread is free for the next frame without waiting on disk or DB.
        let mut dispatch_to_dsp_pool = |payload: ComputePayload| {
            let h = handle.clone();
            crate::runtime::dsp_pool().spawn_fifo(move || {
                crate::actors::dsp_compute::run_compute(payload, h);
            });
        };
        self.ingest_batch_and_dispatch(pending, pending_backlog_depth, &mut dispatch_to_dsp_pool)
            .await;

        self.flush_deferred_source_manifest_consumptions().await;
        pending_backlog_depth
    }

    /// One poll cycle's batch: every payload is dispatched as soon as it is
    /// built (compute overlaps ingest), and the end of the batch is the only
    /// point where parked coverage-wait localizations are decided.
    async fn ingest_batch_and_dispatch(
        &mut self,
        pending: Vec<DspManifest>,
        pending_backlog_depth: usize,
        dispatch: &mut (dyn FnMut(ComputePayload) + Send),
    ) {
        for manifest in pending {
            let ingested_payload = self.ingest_one(manifest, pending_backlog_depth).await;
            // A newer candidate can supersede (and so release) a parked one.
            let ready_payloads = self.take_ready_deferred_localizations(false).await;
            for payload in ingested_payload.into_iter().chain(ready_payloads) {
                for ready in self.prepare_dispatch(payload).await {
                    dispatch(ready);
                }
            }
        }
        // Batch boundary (also the idle poll tick): decide parked windows.
        for payload in self.take_ready_deferred_localizations(true).await {
            for ready in self.prepare_dispatch(payload).await {
                dispatch(ready);
            }
        }
        for ready in self.release_held_renders_at_batch_boundary().await {
            dispatch(ready);
        }
    }

    async fn ingest_one(
        &mut self,
        manifest: DspManifest,
        pending_backlog_depth: usize,
    ) -> Option<ComputePayload> {
        let owned_segments = self.prepare_owned_manifest_audio(manifest).await?;

        for (segment_index, owned) in owned_segments.iter().enumerate() {
            self.publish_raw_audio_frame_for_owned_manifest(
                owned,
                segment_index,
                owned_segments.len(),
            )
            .await;
        }

        // The node's own array failed the geometry gate (too many mics, or
        // missing / collinear offsets): omni-render only the newest contiguous
        // segment rather than falsely joining audio across a capture gap.
        if let Some(reason) = owned_segments.last()?.large_array_omni_fallback_reason {
            return self.dispatch_large_array_manifest(
                owned_segments.into_iter().last()?,
                pending_backlog_depth,
                reason,
            );
        }

        let mut final_owned_and_buffered = None;
        for owned in owned_segments {
            let buffered = self.buffer_owned_manifest_audio(&owned).await?;
            final_owned_and_buffered = Some((owned, buffered));
        }
        let (final_owned, final_buffered) = final_owned_and_buffered?;
        self.dispatch_buffered_manifest(final_owned, final_buffered, pending_backlog_depth)
            .await
    }

    async fn note_failures(&self, count: u64) {
        if count == 0 {
            return;
        }
        let mut st = self.state.write().await;
        st.total_failures += count;
    }

    async fn note_failure(&self) {
        self.note_failures(1).await;
    }

    /// Cross-process diagnostic counter mirroring the Python
    /// `SensorStreamBuffer.reanchor_count`. Surfaced via `/api/v1/dsp/status`
    /// so an operator alerting on long-uptime timeline drift sees the same
    /// signal whether the deployment is using the Python or the Rust ingest.
    async fn note_buffer_reanchors(&self, count: u64) {
        if count == 0 {
            return;
        }
        let mut st = self.state.write().await;
        st.total_buffer_reanchors = st.total_buffer_reanchors.saturating_add(count);
    }

    async fn note_buffer_timeline_corrections(&self, corrections: NodeTimelineCorrections) {
        if corrections.is_empty() {
            return;
        }
        let mut st = self.state.write().await;
        st.total_timeline_step_reanchors = st
            .total_timeline_step_reanchors
            .saturating_add(u64::from(corrections.step_reanchors));
        st.total_timeline_drift_slews = st
            .total_timeline_drift_slews
            .saturating_add(u64::from(corrections.drift_slews));
    }

    /// Evict per-stream entries (buffers, audio state, cadence timestamps) for
    /// streams that have not produced a heartbeat in `stream_inactivity_evict_seconds`.
    /// Mirrors the leases.rs `purge_expired_*` idiom. Called once per minute from
    /// `run_loop` to prevent RSS growth from node-ID churn (reflash, hostname change).
    async fn purge_stale_streams(&mut self) {
        let ttl_ns = (self.config.stream_inactivity_evict_seconds as u128) * 1_000_000_000;
        let now_ns = system_now_ns();
        let cutoff = now_ns.saturating_sub(ttl_ns);
        self.sensor_registry.evict_stale_sensors(now_ns);
        self.node_clock_quality.evict_stale(now_ns);
        self.localization_trigger.evict_stale_node_floors(now_ns, ttl_ns);
        let stale: Vec<String> = self
            .last_heartbeat_ns_by_stream
            .iter()
            .filter(|(_, v)| **v < cutoff)
            .map(|(k, _)| k.clone())
            .collect();
        if stale.is_empty() {
            return;
        }
        for key in &stale {
            self.last_classifier_render_ns_by_stream.remove(key);
            self.last_localization_ns_by_stream.remove(key);
            self.localization_trigger.forget_stream(key);
            self.last_heartbeat_ns_by_stream.remove(key);
        }
        // Gates are per buffer and solve rate; a buffer goes only once no rate
        // group still heartbeats on it.
        let stale_buffer_keys: HashSet<String> = stale
            .iter()
            .map(|key| buffer_key_of_dispatch_gate_key(key).to_string())
            .collect();
        for key in &stale_buffer_keys {
            if self
                .last_heartbeat_ns_by_stream
                .keys()
                .any(|gate_key| buffer_key_of_dispatch_gate_key(gate_key) == key)
            {
                continue;
            }
            self.buffers.remove(key);
            self.buffer_channel_sensor_ids.remove(key);
            self.sensor_registry.forget_buffer(key);
            self.deferred_localizations.forget(key);
            self.recent_event_trigger_solves.forget(key);
            // node_audio_state is keyed by node_id (prefix before "__"), not stream_key.
            // Only remove if no remaining stream for that node still active.
            let node_id = key.split("__").next().unwrap_or(key.as_str());
            let node_still_active = self
                .last_heartbeat_ns_by_stream
                .keys()
                .any(|k| k.starts_with(node_id));
            if !node_still_active {
                self.node_audio_state.remove(node_id);
            }
        }
        let evicted = stale.len() as u64;
        let mut st = self.state.write().await;
        st.total_stale_streams_evicted = st.total_stale_streams_evicted.saturating_add(evicted);
        info!(count = evicted, "Evicted stale per-stream entries");
    }

    /// Mirror of the Python localization `"no_window"` drop reason. Incremented
    /// when `compute_manifest_frame` (or its callers) cannot extract an audio
    /// window for a manifest because the requested time is outside the buffer
    /// coverage. This is the single canonical "silent stall" signal — paired
    /// with `total_buffer_reanchors` it explains *why* the stall happened.
    async fn note_window_underrun_drop(&self) {
        let mut st = self.state.write().await;
        st.total_window_underrun_drops = st.total_window_underrun_drops.saturating_add(1);
    }

    /// Last step before a payload leaves for compute, in dispatch order: a
    /// covering render takes over the event solve (or waits for a parked
    /// one), then attempts are counted. Returns what to dispatch now.
    async fn prepare_dispatch(&mut self, payload: ComputePayload) -> Vec<ComputePayload> {
        let ready = self.route_through_covered_event_solves(payload);
        for payload in &ready {
            self.note_stage_attempts(payload).await;
        }
        ready
    }

    async fn release_held_renders_at_batch_boundary(&mut self) -> Vec<ComputePayload> {
        let ready = self.release_every_held_render_no_longer_waiting();
        for payload in &ready {
            self.note_stage_attempts(payload).await;
        }
        ready
    }

    async fn note_stage_attempts(&self, payload: &ComputePayload) {
        if !payload.run_srp && !payload.run_classifier_render {
            return;
        }
        let mut st = self.state.write().await;
        if payload.run_srp {
            st.total_localization_attempts += 1;
        }
        if payload.run_classifier_render {
            st.total_classification_attempts += 1;
        }
    }
}

pub(crate) fn system_now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .expect("system clock is before UNIX_EPOCH — RTC fault or misconfiguration")
}

#[cfg(test)]
#[path = "dsp_worker_tests.rs"]
mod dsp_worker_tests;

#[cfg(test)]
#[path = "dsp_worker_coverage_race_tests.rs"]
mod dsp_worker_coverage_race_tests;

#[cfg(test)]
#[path = "dsp_worker_never_streamed_member_tests.rs"]
mod dsp_worker_never_streamed_member_tests;

#[cfg(test)]
#[path = "dsp_worker_clock_evidence_tests.rs"]
mod dsp_worker_clock_evidence_tests;
#[cfg(test)]
#[path = "dsp_worker_timeline_discipline_tests.rs"]
mod dsp_worker_timeline_discipline_tests;

#[cfg(test)]
#[path = "dsp_worker_test_node_fixtures.rs"]
mod dsp_worker_test_node_fixtures;

#[cfg(test)]
#[path = "dsp_worker_solve_manifest_inspection.rs"]
mod dsp_worker_solve_manifest_inspection;

#[cfg(test)]
#[path = "dsp_worker_mixed_cluster_solve_tests.rs"]
mod dsp_worker_mixed_cluster_solve_tests;

#[cfg(test)]
#[path = "dsp_worker_covered_render_solve_tests.rs"]
mod dsp_worker_covered_render_solve_tests;

#[cfg(test)]
#[path = "dsp_worker_sample_rate_guard_tests.rs"]
mod dsp_worker_sample_rate_guard_tests;

#[cfg(test)]
#[path = "dsp_worker_localization_trigger_tests.rs"]
mod dsp_worker_localization_trigger_tests;
