//! Worker-level coverage race tests: two tetrahedral nodes share one cluster
//! buffer and node b delivers each frame after node a's next frame.

use super::dsp_worker_tests::{
    drain_published_manifests, encode_pcm16le_b64, raw_manifest_for_payload_with_metadata,
    store_forward_payload_with_channel_count,
};
use super::*;
use crate::{
    derived_cache::{DerivedCache, DerivedCacheConfig},
    dsp_events::{DspEventPublisher, ReplayableDspEvent},
    localization_coverage_wait::{
        CoverageWaitClock, STOP_REASON_ALL_COVERED, STOP_REASON_STALLED, STOP_REASON_TOTAL_CAP,
    },
    manifests::ManifestStore,
};
use std::sync::atomic::AtomicU64;
use tokio::sync::{broadcast, RwLock};

const CLUSTER_ID: &str = "coverage-race";
const CLUSTER_BUFFER_KEY: &str = "cluster::coverage-race::audio_main";
pub(super) const NODE_A: &str = "node-a";
pub(super) const NODE_B: &str = "node-b";
const TETRA_CHANNEL_COUNT: usize = 4;
const CLUSTER_CHANNEL_COUNT: usize = 2 * TETRA_CHANNEL_COUNT;
const FRAME_SAMPLES: u64 = 512;
pub(super) const FRAME_DURATION_NS: u64 = 32_000_000;
const STREAM_START_NS: u64 = 1_000_000_000;
const NODE_SEPARATION_M: f32 = 20.0;
const NODE_HEIGHT_M: f32 = 1.5;
const EVENT_CHANNEL_CAPACITY: usize = 4096;
/// Frames spanning two trigger cooldowns (0.8 s default) after the first.
pub(super) const FRAMES_PER_RUN: usize = 60;
const NANOSECONDS_PER_MILLISECOND_F64: f64 = 1_000_000.0;
/// Segment ids are `seg-<node>-<frame>`; frame 0 precedes node b's stream.
const FIRST_FRAME_SEGMENT_SUFFIX: &str = "-0";

pub(super) struct RaceHarness {
    _tmp: tempfile::TempDir,
    worker: DspWorker,
    pub(super) state: SharedDspState,
    pub(super) events: broadcast::Receiver<ReplayableDspEvent>,
    clock_now_ns: Arc<AtomicU64>,
}

impl RaceHarness {
    pub(super) async fn new(config: DspWorkerConfig) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let manifest_store = ManifestStore::new(tmp.path());
        manifest_store.ensure_initialized().await.unwrap();
        let derived_cache = DerivedCache::new(
            tmp.path(),
            DerivedCacheConfig {
                budget_bytes: 16_777_216,
                admission_reserve_bytes: 0,
            },
        );
        derived_cache.ensure_initialized().await.unwrap();
        let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
        let publisher = DspEventPublisher::new(
            state.clone(),
            EVENT_CHANNEL_CAPACITY,
            EVENT_CHANNEL_CAPACITY,
            50,
        );
        let events = publisher.subscribe();
        let clock_now_ns = Arc::new(AtomicU64::new(0));
        let worker = DspWorker::new(manifest_store, derived_cache, config, state.clone())
            .with_dsp_event_publisher(publisher)
            .with_coverage_wait_clock(CoverageWaitClock::Manual(clock_now_ns.clone()));
        Self {
            _tmp: tmp,
            worker,
            state,
            events,
            clock_now_ns,
        }
    }

    pub(super) fn advance_clock_ms(&self, milliseconds: u64) {
        self.clock_now_ns
            .fetch_add(milliseconds * 1_000_000, Ordering::AcqRel);
    }

    /// Delivers a caller-built manifest (fixture nodes, custom geometry).
    pub(super) async fn deliver_manifest(&mut self, manifest: DspManifest) {
        self.worker.process_one(manifest, 1).await;
    }

    pub(super) fn storage_root(&self) -> &std::path::Path {
        self._tmp.path()
    }

    pub(super) fn has_pending_localization(&self, buffer_key: &str) -> bool {
        self.worker.deferred_localizations.has_pending(buffer_key)
    }

    pub(super) async fn deliver(&mut self, node_id: &str, frame_index: usize) {
        self.deliver_with_cluster_positions(node_id, frame_index, two_tetra_cluster_positions())
            .await;
    }

    async fn deliver_with_cluster_positions(
        &mut self,
        node_id: &str,
        frame_index: usize,
        cluster_sensor_positions: Vec<(String, [f32; 3])>,
    ) {
        let manifest = self
            .frame_manifest(node_id, frame_index, cluster_sensor_positions, false)
            .await;
        self.worker.process_one(manifest, 1).await;
    }

    async fn frame_manifest(
        &self,
        node_id: &str,
        frame_index: usize,
        cluster_sensor_positions: Vec<(String, [f32; 3])>,
        silent: bool,
    ) -> DspManifest {
        let frame_index_u64 = frame_index as u64;
        let mut payload = store_forward_payload_with_channel_count(
            STREAM_START_NS + frame_index_u64 * FRAME_DURATION_NS,
            frame_index_u64 * FRAME_SAMPLES,
            frame_index_u64 + 1,
            TETRA_CHANNEL_COUNT,
        );
        if silent {
            payload = silence_payload_samples(&payload);
        }
        raw_manifest_for_payload_with_metadata(
            self._tmp.path(),
            &format!("manifest-{node_id}-{frame_index}"),
            &format!("seg-{node_id}-{frame_index}"),
            &format!("{node_id}__audio_main__0001"),
            payload,
            Some(CLUSTER_ID.to_string()),
            Some(cluster_sensor_positions),
        )
        .await
    }

    /// Active-channel count of every cadence-triggered localization. Frame
    /// 0's solve (node b not streaming yet) is excluded, so every counted
    /// solve happens after both nodes started streaming.
    fn solved_localization_channel_counts(&mut self) -> Vec<usize> {
        drain_published_manifests(&mut self.events)
            .iter()
            .filter(|manifest| manifest.manifest_type == "localization_result")
            .filter(|manifest| {
                !manifest
                    .source_handles
                    .iter()
                    .any(|handle| handle.segment_id.ends_with(FIRST_FRAME_SEGMENT_SUFFIX))
            })
            .filter(|manifest| {
                manifest.localization.as_ref().is_some_and(|localization| {
                    localization.resolved_algorithm != "localization_cadence_skipped"
                })
            })
            .filter_map(|manifest| manifest.coverage_stats.as_ref())
            .filter(|coverage| coverage.get("window_type").is_none())
            .filter_map(|coverage| coverage.get("active_channels")?.as_array().map(Vec::len))
            .collect()
    }

    async fn stop_reason_count(&self, stop_reason: &str) -> u64 {
        let st = self.state.read().await;
        st.localization_coverage_wait_stop_reasons
            .get(stop_reason)
            .copied()
            .unwrap_or(0)
    }

    async fn max_wait_ms(&self) -> f64 {
        self.state.read().await.localization_coverage_wait_max_ms
    }
}

fn silence_payload_samples(payload: &str) -> String {
    let mut payload_json: serde_json::Value = serde_json::from_str(payload).unwrap();
    let silent_channels = vec![vec![0.0_f32; FRAME_SAMPLES as usize]; TETRA_CHANNEL_COUNT];
    payload_json["buffered_frames"][0]["frame"]["samples_b64"] =
        serde_json::Value::String(encode_pcm16le_b64(&silent_channels));
    payload_json.to_string()
}

fn two_tetra_cluster_positions() -> Vec<(String, [f32; 3])> {
    tetra_cluster_positions(&[(NODE_A, 0.0), (NODE_B, NODE_SEPARATION_M)])
}

fn tetra_cluster_positions(nodes: &[(&str, f32)]) -> Vec<(String, [f32; 3])> {
    nodes
        .iter()
        .flat_map(|(node_id, east_m)| {
            SIRITH_MIC_POSITIONS_M
                .iter()
                .enumerate()
                .map(move |(channel_index, offset_m)| {
                    (
                        format!("{node_id}:ch{channel_index}"),
                        [
                            east_m + offset_m[0],
                            offset_m[1],
                            NODE_HEIGHT_M + offset_m[2],
                        ],
                    )
                })
        })
        .collect()
}

/// Production cadence/cooldown defaults; renders disabled after the first
/// frame so every later result carries localization coverage.
pub(super) fn race_config(coverage_wait_enabled: bool) -> DspWorkerConfig {
    DspWorkerConfig {
        classifier_render_min_interval_seconds: 1.0e6,
        localization_rms_gate: 0.0,
        localization_coverage_wait_all_streaming_nodes_enabled: coverage_wait_enabled,
        ..DspWorkerConfig::default()
    }
}

/// Node b's frame k arrives after node a's frame k+1 (one frame of network
/// skew); node a's frames arrive in real time.
async fn deliver_with_node_b_one_frame_late(harness: &mut RaceHarness, frame_count: usize) {
    for frame_index in 0..frame_count {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        if frame_index > 0 {
            harness.deliver(NODE_B, frame_index - 1).await;
        }
    }
    harness.deliver(NODE_B, frame_count - 1).await;
}

#[tokio::test]
async fn legacy_dispatch_localizes_the_lagging_cluster_alone() {
    // Reproduction of the bug (flag off == pre-fix dispatch): every
    // cooldown-gated solve runs on node a's four channels only.
    let mut harness = RaceHarness::new(race_config(false)).await;
    deliver_with_node_b_one_frame_late(&mut harness, FRAMES_PER_RUN).await;
    let channel_counts = harness.solved_localization_channel_counts();
    assert!(channel_counts.len() >= 2, "{channel_counts:?}");
    assert!(
        channel_counts
            .iter()
            .all(|count| *count == TETRA_CHANNEL_COUNT),
        "legacy dispatch should exclude the late node: {channel_counts:?}"
    );
}

#[tokio::test]
async fn coverage_wait_includes_the_late_node_in_every_cluster_localization() {
    let mut harness = RaceHarness::new(race_config(true)).await;
    deliver_with_node_b_one_frame_late(&mut harness, FRAMES_PER_RUN).await;
    let channel_counts = harness.solved_localization_channel_counts();
    assert!(channel_counts.len() >= 2, "{channel_counts:?}");
    assert!(
        channel_counts
            .iter()
            .all(|count| *count == CLUSTER_CHANNEL_COUNT),
        "every solve after node b started streaming must use both nodes: {channel_counts:?}"
    );
    // Node b lands within the same 32 ms frame tick, so the wall-clock wait
    // is zero on the manual clock; in production it equals the arrival skew.
    assert!(
        harness.max_wait_ms().await <= (FRAME_DURATION_NS as f64 / NANOSECONDS_PER_MILLISECOND_F64)
    );
    assert!(!harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
}

#[tokio::test]
async fn all_nodes_present_adds_no_wait() {
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..FRAMES_PER_RUN {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        harness.deliver(NODE_B, frame_index).await;
    }
    let channel_counts = harness.solved_localization_channel_counts();
    assert!(channel_counts.len() >= 2, "{channel_counts:?}");
    assert!(channel_counts
        .iter()
        .all(|count| *count == CLUSTER_CHANNEL_COUNT));
    assert_eq!(harness.max_wait_ms().await, 0.0);
    assert_eq!(harness.stop_reason_count(STOP_REASON_STALLED).await, 0);
    assert_eq!(harness.stop_reason_count(STOP_REASON_TOTAL_CAP).await, 0);
    assert!(harness.stop_reason_count(STOP_REASON_ALL_COVERED).await >= 2);
}

#[tokio::test]
async fn stalled_node_is_released_within_the_stall_budget_and_then_costs_nothing() {
    const NODE_B_LAST_FRAME: usize = 10;
    const FRAME_COUNT: usize = 80;
    let config = race_config(true);
    let stall_budget_ms = config.localization_coverage_stall_budget_seconds * 1_000.0;
    let mut harness = RaceHarness::new(config).await;
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        if (1..=NODE_B_LAST_FRAME + 1).contains(&frame_index) {
            harness.deliver(NODE_B, frame_index - 1).await;
        }
    }
    let channel_counts = harness.solved_localization_channel_counts();
    // Cadence triggers at frames 25, 50, 75: none is lost, none waits forever.
    assert_eq!(channel_counts.len(), 3, "{channel_counts:?}");
    assert!(channel_counts
        .iter()
        .all(|count| *count == TETRA_CHANNEL_COUNT));
    assert_eq!(harness.stop_reason_count(STOP_REASON_STALLED).await, 3);
    let max_wait_ms = harness.max_wait_ms().await;
    assert!(
        max_wait_ms > 0.0 && max_wait_ms <= stall_budget_ms,
        "dead node must cost at most one stall budget: {max_wait_ms} ms"
    );
    assert!(!harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
}

#[tokio::test]
async fn total_cap_releases_a_window_even_while_the_late_node_keeps_advancing() {
    const TRIGGER_FRAME: usize = 25;
    let config = DspWorkerConfig {
        localization_coverage_stall_budget_seconds: 10.0,
        localization_coverage_total_cap_seconds: 0.1,
        ..race_config(true)
    };
    let cap_ms = config.localization_coverage_total_cap_seconds * 1_000.0;
    let mut harness = RaceHarness::new(config).await;
    // Node b runs two frames behind (64 ms of audio, inside the 100 ms cap),
    // so node a's trigger at frame 25 is held for it.
    for frame_index in 0..=TRIGGER_FRAME {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        if frame_index >= 2 {
            harness.deliver(NODE_B, frame_index - 2).await;
        }
    }
    assert!(harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
    // Node b advances again (still short of the window) only after the cap
    // elapsed in wall time: the cap, not the stall budget, must release it.
    harness.advance_clock_ms(2 * cap_ms as u64);
    harness.deliver(NODE_B, TRIGGER_FRAME - 1).await;
    assert!(!harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
    assert_eq!(harness.stop_reason_count(STOP_REASON_TOTAL_CAP).await, 1);
}

#[tokio::test]
async fn poll_tick_releases_a_pending_window_when_every_member_goes_quiet() {
    let config = race_config(true);
    let stall_budget_ms = (config.localization_coverage_stall_budget_seconds * 1_000.0) as u64;
    let mut harness = RaceHarness::new(config).await;
    // Stop right after node a's second trigger frame: node b never catches up
    // and no further manifest arrives to re-evaluate the pending window.
    for frame_index in 0..=25 {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        if frame_index > 0 && frame_index < 25 {
            harness.deliver(NODE_B, frame_index - 1).await;
        }
    }
    assert!(harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
    assert_eq!(
        harness.worker.run_poll_tick_deferred_localizations().await,
        0
    );
    harness.advance_clock_ms(stall_budget_ms);
    assert_eq!(
        harness.worker.run_poll_tick_deferred_localizations().await,
        1
    );
    assert!(!harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
    let channel_counts = harness.solved_localization_channel_counts();
    assert_eq!(channel_counts.last(), Some(&TETRA_CHANNEL_COUNT));
}

#[tokio::test]
async fn cluster_growth_while_pending_solves_with_the_parked_geometry() {
    // A third node joins the cluster (buffer grows 8 -> 12 channels) while a
    // window is parked. The parked payload carries the 8-sensor geometry, so
    // the release must not hand SRP channel indices it has no positions for
    // (an out-of-bounds index panics the rayon pool and aborts the sidecar).
    const TRIGGER_FRAME: usize = 25;
    const NODE_C: &str = "node-c";
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..=TRIGGER_FRAME {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        if frame_index > 0 {
            harness.deliver(NODE_B, frame_index - 1).await;
        }
    }
    assert!(harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
    let three_tetra_positions = tetra_cluster_positions(&[
        (NODE_A, 0.0),
        (NODE_B, NODE_SEPARATION_M),
        (NODE_C, 2.0 * NODE_SEPARATION_M),
    ]);
    for frame_index in TRIGGER_FRAME - 1..=TRIGGER_FRAME {
        harness
            .deliver_with_cluster_positions(NODE_C, frame_index, three_tetra_positions.clone())
            .await;
    }
    harness.deliver(NODE_B, TRIGGER_FRAME).await;
    assert!(!harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
    let channel_counts = harness.solved_localization_channel_counts();
    assert_eq!(channel_counts.last(), Some(&CLUSTER_CHANNEL_COUNT));
}

#[tokio::test]
async fn backlog_batch_decides_only_after_newest_first_backfill() {
    // One poll batch holds node b's frames 25 and 26 and node a's trigger
    // frame 25. Production processes it newest-first: b26 (silent, so it does
    // not trigger), then a25 (triggers), then b25. Mid-batch, b's buffer end is
    // already past the window while frame 25 is still a hole; deciding there
    // solved on node a alone. The decision must wait for the batch boundary.
    const TRIGGER_FRAME: usize = 25;
    const BELOW_SIGNAL_RMS_GATE: f32 = 0.01;
    let config = DspWorkerConfig {
        localization_rms_gate: BELOW_SIGNAL_RMS_GATE,
        ..race_config(true)
    };
    let mut harness = RaceHarness::new(config).await;
    for frame_index in 0..TRIGGER_FRAME {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        harness.deliver(NODE_B, frame_index).await;
    }
    harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
    let positions = two_tetra_cluster_positions();
    let batch_received_ns = system_now_ns();
    let mut batch = Vec::new();
    for (arrival_rank, node_id, frame_index, silent) in [
        (1, NODE_B, TRIGGER_FRAME, false),
        (2, NODE_A, TRIGGER_FRAME, false),
        (3, NODE_B, TRIGGER_FRAME + 1, true),
    ] {
        let mut manifest = harness
            .frame_manifest(node_id, frame_index, positions.clone(), silent)
            .await;
        manifest.created_ns = batch_received_ns + arrival_rank;
        batch.push(manifest);
    }
    harness.worker.process_batch_inline(batch).await;
    let channel_counts = harness.solved_localization_channel_counts();
    assert_eq!(
        channel_counts.last(),
        Some(&CLUSTER_CHANNEL_COUNT),
        "trigger-frame solve must include node b's backfilled frame: {channel_counts:?}"
    );
    assert!(!harness
        .worker
        .deferred_localizations
        .has_pending(CLUSTER_BUFFER_KEY));
}

#[tokio::test]
async fn lossy_late_node_costs_one_frame_gap_not_the_stall_budget() {
    // Pico nodes lose ~45% of frames. Node b runs one frame late and drops
    // 5 of every 11 frames, including both trigger frames (25 and 50). A lost
    // frame must not hold the window for the stall budget: b's next arriving
    // frame advances its buffer end past the window and releases it.
    const LOST_FRAME_RESIDUES_MOD_11: [usize; 5] = [3, 5, 6, 8, 10];
    const MAX_FRAME_GAPS_WAITED: u64 = 2;
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..FRAMES_PER_RUN {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        let node_b_frame_index = frame_index.wrapping_sub(1);
        if frame_index > 0 && !LOST_FRAME_RESIDUES_MOD_11.contains(&(node_b_frame_index % 11)) {
            harness.deliver(NODE_B, node_b_frame_index).await;
        }
    }
    let channel_counts = harness.solved_localization_channel_counts();
    assert!(channel_counts.len() >= 2, "{channel_counts:?}");
    assert_eq!(harness.stop_reason_count(STOP_REASON_STALLED).await, 0);
    assert_eq!(harness.stop_reason_count(STOP_REASON_TOTAL_CAP).await, 0);
    let max_wait_ms = harness.max_wait_ms().await;
    let frame_gap_ms =
        (MAX_FRAME_GAPS_WAITED * FRAME_DURATION_NS) as f64 / NANOSECONDS_PER_MILLISECOND_F64;
    assert!(
        max_wait_ms > 0.0 && max_wait_ms <= frame_gap_ms,
        "a lost frame must cost at most the gap to the next arriving frame: {max_wait_ms} ms"
    );
}

#[tokio::test]
async fn render_on_a_parked_trigger_frame_publishes_one_result_carrying_the_render_handle() {
    // Renders share the trigger cooldown, so every cadence trigger also has a
    // classifier render due. The render must ride the deferred payload: one
    // real localization_result per trigger, and that one carries the
    // derived_handle Python needs to load the render bundle.
    let base_config = race_config(true);
    let config = DspWorkerConfig {
        classifier_render_min_interval_seconds: base_config.trigger_cooldown_seconds,
        ..base_config
    };
    let mut harness = RaceHarness::new(config).await;
    deliver_with_node_b_one_frame_late(&mut harness, FRAMES_PER_RUN).await;
    let published = drain_published_manifests(&mut harness.events);
    let render_count = published
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .count();
    let mut real_results_by_segment: HashMap<String, Vec<&DspManifest>> = HashMap::new();
    for manifest in published.iter().filter(|manifest| {
        manifest.manifest_type == "localization_result"
            && manifest.localization.as_ref().is_some_and(|localization| {
                localization.resolved_algorithm != "localization_cadence_skipped"
            })
    }) {
        let segment_id = manifest.source_handles[0].segment_id.clone();
        real_results_by_segment
            .entry(segment_id)
            .or_default()
            .push(manifest);
    }
    let trigger_segments = real_results_by_segment
        .keys()
        .filter(|segment_id| !segment_id.ends_with(FIRST_FRAME_SEGMENT_SUFFIX))
        .count();
    assert!(trigger_segments >= 2, "{real_results_by_segment:?}");
    assert!(render_count >= trigger_segments, "renders: {render_count}");
    for (segment_id, results) in &real_results_by_segment {
        assert_eq!(results.len(), 1, "duplicate results for {segment_id}");
        // Frame 0 precedes node b's stream; its render mixes node a alone.
        assert!(
            results[0].derived_handle.is_some(),
            "trigger result for {segment_id} lost the render handle"
        );
    }
    let total_localization_results = harness.state.read().await.total_localization_results;
    assert_eq!(
        total_localization_results,
        real_results_by_segment.len() as u64
    );
}
