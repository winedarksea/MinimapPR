//! Per-cluster-buffer parking lot for localizations held by the coverage wait
//! (see `localization_coverage_wait.rs` for the policy and why it exists).
//!
//! Bounded by construction: at most one pending localization per buffer key
//! (a newer candidate supersedes and dispatches the older one), and buffer
//! keys are evicted together with their stream buffers. Ready payloads go to a
//! small outbox the worker drains after every manifest and poll tick.

use std::collections::HashMap;

use crate::{
    dsp::SensorStreamBuffer,
    dsp_worker::{
        eligible_localization_channels, localization_channel_states_centered, ComputePayload,
    },
    localization_coverage_wait::{
        evaluate_window_coverage, ChannelStreamActivityTracker, CoverageWaitClock,
        CoverageWaitDecision, LocalizationCoverageWaitConfig, LocalizationWindowSpan,
        STOP_REASON_SUPERSEDED,
    },
};

/// One finished coverage decision, for the `/api/v1/dsp/status` counters that
/// mirror Python's `localization_coverage_wait_stop_reasons` / `_max_ms`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoverageWaitOutcome {
    pub(crate) stop_reason: &'static str,
    pub(crate) waited_ns: u128,
}

struct PendingLocalization {
    span: LocalizationWindowSpan,
    first_deferred_ns: u128,
    payload_template: ComputePayload,
}

#[derive(Default)]
struct BufferCoverageWaitState {
    tracker: ChannelStreamActivityTracker,
    pending: Option<PendingLocalization>,
}

pub(crate) struct DeferredLocalizationScheduler {
    config: LocalizationCoverageWaitConfig,
    clock: CoverageWaitClock,
    states_by_buffer_key: HashMap<String, BufferCoverageWaitState>,
    ready_payloads: Vec<ComputePayload>,
    outcomes: Vec<CoverageWaitOutcome>,
}

impl DeferredLocalizationScheduler {
    pub(crate) fn new(config: LocalizationCoverageWaitConfig) -> Self {
        Self {
            config,
            clock: CoverageWaitClock::default(),
            states_by_buffer_key: HashMap::new(),
            ready_payloads: Vec::new(),
            outcomes: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn set_clock(&mut self, clock: CoverageWaitClock) {
        self.clock = clock;
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    #[cfg(test)]
    pub(crate) fn has_pending(&self, buffer_key: &str) -> bool {
        self.states_by_buffer_key
            .get(buffer_key)
            .is_some_and(|state| state.pending.is_some())
    }

    /// Record channel progress for `buffer_key` (stall accounting). Called
    /// after every append to that buffer; never dispatches mid-batch.
    pub(crate) fn observe_channel_progress(
        &mut self,
        buffer_key: &str,
        buffers: &[SensorStreamBuffer],
    ) {
        if !self.config.enabled {
            return;
        }
        let now_ns = self.clock.now_ns();
        self.states_by_buffer_key
            .entry(buffer_key.to_string())
            .or_default()
            .tracker
            .observe(buffers, now_ns);
    }

    /// Batch-boundary path (end of every DSP poll cycle): re-evaluates every
    /// pending window. Deciding only here, never mid-batch, matters because
    /// `merge_pending_manifests_for_batch` processes a backlog newest-first:
    /// mid-batch a member can show frame k+1 with frame k still a hole that
    /// the same batch backfills a moment later.
    pub(crate) fn service_all(
        &mut self,
        buffers_by_key: &HashMap<String, Vec<SensorStreamBuffer>>,
    ) {
        if !self.config.enabled {
            return;
        }
        let now_ns = self.clock.now_ns();
        for (buffer_key, state) in self.states_by_buffer_key.iter_mut() {
            if state.pending.is_none() {
                continue;
            }
            let Some(buffers) = buffers_by_key.get(buffer_key) else {
                state.pending = None;
                continue;
            };
            state.tracker.observe(buffers, now_ns);
            Self::service_state(
                state,
                buffers,
                self.config,
                now_ns,
                &mut self.ready_payloads,
                &mut self.outcomes,
            );
        }
    }

    /// Decision for a fresh localization candidate (no time waited yet).
    pub(crate) fn evaluate_new_candidate(
        &self,
        buffer_key: &str,
        buffers: &[SensorStreamBuffer],
        span: LocalizationWindowSpan,
    ) -> CoverageWaitDecision {
        let Some(state) = self.states_by_buffer_key.get(buffer_key) else {
            return CoverageWaitDecision::Ready {
                stop_reason: crate::localization_coverage_wait::STOP_REASON_ALL_COVERED,
            };
        };
        evaluate_window_coverage(
            buffers,
            span,
            &state.tracker,
            self.config,
            0,
            self.clock.now_ns(),
        )
    }

    /// Park a localization-only payload until `span` is covered. An older
    /// pending localization for the same buffer is dispatched first.
    pub(crate) fn defer(
        &mut self,
        buffer_key: &str,
        buffers: &[SensorStreamBuffer],
        span: LocalizationWindowSpan,
        payload_template: ComputePayload,
    ) {
        let now_ns = self.clock.now_ns();
        let state = self
            .states_by_buffer_key
            .entry(buffer_key.to_string())
            .or_default();
        if let Some(superseded) = state.pending.take() {
            let waited_ns = now_ns.saturating_sub(superseded.first_deferred_ns);
            self.ready_payloads
                .push(finalize_pending_payload(superseded, buffers));
            self.outcomes.push(CoverageWaitOutcome {
                stop_reason: STOP_REASON_SUPERSEDED,
                waited_ns,
            });
        }
        state.pending = Some(PendingLocalization {
            span,
            first_deferred_ns: now_ns,
            payload_template,
        });
    }

    pub(crate) fn record_immediate_outcome(&mut self, stop_reason: &'static str) {
        self.outcomes.push(CoverageWaitOutcome {
            stop_reason,
            waited_ns: 0,
        });
    }

    pub(crate) fn take_ready_payloads(&mut self) -> Vec<ComputePayload> {
        std::mem::take(&mut self.ready_payloads)
    }

    pub(crate) fn take_outcomes(&mut self) -> Vec<CoverageWaitOutcome> {
        std::mem::take(&mut self.outcomes)
    }

    /// Drop all state for an evicted buffer (stale-stream purge).
    pub(crate) fn forget(&mut self, buffer_key: &str) {
        self.states_by_buffer_key.remove(buffer_key);
    }

    fn service_state(
        state: &mut BufferCoverageWaitState,
        buffers: &[SensorStreamBuffer],
        config: LocalizationCoverageWaitConfig,
        now_ns: u128,
        ready_payloads: &mut Vec<ComputePayload>,
        outcomes: &mut Vec<CoverageWaitOutcome>,
    ) {
        let Some(pending) = state.pending.as_ref() else {
            return;
        };
        let waited_ns = now_ns.saturating_sub(pending.first_deferred_ns);
        let decision = evaluate_window_coverage(
            buffers,
            pending.span,
            &state.tracker,
            config,
            waited_ns,
            now_ns,
        );
        if let CoverageWaitDecision::Ready { stop_reason } = decision {
            let pending = state.pending.take().expect("pending checked above");
            ready_payloads.push(finalize_pending_payload(pending, buffers));
            outcomes.push(CoverageWaitOutcome {
                stop_reason,
                waited_ns,
            });
        }
    }
}

/// Re-read the (now fuller) buffers at the original window centre, so the
/// solve uses the trigger frame's span with every channel that arrived.
///
/// Only the channels the parked geometry describes are re-read: a cluster
/// buffer grows (never reorders) when a member joins, and a channel index
/// past `mic_positions_m` would panic SRP on the rayon pool (process abort).
fn finalize_pending_payload(
    pending: PendingLocalization,
    buffers: &[SensorStreamBuffer],
) -> ComputePayload {
    let PendingLocalization {
        span,
        payload_template: mut payload,
        ..
    } = pending;
    let parked_geometry_channel_count = payload.mic_positions_m.len().min(buffers.len());
    payload.channel_states = localization_channel_states_centered(
        &buffers[..parked_geometry_channel_count],
        span.center_time_ns,
        span.window_seconds,
    );
    payload.active_channels =
        eligible_localization_channels(&payload.channel_states, span.min_coverage_ratio);
    payload
}

/// Localization-only copy of a buffered-manifest payload: SRP on, no
/// classifier render, and the source manifest marked memory-path so the
/// original payload stays the only one that consumes a persisted manifest.
pub(crate) fn localization_only_template(payload: &ComputePayload) -> ComputePayload {
    let mut manifest = payload.manifest.clone();
    manifest.raw_payload = Some(Vec::new());
    let channel_count = payload.channel_states.len();
    ComputePayload {
        manifest,
        stream_key: payload.stream_key.clone(),
        channel_states: Vec::new(),
        active_channels: Vec::new(),
        classification_windows: vec![Vec::new(); channel_count],
        listenable_classification_windows: vec![Vec::new(); channel_count],
        classification_coverage: vec![None; channel_count],
        classifier_render_start_ns: None,
        classifier_render_end_ns: None,
        mic_positions_m: payload.mic_positions_m.clone(),
        half_space: payload.half_space,
        source_ids: payload.source_ids.clone(),
        sr: payload.sr,
        now_ns: payload.now_ns,
        effective_sound_speed_mps: payload.effective_sound_speed_mps,
        reported_temperature_c: payload.reported_temperature_c,
        reported_humidity_fraction: payload.reported_humidity_fraction,
        reported_environment_source: payload.reported_environment_source.clone(),
        run_srp: channel_count >= crate::dsp_worker::MIN_SRP_CHANNEL_COUNT,
        run_classifier_render: false,
        skip_localization_result: false,
        omni_fallback_reason: payload.omni_fallback_reason.clone(),
        omni_channels_override: None,
        listenable_omni_channels_override: None,
        manifest_store: payload.manifest_store.clone(),
        derived_cache: payload.derived_cache.clone(),
        state: payload.state.clone(),
        classification_tx: payload.classification_tx.clone(),
        dsp_event_publisher: payload.dsp_event_publisher.clone(),
        config: payload.config.clone(),
        consumed_since_prune: payload.consumed_since_prune.clone(),
        diagnostics: payload.diagnostics.clone(),
    }
}
