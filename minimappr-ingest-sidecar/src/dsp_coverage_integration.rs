//! Worker side of the localization coverage wait: when to park a
//! cadence-triggered solve, draining released payloads, and test seams.

use super::*;

impl DspWorker {
    /// `Some(span)` when this cadence-triggered localization must wait for
    /// streaming cluster members that have not yet delivered the window.
    pub(super) fn coverage_deferral_span(
        &mut self,
        buffer_key: &str,
        localization_center_time_ns: i128,
        on_localization_cadence: bool,
        buffer_channel_count: usize,
    ) -> Option<LocalizationWindowSpan> {
        if !on_localization_cadence
            || !self.deferred_localizations.is_enabled()
            || buffer_channel_count < MIN_SRP_CHANNEL_COUNT
        {
            return None;
        }
        let buffers = self.buffers.get(buffer_key)?;
        let span = LocalizationWindowSpan {
            center_time_ns: localization_center_time_ns,
            window_seconds: self.config.window_seconds,
            min_coverage_ratio: self.config.min_coverage_ratio,
        };
        // Cluster buffers always park until the batch boundary: mid-batch a
        // member may show holes that later frames in the same (newest-first)
        // batch backfill. Costs no wall time; released at the end of the cycle.
        if buffer_key.starts_with(CLUSTER_BUFFER_KEY_PREFIX) {
            return Some(span);
        }
        match self
            .deferred_localizations
            .evaluate_new_candidate(buffer_key, buffers, span)
        {
            CoverageWaitDecision::Wait => Some(span),
            CoverageWaitDecision::Ready { stop_reason } => {
                self.deferred_localizations
                    .record_immediate_outcome(stop_reason);
                None
            }
        }
    }

    /// Drain localizations released by the coverage wait. `poll_tick` also
    /// re-evaluates every pending buffer (covers members that all went quiet).
    pub(super) async fn take_ready_deferred_localizations(
        &mut self,
        poll_tick: bool,
    ) -> Vec<ComputePayload> {
        if poll_tick {
            self.deferred_localizations.service_all(&self.buffers);
        }
        let outcomes = self.deferred_localizations.take_outcomes();
        if !outcomes.is_empty() {
            let mut st = self.state.write().await;
            for outcome in outcomes {
                *st.localization_coverage_wait_stop_reasons
                    .entry(outcome.stop_reason.to_string())
                    .or_insert(0) += 1;
                let waited_ms = outcome.waited_ns as f64 / NANOSECONDS_PER_MILLISECOND_F64;
                st.localization_coverage_wait_max_ms =
                    st.localization_coverage_wait_max_ms.max(waited_ms);
            }
        }
        self.deferred_localizations.take_ready_payloads()
    }

    #[cfg(test)]
    pub(crate) async fn process_one(
        &mut self,
        manifest: DspManifest,
        pending_backlog_depth: usize,
    ) {
        // A test delivery is a batch of one, so it ends at a batch boundary.
        let ingested_payload = self.ingest_one(manifest, pending_backlog_depth).await;
        let ready_payloads = self.take_ready_deferred_localizations(true).await;
        for payload in ingested_payload.into_iter().chain(ready_payloads) {
            self.note_stage_attempts(&payload).await;
            let result = crate::actors::dsp_compute::run_math(payload);
            crate::actors::dsp_compute::run_io(result).await;
        }
        self.flush_deferred_source_manifest_consumptions().await;
    }

    /// Test seam for one `process_pending` batch, newest-first as production
    /// orders it, with compute run inline instead of on the rayon pool.
    #[cfg(test)]
    pub(crate) async fn process_batch_inline(&mut self, manifests: Vec<DspManifest>) {
        let pending = merge_pending_manifests_for_batch(manifests, Vec::new(), usize::MAX);
        let pending_backlog_depth = pending.len();
        let mut dispatched_payloads = Vec::new();
        let mut collect_payload = |payload: ComputePayload| dispatched_payloads.push(payload);
        self.ingest_batch_and_dispatch(pending, pending_backlog_depth, &mut collect_payload)
            .await;
        for payload in dispatched_payloads {
            let result = crate::actors::dsp_compute::run_math(payload);
            crate::actors::dsp_compute::run_io(result).await;
        }
        self.flush_deferred_source_manifest_consumptions().await;
    }

    /// Test seam for the poll-tick release path of `process_pending`.
    #[cfg(test)]
    pub(crate) async fn run_poll_tick_deferred_localizations(&mut self) -> usize {
        let ready_payloads = self.take_ready_deferred_localizations(true).await;
        let released_count = ready_payloads.len();
        for payload in ready_payloads {
            self.note_stage_attempts(&payload).await;
            let result = crate::actors::dsp_compute::run_math(payload);
            crate::actors::dsp_compute::run_io(result).await;
        }
        released_count
    }
}
