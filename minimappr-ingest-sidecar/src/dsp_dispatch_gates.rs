//! Dispatch decisions for a buffered manifest: cadence/cooldown/render
//! gates and the `ComputePayload` they produce.

use super::*;

pub(super) struct BufferedManifestTimingGates {
    pub(super) on_localization_cadence: bool,
    pub(super) on_heartbeat_cadence: bool,
    pub(super) run_classifier_render: bool,
    pub(super) run_srp: bool,
}

impl BufferedManifestTimingGates {
    /// Today's only live trigger is the absolute-RMS gate; a solve forced by a
    /// due classifier render between cadence ticks is a scheduled render.
    fn solve_attempt_kind(&self) -> SolveAttemptKind {
        if !self.run_srp {
            SolveAttemptKind::NotAttempted
        } else if self.on_localization_cadence {
            SolveAttemptKind::EventTriggerAbsoluteRms
        } else {
            SolveAttemptKind::ScheduledClassifierRender
        }
    }
}

impl DspWorker {
    pub(super) fn dispatch_large_array_manifest(
        &mut self,
        owned: OwnedManifestAudio,
        pending_backlog_depth: usize,
        reason: &str,
    ) -> Option<ComputePayload> {
        if !self.should_publish_classifier_render(
            &owned.buffer_key,
            owned.now_ns,
            pending_backlog_depth,
        ) {
            self.defer_source_manifest_consumption(&owned.manifest);
            return None;
        }

        let omni_channels = owned.decoded.channels.clone();
        Some(ComputePayload {
            manifest: owned.manifest,
            stream_key: owned.buffer_key,
            channel_states: (0..owned.buffer_channel_count)
                .map(|_| LocalizationChannelState {
                    coverage: None,
                    window: Vec::new(),
                })
                .collect(),
            active_channels: Vec::new(),
            classification_windows: (0..owned.buffer_channel_count)
                .map(|_| Vec::new())
                .collect(),
            listenable_classification_windows: (0..owned.buffer_channel_count)
                .map(|_| Vec::new())
                .collect(),
            classification_coverage: (0..owned.buffer_channel_count).map(|_| None).collect(),
            classifier_render_start_ns: Some(owned.buffer_start_time_ns.max(0) as u128),
            classifier_render_end_ns: Some(owned.buffer_end_time_ns.max(0) as u128),
            localization_window_center_time_ns: owned.buffer_start_time_ns
                + (owned.buffer_end_time_ns - owned.buffer_start_time_ns) / 2,
            sensor_layout: owned.sensor_layout,
            attempt_kind: SolveAttemptKind::NotAttempted,
            pair_admission: SolvePairAdmission::admit_all(),
            source_ids: owned.source_ids,
            sr: owned.sr,
            now_ns: owned.now_ns,
            effective_sound_speed_mps: owned.effective_sound_speed_mps,
            reported_temperature_c: owned.decoded.temperature_c,
            reported_humidity_fraction: owned.decoded.humidity_fraction,
            reported_environment_source: owned.decoded.environment_source,
            run_srp: false,
            run_classifier_render: true,
            skip_localization_result: true,
            omni_fallback_reason: Some(reason.to_string()),
            omni_channels_override: Some(omni_channels.clone()),
            listenable_omni_channels_override: Some(omni_channels),
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            dsp_event_publisher: self.dsp_event_publisher.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
            diagnostics: self.diagnostics.clone(),
        })
    }

    pub(super) fn resolve_buffered_manifest_timing_gates(
        &mut self,
        stream_key: &str,
        channel_count: usize,
        audio_end_ns: u128,
        pending_backlog_depth: usize,
        windows: &[Vec<f32>],
    ) -> BufferedManifestTimingGates {
        let on_localization_cadence =
            self.should_run_localization(stream_key, audio_end_ns, windows);
        let run_classifier_render =
            self.should_publish_classifier_render(stream_key, audio_end_ns, pending_backlog_depth);
        let run_srp = channel_count >= MIN_SRP_CHANNEL_COUNT
            && (on_localization_cadence || run_classifier_render);
        let on_heartbeat_cadence =
            on_localization_cadence || self.should_emit_heartbeat(stream_key, audio_end_ns);
        BufferedManifestTimingGates {
            on_localization_cadence,
            on_heartbeat_cadence,
            run_classifier_render,
            run_srp,
        }
    }

    pub(super) async fn dispatch_buffered_manifest(
        &mut self,
        owned: OwnedManifestAudio,
        buffered: BufferedManifestAudio,
        pending_backlog_depth: usize,
    ) -> Option<ComputePayload> {
        if let Some(buffers) = self.buffers.get(&owned.buffer_key) {
            self.deferred_localizations
                .observe_channel_progress(&owned.buffer_key, buffers);
        }
        if buffered
            .channel_states
            .iter()
            .all(|state| state.coverage.is_none())
        {
            let run_classifier_render = self.should_publish_classifier_render(
                &owned.buffer_key,
                buffered.audio_end_ns,
                pending_backlog_depth,
            );
            if run_classifier_render {
                debug!(
                    manifest_id = %owned.manifest.manifest_id,
                    "DSP worker: no localization coverage window after buffering; publishing omni fallback render"
                );
                let Some(buffers) = self.buffers.get(&owned.buffer_key) else {
                    error!(stream_key = %owned.stream_key, buffer_key = %owned.buffer_key, "stream buffers missing after append in omni fallback path — skipping");
                    return None;
                };
                let fallback_render_windows = channel_windows_ending_at(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                );
                let listenable_fallback_render_windows =
                    channel_windows_ending_at_with_gap_concealment(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    );
                let mut fallback_render_coverage = channel_coverage_ending_at(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                );
                let layout = &owned.sensor_layout;
                let mut fallback_render_windows = fallback_render_windows;
                let mut listenable_fallback_render_windows = listenable_fallback_render_windows;
                layout.clear_solve_excluded_channels(&mut fallback_render_windows);
                layout.clear_solve_excluded_channels(&mut listenable_fallback_render_windows);
                layout.clear_solve_excluded_channels(&mut fallback_render_coverage);
                let fallback_render_channels = if fallback_render_windows
                    .iter()
                    .any(|window| !window.is_empty())
                {
                    fallback_render_windows.clone()
                } else {
                    owned.decoded.channels.clone()
                };
                let listenable_fallback_render_channels = if listenable_fallback_render_windows
                    .iter()
                    .any(|window| !window.is_empty())
                {
                    listenable_fallback_render_windows.clone()
                } else {
                    fallback_render_channels.clone()
                };
                let (classifier_render_start_ns, classifier_render_end_ns) =
                    classifier_render_bounds_from_windows(
                        buffered.end_ns,
                        &fallback_render_channels,
                        owned.sr,
                    );
                return Some(ComputePayload {
                    manifest: owned.manifest,
                    stream_key: owned.buffer_key,
                    channel_states: buffered.channel_states,
                    active_channels: eligible_coverage_channels(
                        &fallback_render_coverage,
                        self.config.min_coverage_ratio,
                    ),
                    classification_windows: (0..owned.buffer_channel_count)
                        .map(|_| Vec::new())
                        .collect(),
                    listenable_classification_windows: (0..owned.buffer_channel_count)
                        .map(|_| Vec::new())
                        .collect(),
                    classification_coverage: fallback_render_coverage,
                    classifier_render_start_ns,
                    classifier_render_end_ns,
                    localization_window_center_time_ns: buffered.localization_center_time_ns,
                    sensor_layout: owned.sensor_layout,
                    attempt_kind: SolveAttemptKind::NotAttempted,
                    pair_admission: SolvePairAdmission::admit_all(),
                    source_ids: owned.source_ids,
                    sr: owned.sr,
                    now_ns: owned.now_ns,
                    effective_sound_speed_mps: owned.effective_sound_speed_mps,
                    reported_temperature_c: owned.decoded.temperature_c,
                    reported_humidity_fraction: owned.decoded.humidity_fraction,
                    reported_environment_source: owned.decoded.environment_source,
                    run_srp: false,
                    run_classifier_render: true,
                    skip_localization_result: true,
                    omni_fallback_reason: Some("localization_coverage_unavailable".to_string()),
                    omni_channels_override: Some(fallback_render_channels),
                    listenable_omni_channels_override: Some(listenable_fallback_render_channels),
                    manifest_store: self.manifest_store.clone(),
                    derived_cache: self.derived_cache.clone(),
                    state: self.state.clone(),
                    classification_tx: self.classification_tx.clone(),
                    dsp_event_publisher: self.dsp_event_publisher.clone(),
                    config: self.config.clone(),
                    consumed_since_prune: self.consumed_manifests_since_prune.clone(),
                    diagnostics: self.diagnostics.clone(),
                });
            }
        }

        let active_channels = eligible_localization_channels(
            &buffered.channel_states,
            self.config.min_coverage_ratio,
        );
        let windows: Vec<Vec<f32>> = buffered
            .channel_states
            .iter()
            .map(|state| state.window.clone())
            .collect();
        let mut timing_gates = self.resolve_buffered_manifest_timing_gates(
            &owned.buffer_key,
            owned.buffer_channel_count,
            buffered.audio_end_ns,
            pending_backlog_depth,
            &windows,
        );
        // Taken before a coverage deferral clears `run_srp`: the parked
        // template solves for this same trigger.
        let trigger_attempt_kind = timing_gates.solve_attempt_kind();
        let coverage_deferral_span = self.coverage_deferral_span(
            &owned.buffer_key,
            buffered.localization_center_time_ns,
            timing_gates.on_localization_cadence,
            owned.buffer_channel_count,
        );
        let mut deferred_classifier_render = None;
        if coverage_deferral_span.is_some() {
            // The cadence-triggered solve and any classifier render due on this
            // frame move to the deferred payload, so the trigger publishes one
            // localization_result (carrying the render handle). This frame
            // still emits its heartbeat.
            deferred_classifier_render =
                timing_gates
                    .run_classifier_render
                    .then_some(DeferredClassifierRender {
                        end_ns: buffered.end_ns,
                        window_seconds: owned.classification_window_sec,
                    });
            timing_gates.run_srp = false;
            timing_gates.run_classifier_render = false;
        }

        if !timing_gates.on_heartbeat_cadence && !timing_gates.run_classifier_render {
            // Parity with Python's localization "no_window" silent-drop reason:
            // when every per-channel window is empty AND no cadence wants a
            // result, the frame is dropped invisibly. The Python pipeline now
            // counts the corresponding state via
            // `localization_drops_by_reason["no_window"]`; bump the equivalent
            // here so a Rust-backed deployment exposes the same signal via
            // /api/v1/dsp/status.total_window_underrun_drops. Heartbeat /
            // classifier-render dispatches are intentional emissions and not
            // counted as a drop.
            let all_windows_empty = windows.iter().all(|window| window.is_empty());
            if all_windows_empty {
                self.note_window_underrun_drop().await;
            }
            self.defer_source_manifest_consumption(&owned.manifest);
            return None;
        }

        let (
            mut classification_windows,
            mut listenable_classification_windows,
            mut classification_coverage,
        ) = if timing_gates.run_classifier_render {
            let Some(buffers) = self.buffers.get(&owned.buffer_key) else {
                error!(stream_key = %owned.stream_key, buffer_key = %owned.buffer_key, "stream buffers missing after append in classifier render path — skipping");
                return None;
            };
            (
                channel_windows_ending_at(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                ),
                channel_windows_ending_at_with_gap_concealment(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                ),
                channel_coverage_ending_at(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                ),
            )
        } else {
            (
                buffered.channel_states.iter().map(|_| Vec::new()).collect(),
                buffered.channel_states.iter().map(|_| Vec::new()).collect(),
                buffered.channel_states.iter().map(|_| None).collect(),
            )
        };
        let layout = &owned.sensor_layout;
        layout.clear_solve_excluded_channels(&mut classification_windows);
        layout.clear_solve_excluded_channels(&mut listenable_classification_windows);
        layout.clear_solve_excluded_channels(&mut classification_coverage);
        let (classifier_render_start_ns, classifier_render_end_ns) =
            if timing_gates.run_classifier_render {
                classifier_render_bounds_from_windows(
                    buffered.end_ns,
                    &classification_windows,
                    owned.sr,
                )
            } else {
                (None, None)
            };

        let payload = ComputePayload {
            manifest: owned.manifest,
            stream_key: owned.buffer_key,
            channel_states: buffered.channel_states,
            active_channels,
            classification_windows,
            listenable_classification_windows,
            classification_coverage,
            classifier_render_start_ns,
            classifier_render_end_ns,
            localization_window_center_time_ns: buffered.localization_center_time_ns,
            sensor_layout: owned.sensor_layout,
            attempt_kind: if timing_gates.run_srp {
                trigger_attempt_kind
            } else {
                SolveAttemptKind::NotAttempted
            },
            pair_admission: SolvePairAdmission::admit_all(),
            source_ids: owned.source_ids,
            sr: owned.sr,
            now_ns: owned.now_ns,
            effective_sound_speed_mps: owned.effective_sound_speed_mps,
            reported_temperature_c: owned.decoded.temperature_c,
            reported_humidity_fraction: owned.decoded.humidity_fraction,
            reported_environment_source: owned.decoded.environment_source,
            run_srp: timing_gates.run_srp,
            run_classifier_render: timing_gates.run_classifier_render,
            skip_localization_result: false,
            omni_fallback_reason: (owned.buffer_channel_count < 4)
                .then(|| "single_point_node".to_string()),
            omni_channels_override: None,
            listenable_omni_channels_override: None,
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            dsp_event_publisher: self.dsp_event_publisher.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
            diagnostics: self.diagnostics.clone(),
        };
        if let (Some(span), Some(buffers)) = (
            coverage_deferral_span,
            self.buffers.get(&payload.stream_key),
        ) {
            let template = localization_only_template(&payload, trigger_attempt_kind);
            self.deferred_localizations.defer(
                &payload.stream_key,
                buffers,
                span,
                template,
                deferred_classifier_render,
            );
        }
        Some(payload)
    }

    pub(super) fn should_run_localization(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        windows: &[Vec<f32>],
    ) -> bool {
        if !self.should_trigger(stream_key, audio_ns, windows) {
            return false;
        }
        let cadence_ns = (self.config.localization_cadence_ms as u128) * 1_000_000;
        if cadence_ns == 0 {
            return true;
        }
        match self.last_localization_ns_by_stream.get(stream_key).copied() {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < cadence_ns => false,
            _ => {
                self.last_localization_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }

    /// Combined RMS energy gate + trigger cooldown, matching Python's IngestProcessor
    /// trigger evaluation. Returns true when the audio is loud enough and the cooldown
    /// period has elapsed since the last trigger for this stream.
    pub(super) fn should_trigger(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        windows: &[Vec<f32>],
    ) -> bool {
        if self.config.localization_rms_gate > 0.0 {
            let max_rms = windows
                .iter()
                .map(|w| {
                    if w.is_empty() {
                        0.0_f32
                    } else {
                        (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt()
                    }
                })
                .fold(0.0_f32, f32::max);
            if max_rms < self.config.localization_rms_gate {
                return false;
            }
        }
        let cooldown_ns = (self.config.trigger_cooldown_seconds * 1_000_000_000.0).round() as u128;
        if cooldown_ns == 0 {
            return true;
        }
        match self.last_trigger_ns_by_stream.get(stream_key).copied() {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < cooldown_ns => false,
            _ => {
                self.last_trigger_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }

    /// Time-only cadence check for node liveness heartbeats.
    /// Never blocked by the RMS energy gate or trigger cooldown.
    pub(super) fn should_emit_heartbeat(&mut self, stream_key: &str, audio_ns: u128) -> bool {
        let cadence_ns = (self.config.localization_cadence_ms as u128) * 1_000_000;
        if cadence_ns == 0 {
            return true;
        }
        match self.last_heartbeat_ns_by_stream.get(stream_key).copied() {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < cadence_ns => false,
            _ => {
                self.last_heartbeat_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }

    pub(super) fn should_publish_classifier_render(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        pending_backlog_depth: usize,
    ) -> bool {
        let min_interval_ns = classifier_render_min_interval_ns(
            self.config.classifier_render_min_interval_seconds,
            pending_backlog_depth,
        );
        if min_interval_ns == 0 {
            return true;
        }
        match self
            .last_classifier_render_ns_by_stream
            .get(stream_key)
            .copied()
        {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < min_interval_ns => false,
            _ => {
                self.last_classifier_render_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }
}
