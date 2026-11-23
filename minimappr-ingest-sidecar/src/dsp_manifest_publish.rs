//! Free publish/consume helpers shared by the ingest path and
//! `actors/dsp_compute`.

use super::*;

pub(super) struct RawAudioFramePublishRequest<'a> {
    pub(super) publisher: &'a Option<DspEventPublisher>,
    pub(super) source_manifest: &'a DspManifest,
    pub(super) decoded: &'a DecodedAudioPayload,
    pub(super) stream_key: &'a str,
    pub(super) sample_rate_hz: u32,
    pub(super) start_time_ns: i128,
    pub(super) end_time_ns: i128,
    pub(super) created_ns: u128,
    pub(super) segment_index: usize,
    pub(super) segment_count: usize,
}

impl DspWorker {
    pub(super) fn defer_source_manifest_consumption(&mut self, manifest: &DspManifest) {
        if !source_manifest_was_persisted(manifest) {
            return;
        }
        self.deferred_source_manifest_ids
            .push(manifest.manifest_id.clone());
    }

    pub(super) async fn mark_source_manifest_consumed_if_persisted(&self, manifest: &DspManifest) {
        if !source_manifest_was_persisted(manifest) {
            return;
        }
        let _ = self
            .manifest_store
            .mark_consumed(&manifest.manifest_id)
            .await;
    }

    pub(super) async fn flush_deferred_source_manifest_consumptions(&mut self) {
        if self.deferred_source_manifest_ids.is_empty() {
            return;
        }
        let manifest_ids = std::mem::take(&mut self.deferred_source_manifest_ids);
        let mut unique_manifest_ids = HashSet::with_capacity(manifest_ids.len());
        for id in manifest_ids {
            unique_manifest_ids.insert(id);
        }

        // Run all renames concurrently — on tmpfs these are in-memory directory
        // entry updates and complete in microseconds even at batch depth 128.
        let mut set = tokio::task::JoinSet::new();
        for manifest_id in unique_manifest_ids {
            let store = self.manifest_store.clone();
            set.spawn(async move {
                let result = store.mark_consumed(&manifest_id).await;
                (manifest_id, result)
            });
        }
        let mut successful_consumes = 0_u64;
        while let Some(result) = set.join_next().await {
            match result {
                Ok((_, Ok(()))) => successful_consumes += 1,
                Ok((manifest_id, Err(err))) => {
                    warn!(
                        manifest_id = %manifest_id,
                        error = %err,
                        "DSP worker failed to mark manifest consumed"
                    );
                }
                Err(join_err) => {
                    warn!(error = %join_err, "mark_consumed task panicked");
                }
            }
        }

        if successful_consumes == 0 {
            return;
        }

        let prune_interval = self.config.consumed_manifest_prune_interval.max(1);
        let previous = self
            .consumed_manifests_since_prune
            .fetch_add(successful_consumes, Ordering::Relaxed);
        let current = previous.saturating_add(successful_consumes);
        let count_due = previous / prune_interval < current / prune_interval;
        let prune_max_age_ns =
            (self.config.consumed_manifest_prune_max_age_seconds as u128) * 1_000_000_000;
        let time_due =
            system_now_ns().saturating_sub(self.last_consumed_prune_ns) >= prune_max_age_ns;
        if count_due || time_due {
            self.last_consumed_prune_ns = system_now_ns();
            let store = self.manifest_store.clone();
            let retention_max_files = self.config.consumed_manifest_retention_max_files;
            tokio::spawn(async move {
                if let Err(error) = store.prune_consumed_manifests(retention_max_files).await {
                    warn!(error = %error, "DSP worker failed to prune consumed manifests");
                }
            });
        }
    }
}

pub(crate) async fn dispatch_classification_result_standalone(
    result: crate::dsp_render_output::RenderPublishResult,
    classification_tx: &Option<flume::Sender<ClassificationRequest>>,
    state: &SharedDspState,
    raw_render_bytes: Option<String>,
) -> Option<DspManifest> {
    let pending = result.pending_manifest?;
    if let Some(tx) = classification_tx {
        let pcm_bytes = result.pcm_bytes.unwrap_or_default();
        let mut pending_with_bytes = pending.clone();
        pending_with_bytes.raw_render_bytes = raw_render_bytes.clone();
        let req = ClassificationRequest {
            pcm_bytes,
            sample_rate_hz: result.sample_rate_hz,
            pending_manifest: pending_with_bytes,
            raw_render_bytes: raw_render_bytes.clone(),
        };
        match tx.try_send(req) {
            Ok(()) => {
                // ClassificationWorker will publish the single classifier_render event.
                return None;
            }
            Err(flume::TrySendError::Full(_)) => {
                let mut st = state.write().await;
                st.total_classification_drops += 1;
                warn!(
                    drops = st.total_classification_drops,
                    "ClassificationWorker channel full; returning manifest without BirdNET labels"
                );
                // No disk write — caller broadcasts via SSE without labels.
            }
            Err(flume::TrySendError::Disconnected(_)) => {
                warn!(
                    "ClassificationWorker channel closed; returning manifest without BirdNET labels"
                );
                // No disk write — caller broadcasts via SSE without labels.
            }
        }
    }
    let mut fallback = pending;
    fallback.raw_render_bytes = raw_render_bytes;
    // No ClassificationWorker configured — return manifest unlabeled; caller broadcasts via SSE.
    Some(fallback)
}

pub(super) async fn publish_raw_audio_frame_event(request: RawAudioFramePublishRequest<'_>) {
    let RawAudioFramePublishRequest {
        publisher,
        source_manifest,
        decoded,
        stream_key,
        sample_rate_hz,
        start_time_ns,
        end_time_ns,
        created_ns,
        segment_index,
        segment_count,
    } = request;
    let Some(publisher) = publisher else {
        return;
    };
    let Some(raw_pcm_bytes) = encode_interleaved_pcm16le(&decoded.channels) else {
        return;
    };
    let channel_count = decoded.channels.len();
    let sample_count = decoded.channels.iter().map(Vec::len).min().unwrap_or(0);
    // raw_audio_frame is a freshly decoded contiguous payload — every sample
    // is present at this point, so coverage is all-true. Matches Python's
    // AudioCoverageStats.to_json() 9-field shape at audio_buffer.py:30.
    let frame_coverage = vec![true; sample_count];
    let coverage_stats_json =
        serde_json::to_value(coverage_stats(&frame_coverage, sample_rate_hz.max(1))).ok();
    let raw_audio_manifest_id = if segment_count == 1 {
        format!("raw-audio-{}", source_manifest.manifest_id)
    } else {
        format!(
            "raw-audio-{}-segment-{}",
            source_manifest.manifest_id, segment_index
        )
    };
    let raw_manifest = DspManifest {
        manifest_id: raw_audio_manifest_id,
        manifest_type: "raw_audio_frame".to_string(),
        created_ns,
        source_handles: source_manifest.source_handles.clone(),
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: coverage_stats_json,
        promotion_ready: false,
        env_samples: None,
        node_context: source_manifest.node_context.clone(),
        cluster_id: source_manifest.cluster_id.clone(),
        cluster_sensor_positions: source_manifest.cluster_sensor_positions.clone(),
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: Some(serde_json::json!({
            "stream_key": stream_key,
            "sample_rate_hz": sample_rate_hz,
            "channel_count": channel_count,
            "channels": channel_count,
            "sample_count": sample_count,
            "sample_format": "pcm16le",
            "start_time_ns": start_time_ns.max(0) as u128,
            "end_time_ns": end_time_ns.max(0) as u128,
            "start_sample_index": decoded.start_sample_index,
            "end_sample_index": decoded.end_sample_index,
            "source_manifest_id": source_manifest.manifest_id,
        })),
        raw_audio_bytes: Some(BASE64.encode(raw_pcm_bytes)),
    };
    let _ = publisher.publish(raw_manifest).await;
}

pub(super) fn encode_interleaved_pcm16le(channels: &[Vec<f32>]) -> Option<Vec<u8>> {
    let sample_count = channels.iter().map(Vec::len).min().unwrap_or(0);
    if channels.is_empty() || sample_count == 0 {
        return None;
    }
    let mut raw = Vec::with_capacity(sample_count * channels.len() * std::mem::size_of::<i16>());
    for sample_index in 0..sample_count {
        for channel in channels {
            let scaled = (channel[sample_index].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            raw.extend_from_slice(&scaled.to_le_bytes());
        }
    }
    Some(raw)
}

pub(super) fn merge_pending_manifests_for_batch(
    channel_manifests: Vec<DspManifest>,
    disk_manifests: Vec<DspManifest>,
    batch_limit: usize,
) -> Vec<DspManifest> {
    let mut dedupe_manifest_ids = HashSet::new();
    let mut merged = Vec::with_capacity(channel_manifests.len() + disk_manifests.len());
    for manifest in channel_manifests.into_iter().chain(disk_manifests) {
        if dedupe_manifest_ids.insert(manifest.manifest_id.clone()) {
            merged.push(manifest);
        }
    }

    // Prioritize freshest manifests so live audio stays current even while a
    // backlog exists in either source. This keeps the 30-second BirdNET window
    // populated with recent audio rather than draining an old backlog first.
    merged.sort_unstable_by(|left, right| right.created_ns.cmp(&left.created_ns));
    merged.truncate(batch_limit);
    merged
}

pub(super) fn should_sleep_after_poll_cycle(processed_manifest_count: usize) -> bool {
    processed_manifest_count == 0
}

pub(crate) async fn consume_manifest_standalone(
    manifest: &DspManifest,
    manifest_store: &ManifestStore,
    consumed_since_prune: &Arc<AtomicU64>,
    prune_interval: u64,
    retention_max_files: usize,
) {
    if !source_manifest_was_persisted(manifest) {
        return;
    }
    if let Err(err) = manifest_store.mark_consumed(&manifest.manifest_id).await {
        warn!(
            manifest_id = %manifest.manifest_id,
            error = %err,
            "DSP worker failed to mark manifest consumed"
        );
        return;
    }
    let count = consumed_since_prune.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_multiple_of(prune_interval) {
        let store = manifest_store.clone();
        tokio::spawn(async move {
            if let Err(error) = store.prune_consumed_manifests(retention_max_files).await {
                warn!(error = %error, "DSP worker failed to prune consumed manifests");
            }
        });
    }
}

pub(crate) fn source_manifest_was_persisted(manifest: &DspManifest) -> bool {
    // Manifests read from disk never include raw_payload because the field is
    // serde-skipped. Channel-delivered manifests include raw_payload and do not
    // require mark_consumed filesystem operations.
    manifest.raw_payload.is_none()
}

pub(crate) fn render_coverage_json(
    channel_coverage: &[Option<AudioCoverageStats>],
    active_channels: &[usize],
    min_coverage_ratio: f64,
    window_type: &str,
) -> Option<serde_json::Value> {
    serde_json::to_value(serde_json::json!({
        "window_type": window_type,
        "per_channel": channel_coverage.to_vec(),
        "active_channels": active_channels,
        "threshold": min_coverage_ratio,
    }))
    .ok()
}
