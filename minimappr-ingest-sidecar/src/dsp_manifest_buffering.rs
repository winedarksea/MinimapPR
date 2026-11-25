//! Manifest ingest: decode, geometry/clock resolution and buffer append.

use super::*;
use minimappr_core::localization_sensor_entry::SolveBufferScope;

pub(super) struct OwnedManifestAudio {
    pub(super) manifest: DspManifest,
    pub(super) stream_key: String,
    pub(super) buffer_key: String,
    pub(super) decoded: DecodedAudioPayload,
    pub(super) source_ids: Vec<String>,
    pub(super) sr: u32,
    pub(super) now_ns: u128,
    pub(super) buffer_channel_count: usize,
    pub(super) buffer_channel_indices: Vec<usize>,
    /// Sensor id behind every buffer channel (the registry's current order).
    pub(super) buffer_sensor_ids: Arc<Vec<String>>,
    pub(super) classification_window_sec: f64,
    pub(super) sensor_layout: Arc<SolveSensorLayout>,
    /// Set when the node's own array failed the geometry gate.
    pub(super) large_array_omni_fallback_reason: Option<&'static str>,
    pub(super) effective_sound_speed_mps: f32,
    pub(super) buffer_start_time_ns: i128,
    pub(super) buffer_end_time_ns: i128,
    pub(super) start_sample_index: Option<i64>,
    pub(super) end_sample_index: Option<i64>,
    pub(super) node_timestamp_is_available: bool,
    pub(super) buffer_uses_receipt_time: bool,
}

pub(super) struct BufferedManifestAudio {
    pub(super) audio_end_ns: u128,
    pub(super) end_ns: i128,
    pub(super) localization_center_time_ns: i128,
    pub(super) channel_states: Vec<LocalizationChannelState>,
}

impl DspWorker {
    pub(super) async fn prepare_owned_manifest_audio(
        &mut self,
        mut manifest: DspManifest,
    ) -> Option<Vec<OwnedManifestAudio>> {
        let now_ns = system_now_ns();
        let Some(first_handle) = manifest.source_handles.first().cloned() else {
            self.mark_source_manifest_consumed_if_persisted(&manifest)
                .await;
            return None;
        };

        if let (Some(diagnostics), Some(received_ns)) =
            (&self.diagnostics, first_handle.received_ns)
        {
            let wait_ms = now_ns.saturating_sub(received_ns) / 1_000_000;
            diagnostics.record_queue_wait_ms(wait_ms as u64);
        }

        let stream_key = first_handle.stream_key.clone();
        let arrived_via_channel = manifest.raw_payload.is_some();
        if self
            .should_skip_stale_manifest_for_live_buffer(&manifest, now_ns, arrived_via_channel)
            .await
        {
            return None;
        }
        let raw_payload = if let Some(bytes) = manifest.raw_payload.clone() {
            bytes
        } else {
            warn!(
                manifest_id = %manifest.manifest_id,
                "DSP worker: raw manifest arrived without in-memory payload; frame dropped"
            );
            self.defer_source_manifest_consumption(&manifest);
            return None;
        };

        let decoded_segments = match decode_audio_payload_segments(&raw_payload) {
            Ok(decoded_segments) => decoded_segments,
            Err(err) => {
                self.note_failure().await;
                warn!(
                    manifest_id = %manifest.manifest_id,
                    error = %err,
                    "DSP worker failed to decode ingest audio; consuming manifest"
                );
                self.mark_source_manifest_consumed_if_persisted(&manifest)
                    .await;
                return None;
            }
        };
        if decoded_segments.is_empty()
            || decoded_segments
                .iter()
                .any(|decoded| decoded.channels.is_empty())
        {
            self.mark_source_manifest_consumed_if_persisted(&manifest)
                .await;
            return None;
        }

        let channel_count = decoded_segments[0].channels.len();
        // The geometry write lock is held only for routing: placement may
        // adopt the automatic frame's origin, and a concurrent push must not
        // land between placing one member and another.
        let shared_state = self.state.clone();
        let mut state = shared_state.write().await;
        let routing = self.sensor_registry.route_node(
            &NodeRoutingRequest {
                stream_key: &stream_key,
                manifest: &manifest,
                channel_count,
                sample_rate_hz: decoded_segments[0].sample_rate_hz.max(1),
                now_ns,
            },
            &mut state.site_geometry,
        );
        drop(state);
        let route = routing.route;
        if route.layout.buffer_scope != SolveBufferScope::NodeLocal {
            // Published results carry the buffer-ordered list, so a legacy
            // reader indexing `cluster_sensor_positions` by channel stays right.
            manifest.cluster_id = route.layout.cluster_id.clone();
            manifest.cluster_sensor_positions = Some(route.layout.cluster_sensor_positions());
        }
        let source_ids = manifest
            .source_handles
            .iter()
            .map(|handle| handle.segment_id.clone())
            .collect::<Vec<_>>();
        let classification_window_sec = self
            .config
            .classification_window_seconds
            .max(MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS)
            .max(self.config.window_seconds);
        let buffer_channel_count = route.buffer_sensor_ids.len();
        let segment_count = decoded_segments.len();
        let mut manifest_for_final_segment = Some(manifest);
        let mut owned_segments = Vec::with_capacity(segment_count);
        for (segment_index, mut decoded) in decoded_segments.into_iter().enumerate() {
            {
                let node_id_key = stream_key.split("__").next().unwrap_or(&stream_key);
                let st = self.state.read().await;
                let cfg = st.node_audio_overrides.get(node_id_key).cloned();
                drop(st);
                let cfg = cfg.unwrap_or_else(|| self.config.default_audio_config.clone());
                if !cfg.effective_stages().is_empty() {
                    let node_state = self
                        .node_audio_state
                        .entry(node_id_key.to_string())
                        .or_default();
                    node_state.apply(&mut decoded.channels, &cfg, decoded.sample_rate_hz);
                }
            }

            let sr = decoded.sample_rate_hz.max(1);
            let (env_temp_c, env_humidity_fraction) =
                if decoded.temperature_c.is_none() && decoded.humidity_fraction.is_none() {
                    if let Some(cache) = &self.env_cache {
                        let query_ns = first_handle.toa_ns.unwrap_or(now_ns as u64);
                        let node_id = stream_key.split("__").next().unwrap_or(&stream_key);
                        cache
                            .interpolate(node_id, query_ns)
                            .await
                            .map(|(temperature_c, humidity_fraction)| {
                                (Some(temperature_c), Some(humidity_fraction))
                            })
                            .unwrap_or((None, None))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

            let effective_temperature_c = decoded.temperature_c.or(env_temp_c);
            let effective_humidity_fraction = decoded.humidity_fraction.or(env_humidity_fraction);
            let effective_sound_speed_mps = resolve_effective_sound_speed_mps(
                &self.config,
                effective_temperature_c,
                effective_humidity_fraction,
            );

            let frames_all = decoded.channels.iter().map(Vec::len).min().unwrap_or(0);
            let render_duration_ns =
                (frames_all as i128).saturating_mul(1_000_000_000) / i128::from(sr.max(1));
            let start_time_ns = resolve_buffer_start_time_ns(&decoded, &first_handle, sr, now_ns);
            let node_timestamp_is_available =
                decoded.start_time_ns.is_some_and(|start_ns| start_ns > 0)
                    || first_handle.toa_ns.is_some();
            let skew_ns = (start_time_ns - now_ns as i128).unsigned_abs();
            let max_skew_ns =
                (self.config.max_trusted_node_clock_skew_seconds * 1_000_000_000.0).round() as u128;
            let buffer_uses_receipt_time = should_use_receipt_time_alignment(
                node_timestamp_is_available,
                skew_ns,
                max_skew_ns,
            );
            let (buffer_start_time_ns, buffer_end_time_ns) = if buffer_uses_receipt_time {
                let anchor_ns = first_handle.tor_ns.unwrap_or(now_ns as u64) as i128;
                let start_ns = anchor_ns.saturating_sub(render_duration_ns).max(1);
                (start_ns, anchor_ns)
            } else {
                (start_time_ns, start_time_ns + render_duration_ns)
            };

            let start_sample_index = decoded.start_sample_index;
            let end_sample_index = decoded.end_sample_index;
            let segment_manifest = if segment_index + 1 == segment_count {
                manifest_for_final_segment
                    .take()
                    .expect("final segment owns source manifest")
            } else {
                manifest_for_final_segment
                    .as_ref()
                    .expect("source manifest remains available")
                    .clone()
            };

            owned_segments.push(OwnedManifestAudio {
                manifest: segment_manifest,
                stream_key: stream_key.clone(),
                buffer_key: route.buffer_key.clone(),
                decoded,
                source_ids: source_ids.clone(),
                sr,
                now_ns,
                buffer_channel_count,
                buffer_channel_indices: route.buffer_channel_indices.clone(),
                buffer_sensor_ids: route.buffer_sensor_ids.clone(),
                classification_window_sec,
                sensor_layout: route.layout.clone(),
                large_array_omni_fallback_reason: routing.large_array_omni_fallback_reason,
                effective_sound_speed_mps,
                buffer_start_time_ns,
                buffer_end_time_ns,
                start_sample_index,
                end_sample_index,
                node_timestamp_is_available,
                buffer_uses_receipt_time,
            });
        }
        Some(owned_segments)
    }

    pub(super) async fn publish_raw_audio_frame_for_owned_manifest(
        &self,
        owned: &OwnedManifestAudio,
        segment_index: usize,
        segment_count: usize,
    ) {
        publish_raw_audio_frame_event(RawAudioFramePublishRequest {
            publisher: &self.dsp_event_publisher,
            source_manifest: &owned.manifest,
            decoded: &owned.decoded,
            stream_key: &owned.stream_key,
            sample_rate_hz: owned.sr,
            start_time_ns: owned.buffer_start_time_ns,
            end_time_ns: owned.buffer_end_time_ns,
            created_ns: owned.now_ns,
            segment_index,
            segment_count,
        })
        .await;
    }

    /// Checked before routing: a stale manifest must not refresh the sensor
    /// registry (or let the automatic frame adopt an origin from old GPS).
    async fn should_skip_stale_manifest_for_live_buffer(
        &mut self,
        manifest: &DspManifest,
        now_ns: u128,
        arrived_via_channel: bool,
    ) -> bool {
        let source_manifest_is_stale =
            manifest_is_older_than_buffer_horizon(manifest, now_ns, self.config.max_buffer_seconds);
        if source_manifest_is_stale
            && self.config.skip_stale_manifests_for_live_buffer
            && !arrived_via_channel
        {
            debug!(
                manifest_id = %manifest.manifest_id,
                "DSP worker skipped stale disk manifest to protect live buffer continuity"
            );
            self.defer_source_manifest_consumption(manifest);
            let mut st = self.state.write().await;
            st.total_stale_manifest_skips += 1;
            st.last_processed_ns = Some(now_ns);
            return true;
        }
        false
    }

    pub(super) async fn buffer_owned_manifest_audio(
        &mut self,
        owned: &OwnedManifestAudio,
    ) -> Option<BufferedManifestAudio> {
        self.reconcile_buffer_channels(owned);
        let max_buffer_seconds = self.config.max_buffer_seconds;
        let buffers = self
            .buffers
            .get_mut(&owned.buffer_key)
            .expect("reconcile_buffer_channels creates the buffer set");
        for &buffer_channel_index in &owned.buffer_channel_indices {
            // A buffer runs at its own sensor's rate. A placeholder created by
            // another member (or a node that changed rate) is replaced, never
            // written at the wrong rate.
            if let Some(buffer) = buffers.get_mut(buffer_channel_index) {
                if buffer.sample_rate_hz_value() != owned.sr {
                    *buffer = SensorStreamBuffer::new(owned.sr, max_buffer_seconds);
                }
            }
        }
        let reference_buffer_index = owned.buffer_channel_indices.first().copied().unwrap_or(0);

        let existing_sample_timeline_start_time_ns =
            if !owned.node_timestamp_is_available && !owned.buffer_uses_receipt_time {
                owned.start_sample_index.and_then(|sample_index| {
                    buffers
                        .get(reference_buffer_index)
                        .and_then(|buffer| buffer.time_for_sample_index(sample_index))
                })
            } else {
                None
            };
        let (buffer_start_time_ns, buffer_end_time_ns) = if let Some(existing_start_time_ns) =
            existing_sample_timeline_start_time_ns
        {
            (
                existing_start_time_ns,
                existing_start_time_ns + (owned.buffer_end_time_ns - owned.buffer_start_time_ns),
            )
        } else {
            (owned.buffer_start_time_ns, owned.buffer_end_time_ns)
        };

        let mut reanchor_delta: u64 = 0;
        for (channel_index, &buffer_channel_index) in
            owned.buffer_channel_indices.iter().enumerate()
        {
            let Some(channel_samples) = owned.decoded.channels.get(channel_index) else {
                break;
            };
            let Some(buffer) = buffers.get_mut(buffer_channel_index) else {
                self.note_failure().await;
                warn!(
                    manifest_id = %owned.manifest.manifest_id,
                    stream_key = %owned.stream_key,
                    buffer_key = %owned.buffer_key,
                    buffer_channel_index,
                    "DSP worker resolved an invalid cluster buffer channel index"
                );
                self.defer_source_manifest_consumption(&owned.manifest);
                return None;
            };
            let pre_reanchor = buffer.reanchor_count();
            if let Err(err) = buffer.append(
                buffer_start_time_ns,
                channel_samples,
                owned.start_sample_index,
                owned.end_sample_index,
            ) {
                self.note_failure().await;
                warn!(
                    manifest_id = %owned.manifest.manifest_id,
                    channel = channel_index,
                    error = %err,
                    "DSP worker failed to append decoded audio; consuming malformed source manifest"
                );
                self.defer_source_manifest_consumption(&owned.manifest);
                return None;
            }
            reanchor_delta =
                reanchor_delta.saturating_add(buffer.reanchor_count().saturating_sub(pre_reanchor));
        }

        let end_ns = resolve_buffer_end_time_ns(
            buffers,
            owned.end_sample_index,
            buffer_end_time_ns,
            owned.sr,
            reference_buffer_index,
        );
        // With the coverage wait on, clamp against the arriving node's own
        // channel: the window must end at the trigger frame and the wait then
        // brings the other members in. Legacy clamps against channel 0, which
        // in a cluster buffer is whichever node joined first.
        let center_clamp_buffer_index = if self.deferred_localizations.is_enabled() {
            reference_buffer_index
        } else {
            0
        };
        let center_time_ns = resolve_localization_center_time_ns(
            buffers,
            center_clamp_buffer_index,
            buffer_start_time_ns,
            end_ns,
            self.config.window_seconds,
        );
        let mut channel_states = localization_channel_states_centered(
            buffers,
            center_time_ns,
            self.config.window_seconds,
        );
        owned
            .sensor_layout
            .clear_solve_excluded_channels(&mut channel_states);
        // `buffers` borrow ends here; safe to take &self again for the
        // shared-state write.
        if reanchor_delta > 0 {
            self.note_buffer_reanchors(reanchor_delta).await;
        }
        Some(BufferedManifestAudio {
            audio_end_ns: end_ns.max(0) as u128,
            end_ns,
            localization_center_time_ns: center_time_ns,
            channel_states,
        })
    }

    /// Brings `buffers[owned.buffer_key]` to the registry's sensor order. A
    /// reorder first releases any parked solve against the old order, since
    /// its layout snapshot indexes the channels as they were.
    fn reconcile_buffer_channels(&mut self, owned: &OwnedManifestAudio) {
        let current_sensor_ids = self
            .buffer_channel_sensor_ids
            .get(&owned.buffer_key)
            .filter(|_| self.buffers.contains_key(&owned.buffer_key))
            .cloned()
            .unwrap_or_default();
        let max_buffer_seconds = self.config.max_buffer_seconds;
        let new_buffer = || SensorStreamBuffer::new(owned.sr, max_buffer_seconds);
        match plan_buffer_channel_reconcile(&current_sensor_ids, &owned.buffer_sensor_ids) {
            BufferChannelReconcile::Unchanged => {}
            BufferChannelReconcile::AppendChannels => {
                let buffers = self.buffers.entry(owned.buffer_key.clone()).or_default();
                buffers.resize_with(owned.buffer_channel_count, new_buffer);
            }
            BufferChannelReconcile::Reorder {
                source_channel_by_new_channel,
            } => {
                if let Some(buffers) = self.buffers.get(&owned.buffer_key) {
                    self.deferred_localizations
                        .release_pending_and_forget(&owned.buffer_key, buffers);
                }
                let mut previous: Vec<Option<SensorStreamBuffer>> = self
                    .buffers
                    .remove(&owned.buffer_key)
                    .unwrap_or_default()
                    .into_iter()
                    .map(Some)
                    .collect();
                let reordered = source_channel_by_new_channel
                    .into_iter()
                    .map(|source| {
                        source
                            .and_then(|index| previous.get_mut(index)?.take())
                            .unwrap_or_else(new_buffer)
                    })
                    .collect();
                self.buffers.insert(owned.buffer_key.clone(), reordered);
            }
        }
        self.buffers.entry(owned.buffer_key.clone()).or_default();
        self.buffer_channel_sensor_ids
            .insert(owned.buffer_key.clone(), owned.buffer_sensor_ids.clone());
    }
}
