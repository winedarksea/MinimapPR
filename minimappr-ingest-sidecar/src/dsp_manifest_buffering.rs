//! Manifest ingest: decode, geometry/clock resolution and buffer append.

use super::*;

pub(super) struct OwnedManifestAudio {
    pub(super) manifest: DspManifest,
    pub(super) stream_key: String,
    pub(super) buffer_key: String,
    pub(super) decoded: DecodedAudioPayload,
    pub(super) source_ids: Vec<String>,
    pub(super) sr: u32,
    pub(super) now_ns: u128,
    pub(super) channel_count: usize,
    pub(super) buffer_channel_count: usize,
    pub(super) buffer_channel_indices: Vec<usize>,
    pub(super) classification_window_sec: f64,
    pub(super) sensor_layout: Arc<SolveSensorLayout>,
    pub(super) effective_sound_speed_mps: f32,
    pub(super) buffer_start_time_ns: i128,
    pub(super) buffer_end_time_ns: i128,
    pub(super) start_sample_index: Option<i64>,
    pub(super) end_sample_index: Option<i64>,
    pub(super) node_timestamp_is_available: bool,
    pub(super) buffer_uses_receipt_time: bool,
    pub(super) arrived_via_channel: bool,
}

pub(super) struct BufferedManifestAudio {
    pub(super) audio_end_ns: u128,
    pub(super) end_ns: i128,
    pub(super) localization_center_time_ns: i128,
    pub(super) channel_states: Vec<LocalizationChannelState>,
}

pub(super) struct ManifestBufferRouting {
    pub(super) buffer_key: String,
    pub(super) buffer_channel_count: usize,
    pub(super) buffer_channel_indices: Vec<usize>,
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
        self.apply_automatic_gps_geometry(&mut manifest, &stream_key, channel_count);
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
        let sensor_layout = Arc::new(solve_sensor_layout_from_manifest(
            &manifest,
            stream_key_node_id(&stream_key),
            decoded_segments[0].sample_rate_hz.max(1),
            self.automatic_gps_geometry.origin_as_geodetic(),
        ));
        let buffer_routing = resolve_cluster_buffer_routing(&manifest, &stream_key, channel_count);
        let (buffer_key, buffer_channel_count, buffer_channel_indices) = buffer_routing
            .map(|routing| {
                (
                    routing.buffer_key,
                    routing.buffer_channel_count,
                    routing.buffer_channel_indices,
                )
            })
            .unwrap_or_else(|| {
                (
                    stream_key.clone(),
                    channel_count,
                    (0..channel_count).collect(),
                )
            });
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
                buffer_key: buffer_key.clone(),
                decoded,
                source_ids: source_ids.clone(),
                sr,
                now_ns,
                channel_count,
                buffer_channel_count,
                buffer_channel_indices: buffer_channel_indices.clone(),
                classification_window_sec,
                sensor_layout: sensor_layout.clone(),
                effective_sound_speed_mps,
                buffer_start_time_ns,
                buffer_end_time_ns,
                start_sample_index,
                end_sample_index,
                node_timestamp_is_available,
                buffer_uses_receipt_time,
                arrived_via_channel,
            });
        }
        Some(owned_segments)
    }

    pub(super) fn apply_automatic_gps_geometry(
        &mut self,
        manifest: &mut DspManifest,
        stream_key: &str,
        channel_count: usize,
    ) {
        // Explicit cluster geometry is still an operator-directed boundary.
        // Otherwise every GPS-reporting node on the same audio stream joins the
        // automatic array, exactly as the Python global sensor pool does.
        if manifest.cluster_id.is_some() || manifest.cluster_sensor_positions.is_some() {
            return;
        }
        let node_id = stream_key_node_id(stream_key);
        let Some(sensor_positions) =
            self.automatic_gps_geometry
                .update_from_manifest(manifest, node_id, channel_count)
        else {
            return;
        };
        manifest.cluster_id = Some(AUTOMATIC_GPS_CLUSTER_ID.to_string());
        manifest.cluster_sensor_positions = Some(sensor_positions);
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

    pub(super) async fn should_skip_stale_manifest_for_live_buffer(
        &mut self,
        owned: &OwnedManifestAudio,
    ) -> bool {
        let source_manifest_is_stale = manifest_is_older_than_buffer_horizon(
            &owned.manifest,
            owned.now_ns,
            self.config.max_buffer_seconds,
        );
        if source_manifest_is_stale
            && self.config.skip_stale_manifests_for_live_buffer
            && !owned.arrived_via_channel
        {
            debug!(
                manifest_id = %owned.manifest.manifest_id,
                "DSP worker skipped stale disk manifest to protect live buffer continuity"
            );
            self.defer_source_manifest_consumption(&owned.manifest);
            let mut st = self.state.write().await;
            st.total_stale_manifest_skips += 1;
            st.last_processed_ns = Some(owned.now_ns);
            return true;
        }
        false
    }

    pub(super) async fn buffer_owned_manifest_audio(
        &mut self,
        owned: &OwnedManifestAudio,
    ) -> Option<BufferedManifestAudio> {
        let buffers = self
            .buffers
            .entry(owned.buffer_key.clone())
            .or_insert_with(|| {
                (0..owned.buffer_channel_count)
                    .map(|_| SensorStreamBuffer::new(owned.sr, self.config.max_buffer_seconds))
                    .collect()
            });
        while buffers.len() < owned.buffer_channel_count {
            buffers.push(SensorStreamBuffer::new(
                owned.sr,
                self.config.max_buffer_seconds,
            ));
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
        let channel_states = localization_channel_states_centered(
            buffers,
            center_time_ns,
            self.config.window_seconds,
        );
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
}
