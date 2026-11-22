use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type BoxedResult<T> = Result<T, BoxedError>;

#[derive(Clone, Debug)]
pub struct DecodedAudioPayload {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate_hz: u32,
    pub start_time_ns: Option<i128>,
    pub start_sample_index: Option<i64>,
    pub end_sample_index: Option<i64>,
    pub temperature_c: Option<f32>,
    pub humidity_fraction: Option<f32>,
    pub environment_source: Option<String>,
}

struct BinaryReader<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self { payload, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.payload.len().saturating_sub(self.offset)
    }

    fn read(&mut self, length: usize) -> BoxedResult<&'a [u8]> {
        if self.offset.saturating_add(length) > self.payload.len() {
            return Err("binary ingest payload ended unexpectedly".into());
        }
        let start = self.offset;
        self.offset += length;
        Ok(&self.payload[start..self.offset])
    }

    fn u8(&mut self) -> BoxedResult<u8> {
        Ok(self.read(1)?[0])
    }

    fn u16(&mut self) -> BoxedResult<u16> {
        Ok(u16::from_le_bytes(self.read(2)?.try_into()?))
    }

    fn u32(&mut self) -> BoxedResult<u32> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into()?))
    }

    fn u64(&mut self) -> BoxedResult<u64> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into()?))
    }

    fn i32(&mut self) -> BoxedResult<i32> {
        Ok(i32::from_le_bytes(self.read(4)?.try_into()?))
    }

    fn i64(&mut self) -> BoxedResult<i64> {
        Ok(i64::from_le_bytes(self.read(8)?.try_into()?))
    }

    fn f32(&mut self) -> BoxedResult<f32> {
        Ok(f32::from_le_bytes(self.read(4)?.try_into()?))
    }

    fn f64(&mut self) -> BoxedResult<f64> {
        Ok(f64::from_le_bytes(self.read(8)?.try_into()?))
    }

    fn string(&mut self) -> BoxedResult<String> {
        let length = usize::from(self.u8()?);
        let raw = self.read(length)?;
        Ok(std::str::from_utf8(raw)?.to_string())
    }
}

fn read_binary_ingest_version(reader: &mut BinaryReader<'_>) -> BoxedResult<u8> {
    let expected_version = match reader.read(4)? {
        magic if magic == b"MMB2" => 2,
        magic if magic == b"MMB3" => 3,
        _ => return Err("invalid binary ingest magic".into()),
    };
    let version = reader.u8()?;
    if version != expected_version {
        return Err(format!("unsupported binary ingest version {version}").into());
    }
    Ok(version)
}

pub fn decode_audio_payload_segments(raw_payload: &[u8]) -> BoxedResult<Vec<DecodedAudioPayload>> {
    if raw_payload.starts_with(b"MMB2") || raw_payload.starts_with(b"MMB3") {
        return decode_binary_audio(raw_payload);
    }
    decode_store_forward_audio(raw_payload)
}

fn decode_binary_audio(raw_payload: &[u8]) -> BoxedResult<Vec<DecodedAudioPayload>> {
    let mut reader = BinaryReader::new(raw_payload);
    let _version = read_binary_ingest_version(&mut reader)?;

    let _sort_by_toa = reader.u8()?;
    let frame_count = reader.u16()?;
    if frame_count == 0 || frame_count > 2048 {
        return Err("binary ingest frame count must be between 1 and 2048".into());
    }

    skip_binary_node(&mut reader)?;

    let mut segments = Vec::new();
    let mut current_segment: Option<DecodedAudioPayload> = None;
    let mut sample_rate_hz: Option<u32> = None;
    let mut channel_count: Option<usize> = None;
    for _ in 0..frame_count {
        let frame = read_binary_audio_frame(&mut reader)?;
        if let Some(existing_rate) = sample_rate_hz {
            if existing_rate != frame.sample_rate_hz {
                return Err("binary ingest frames mixed sample rates".into());
            }
        }
        sample_rate_hz = Some(frame.sample_rate_hz);
        if channel_count.is_some_and(|existing| existing != frame.channels) {
            return Err("binary ingest frames mixed channel counts".into());
        }
        channel_count = Some(frame.channels);
        let frame_start_sample_index = i64::try_from(frame.start_sample_index)?;
        let frame_end_sample_index = i64::try_from(frame.end_sample_index)?;
        let starts_new_segment = current_segment
            .as_ref()
            .is_some_and(|segment| segment.end_sample_index != Some(frame_start_sample_index));
        if starts_new_segment {
            segments.push(current_segment.take().expect("current segment exists"));
        }
        let segment = current_segment.get_or_insert_with(|| DecodedAudioPayload {
            channels: Vec::new(),
            sample_rate_hz: frame.sample_rate_hz,
            start_time_ns: Some(frame.start_time_ns as i128),
            start_sample_index: Some(frame_start_sample_index),
            end_sample_index: Some(frame_start_sample_index),
            temperature_c: None,
            humidity_fraction: None,
            environment_source: None,
        });
        append_channels(&mut segment.channels, frame.channels, frame.samples);
        segment.end_sample_index = Some(frame_end_sample_index);
        update_segment_environment(
            segment,
            frame.temperature_c,
            frame.humidity_fraction,
            frame.environment_source,
        );
    }
    if reader.remaining() != 0 {
        return Err("binary ingest payload has trailing bytes".into());
    }
    if let Some(segment) = current_segment {
        segments.push(segment);
    }
    sample_rate_hz.ok_or("binary ingest payload contained no audio")?;
    Ok(segments)
}

fn skip_binary_node(reader: &mut BinaryReader<'_>) -> BoxedResult<()> {
    let _node_id = reader.string()?;
    let _node_type_code = reader.u8()?;
    if reader.u8()? != 0 {
        let _lat = reader.f32()?;
        let _lon = reader.f32()?;
        let _alt = reader.f32()?;
    }
    let sensor_count = reader.u8()?;
    if sensor_count == 0 {
        return Err("binary node must declare at least one sensor".into());
    }
    for _ in 0..sensor_count {
        let _x = reader.f32()?;
        let _y = reader.f32()?;
        let _z = reader.f32()?;
    }
    let capability_count = reader.u8()?;
    for _ in 0..capability_count {
        let _capability = reader.string()?;
    }
    let _hardware = reader.string()?;
    let _firmware = reader.string()?;
    let _gps_signal = reader.string()?;
    let _position_source = reader.string()?;
    let _boot_count = reader.u32()?;
    Ok(())
}

struct BinaryAudioFrame {
    sample_rate_hz: u32,
    channels: usize,
    start_time_ns: u64,
    start_sample_index: u64,
    end_sample_index: u64,
    samples: Vec<f32>,
    temperature_c: Option<f32>,
    humidity_fraction: Option<f32>,
    environment_source: Option<String>,
}

fn read_binary_audio_frame(reader: &mut BinaryReader<'_>) -> BoxedResult<BinaryAudioFrame> {
    let start_time_ns = reader.u64()?;
    let _end_time_ns = reader.u64()?;
    let start_sample_index = reader.u64()?;
    let end_sample_index = reader.u64()?;
    let sample_rate_hz = reader.u32()?;
    let channels = usize::from(reader.u8()?);
    if channels == 0 {
        return Err("binary frame must include at least one channel".into());
    }
    if reader.remaining() == 0 {
        return Err("binary frame ended before sequence".into());
    }
    let version3_source_type = if reader.payload.starts_with(b"MMB3") {
        Some(reader.u8()?)
    } else {
        None
    };
    let _sequence = reader.u64()?;
    let _toa_ns = reader.u64()?;
    let _tor_ns = reader.u64()?;
    let _time_quality = reader.u8()?;
    let environment;
    let samples_per_channel;
    if version3_source_type.is_some() {
        samples_per_channel = usize::try_from(reader.u32()?)?;
        environment = read_binary_v3_sections(reader)?;
    } else {
        skip_binary_timing_diagnostics(reader)?;
        environment = read_binary_environment(reader)?;
        samples_per_channel = usize::try_from(reader.u32()?)?;
    }
    if samples_per_channel == 0 {
        return Err("binary frame must include at least one channel and sample".into());
    }
    let expected_end = start_sample_index.saturating_add(samples_per_channel as u64);
    if end_sample_index != expected_end {
        return Err("binary frame sample indices do not match samples_per_channel".into());
    }
    let raw_audio = reader.read(
        samples_per_channel
            .saturating_mul(channels)
            .saturating_mul(2),
    )?;
    Ok(BinaryAudioFrame {
        sample_rate_hz,
        channels,
        start_time_ns,
        start_sample_index,
        end_sample_index,
        samples: pcm16le_to_f32(raw_audio),
        temperature_c: environment.temperature_c,
        humidity_fraction: environment.humidity_fraction,
        environment_source: environment.source,
    })
}

fn read_binary_v3_sections(reader: &mut BinaryReader<'_>) -> BoxedResult<BinaryEnvironmentSample> {
    let section_flags = reader.u16()?;
    let mut environment = BinaryEnvironmentSample {
        temperature_c: None,
        humidity_fraction: None,
        source: None,
    };
    for bit_index in 0..16 {
        let section_bit = 1_u16 << bit_index;
        if section_flags & section_bit == 0 {
            continue;
        }
        let section_len = usize::from(reader.u16()?);
        let section_payload = reader.read(section_len)?;
        let mut section_reader = BinaryReader::new(section_payload);
        match section_bit {
            0x0001 => skip_binary_timing_diagnostics_v3(&mut section_reader)?,
            0x0002 => environment = read_binary_environment(&mut section_reader)?,
            0x0004 => skip_binary_transport_health(&mut section_reader)?,
            _ => {
                // Forward-compatibility: consume unknown/future sections (e.g.
                // 0x0008 aux sensors, 0x0010 clock holdover) so newer firmware
                // does not break ingest on an older sidecar. The section was
                // already length-delimited above; just skip its bytes. The
                // strict trailing-bytes check below still applies to the known
                // sections that under-read.
                let remaining = section_reader.remaining();
                let _ = section_reader.read(remaining)?;
            }
        }
        if section_reader.remaining() != 0 {
            return Err(
                format!("binary ingest section 0x{section_bit:04x} has trailing bytes").into(),
            );
        }
    }
    Ok(environment)
}

fn skip_binary_timing_diagnostics_v3(reader: &mut BinaryReader<'_>) -> BoxedResult<()> {
    let _gps_anchor = reader.u8()?;
    let _pps_edge_count = reader.u32()?;
    let _dma_ring_slot_index = reader.u32()?;
    let _pps_phase_error_ns = reader.i64()?;
    let _estimated_ppm = reader.f64()?;
    let _runner_frames_captured = reader.u64()?;
    let _runner_frames_dropped = reader.u64()?;
    let _runner_continuity_violations = reader.u64()?;
    let _runner_publish_errors = reader.u64()?;
    let _runner_queue_depth = reader.u32()?;
    let _runner_queue_overflows = reader.u64()?;
    let _runner_last_publish_status = reader.i32()?;
    let _packet_age_us = reader.u64()?;
    let _runner_last_publish_failure_stage = reader.u8()?;
    let _runner_last_publish_lwip_error = reader.i32()?;
    let _runner_consecutive_publish_failures = reader.u32()?;
    let _runner_publish_timeout_failures = reader.u64()?;
    let _runner_publish_connect_or_reset_failures = reader.u64()?;
    let _runner_publish_dns_failures = reader.u64()?;
    let _runner_publish_wifi_down_failures = reader.u64()?;
    Ok(())
}

fn skip_binary_transport_health(reader: &mut BinaryReader<'_>) -> BoxedResult<()> {
    let _ring_frames_high_water = reader.u16()?;
    let _ring_frames_capacity = reader.u16()?;
    let _queue_slots_high_water = reader.u16()?;
    let _queue_slots_capacity = reader.u16()?;
    let _publish_latency_last_ms = reader.u16()?;
    let _publish_latency_ewma_ms = reader.u16()?;
    let _publish_latency_max_ms = reader.u16()?;
    let _wifi_rssi_dbm = reader.u8()?;
    let _heap_free_bytes = reader.u32()?;
    let _boot_id = reader.u32()?;
    Ok(())
}

fn skip_binary_timing_diagnostics(reader: &mut BinaryReader<'_>) -> BoxedResult<()> {
    if reader.u8()? == 0 {
        return Ok(());
    }
    let _gps_anchor = reader.u8()?;
    let _pps_edge_count = reader.u32()?;
    let _dma_ring_slot_index = reader.u32()?;
    let _pps_phase_error_ns = reader.i64()?;
    let _estimated_ppm = reader.f64()?;
    let _runner_frames_captured = reader.u64()?;
    let _runner_frames_dropped = reader.u64()?;
    let _runner_continuity_violations = reader.u64()?;
    let _runner_publish_errors = reader.u64()?;
    let _runner_queue_depth = reader.u32()?;
    let _runner_queue_overflows = reader.u64()?;
    let _runner_last_publish_status = reader.i32()?;
    let _packet_age_us = reader.u64()?;
    let _runner_last_publish_failure_stage = reader.u8()?;
    let _runner_last_publish_lwip_error = reader.i32()?;
    let _runner_consecutive_publish_failures = reader.u32()?;
    let _runner_publish_timeout_failures = reader.u64()?;
    let _runner_publish_connect_or_reset_failures = reader.u64()?;
    let _runner_publish_dns_failures = reader.u64()?;
    let _runner_publish_wifi_down_failures = reader.u64()?;
    Ok(())
}

struct BinaryEnvironmentSample {
    temperature_c: Option<f32>,
    humidity_fraction: Option<f32>,
    source: Option<String>,
}

fn read_binary_environment(reader: &mut BinaryReader<'_>) -> BoxedResult<BinaryEnvironmentSample> {
    let flags = reader.u8()?;
    let mut temperature_c = None;
    let mut humidity_fraction = None;
    let mut source = None;
    if flags & 0x01 != 0 {
        temperature_c = Some(reader.f32()?);
    }
    if flags & 0x02 != 0 {
        humidity_fraction = Some(reader.f32()?);
    }
    if flags & 0x04 != 0 {
        source = Some(reader.string()?);
    }
    Ok(BinaryEnvironmentSample {
        temperature_c,
        humidity_fraction,
        source,
    })
}

#[derive(Debug, Deserialize)]
struct StoreForwardEnvelope {
    buffered_frames: Vec<StoreForwardBufferedFrame>,
}

#[derive(Debug, Deserialize)]
struct StoreForwardBufferedFrame {
    frame: StoreForwardFrame,
    #[serde(default)]
    environment: Option<StoreForwardEnvironment>,
}

#[derive(Debug, Deserialize)]
struct StoreForwardEnvironment {
    temperature_c: Option<f32>,
    humidity_fraction: Option<f32>,
    source: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StoreForwardFrame {
    sample_rate_hz: u32,
    channels: usize,
    start_time_ns: Option<u64>,
    utc_start_ns: Option<u64>,
    start_sample_index: Option<u64>,
    end_sample_index: Option<u64>,
    samples_per_channel: Option<u32>,
    samples_b64: String,
}

fn decode_store_forward_audio(raw_payload: &[u8]) -> BoxedResult<Vec<DecodedAudioPayload>> {
    let envelope: StoreForwardEnvelope = serde_json::from_slice(raw_payload)?;
    let mut segments = Vec::new();
    let mut current_segment: Option<DecodedAudioPayload> = None;
    let mut previous_frame_end_sample_index: Option<i64> = None;
    let mut current_segment_has_complete_indices = true;
    let mut sample_rate_hz: Option<u32> = None;
    let mut channel_count: Option<usize> = None;
    for buffered in envelope.buffered_frames {
        let environment = buffered.environment;
        let frame = buffered.frame;
        if frame.channels == 0 {
            return Err("store-forward frame must have at least one channel".into());
        }
        if let Some(existing_rate) = sample_rate_hz {
            if existing_rate != frame.sample_rate_hz {
                return Err("store-forward frames mixed sample rates".into());
            }
        }
        if channel_count.is_some_and(|existing| existing != frame.channels) {
            return Err("store-forward frames mixed channel counts".into());
        }
        channel_count = Some(frame.channels);
        let raw_audio = STANDARD.decode(frame.samples_b64.as_bytes())?;
        let decoded = pcm16le_to_f32(&raw_audio);
        if !decoded.len().is_multiple_of(frame.channels) {
            return Err("store-forward PCM sample count is not divisible by channels".into());
        }
        let frame_samples = decoded.len() / frame.channels;
        if let Some(expected) = frame.samples_per_channel {
            if usize::try_from(expected)? != frame_samples {
                return Err("store-forward samples_per_channel does not match PCM payload".into());
            }
        }
        let frame_start_sample_index = frame.start_sample_index.map(i64::try_from).transpose()?;
        let frame_end_sample_index = match (frame_start_sample_index, frame.end_sample_index) {
            (Some(start), Some(end)) => {
                let end = i64::try_from(end)?;
                if end < start || usize::try_from(end - start)? != frame_samples {
                    return Err(
                        "store-forward frame sample indices do not match PCM payload".into(),
                    );
                }
                Some(end)
            }
            (Some(start), None) => Some(start.saturating_add(i64::try_from(frame_samples)?)),
            (None, _) => None,
        };
        sample_rate_hz = Some(frame.sample_rate_hz);
        let starts_new_segment = previous_frame_end_sample_index
            .zip(frame_start_sample_index)
            .is_some_and(|(previous_end, next_start)| previous_end != next_start);
        if starts_new_segment {
            let mut completed_segment = current_segment.take().expect("current segment exists");
            if !current_segment_has_complete_indices {
                completed_segment.start_sample_index = None;
                completed_segment.end_sample_index = None;
            }
            segments.push(completed_segment);
            current_segment_has_complete_indices = true;
        }
        let segment = current_segment.get_or_insert_with(|| DecodedAudioPayload {
            channels: Vec::new(),
            sample_rate_hz: frame.sample_rate_hz,
            start_time_ns: Some(
                frame
                    .start_time_ns
                    .or(frame.utc_start_ns)
                    .unwrap_or_default() as i128,
            ),
            start_sample_index: frame_start_sample_index,
            end_sample_index: frame_end_sample_index,
            temperature_c: None,
            humidity_fraction: None,
            environment_source: None,
        });
        append_channels(&mut segment.channels, frame.channels, decoded);
        current_segment_has_complete_indices &=
            frame_start_sample_index.is_some() && frame_end_sample_index.is_some();
        segment.end_sample_index = frame_end_sample_index;
        if let Some(environment) = environment {
            update_segment_environment(
                segment,
                environment.temperature_c,
                environment.humidity_fraction,
                environment.source,
            );
        }
        previous_frame_end_sample_index = frame_end_sample_index;
    }
    if let Some(mut segment) = current_segment {
        if !current_segment_has_complete_indices {
            segment.start_sample_index = None;
            segment.end_sample_index = None;
        }
        segments.push(segment);
    }
    sample_rate_hz.ok_or("store-forward payload contained no audio")?;
    Ok(segments)
}

fn update_segment_environment(
    segment: &mut DecodedAudioPayload,
    temperature_c: Option<f32>,
    humidity_fraction: Option<f32>,
    environment_source: Option<String>,
) {
    if temperature_c.is_some() {
        segment.temperature_c = temperature_c;
    }
    if humidity_fraction.is_some() {
        segment.humidity_fraction = humidity_fraction;
    }
    if environment_source.is_some() {
        segment.environment_source = environment_source;
    }
}

fn append_channels(
    target: &mut Vec<Vec<f32>>,
    channel_count: usize,
    interleaved_samples: Vec<f32>,
) {
    if target.is_empty() {
        target.resize_with(channel_count, Vec::new);
    }
    for (index, sample) in interleaved_samples.into_iter().enumerate() {
        let channel_index = index % channel_count;
        target[channel_index].push(sample);
    }
}

fn pcm16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::decode_audio_payload_segments;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use serde_json::json;

    fn push_string(payload: &mut Vec<u8>, value: &str) {
        payload.push(u8::try_from(value.len()).expect("test string fits in length byte"));
        payload.extend_from_slice(value.as_bytes());
    }

    fn push_f32(payload: &mut Vec<u8>, value: f32) {
        payload.extend_from_slice(&value.to_le_bytes());
    }

    fn push_binary_node(payload: &mut Vec<u8>) {
        push_string(payload, "sirith-tetra-1a15");
        payload.push(1); // sirith_tetra
        payload.push(1); // has_geo_position
        push_f32(payload, 44.987);
        push_f32(payload, -93.258);
        push_f32(payload, 281.5);
        payload.push(1); // sensor_count
        push_f32(payload, 0.0);
        push_f32(payload, 0.0);
        push_f32(payload, 0.0);
        payload.push(1); // capability_count
        push_string(payload, "audio");
        push_string(payload, "sirith-test-hardware");
        push_string(payload, "sirith-test-firmware");
        push_string(payload, "fix_3d");
        push_string(payload, "gps_nmea_uart");
        payload.extend_from_slice(&7_u32.to_le_bytes());
    }

    fn push_binary_section(payload: &mut Vec<u8>, section: &[u8]) {
        payload.extend_from_slice(
            &u16::try_from(section.len())
                .expect("test section length fits in u16")
                .to_le_bytes(),
        );
        payload.extend_from_slice(section);
    }

    fn binary_timing_section() -> Vec<u8> {
        let mut section = Vec::new();
        section.push(1); // has_gps_anchor
        section.extend_from_slice(&11_u32.to_le_bytes());
        section.extend_from_slice(&3_u32.to_le_bytes());
        section.extend_from_slice(&(-123_i64).to_le_bytes());
        section.extend_from_slice(&0.25_f64.to_le_bytes());
        section.extend_from_slice(&42_u64.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&2_u32.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&200_i32.to_le_bytes());
        section.extend_from_slice(&500_u64.to_le_bytes());
        section.push(0);
        section.extend_from_slice(&0_i32.to_le_bytes());
        section.extend_from_slice(&0_u32.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section.extend_from_slice(&0_u64.to_le_bytes());
        section
    }

    fn binary_environment_section() -> Vec<u8> {
        let mut section = Vec::new();
        section.push(0x07);
        section.extend_from_slice(&21.5_f32.to_le_bytes());
        section.extend_from_slice(&0.55_f32.to_le_bytes());
        push_string(&mut section, "bme280");
        section
    }

    fn binary_transport_health_section() -> Vec<u8> {
        let mut section = Vec::new();
        section.extend_from_slice(&4_u16.to_le_bytes());
        section.extend_from_slice(&16_u16.to_le_bytes());
        section.extend_from_slice(&12_u16.to_le_bytes());
        section.extend_from_slice(&40_u16.to_le_bytes());
        section.extend_from_slice(&25_u16.to_le_bytes());
        section.extend_from_slice(&31_u16.to_le_bytes());
        section.extend_from_slice(&90_u16.to_le_bytes());
        section.push((-62_i8) as u8);
        section.extend_from_slice(&128_000_u32.to_le_bytes());
        section.extend_from_slice(&0x1234_5678_u32.to_le_bytes());
        section
    }

    fn push_binary_frame_at(
        payload: &mut Vec<u8>,
        version: u8,
        start_time_ns: u64,
        start_sample_index: u64,
        sequence: u64,
    ) {
        payload.extend_from_slice(&start_time_ns.to_le_bytes());
        payload.extend_from_slice(&(start_time_ns + 125_000).to_le_bytes());
        payload.extend_from_slice(&start_sample_index.to_le_bytes());
        payload.extend_from_slice(&(start_sample_index + 2).to_le_bytes());
        payload.extend_from_slice(&16_000_u32.to_le_bytes());
        payload.push(1); // channels
        if version == 3 {
            payload.push(3); // synthetic audio source
        }
        payload.extend_from_slice(&sequence.to_le_bytes());
        payload.extend_from_slice(&start_time_ns.to_le_bytes());
        payload.extend_from_slice(&(start_time_ns + 250).to_le_bytes());
        payload.push(0); // gps_locked
        if version == 3 {
            payload.extend_from_slice(&2_u32.to_le_bytes());
            payload.extend_from_slice(&0x0007_u16.to_le_bytes());
            push_binary_section(payload, &binary_timing_section());
            push_binary_section(payload, &binary_environment_section());
            push_binary_section(payload, &binary_transport_health_section());
        } else {
            payload.push(0); // no timing diagnostics
            payload.push(0); // no environment fields
            payload.extend_from_slice(&2_u32.to_le_bytes());
        }
        payload.extend_from_slice(&0_i16.to_le_bytes());
        payload.extend_from_slice(&32_767_i16.to_le_bytes());
    }

    fn push_binary_frame(payload: &mut Vec<u8>, version: u8) {
        push_binary_frame_at(payload, version, 1_000, 0, 1);
    }

    fn binary_payload(version: u8) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(match version {
            2 => b"MMB2",
            3 => b"MMB3",
            _ => panic!("unsupported test version"),
        });
        payload.push(version);
        payload.push(0); // sort_by_toa
        payload.extend_from_slice(&1_u16.to_le_bytes());
        push_binary_node(&mut payload);
        push_binary_frame(&mut payload, version);
        payload
    }

    fn binary_payload_with_sample_starts(version: u8, sample_starts: &[u64]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(match version {
            2 => b"MMB2",
            3 => b"MMB3",
            _ => panic!("unsupported test version"),
        });
        payload.push(version);
        payload.push(0);
        payload.extend_from_slice(
            &u16::try_from(sample_starts.len())
                .expect("test frame count fits in u16")
                .to_le_bytes(),
        );
        push_binary_node(&mut payload);
        for (frame_index, start_sample_index) in sample_starts.iter().copied().enumerate() {
            push_binary_frame_at(
                &mut payload,
                version,
                1_000 + u64::try_from(frame_index).unwrap() * 125_000,
                start_sample_index,
                u64::try_from(frame_index).unwrap() + 1,
            );
        }
        payload
    }

    #[test]
    fn decode_audio_payload_accepts_mmb2_binary_ingest() {
        let decoded_segments =
            decode_audio_payload_segments(&binary_payload(2)).expect("MMB2 payload should decode");
        let decoded = &decoded_segments[0];

        assert_eq!(decoded.sample_rate_hz, 16_000);
        assert_eq!(decoded.channels.len(), 1);
        assert_eq!(decoded.channels[0].len(), 2);
        assert_eq!(decoded.start_time_ns, Some(1_000));
        assert_eq!(decoded.start_sample_index, Some(0));
        assert_eq!(decoded.end_sample_index, Some(2));
        assert_eq!(decoded.channels[0][0], 0.0);
        assert!(decoded.channels[0][1] > 0.99);
    }

    #[test]
    fn decode_audio_payload_accepts_mmb3_binary_ingest() {
        let decoded_segments =
            decode_audio_payload_segments(&binary_payload(3)).expect("MMB3 payload should decode");
        let decoded = &decoded_segments[0];

        assert_eq!(decoded.sample_rate_hz, 16_000);
        assert_eq!(decoded.channels.len(), 1);
        assert_eq!(decoded.channels[0].len(), 2);
        assert_eq!(decoded.start_time_ns, Some(1_000));
        assert_eq!(decoded.start_sample_index, Some(0));
        assert_eq!(decoded.end_sample_index, Some(2));
        assert_eq!(decoded.temperature_c, Some(21.5));
        assert_eq!(decoded.humidity_fraction, Some(0.55));
        assert_eq!(decoded.environment_source.as_deref(), Some("bme280"));
    }

    #[test]
    fn binary_decoder_splits_gap_and_overlap_boundaries() {
        for version in [2, 3] {
            let decoded_segments = decode_audio_payload_segments(
                &binary_payload_with_sample_starts(version, &[0, 2, 8, 6]),
            )
            .expect("gapped binary payload should decode into segments");

            assert_eq!(decoded_segments.len(), 3);
            assert_eq!(decoded_segments[0].start_sample_index, Some(0));
            assert_eq!(decoded_segments[0].end_sample_index, Some(4));
            assert_eq!(decoded_segments[0].channels[0].len(), 4);
            assert_eq!(decoded_segments[1].start_sample_index, Some(8));
            assert_eq!(decoded_segments[1].end_sample_index, Some(10));
            assert_eq!(decoded_segments[2].start_sample_index, Some(6));
            assert_eq!(decoded_segments[2].end_sample_index, Some(8));
        }
    }

    #[test]
    fn store_forward_decoder_splits_gaps_and_preserves_missing_index_compatibility() {
        let pcm = STANDARD.encode(
            [0_i16, 1_i16]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let indexed_payload = json!({
            "buffered_frames": [
                {"frame": {"sample_rate_hz": 16_000, "channels": 1, "start_time_ns": 1_000,
                    "start_sample_index": 0, "end_sample_index": 2, "samples_per_channel": 2,
                    "samples_b64": pcm}},
                {"frame": {"sample_rate_hz": 16_000, "channels": 1, "start_time_ns": 2_000,
                    "start_sample_index": 6, "end_sample_index": 8, "samples_per_channel": 2,
                    "samples_b64": pcm}}
            ]
        });
        let segments =
            decode_audio_payload_segments(&serde_json::to_vec(&indexed_payload).unwrap())
                .expect("indexed store-forward payload should decode");
        assert_eq!(segments.len(), 2);
        assert_eq!(
            (segments[0].start_sample_index, segments[0].end_sample_index),
            (Some(0), Some(2))
        );
        assert_eq!(
            (segments[1].start_sample_index, segments[1].end_sample_index),
            (Some(6), Some(8))
        );

        let unindexed_payload = json!({
            "buffered_frames": [
                {"frame": {"sample_rate_hz": 16_000, "channels": 1, "start_time_ns": 1_000,
                    "samples_per_channel": 2, "samples_b64": pcm}},
                {"frame": {"sample_rate_hz": 16_000, "channels": 1, "start_time_ns": 9_000,
                    "samples_per_channel": 2, "samples_b64": pcm}}
            ]
        });
        let segments =
            decode_audio_payload_segments(&serde_json::to_vec(&unindexed_payload).unwrap())
                .expect("unindexed store-forward payload should retain legacy concatenation");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].channels[0].len(), 4);
        assert_eq!(segments[0].start_sample_index, None);
        assert_eq!(segments[0].end_sample_index, None);
    }

    #[test]
    fn store_forward_decoder_rejects_frame_coverage_mismatch() {
        let pcm = STANDARD.encode(
            [0_i16, 1_i16]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let payload = json!({
            "buffered_frames": [{"frame": {
                "sample_rate_hz": 16_000, "channels": 1, "start_time_ns": 1_000,
                "start_sample_index": 0, "end_sample_index": 4, "samples_per_channel": 2,
                "samples_b64": pcm
            }}]
        });

        let error = decode_audio_payload_segments(&serde_json::to_vec(&payload).unwrap())
            .expect_err("mismatched coverage must be rejected");
        assert!(error.to_string().contains("sample indices"));
    }
}
