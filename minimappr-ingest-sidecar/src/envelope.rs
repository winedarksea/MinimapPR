use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct CaptureEnvelope {
    pub node_id: String,
    pub stream_id: String,
    pub stream_key: String,
    pub sensor_type: String,
    pub source_type: String,
    pub transport: String,
    pub toa_ns: Option<u64>,
    pub tor_ns: Option<u64>,
    pub time_quality: Option<String>,
    pub clock_domain: Option<String>,
    pub sync_source: Option<String>,
    pub clock_correction_ns: Option<i64>,
    pub clock_drift_ppm: Option<f64>,
    pub sample_rate_hz: Option<u32>,
    pub channel_count: Option<u16>,
    pub channel_layout: Option<String>,
    pub sample_index_start: Option<u64>,
    pub sample_count: Option<u64>,
    pub geometry_version: Option<String>,
    pub orientation_version: Option<String>,
    pub calibration_version: Option<String>,
    pub retention_hint: Option<String>,
    pub payload_codec: String,
    pub integrity_hash: String,
}

#[derive(Debug, Deserialize)]
struct StoreForwardEnvelope {
    node: StoreForwardNode,
    buffered_frames: Vec<StoreForwardBufferedFrame>,
}

#[derive(Debug, Deserialize)]
struct StoreForwardNode {
    id: String,
    #[serde(default)]
    sensor_offsets_m: Vec<[f32; 3]>,
    #[serde(default)]
    metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct StoreForwardBufferedFrame {
    frame: StoreForwardFrame,
}

#[derive(Debug, Deserialize)]
struct StoreForwardFrame {
    sample_rate_hz: u32,
    channels: u16,
    #[serde(default)]
    start_sample_index: Option<u64>,
    #[serde(default)]
    end_sample_index: Option<u64>,
    #[serde(default)]
    samples_per_channel: Option<u32>,
    #[serde(default)]
    time_quality: Option<String>,
    #[serde(default)]
    toa_ns: Option<u64>,
    #[serde(default)]
    tor_ns: Option<u64>,
    #[serde(default)]
    source_type: Option<String>,
    #[serde(default)]
    encoding: Option<String>,
}

#[derive(Clone, Copy, Debug)]
enum BinaryTimeQuality {
    GpsLocked,
    GpsHoldover,
    NtpDisciplined,
    FreeRunning,
}

impl BinaryTimeQuality {
    fn as_str(self) -> &'static str {
        match self {
            Self::GpsLocked => "gps_locked",
            Self::GpsHoldover => "gps_holdover",
            Self::NtpDisciplined => "ntp_disciplined",
            Self::FreeRunning => "free_running",
        }
    }
}

#[derive(Clone, Debug)]
struct BinaryFrameSummary {
    sample_rate_hz: u32,
    channels: u16,
    sample_index_start: u64,
    sample_count: u64,
    toa_ns: u64,
    tor_ns: u64,
    time_quality: BinaryTimeQuality,
    clock_correction_ns: Option<i64>,
    clock_drift_ppm: Option<f64>,
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

    fn read(&mut self, length: usize) -> Result<&'a [u8], String> {
        if self.offset.saturating_add(length) > self.payload.len() {
            return Err("binary ingest payload ended unexpectedly".to_string());
        }
        let start = self.offset;
        self.offset += length;
        Ok(&self.payload[start..self.offset])
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.read(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let bytes: [u8; 2] = self.read(2)?.try_into().map_err(|_| "invalid u16 field")?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes: [u8; 4] = self.read(4)?.try_into().map_err(|_| "invalid u32 field")?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let bytes: [u8; 8] = self.read(8)?.try_into().map_err(|_| "invalid u64 field")?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn i32(&mut self) -> Result<i32, String> {
        let bytes: [u8; 4] = self.read(4)?.try_into().map_err(|_| "invalid i32 field")?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64, String> {
        let bytes: [u8; 8] = self.read(8)?.try_into().map_err(|_| "invalid i64 field")?;
        Ok(i64::from_le_bytes(bytes))
    }

    fn f32(&mut self) -> Result<f32, String> {
        let bytes: [u8; 4] = self.read(4)?.try_into().map_err(|_| "invalid f32 field")?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, String> {
        let bytes: [u8; 8] = self.read(8)?.try_into().map_err(|_| "invalid f64 field")?;
        Ok(f64::from_le_bytes(bytes))
    }

    fn string(&mut self) -> Result<String, String> {
        let length = usize::from(self.u8()?);
        let raw = self.read(length)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| "binary ingest string field was not utf-8".to_string())
    }
}

pub fn parse_capture_envelope(
    endpoint: &str,
    raw_payload: &[u8],
) -> Result<CaptureEnvelope, String> {
    match endpoint {
        "/api/v1/ingest/binary" => parse_binary_capture_envelope(raw_payload),
        "/api/v1/ingest/store-forward" => parse_store_forward_capture_envelope(raw_payload),
        other => Err(format!("unsupported ingest endpoint {other}")),
    }
}

fn parse_binary_capture_envelope(raw_payload: &[u8]) -> Result<CaptureEnvelope, String> {
    let mut reader = BinaryReader::new(raw_payload);
    if reader.read(4)? != b"MMB1" {
        return Err("invalid binary ingest magic".to_string());
    }
    let version = reader.u8()?;
    if version != 1 {
        return Err(format!("unsupported binary ingest version {version}"));
    }

    let _sort_by_toa = reader.u8()?;
    let frame_count = reader.u16()?;
    if frame_count == 0 || frame_count > 2048 {
        return Err("binary ingest frame count must be between 1 and 2048".to_string());
    }

    let node_id = reader.string()?;
    let _node_type_code = reader.u8()?;
    let _position_x = reader.f32()?;
    let _position_y = reader.f32()?;
    let _position_z = reader.f32()?;
    let has_geo_position = reader.u8()? != 0;
    if has_geo_position {
        let _lat = reader.f32()?;
        let _lon = reader.f32()?;
        let _alt = reader.f32()?;
    }
    let sensor_count = reader.u8()?;
    if sensor_count == 0 {
        return Err("binary node must declare at least one sensor".to_string());
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

    let mut first_frame: Option<BinaryFrameSummary> = None;
    let mut total_sample_count = 0_u64;
    for _ in 0..frame_count {
        let frame = read_binary_frame_summary(&mut reader)?;
        total_sample_count = total_sample_count.saturating_add(frame.sample_count);
        if first_frame.is_none() {
            first_frame = Some(frame);
        }
    }

    if reader.remaining() != 0 {
        return Err("binary ingest payload has trailing bytes".to_string());
    }

    let first_frame =
        first_frame.ok_or_else(|| "binary ingest payload contained no frames".to_string())?;
    let time_quality = first_frame.time_quality.as_str().to_string();
    let stream_id = "audio_main".to_string();
    let channel_count = Some(first_frame.channels.max(u16::from(sensor_count)));
    Ok(CaptureEnvelope {
        node_id: node_id.clone(),
        stream_id: stream_id.clone(),
        stream_key: stream_key_from_parts(&node_id, &stream_id),
        sensor_type: "audio".to_string(),
        source_type: "raw_sensor".to_string(),
        transport: "http_binary".to_string(),
        toa_ns: Some(first_frame.toa_ns),
        tor_ns: Some(first_frame.tor_ns),
        time_quality: Some(time_quality.clone()),
        clock_domain: Some("utc".to_string()),
        sync_source: sync_source_from_time_quality(Some(time_quality.as_str())),
        clock_correction_ns: first_frame.clock_correction_ns,
        clock_drift_ppm: first_frame.clock_drift_ppm,
        sample_rate_hz: Some(first_frame.sample_rate_hz),
        channel_count,
        channel_layout: channel_layout(channel_count),
        sample_index_start: Some(first_frame.sample_index_start),
        sample_count: Some(total_sample_count),
        geometry_version: None,
        orientation_version: None,
        calibration_version: None,
        retention_hint: Some("ephemeral".to_string()),
        payload_codec: "binary_mmb1_pcm16le".to_string(),
        integrity_hash: sha256_hex(raw_payload),
    })
}

fn read_binary_frame_summary(reader: &mut BinaryReader<'_>) -> Result<BinaryFrameSummary, String> {
    let _start_time_ns = reader.u64()?;
    let _end_time_ns = reader.u64()?;
    let start_sample_index = reader.u64()?;
    let end_sample_index = reader.u64()?;
    let sample_rate_hz = reader.u32()?;
    let channels = reader.u8()?;
    let _sequence = reader.u64()?;
    let toa_ns = reader.u64()?;
    let tor_ns = reader.u64()?;
    let time_quality = read_binary_time_quality(reader.u8()?)?;
    let (clock_correction_ns, clock_drift_ppm) = read_binary_timing_diagnostics(reader)?;
    skip_binary_environment(reader)?;
    let samples_per_channel = reader.u32()?;
    if channels == 0 {
        return Err("binary frame must have at least one channel".to_string());
    }
    if samples_per_channel == 0 {
        return Err("binary frame must include at least one sample".to_string());
    }
    let expected_end_sample_index =
        start_sample_index.saturating_add(u64::from(samples_per_channel));
    if end_sample_index != expected_end_sample_index {
        return Err("binary frame sample indices do not match samples_per_channel".to_string());
    }
    let sample_bytes = usize::from(channels)
        .saturating_mul(
            usize::try_from(samples_per_channel).map_err(|_| "samples_per_channel overflow")?,
        )
        .saturating_mul(2);
    let _raw_audio = reader.read(sample_bytes)?;
    Ok(BinaryFrameSummary {
        sample_rate_hz,
        channels: u16::from(channels),
        sample_index_start: start_sample_index,
        sample_count: u64::from(samples_per_channel),
        toa_ns,
        tor_ns,
        time_quality,
        clock_correction_ns,
        clock_drift_ppm,
    })
}

fn read_binary_time_quality(value: u8) -> Result<BinaryTimeQuality, String> {
    match value {
        0 => Ok(BinaryTimeQuality::GpsLocked),
        1 => Ok(BinaryTimeQuality::GpsHoldover),
        2 => Ok(BinaryTimeQuality::NtpDisciplined),
        3 => Ok(BinaryTimeQuality::FreeRunning),
        _ => Err(format!("unsupported binary time quality {value}")),
    }
}

fn read_binary_timing_diagnostics(
    reader: &mut BinaryReader<'_>,
) -> Result<(Option<i64>, Option<f64>), String> {
    if reader.u8()? == 0 {
        return Ok((None, None));
    }
    let _gps_anchor = reader.u8()?;
    let _pps_edge_count = reader.u32()?;
    let _dma_ring_slot_index = reader.u32()?;
    let pps_phase_error_ns = reader.i64()?;
    let estimated_ppm = reader.f64()?;
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
    Ok((Some(pps_phase_error_ns), Some(estimated_ppm)))
}

fn skip_binary_environment(reader: &mut BinaryReader<'_>) -> Result<(), String> {
    let flags = reader.u8()?;
    if flags & 0x01 != 0 {
        let _temperature_c = reader.f32()?;
    }
    if flags & 0x02 != 0 {
        let _humidity_fraction = reader.f32()?;
    }
    if flags & 0x04 != 0 {
        let _source = reader.string()?;
    }
    Ok(())
}

fn parse_store_forward_capture_envelope(raw_payload: &[u8]) -> Result<CaptureEnvelope, String> {
    let envelope: StoreForwardEnvelope = serde_json::from_slice(raw_payload)
        .map_err(|error| format!("invalid store-forward ingest payload: {error}"))?;
    let first_frame = envelope.buffered_frames.first().ok_or_else(|| {
        "store-forward ingest payload must include at least one buffered frame".to_string()
    })?;

    let total_sample_count = envelope
        .buffered_frames
        .iter()
        .fold(Some(0_u64), |running_total, buffered_frame| {
            let frame = &buffered_frame.frame;
            let frame_samples = frame
                .samples_per_channel
                .map(u64::from)
                .or_else(|| match (frame.start_sample_index, frame.end_sample_index) {
                    (Some(start), Some(end)) if end >= start => Some(end - start),
                    _ => None,
                });
            match (running_total, frame_samples) {
                (Some(total), Some(samples)) => Some(total.saturating_add(samples)),
                _ => None,
            }
        });

    let node_id = envelope.node.id.clone();
    let stream_id = "audio_main".to_string();
    let time_quality = first_frame.frame.time_quality.clone();
    let encoding = first_frame
        .frame
        .encoding
        .clone()
        .unwrap_or_else(|| "pcm16le".to_string());
    let source_type = first_frame
        .frame
        .source_type
        .clone()
        .unwrap_or_else(|| "raw_sensor".to_string());
    let channel_count = if first_frame.frame.channels > 0 {
        Some(first_frame.frame.channels)
    } else {
        u16::try_from(envelope.node.sensor_offsets_m.len()).ok()
    };
    let metadata = &envelope.node.metadata;
    Ok(CaptureEnvelope {
        node_id: node_id.clone(),
        stream_id: stream_id.clone(),
        stream_key: stream_key_from_parts(&node_id, &stream_id),
        sensor_type: "audio".to_string(),
        source_type,
        transport: "http_store_forward".to_string(),
        toa_ns: first_frame.frame.toa_ns,
        tor_ns: first_frame.frame.tor_ns,
        time_quality: time_quality.clone(),
        clock_domain: Some("utc".to_string()),
        sync_source: sync_source_from_time_quality(time_quality.as_deref()),
        clock_correction_ns: None,
        clock_drift_ppm: metadata
            .get("gps")
            .and_then(|value| value.get("estimated_ppm"))
            .and_then(serde_json::Value::as_f64),
        sample_rate_hz: Some(first_frame.frame.sample_rate_hz),
        channel_count,
        channel_layout: channel_layout(channel_count),
        sample_index_start: first_frame.frame.start_sample_index,
        sample_count: total_sample_count,
        geometry_version: metadata_string(metadata.get("geometry_version")),
        orientation_version: metadata_string(metadata.get("orientation_version")),
        calibration_version: metadata_string(metadata.get("calibration_version")),
        retention_hint: Some("ephemeral".to_string()),
        payload_codec: format!("store_forward_{encoding}"),
        integrity_hash: sha256_hex(raw_payload),
    })
}

fn metadata_string(value: Option<&serde_json::Value>) -> Option<String> {
    value.and_then(serde_json::Value::as_str).map(str::to_owned)
}

fn sync_source_from_time_quality(time_quality: Option<&str>) -> Option<String> {
    match time_quality {
        Some("gps_locked") | Some("gps_holdover") => Some("gps".to_string()),
        Some("ntp_disciplined") | Some("ntp_sync") => Some("ntp".to_string()),
        Some("free_running") => Some("free_running".to_string()),
        Some(other) => Some(other.to_string()),
        None => None,
    }
}

fn channel_layout(channel_count: Option<u16>) -> Option<String> {
    match channel_count {
        Some(1) => Some("mono".to_string()),
        Some(2) => Some("stereo".to_string()),
        Some(4) => Some("tetrahedral".to_string()),
        Some(channels) => Some(format!("{channels}ch_interleaved")),
        None => None,
    }
}

fn sha256_hex(raw_payload: &[u8]) -> String {
    let digest = Sha256::digest(raw_payload);
    hex_encode(&digest)
}

fn stream_key_from_parts(node_id: &str, stream_id: &str) -> String {
    let suffix = short_hash_hex(format!("{node_id}:{stream_id}").as_bytes());
    format!(
        "{}__{}__{}",
        sanitize_component(node_id),
        sanitize_component(stream_id),
        suffix
    )
}

fn short_hash_hex(raw_payload: &[u8]) -> String {
    let digest = Sha256::digest(raw_payload);
    hex_encode(&digest[..4])
}

fn sanitize_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        "stream".to_string()
    } else {
        trimmed.to_string()
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::parse_capture_envelope;

    #[test]
    fn store_forward_accepts_missing_sensor_offsets_for_point_nodes() {
        let payload = serde_json::json!({
            "node": {
                "id": "point-node-1",
                "metadata": {}
            },
            "buffered_frames": [
                {
                    "frame": {
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "samples_per_channel": 4,
                        "time_quality": "gps_locked",
                        "toa_ns": 1000,
                        "tor_ns": 2000,
                        "encoding": "pcm16le",
                        "source_type": "raw_sensor"
                    }
                }
            ]
        });

        let parsed = parse_capture_envelope(
            "/api/v1/ingest/store-forward",
            payload.to_string().as_bytes(),
        )
        .expect("store-forward envelope should parse without sensor offsets");

        assert_eq!(parsed.node_id, "point-node-1");
        assert_eq!(parsed.channel_count, Some(1));
        assert_eq!(parsed.sample_count, Some(4));
    }

    #[test]
    fn store_forward_missing_sample_coverage_degrades_sample_count_to_none() {
        let payload = serde_json::json!({
            "node": {
                "id": "point-node-2",
                "metadata": {}
            },
            "buffered_frames": [
                {
                    "frame": {
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "time_quality": "gps_locked",
                        "toa_ns": 1000,
                        "tor_ns": 2000,
                        "encoding": "pcm16le",
                        "source_type": "raw_sensor"
                    }
                }
            ]
        });

        let parsed = parse_capture_envelope(
            "/api/v1/ingest/store-forward",
            payload.to_string().as_bytes(),
        )
        .expect("store-forward envelope should parse even without sample coverage fields");

        assert_eq!(parsed.node_id, "point-node-2");
        assert_eq!(parsed.sample_count, None);
    }
}
