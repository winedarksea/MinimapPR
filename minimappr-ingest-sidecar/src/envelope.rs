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
    /// Firmware per-frame sequence counter — sequence of the *first* frame in
    /// the payload. Sidecar uses (first_sequence, last_sequence) to detect
    /// gaps across consecutive payloads from the same (node_id, boot_session).
    /// `None` for transports that don't carry a sequence (e.g. JSON store-and-forward).
    pub first_sequence: Option<u64>,
    /// Sequence of the *last* frame in the payload. Equal to first_sequence
    /// for single-frame payloads.
    pub last_sequence: Option<u64>,
    /// Firmware boot-session counter (resets on reboot). Scopes the sequence
    /// tracker so a reboot does not surface as a spurious gap warning.
    pub boot_session: Option<u32>,
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
    sequence: u64,
    toa_ns: u64,
    tor_ns: u64,
    time_quality: BinaryTimeQuality,
    clock_correction_ns: Option<i64>,
    clock_drift_ppm: Option<f64>,
}

fn binary_audio_source_type(value: u8) -> &'static str {
    match value {
        0 => "tdm",
        1 => "i2s_mono",
        2 => "pdm_direct",
        3 => "synthetic",
        4 => "silence",
        _ => "unknown",
    }
}

fn binary_publish_failure_stage(value: u8) -> &'static str {
    match value {
        1 => "dns",
        2 => "connect",
        3 => "send",
        4 => "recv",
        5 => "response_parse",
        6 => "timeout",
        7 => "wifi_disconnected",
        _ => "none",
    }
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

fn read_binary_ingest_version(reader: &mut BinaryReader<'_>) -> Result<u8, String> {
    let expected_version = match reader.read(4)? {
        magic if magic == b"MMB2" => 2,
        magic if magic == b"MMB3" => 3,
        _ => return Err("invalid binary ingest magic".to_string()),
    };
    let version = reader.u8()?;
    if version != expected_version {
        return Err(format!("unsupported binary ingest version {version}"));
    }
    Ok(version)
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

/// Extracts node spec fields from the binary header as a JSON value compatible
/// with NodeSpec model_validate. Returns None on parse error (binary is still valid
/// for audio processing — this is best-effort metadata extraction only).
pub fn extract_binary_node_json(raw_bytes: &[u8]) -> Option<serde_json::Value> {
    let mut reader = BinaryReader::new(raw_bytes);
    let _version = read_binary_ingest_version(&mut reader).ok()?;
    reader.u8().ok()?; // sort_by_toa
    reader.u16().ok()?; // frame_count

    let node_id = reader.string().ok()?;
    let node_type_code = reader.u8().ok()?;
    let has_geo_position = reader.u8().ok()? != 0;
    let position_geo: Option<(f32, f32, f32)> = if has_geo_position {
        let lat = reader.f32().ok()?;
        let lon = reader.f32().ok()?;
        let alt = reader.f32().ok()?;
        Some((lat, lon, alt))
    } else {
        None
    };
    let sensor_count = reader.u8().ok()?;
    let mut sensor_offsets: Vec<[f32; 3]> = Vec::with_capacity(usize::from(sensor_count));
    for _ in 0..sensor_count {
        let x = reader.f32().ok()?;
        let y = reader.f32().ok()?;
        let z = reader.f32().ok()?;
        sensor_offsets.push([x, y, z]);
    }
    let capability_count = reader.u8().ok()?;
    let mut capabilities: Vec<String> = Vec::with_capacity(usize::from(capability_count));
    for _ in 0..capability_count {
        if let Ok(cap) = reader.string() {
            capabilities.push(cap);
        } else {
            break;
        }
    }
    // Keep this metadata shape aligned with Python's binary_ingest._read_node;
    // stream consumers upsert it directly into the API node status view.
    let hardware = reader.string().unwrap_or_default();
    let firmware = reader.string().unwrap_or_default();
    let gps_signal = reader.string().unwrap_or_default();
    let position_source = reader.string().unwrap_or_default();
    let boot_count = reader.u32().unwrap_or_default();

    // node_type_code: 0=point, 1=sirith_tetra, 2=array, 3=gateway
    let node_type_str = match node_type_code {
        0 => "point",
        1 => "sirith_tetra",
        2 => "array",
        3 => "gateway",
        _ => "point",
    };

    let mut gps_meta = serde_json::Map::new();
    if !gps_signal.is_empty() {
        gps_meta.insert("signal".to_string(), serde_json::Value::String(gps_signal));
    }
    if !position_source.is_empty() {
        gps_meta.insert(
            "position_source".to_string(),
            serde_json::Value::String(position_source),
        );
    }
    let mut metadata_map = serde_json::Map::new();
    metadata_map.insert(
        "hardware".to_string(),
        serde_json::Value::String(if hardware.is_empty() {
            "unknown".to_string()
        } else {
            hardware
        }),
    );
    metadata_map.insert(
        "firmware".to_string(),
        serde_json::Value::String(if firmware.is_empty() {
            "dev".to_string()
        } else {
            firmware
        }),
    );
    metadata_map.insert(
        "boot_count".to_string(),
        serde_json::Value::from(boot_count),
    );
    if !gps_meta.is_empty() {
        metadata_map.insert("gps".to_string(), serde_json::Value::Object(gps_meta));
    }

    let mut node_json = serde_json::json!({
        "id": node_id,
        "node_type": node_type_str,
        "sensor_offsets_m": sensor_offsets,
        "capabilities": capabilities,
        "metadata": serde_json::Value::Object(metadata_map),
    });
    if let Some((lat, lon, alt)) = position_geo {
        node_json["position_geo"] = serde_json::json!({
            "lat": lat,
            "lon": lon,
            "alt_m": alt,
        });
    }
    Some(node_json)
}

pub fn extract_binary_first_frame_timing_json(raw_bytes: &[u8]) -> Option<serde_json::Value> {
    let mut reader = BinaryReader::new(raw_bytes);
    let version = read_binary_ingest_version(&mut reader).ok()?;
    if version < 3 {
        return None;
    }
    reader.u8().ok()?; // sort_by_toa
    let frame_count = reader.u16().ok()?;
    if frame_count == 0 {
        return None;
    }
    skip_binary_node_header_after_frame_count(&mut reader).ok()?;
    read_binary_first_frame_timing_json(&mut reader).ok()
}

fn skip_binary_node_header_after_frame_count(reader: &mut BinaryReader<'_>) -> Result<(), String> {
    let _node_id = reader.string()?;
    let _node_type_code = reader.u8()?;
    let has_geo_position = reader.u8()? != 0;
    if has_geo_position {
        let _lat = reader.f32()?;
        let _lon = reader.f32()?;
        let _alt = reader.f32()?;
    }
    let sensor_count = reader.u8()?;
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

fn read_binary_first_frame_timing_json(
    reader: &mut BinaryReader<'_>,
) -> Result<serde_json::Value, String> {
    let _start_time_ns = reader.u64()?;
    let _end_time_ns = reader.u64()?;
    let start_sample_index = reader.u64()?;
    let end_sample_index = reader.u64()?;
    let sample_rate_hz = reader.u32()?;
    let channels = reader.u8()?;
    let audio_source_type = reader.u8()?;
    let sequence = reader.u64()?;
    let toa_ns = reader.u64()?;
    let tor_ns = reader.u64()?;
    let time_quality = read_binary_time_quality(reader.u8()?)?;
    let samples_per_channel = reader.u32()?;
    let section_flags = reader.u16()?;

    let mut timing = serde_json::Map::new();
    timing.insert(
        "audio_source_type".to_string(),
        serde_json::Value::String(binary_audio_source_type(audio_source_type).to_string()),
    );
    timing.insert("sequence".to_string(), serde_json::Value::from(sequence));
    timing.insert("toa_ns".to_string(), serde_json::Value::from(toa_ns));
    timing.insert("tor_ns".to_string(), serde_json::Value::from(tor_ns));
    timing.insert(
        "time_quality".to_string(),
        serde_json::Value::String(time_quality.as_str().to_string()),
    );
    timing.insert(
        "sample_rate_hz".to_string(),
        serde_json::Value::from(sample_rate_hz),
    );
    timing.insert("channels".to_string(), serde_json::Value::from(channels));
    timing.insert(
        "start_sample_index".to_string(),
        serde_json::Value::from(start_sample_index),
    );
    timing.insert(
        "end_sample_index".to_string(),
        serde_json::Value::from(end_sample_index),
    );
    timing.insert(
        "samples_per_channel".to_string(),
        serde_json::Value::from(samples_per_channel),
    );

    for bit_index in 0..16 {
        let section_bit = 1_u16 << bit_index;
        if section_flags & section_bit == 0 {
            continue;
        }
        let section_len = usize::from(reader.u16()?);
        let section_payload = reader.read(section_len)?;
        let mut section_reader = BinaryReader::new(section_payload);
        match section_bit {
            0x0001 => {
                timing.extend(read_binary_timing_diagnostics_v3_json(&mut section_reader)?);
            }
            0x0002 => skip_binary_environment(&mut section_reader)?,
            0x0004 => {
                timing.insert(
                    "transport_health".to_string(),
                    read_binary_transport_health_json(&mut section_reader)?,
                );
            }
            0x0010 => {
                timing.insert(
                    "clock_holdover".to_string(),
                    read_binary_clock_holdover_json(&mut section_reader)?,
                );
            }
            _ => {
                // Forward-compatibility: consume unknown/future sections so newer
                // firmware does not break sidecar ingest. The section was already
                // length-delimited above.
                let remaining = section_reader.remaining();
                let _ = section_reader.read(remaining)?;
            }
        }
        if section_reader.remaining() != 0 {
            return Err(format!(
                "binary ingest section 0x{section_bit:04x} has trailing bytes"
            ));
        }
    }
    Ok(serde_json::Value::Object(timing))
}

fn read_binary_clock_holdover_json(
    reader: &mut BinaryReader<'_>,
) -> Result<serde_json::Value, String> {
    let flags = reader.u8()?;
    let holdover_age_ms = reader.u32()?;
    let predicted_error_ns = reader.u32()?;
    let lt_ppm = reader.f32()?;
    let lt_ppm_sigma = reader.f32()?;
    let temp_slope_ppm_per_c = reader.f32()?;
    let temp_resid_rms_ppm = reader.f32()?;

    let mut clock = serde_json::Map::new();
    clock.insert(
        "holdover_active".to_string(),
        serde_json::Value::from(flags & 0x01 != 0),
    );
    clock.insert(
        "lt_valid".to_string(),
        serde_json::Value::from(flags & 0x02 != 0),
    );
    clock.insert(
        "temp_model_valid".to_string(),
        serde_json::Value::from(flags & 0x04 != 0),
    );
    clock.insert(
        "temp_comp_applied".to_string(),
        serde_json::Value::from(flags & 0x08 != 0),
    );
    clock.insert(
        "holdover_age_ms".to_string(),
        serde_json::Value::from(holdover_age_ms),
    );
    clock.insert(
        "predicted_error_ns".to_string(),
        serde_json::Value::from(predicted_error_ns),
    );
    clock.insert(
        "lt_ppm".to_string(),
        serde_json::Value::from(f64::from(lt_ppm)),
    );
    clock.insert(
        "lt_ppm_sigma".to_string(),
        serde_json::Value::from(f64::from(lt_ppm_sigma)),
    );
    clock.insert(
        "temp_slope_ppm_per_c".to_string(),
        serde_json::Value::from(f64::from(temp_slope_ppm_per_c)),
    );
    clock.insert(
        "temp_resid_rms_ppm".to_string(),
        serde_json::Value::from(f64::from(temp_resid_rms_ppm)),
    );
    Ok(serde_json::Value::Object(clock))
}

fn read_binary_timing_diagnostics_v3_json(
    reader: &mut BinaryReader<'_>,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let gps_anchor = reader.u8()? != 0;
    let pps_edge_count = reader.u32()?;
    let dma_ring_slot_index = reader.u32()?;
    let pps_phase_error_ns = reader.i64()?;
    let estimated_ppm = reader.f64()?;
    let runner_frames_captured = reader.u64()?;
    let runner_frames_dropped = reader.u64()?;
    let runner_continuity_violations = reader.u64()?;
    let runner_publish_errors = reader.u64()?;
    let runner_queue_depth = reader.u32()?;
    let runner_queue_overflows = reader.u64()?;
    let runner_last_publish_status = reader.i32()?;
    let packet_age_us = reader.u64()?;
    let runner_last_publish_failure_stage = reader.u8()?;
    let runner_last_publish_lwip_error = reader.i32()?;
    let runner_consecutive_publish_failures = reader.u32()?;
    let runner_publish_timeout_failures = reader.u64()?;
    let runner_publish_connect_or_reset_failures = reader.u64()?;
    let runner_publish_dns_failures = reader.u64()?;
    let runner_publish_wifi_down_failures = reader.u64()?;

    let mut timing = serde_json::Map::new();
    timing.insert(
        "gps_anchor".to_string(),
        serde_json::Value::from(gps_anchor),
    );
    timing.insert(
        "pps_edge_count".to_string(),
        serde_json::Value::from(pps_edge_count),
    );
    timing.insert(
        "dma_ring_slot_index".to_string(),
        serde_json::Value::from(dma_ring_slot_index),
    );
    timing.insert(
        "pps_phase_error_ns".to_string(),
        serde_json::Value::from(pps_phase_error_ns),
    );
    timing.insert(
        "estimated_ppm".to_string(),
        serde_json::Value::from(estimated_ppm),
    );
    timing.insert(
        "runner_frames_captured".to_string(),
        serde_json::Value::from(runner_frames_captured),
    );
    timing.insert(
        "runner_frames_dropped".to_string(),
        serde_json::Value::from(runner_frames_dropped),
    );
    timing.insert(
        "runner_continuity_violations".to_string(),
        serde_json::Value::from(runner_continuity_violations),
    );
    timing.insert(
        "runner_publish_errors".to_string(),
        serde_json::Value::from(runner_publish_errors),
    );
    timing.insert(
        "runner_queue_depth".to_string(),
        serde_json::Value::from(runner_queue_depth),
    );
    timing.insert(
        "runner_queue_overflows".to_string(),
        serde_json::Value::from(runner_queue_overflows),
    );
    timing.insert(
        "runner_last_publish_status".to_string(),
        serde_json::Value::from(runner_last_publish_status),
    );
    timing.insert(
        "packet_age_us".to_string(),
        serde_json::Value::from(packet_age_us),
    );
    timing.insert(
        "runner_last_publish_failure_stage".to_string(),
        serde_json::Value::String(
            binary_publish_failure_stage(runner_last_publish_failure_stage).to_string(),
        ),
    );
    timing.insert(
        "runner_last_publish_failure_stage_code".to_string(),
        serde_json::Value::from(runner_last_publish_failure_stage),
    );
    timing.insert(
        "runner_last_publish_lwip_error".to_string(),
        serde_json::Value::from(runner_last_publish_lwip_error),
    );
    timing.insert(
        "runner_consecutive_publish_failures".to_string(),
        serde_json::Value::from(runner_consecutive_publish_failures),
    );
    timing.insert(
        "runner_publish_timeout_failures".to_string(),
        serde_json::Value::from(runner_publish_timeout_failures),
    );
    timing.insert(
        "runner_publish_connect_or_reset_failures".to_string(),
        serde_json::Value::from(runner_publish_connect_or_reset_failures),
    );
    timing.insert(
        "runner_publish_dns_failures".to_string(),
        serde_json::Value::from(runner_publish_dns_failures),
    );
    timing.insert(
        "runner_publish_wifi_down_failures".to_string(),
        serde_json::Value::from(runner_publish_wifi_down_failures),
    );
    Ok(timing)
}

fn read_binary_transport_health_json(
    reader: &mut BinaryReader<'_>,
) -> Result<serde_json::Value, String> {
    let ring_frames_high_water = reader.u16()?;
    let ring_frames_capacity = reader.u16()?;
    let queue_slots_high_water = reader.u16()?;
    let queue_slots_capacity = reader.u16()?;
    let publish_latency_last_ms = reader.u16()?;
    let publish_latency_ewma_ms = reader.u16()?;
    let publish_latency_max_ms = reader.u16()?;
    let wifi_rssi_dbm = i8::from_le_bytes([reader.u8()?]);
    let heap_free_bytes = reader.u32()?;
    let boot_id = reader.u32()?;

    Ok(serde_json::json!({
        "ring_frames_high_water": ring_frames_high_water,
        "ring_frames_capacity": ring_frames_capacity,
        "queue_slots_high_water": queue_slots_high_water,
        "queue_slots_capacity": queue_slots_capacity,
        "publish_latency_last_ms": publish_latency_last_ms,
        "publish_latency_ewma_ms": publish_latency_ewma_ms,
        "publish_latency_max_ms": publish_latency_max_ms,
        "wifi_rssi_dbm": wifi_rssi_dbm,
        "heap_free_bytes": heap_free_bytes,
        "boot_id": boot_id,
    }))
}

fn parse_binary_capture_envelope(raw_payload: &[u8]) -> Result<CaptureEnvelope, String> {
    let mut reader = BinaryReader::new(raw_payload);
    let version = read_binary_ingest_version(&mut reader)?;

    let _sort_by_toa = reader.u8()?;
    let frame_count = reader.u16()?;
    if frame_count == 0 || frame_count > 2048 {
        return Err("binary ingest frame count must be between 1 and 2048".to_string());
    }

    let node_id = reader.string()?;
    let _node_type_code = reader.u8()?;
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
    let boot_count = reader.u32()?;

    let mut first_frame: Option<BinaryFrameSummary> = None;
    let mut last_sequence: Option<u64> = None;
    let mut total_sample_count = 0_u64;
    for _ in 0..frame_count {
        let frame = read_binary_frame_summary(&mut reader, version)?;
        total_sample_count = total_sample_count.saturating_add(frame.sample_count);
        last_sequence = Some(frame.sequence);
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
        payload_codec: format!("binary_mmb{version}_pcm16le"),
        integrity_hash: sha256_hex(raw_payload),
        first_sequence: Some(first_frame.sequence),
        last_sequence,
        boot_session: Some(boot_count),
    })
}

fn read_binary_frame_summary(
    reader: &mut BinaryReader<'_>,
    version: u8,
) -> Result<BinaryFrameSummary, String> {
    let _start_time_ns = reader.u64()?;
    let _end_time_ns = reader.u64()?;
    let start_sample_index = reader.u64()?;
    let end_sample_index = reader.u64()?;
    let sample_rate_hz = reader.u32()?;
    let channels = reader.u8()?;
    if version >= 3 {
        let _audio_source_type = reader.u8()?;
    }
    let sequence = reader.u64()?;
    let toa_ns = reader.u64()?;
    let tor_ns = reader.u64()?;
    let time_quality = read_binary_time_quality(reader.u8()?)?;
    let samples_per_channel;
    let clock_correction_ns;
    let clock_drift_ppm;
    if version >= 3 {
        samples_per_channel = reader.u32()?;
        let timing = read_binary_v3_summary_sections(reader)?;
        clock_correction_ns = timing.0;
        clock_drift_ppm = timing.1;
    } else {
        let timing = read_binary_timing_diagnostics(reader)?;
        clock_correction_ns = timing.0;
        clock_drift_ppm = timing.1;
        skip_binary_environment(reader)?;
        samples_per_channel = reader.u32()?;
    }
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
        sequence,
        toa_ns,
        tor_ns,
        time_quality,
        clock_correction_ns,
        clock_drift_ppm,
    })
}

fn read_binary_v3_summary_sections(
    reader: &mut BinaryReader<'_>,
) -> Result<(Option<i64>, Option<f64>), String> {
    let section_flags = reader.u16()?;
    let mut clock_correction_ns = None;
    let mut clock_drift_ppm = None;
    for bit_index in 0..16 {
        let section_bit = 1_u16 << bit_index;
        if section_flags & section_bit == 0 {
            continue;
        }
        let section_len = usize::from(reader.u16()?);
        let section_payload = reader.read(section_len)?;
        let mut section_reader = BinaryReader::new(section_payload);
        match section_bit {
            0x0001 => {
                let timing = read_binary_timing_diagnostics_v3(&mut section_reader)?;
                clock_correction_ns = timing.0;
                clock_drift_ppm = timing.1;
            }
            0x0002 => skip_binary_environment(&mut section_reader)?,
            0x0004 => skip_binary_transport_health(&mut section_reader)?,
            _ => {
                // Forward-compatibility: consume unknown/future sections (e.g.
                // 0x0010 clock holdover, not needed for this summary).
                let remaining = section_reader.remaining();
                let _ = section_reader.read(remaining)?;
            }
        }
        if section_reader.remaining() != 0 {
            return Err(format!(
                "binary ingest section 0x{section_bit:04x} has trailing bytes"
            ));
        }
    }
    Ok((clock_correction_ns, clock_drift_ppm))
}

fn read_binary_timing_diagnostics_v3(
    reader: &mut BinaryReader<'_>,
) -> Result<(Option<i64>, Option<f64>), String> {
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

fn skip_binary_transport_health(reader: &mut BinaryReader<'_>) -> Result<(), String> {
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

    let total_sample_count =
        envelope
            .buffered_frames
            .iter()
            .try_fold(0_u64, |running_total, buffered_frame| {
                let frame = &buffered_frame.frame;
                let frame_samples = frame.samples_per_channel.map(u64::from).or_else(|| {
                    match (frame.start_sample_index, frame.end_sample_index) {
                        (Some(start), Some(end)) if end >= start => Some(end - start),
                        _ => None,
                    }
                });
                match (Some(running_total), frame_samples) {
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
        // JSON store-and-forward transport does not currently carry sequence
        // or boot-session metadata. Sequence-gap tracking is therefore a no-op
        // for this transport.
        first_sequence: None,
        last_sequence: None,
        boot_session: None,
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
    use super::{
        extract_binary_first_frame_timing_json, extract_binary_node_json, parse_capture_envelope,
    };

    fn push_string(payload: &mut Vec<u8>, value: &str) {
        payload.push(u8::try_from(value.len()).expect("test string fits in length byte"));
        payload.extend_from_slice(value.as_bytes());
    }

    fn push_f32(payload: &mut Vec<u8>, value: f32) {
        payload.extend_from_slice(&value.to_le_bytes());
    }

    fn push_binary_node_header(payload: &mut Vec<u8>) {
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
        payload.push(2); // capability_count
        push_string(payload, "audio");
        push_string(payload, "gps_optional");
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

    fn binary_clock_holdover_section() -> Vec<u8> {
        let mut section = Vec::new();
        // flags: holdover_active | lt_valid | temp_model_valid | temp_comp_applied
        section.push(0x0F);
        section.extend_from_slice(&34_000_u32.to_le_bytes()); // holdover_age_ms
        section.extend_from_slice(&12_300_u32.to_le_bytes()); // predicted_error_ns
        section.extend_from_slice(&(-3.5_f32).to_le_bytes()); // lt_ppm
        section.extend_from_slice(&0.25_f32.to_le_bytes()); // lt_ppm_sigma
        section.extend_from_slice(&(-0.8_f32).to_le_bytes()); // temp_slope_ppm_per_c
        section.extend_from_slice(&0.05_f32.to_le_bytes()); // temp_resid_rms_ppm
        section
    }

    // Builds an MMB3 single-frame payload carrying the timing (0x0001),
    // transport-health (0x0004), and clock-holdover (0x0010) sections, plus an
    // optional unknown future section (0x0020) to exercise forward-compat.
    fn binary_single_frame_payload_with_holdover(include_unknown: bool) -> Vec<u8> {
        let mut payload = binary_header_only_payload(3);
        payload.extend_from_slice(&1_000_u64.to_le_bytes());
        payload.extend_from_slice(&2_000_u64.to_le_bytes());
        payload.extend_from_slice(&0_u64.to_le_bytes());
        payload.extend_from_slice(&2_u64.to_le_bytes());
        payload.extend_from_slice(&16_000_u32.to_le_bytes());
        payload.push(1); // channels
        payload.push(3); // synthetic audio source
        payload.extend_from_slice(&1_u64.to_le_bytes());
        payload.extend_from_slice(&1_000_u64.to_le_bytes());
        payload.extend_from_slice(&1_250_u64.to_le_bytes());
        payload.push(0); // gps_locked
        payload.extend_from_slice(&2_u32.to_le_bytes());
        let flags: u16 = 0x0001 | 0x0004 | 0x0010 | if include_unknown { 0x0020 } else { 0 };
        payload.extend_from_slice(&flags.to_le_bytes());
        // Sections in ascending bit order, matching the firmware emission.
        push_binary_section(&mut payload, &binary_timing_section());
        push_binary_section(&mut payload, &binary_transport_health_section());
        push_binary_section(&mut payload, &binary_clock_holdover_section());
        if include_unknown {
            push_binary_section(&mut payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
        }
        payload.extend_from_slice(&0_i16.to_le_bytes());
        payload.extend_from_slice(&32_767_i16.to_le_bytes());
        payload
    }

    #[test]
    fn timing_json_decodes_clock_holdover_section() {
        let payload = binary_single_frame_payload_with_holdover(false);
        let timing = extract_binary_first_frame_timing_json(&payload).expect("timing json");
        let clock = timing.get("clock_holdover").expect("clock_holdover present");
        assert_eq!(clock["holdover_active"], serde_json::json!(true));
        assert_eq!(clock["lt_valid"], serde_json::json!(true));
        assert_eq!(clock["temp_model_valid"], serde_json::json!(true));
        assert_eq!(clock["temp_comp_applied"], serde_json::json!(true));
        assert_eq!(clock["holdover_age_ms"], serde_json::json!(34_000));
        assert_eq!(clock["predicted_error_ns"], serde_json::json!(12_300));
        assert!((clock["lt_ppm"].as_f64().unwrap() - (-3.5)).abs() < 1e-5);
        assert!((clock["lt_ppm_sigma"].as_f64().unwrap() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn timing_json_skips_unknown_future_section_without_error() {
        // An unknown 0x0020 section must not break decoding of the known ones.
        let payload = binary_single_frame_payload_with_holdover(true);
        let timing = extract_binary_first_frame_timing_json(&payload).expect("timing json");
        assert!(timing.get("clock_holdover").is_some());
        assert!(timing.get("transport_health").is_some());
    }

    fn push_binary_frame(payload: &mut Vec<u8>, version: u8) {
        payload.extend_from_slice(&1_000_u64.to_le_bytes());
        payload.extend_from_slice(&2_000_u64.to_le_bytes());
        payload.extend_from_slice(&0_u64.to_le_bytes());
        payload.extend_from_slice(&2_u64.to_le_bytes());
        payload.extend_from_slice(&16_000_u32.to_le_bytes());
        payload.push(1); // channels
        if version == 3 {
            payload.push(3); // synthetic audio source
        }
        payload.extend_from_slice(&1_u64.to_le_bytes());
        payload.extend_from_slice(&1_000_u64.to_le_bytes());
        payload.extend_from_slice(&1_250_u64.to_le_bytes());
        payload.push(0); // gps_locked
        if version == 3 {
            payload.extend_from_slice(&2_u32.to_le_bytes());
            payload.extend_from_slice(&0x0007_u16.to_le_bytes());
            push_binary_section(payload, &binary_timing_section());
            push_binary_section(payload, &binary_environment_section());
            push_binary_section(payload, &binary_transport_health_section());
        } else {
            payload.push(0); // no timing diagnostics
            payload.push(0); // no environment
            payload.extend_from_slice(&2_u32.to_le_bytes());
        }
        payload.extend_from_slice(&0_i16.to_le_bytes());
        payload.extend_from_slice(&32_767_i16.to_le_bytes());
    }

    fn binary_header_only_payload(version: u8) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(match version {
            2 => b"MMB2",
            3 => b"MMB3",
            _ => panic!("unsupported test version"),
        });
        payload.push(version);
        payload.push(1); // sort_by_toa
        payload.extend_from_slice(&1_u16.to_le_bytes());
        push_binary_node_header(&mut payload);
        payload
    }

    fn binary_single_frame_payload(version: u8) -> Vec<u8> {
        let mut payload = binary_header_only_payload(version);
        push_binary_frame(&mut payload, version);
        payload
    }

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

    #[test]
    fn binary_node_extraction_preserves_python_metadata_shape_for_gps_status() {
        let node = extract_binary_node_json(&binary_header_only_payload(2))
            .expect("MMB2 node header should extract");

        assert_eq!(node["id"], "sirith-tetra-1a15");
        assert_eq!(node["node_type"], "sirith_tetra");
        assert!(node.get("position_m").is_none());
        assert_eq!(node["metadata"]["hardware"], "sirith-test-hardware");
        assert_eq!(node["metadata"]["firmware"], "sirith-test-firmware");
        assert_eq!(node["metadata"]["boot_count"], 7);
        assert_eq!(node["metadata"]["gps"]["signal"], "fix_3d");
        assert_eq!(node["metadata"]["gps"]["position_source"], "gps_nmea_uart");
        assert_eq!(node["capabilities"][1], "gps_optional");
        assert_eq!(node["position_geo"]["lat"].as_f64().unwrap() as f32, 44.987);
    }

    #[test]
    fn binary_mmb2_node_extraction_omits_position_m_but_preserves_geo() {
        let node = extract_binary_node_json(&binary_header_only_payload(2))
            .expect("MMB2 node header should extract");

        assert_eq!(node["id"], "sirith-tetra-1a15");
        assert!(node.get("position_m").is_none());
        assert_eq!(node["metadata"]["gps"]["position_source"], "gps_nmea_uart");
        assert_eq!(
            node["position_geo"]["lon"].as_f64().unwrap() as f32,
            -93.258
        );
    }

    #[test]
    fn binary_capture_envelope_accepts_mmb2_payloads() {
        let parsed =
            parse_capture_envelope("/api/v1/ingest/binary", &binary_single_frame_payload(2))
                .expect("MMB2 binary envelope should parse");

        assert_eq!(parsed.node_id, "sirith-tetra-1a15");
        assert_eq!(parsed.sample_rate_hz, Some(16_000));
        assert_eq!(parsed.channel_count, Some(1));
        assert_eq!(parsed.sample_count, Some(2));
        assert_eq!(parsed.payload_codec, "binary_mmb2_pcm16le");
    }

    #[test]
    fn binary_capture_envelope_accepts_mmb3_payloads() {
        let parsed =
            parse_capture_envelope("/api/v1/ingest/binary", &binary_single_frame_payload(3))
                .expect("MMB3 binary envelope should parse");

        assert_eq!(parsed.node_id, "sirith-tetra-1a15");
        assert_eq!(parsed.sample_rate_hz, Some(16_000));
        assert_eq!(parsed.channel_count, Some(1));
        assert_eq!(parsed.sample_count, Some(2));
        assert_eq!(parsed.payload_codec, "binary_mmb3_pcm16le");
        assert_eq!(parsed.clock_correction_ns, Some(-123));
        assert_eq!(parsed.clock_drift_ppm, Some(0.25));
    }
}
