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

pub fn decode_audio_payload(raw_payload: &[u8]) -> BoxedResult<DecodedAudioPayload> {
    if raw_payload.starts_with(b"MMB1") {
        return decode_binary_mmb1_audio(raw_payload);
    }
    decode_store_forward_audio(raw_payload)
}

fn decode_binary_mmb1_audio(raw_payload: &[u8]) -> BoxedResult<DecodedAudioPayload> {
    let mut reader = BinaryReader::new(raw_payload);
    if reader.read(4)? != b"MMB1" {
        return Err("invalid binary ingest magic".into());
    }
    let version = reader.u8()?;
    if version != 1 {
        return Err(format!("unsupported binary ingest version {version}").into());
    }

    let _sort_by_toa = reader.u8()?;
    let frame_count = reader.u16()?;
    if frame_count == 0 || frame_count > 2048 {
        return Err("binary ingest frame count must be between 1 and 2048".into());
    }

    skip_binary_node(&mut reader)?;

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sample_rate_hz: Option<u32> = None;
    let mut first_start_time_ns: Option<i128> = None;
    let mut first_start_sample_index: Option<i64> = None;
    let mut last_end_sample_index: Option<i64> = None;
    let mut latest_temperature_c: Option<f32> = None;
    let mut latest_humidity_fraction: Option<f32> = None;
    for _ in 0..frame_count {
        let frame = read_binary_audio_frame(&mut reader)?;
        if let Some(existing_rate) = sample_rate_hz {
            if existing_rate != frame.sample_rate_hz {
                return Err("binary ingest frames mixed sample rates".into());
            }
        }
        sample_rate_hz = Some(frame.sample_rate_hz);
        first_start_time_ns.get_or_insert(frame.start_time_ns as i128);
        first_start_sample_index.get_or_insert(frame.start_sample_index as i64);
        last_end_sample_index = Some(frame.end_sample_index as i64);
        if frame.temperature_c.is_some() {
            latest_temperature_c = frame.temperature_c;
        }
        if frame.humidity_fraction.is_some() {
            latest_humidity_fraction = frame.humidity_fraction;
        }
        append_channels(&mut channels, frame.channels, frame.samples);
    }
    if reader.remaining() != 0 {
        return Err("binary ingest payload has trailing bytes".into());
    }
    Ok(DecodedAudioPayload {
        channels,
        sample_rate_hz: sample_rate_hz.ok_or("binary ingest payload contained no audio")?,
        start_time_ns: first_start_time_ns,
        start_sample_index: first_start_sample_index,
        end_sample_index: last_end_sample_index,
        temperature_c: latest_temperature_c,
        humidity_fraction: latest_humidity_fraction,
    })
}

fn skip_binary_node(reader: &mut BinaryReader<'_>) -> BoxedResult<()> {
    let _node_id = reader.string()?;
    let _node_type_code = reader.u8()?;
    let _position_x = reader.f32()?;
    let _position_y = reader.f32()?;
    let _position_z = reader.f32()?;
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
}

fn read_binary_audio_frame(reader: &mut BinaryReader<'_>) -> BoxedResult<BinaryAudioFrame> {
    let start_time_ns = reader.u64()?;
    let _end_time_ns = reader.u64()?;
    let start_sample_index = reader.u64()?;
    let end_sample_index = reader.u64()?;
    let sample_rate_hz = reader.u32()?;
    let channels = usize::from(reader.u8()?);
    let _sequence = reader.u64()?;
    let _toa_ns = reader.u64()?;
    let _tor_ns = reader.u64()?;
    let _time_quality = reader.u8()?;
    skip_binary_timing_diagnostics(reader)?;
    let environment = read_binary_environment(reader)?;
    let samples_per_channel = usize::try_from(reader.u32()?)?;
    if channels == 0 || samples_per_channel == 0 {
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
    })
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
}

fn read_binary_environment(reader: &mut BinaryReader<'_>) -> BoxedResult<BinaryEnvironmentSample> {
    let flags = reader.u8()?;
    let mut temperature_c = None;
    let mut humidity_fraction = None;
    if flags & 0x01 != 0 {
        temperature_c = Some(reader.f32()?);
    }
    if flags & 0x02 != 0 {
        humidity_fraction = Some(reader.f32()?);
    }
    if flags & 0x04 != 0 {
        let _source = reader.string()?;
    }
    Ok(BinaryEnvironmentSample {
        temperature_c,
        humidity_fraction,
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

fn decode_store_forward_audio(raw_payload: &[u8]) -> BoxedResult<DecodedAudioPayload> {
    let envelope: StoreForwardEnvelope = serde_json::from_slice(raw_payload)?;
    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sample_rate_hz: Option<u32> = None;
    let mut first_start_time_ns: Option<i128> = None;
    let mut first_start_sample_index: Option<i64> = None;
    let mut last_end_sample_index: Option<i64> = None;
    let mut latest_temperature_c: Option<f32> = None;
    let mut latest_humidity_fraction: Option<f32> = None;
    for buffered in envelope.buffered_frames {
        if let Some(environment) = buffered.environment {
            if environment.temperature_c.is_some() {
                latest_temperature_c = environment.temperature_c;
            }
            if environment.humidity_fraction.is_some() {
                latest_humidity_fraction = environment.humidity_fraction;
            }
        }
        let frame = buffered.frame;
        if frame.channels == 0 {
            return Err("store-forward frame must have at least one channel".into());
        }
        if let Some(existing_rate) = sample_rate_hz {
            if existing_rate != frame.sample_rate_hz {
                return Err("store-forward frames mixed sample rates".into());
            }
        }
        let raw_audio = STANDARD.decode(frame.samples_b64.as_bytes())?;
        let decoded = pcm16le_to_f32(&raw_audio);
        if decoded.len() % frame.channels != 0 {
            return Err("store-forward PCM sample count is not divisible by channels".into());
        }
        let frame_samples = decoded.len() / frame.channels;
        if let Some(expected) = frame.samples_per_channel {
            if usize::try_from(expected)? != frame_samples {
                return Err("store-forward samples_per_channel does not match PCM payload".into());
            }
        }
        sample_rate_hz = Some(frame.sample_rate_hz);
        first_start_time_ns.get_or_insert(
            frame
                .start_time_ns
                .or(frame.utc_start_ns)
                .unwrap_or_default() as i128,
        );
        if let Some(start) = frame.start_sample_index {
            first_start_sample_index.get_or_insert(start as i64);
            last_end_sample_index = Some(
                frame
                    .end_sample_index
                    .unwrap_or(start.saturating_add(frame_samples as u64)) as i64,
            );
        }
        append_channels(&mut channels, frame.channels, decoded);
    }
    Ok(DecodedAudioPayload {
        channels,
        sample_rate_hz: sample_rate_hz.ok_or("store-forward payload contained no audio")?,
        start_time_ns: first_start_time_ns,
        start_sample_index: first_start_sample_index,
        end_sample_index: last_end_sample_index,
        temperature_c: latest_temperature_c,
        humidity_fraction: latest_humidity_fraction,
    })
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
