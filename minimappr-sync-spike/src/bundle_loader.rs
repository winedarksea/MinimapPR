//! Loads one phone's bundle: a tolerant WAV reader (the app may be killed
//! before it patches the header, so the data chunk runs to end of file),
//! `session.json`, `events.jsonl`, and the capture timeline that maps
//! monotonic nanoseconds to capture samples (`capture_timeline.rs`).

use std::fs;
use std::path::{Path, PathBuf};

use crate::analysis_config::AnalysisConfig;
use crate::bundle_schema::{BundleEvent, SessionDescription};
use crate::capture_timeline::CaptureTimeline;

const WAV_HEADER_SEARCH_LIMIT_BYTES: usize = 4_096;
const NOMINAL_RATE_HZ: f64 = 48_000.0;
const NANOSECONDS_PER_SECOND: f64 = 1e9;

#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    /// Data length the header claims (0 or wrong when the app was killed).
    pub header_declared_data_bytes: u32,
    pub actual_data_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineAnchor {
    pub reference_frame: f64,
    pub ns_at_reference: f64,
    pub ns_per_frame: f64,
    pub residual_rms_ns: f64,
    pub anchor_count: usize,
}

impl TimelineAnchor {
    pub fn frame_at_ns(&self, ns: f64) -> f64 {
        self.reference_frame + (ns - self.ns_at_reference) / self.ns_per_frame
    }

    pub fn ns_at_frame(&self, frame: f64) -> f64 {
        self.ns_at_reference + (frame - self.reference_frame) * self.ns_per_frame
    }

    /// Capture clock rate relative to the monotonic clock, ppm.
    pub fn rate_offset_ppm(&self) -> f64 {
        (NANOSECONDS_PER_SECOND / NOMINAL_RATE_HZ / self.ns_per_frame - 1.0) * 1e6
    }
}

/// Least-squares line through `(frame, ns)` points.
pub fn fit_timeline(points: &[(f64, f64)]) -> Option<TimelineAnchor> {
    let first = points.first()?;
    if points.len() == 1 {
        return Some(TimelineAnchor {
            reference_frame: first.0,
            ns_at_reference: first.1,
            ns_per_frame: NANOSECONDS_PER_SECOND / NOMINAL_RATE_HZ,
            residual_rms_ns: 0.0,
            anchor_count: 1,
        });
    }
    let count = points.len() as f64;
    let (x0, y0) = *first;
    let mean_x = points.iter().map(|p| p.0 - x0).sum::<f64>() / count;
    let mean_y = points.iter().map(|p| p.1 - y0).sum::<f64>() / count;
    let sxx: f64 = points.iter().map(|p| (p.0 - x0 - mean_x).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - x0 - mean_x) * (p.1 - y0 - mean_y)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let residual_rms_ns = (points.iter().map(|p| (p.1 - y0 - intercept - slope * (p.0 - x0)).powi(2)).sum::<f64>() / count).sqrt();
    Some(TimelineAnchor { reference_frame: x0, ns_at_reference: y0 + intercept, ns_per_frame: slope, residual_rms_ns, anchor_count: points.len() })
}

pub fn read_wav_tolerant(path: &Path) -> Result<WavAudio, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF/WAVE file", path.display()));
    }
    let mut cursor = 12;
    let mut format: Option<(u16, u16, u32, u16)> = None;
    while cursor + 8 <= bytes.len().min(WAV_HEADER_SEARCH_LIMIT_BYTES + 8) {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().expect("4 bytes"));
        let body = cursor + 8;
        if chunk_id == b"fmt " && body + 16 <= bytes.len() {
            let read_u16 = |offset: usize| u16::from_le_bytes([bytes[body + offset], bytes[body + offset + 1]]);
            let sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().expect("4 bytes"));
            format = Some((read_u16(0), read_u16(2), sample_rate, read_u16(14)));
        }
        if chunk_id == b"data" {
            let (audio_format, channel_count, sample_rate_hz, bits) = format.ok_or("data chunk before fmt chunk")?;
            if audio_format != 1 || bits != 16 {
                return Err(format!("{}: only PCM16 is supported (format {audio_format}, {bits} bits)", path.display()));
            }
            let data = &bytes[body..];
            let usable = data.len() - data.len() % (2 * channel_count as usize);
            let samples = data[..usable]
                .chunks_exact(2)
                .step_by(channel_count as usize)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / i16::MAX as f32)
                .collect();
            return Ok(WavAudio { samples, sample_rate_hz, channel_count, header_declared_data_bytes: chunk_size, actual_data_bytes: data.len() });
        }
        cursor = body + chunk_size as usize + (chunk_size as usize & 1);
    }
    Err(format!("{}: no data chunk", path.display()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct WallClockMapping {
    capture: CaptureTimeline,
    monotonic_to_wall: TimelineAnchor,
}

impl WallClockMapping {
    pub fn wall_ns_at_frame(&self, frame: f64) -> f64 {
        self.monotonic_to_wall.ns_at_frame(self.capture.ns_at_frame(frame))
    }

    pub fn frame_at_wall_ns(&self, wall_ns: f64) -> f64 {
        self.capture.frame_at_ns(self.monotonic_to_wall.frame_at_ns(wall_ns))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmissionRecord {
    pub emission_index: u64,
    pub output_frame_index: u64,
    pub target_ns: Option<i64>,
    pub played: bool,
    pub t_ns: i64,
    pub channel: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PhoneBundle {
    pub directory: PathBuf,
    pub session: SessionDescription,
    pub audio: WavAudio,
    pub events: Vec<BundleEvent>,
    pub unparsed_event_lines: usize,
    pub capture_timeline: Option<CaptureTimeline>,
}

impl PhoneBundle {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let session_text = fs::read_to_string(directory.join("session.json")).map_err(|e| format!("{}: session.json: {e}", directory.display()))?;
        let session: SessionDescription = serde_json::from_str(&session_text).map_err(|e| format!("{}: session.json: {e}", directory.display()))?;
        let audio = read_wav_tolerant(&directory.join("capture.wav"))?;
        let events_text = fs::read_to_string(directory.join("events.jsonl")).unwrap_or_default();
        let mut events = Vec::new();
        let mut unparsed_event_lines = 0;
        for line in events_text.lines().filter(|line| !line.trim().is_empty()) {
            match BundleEvent::from_json_line(line) {
                Some(event) => events.push(event),
                None => unparsed_event_lines += 1,
            }
        }
        let anchor_points: Vec<(f64, f64)> = events
            .iter()
            .filter_map(|event| match event {
                BundleEvent::CaptureTimestamp { frame_position, nano_time, .. } => Some((*frame_position as f64, *nano_time as f64)),
                _ => None,
            })
            .collect();
        let step_threshold_ns = AnalysisConfig::default().capture_timestamp_step_threshold_s * NANOSECONDS_PER_SECOND;
        let capture_timeline = CaptureTimeline::from_points(&anchor_points, step_threshold_ns);
        Ok(Self { directory: directory.to_path_buf(), session, audio, events, unparsed_event_lines, capture_timeline })
    }

    pub fn emissions(&self) -> Vec<EmissionRecord> {
        self.events
            .iter()
            .filter_map(|event| match event {
                BundleEvent::Emission { emission_index, output_frame_index, target_ns, played, t_ns, channel } => Some(EmissionRecord {
                    emission_index: *emission_index,
                    output_frame_index: *output_frame_index,
                    target_ns: *target_ns,
                    played: *played,
                    t_ns: *t_ns,
                    // Old bundles stamped no channel per emission: the schedule's is the one played.
                    channel: channel.clone().or_else(|| Some(self.session.schedule.channel.clone()).filter(|c| !c.is_empty())),
                }),
                _ => None,
            })
            .collect()
    }

    /// Monotonic ns → wall-clock ns from the capture timestamps (NTP-level
    /// accuracy: resolves which chirp period is which across phones). The
    /// anchor's "frame" axis is monotonic ns here. It is fitted against
    /// nano_time, not frame_position, because the monotonic clock does not
    /// jump at a silent capture drop.
    fn monotonic_to_wall_anchor(&self) -> Option<TimelineAnchor> {
        let points: Vec<(f64, f64)> = self
            .events
            .iter()
            .filter_map(|event| match event {
                BundleEvent::CaptureTimestamp { nano_time, wall_ms: Some(wall_ms), .. } => Some((*nano_time as f64, *wall_ms as f64 * 1e6)),
                _ => None,
            })
            .collect();
        fit_timeline(&points)
    }

    /// Capture frame ↔ wall-clock ns, through the segmented capture timeline.
    pub fn wall_clock_mapping(&self) -> Option<WallClockMapping> {
        Some(WallClockMapping { capture: self.capture_timeline.clone()?, monotonic_to_wall: self.monotonic_to_wall_anchor()? })
    }

    pub fn duration_s(&self) -> f64 {
        self.audio.samples.len() as f64 / f64::from(self.audio.sample_rate_hz.max(1))
    }

    pub fn markers(&self) -> Vec<(i64, String)> {
        self.events
            .iter()
            .filter_map(|event| match event {
                BundleEvent::Marker { t_ns, label } => Some((*t_ns, label.clone())),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_header_reads_to_end_of_file() {
        let directory = std::env::temp_dir().join(format!("sync_spike_wav_{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("capture.wav");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&48_000u32.to_le_bytes());
        bytes.extend_from_slice(&96_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data\0\0\0\0");
        for sample in [0i16, 16_384, -16_384, 1] {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes.push(7);
        fs::write(&path, bytes).unwrap();
        let audio = read_wav_tolerant(&path).unwrap();
        assert_eq!(audio.samples.len(), 4);
        assert_eq!(audio.header_declared_data_bytes, 0);
        assert!((audio.samples[1] - 0.5).abs() < 1e-3);
        fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn timeline_fit_recovers_rate() {
        let points: Vec<(f64, f64)> = (0..20).map(|k| (k as f64 * 48_000.0, 1e12 + k as f64 * 1e9 / (1.0 + 30e-6))).collect();
        let anchor = fit_timeline(&points).unwrap();
        assert!((anchor.rate_offset_ppm() - 30.0).abs() < 0.01);
        assert!((anchor.frame_at_ns(1e12 + 5e9 / (1.0 + 30e-6)) - 240_000.0).abs() < 1e-3);
    }
}
