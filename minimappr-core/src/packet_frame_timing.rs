//! Per-frame clock evidence kept from decode (localization overhaul R5).
//!
//! Decode merges sample-contiguous frames into one segment so audio assembly
//! is unchanged, but each frame's own timestamp, clock class, holdover error
//! and boot session survive here: a timestamp step inside one upload is only
//! visible frame by frame.

use crate::localization_sensor_entry::NodeTimeQuality;

/// Timebase of firmware packet timestamps (UTC epoch nanoseconds).
pub const PACKET_TIMESTAMP_CLOCK_DOMAIN_UTC: &str = "utc";

/// Clock evidence of one decoded frame, in segment order.
#[derive(Clone, Debug, PartialEq)]
pub struct PacketFrameTiming {
    /// First sample of this frame, counted from the segment's first sample.
    pub sample_offset: usize,
    /// Packet timestamp of the frame's first sample; `None` when the frame
    /// carries none (a zero timestamp means "unset" on the wire).
    pub packet_timestamp_ns: Option<i128>,
    pub time_quality: NodeTimeQuality,
    /// Firmware-predicted clock error while in GPS holdover (MMB3 clock
    /// holdover section); `None` when the packet reports no active holdover.
    pub reported_holdover_error_ns: Option<u64>,
    /// Firmware boot counter of the upload; `None` for transports without one.
    pub boot_session: Option<u32>,
}

impl PacketFrameTiming {
    /// Absolute node sample index of the frame's first sample.
    pub fn start_sample_index(&self, segment_start_sample_index: i64) -> i64 {
        segment_start_sample_index.saturating_add(self.sample_offset as i64)
    }
}

/// Binary ingest `time_quality` wire code (firmware enum order).
pub fn time_quality_from_binary_wire_code(code: u8) -> NodeTimeQuality {
    match code {
        0 => NodeTimeQuality::GpsLocked,
        1 => NodeTimeQuality::GpsHoldover,
        2 => NodeTimeQuality::NtpDisciplined,
        3 => NodeTimeQuality::FreeRunning,
        _ => NodeTimeQuality::Unknown,
    }
}
