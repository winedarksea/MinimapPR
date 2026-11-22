//! Clip codec: decimate a window of the 48 kHz rolling buffer to 24 kHz and
//! wrap it as FLAC inside an MMCL container, and the inverse on the host.

use minimappr_core::clip_codec::field_clip::{decode_field_clip, encode_field_clip};
use minimappr_core::clip_codec::{DecimationFilterId, FieldClipDecimator};

use crate::error::FieldFfiError;
use crate::pcm16_bytes::{pcm16_little_endian_bytes, pcm16_samples_from_little_endian_bytes};

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldDecimationFilter {
    /// Bird band: anti-alias lowpass plus the ~25 Hz linear-phase rumble cut.
    BirdBandWithRumbleCut,
    /// Flat to DC for drone and gunshot modes.
    FlatToDirectCurrentForDroneAndGunshot,
}

impl From<FieldDecimationFilter> for DecimationFilterId {
    fn from(filter: FieldDecimationFilter) -> Self {
        match filter {
            FieldDecimationFilter::BirdBandWithRumbleCut => DecimationFilterId::BirdBandWithRumbleCut,
            FieldDecimationFilter::FlatToDirectCurrentForDroneAndGunshot => {
                DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot
            }
        }
    }
}

/// A decoded clip: 24 kHz PCM16 plus the 48 kHz device-clock anchor.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct FieldDecodedClip {
    pub first_sample_index_device_clock: u64,
    pub decimation_factor: u32,
    pub output_sample_rate_hz: u32,
    /// Little-endian PCM16 at the output rate (see `pcm16_bytes`).
    pub pcm16_little_endian_bytes_output_rate: Vec<u8>,
}

/// Encodes `[window_first, window_first + window_sample_count)` of the device
/// clock, reading filter context from `rolling_buffer` (whose element 0 is
/// device sample `rolling_buffer_first_sample_index`), as MMCL+FLAC bytes.
#[uniffi::export]
pub fn encode_clip_from_rolling_buffer(
    rolling_buffer_pcm16_little_endian_bytes: Vec<u8>,
    rolling_buffer_first_sample_index: u64,
    window_first_sample_index: u64,
    window_sample_count: u32,
    decimation_filter: FieldDecimationFilter,
) -> Result<Vec<u8>, FieldFfiError> {
    let rolling_buffer = pcm16_samples_from_little_endian_bytes(&rolling_buffer_pcm16_little_endian_bytes)?;
    // The decimator accepts a window that merely overlaps the buffer and
    // allocates per window sample, so an unchecked Kotlin `window_sample_count`
    // (up to u32::MAX) could request gigabytes and abort on OOM. Requiring the
    // window inside the buffer bounds the allocation by the bytes already
    // passed in, and matches the mesh node's own `window_is_buffered` rule.
    let rolling_buffer_end_exclusive = rolling_buffer_first_sample_index.checked_add(rolling_buffer.len() as u64);
    let window_end_exclusive = window_first_sample_index.checked_add(u64::from(window_sample_count));
    let window_is_inside_rolling_buffer = match (rolling_buffer_end_exclusive, window_end_exclusive) {
        (Some(buffer_end), Some(window_end)) => {
            window_sample_count > 0 && window_first_sample_index >= rolling_buffer_first_sample_index && window_end <= buffer_end
        }
        _ => false,
    };
    if !window_is_inside_rolling_buffer {
        return Err(FieldFfiError::InvalidArgument {
            detail: format!(
                "clip window [{window_first_sample_index}, +{window_sample_count}) is not inside the rolling buffer \
                 [{rolling_buffer_first_sample_index}, +{})",
                rolling_buffer.len()
            ),
        });
    }
    encode_clip_from_rolling_buffer_slice(
        &rolling_buffer,
        rolling_buffer_first_sample_index,
        window_first_sample_index,
        window_sample_count,
        decimation_filter.into(),
    )
}

/// Slice form shared with the mesh node's rolling buffer (no copy).
pub(crate) fn encode_clip_from_rolling_buffer_slice(
    rolling_buffer: &[i16],
    rolling_buffer_first_sample_index: u64,
    window_first_sample_index: u64,
    window_sample_count: u32,
    decimation_filter_id: DecimationFilterId,
) -> Result<Vec<u8>, FieldFfiError> {
    let decimated_window = FieldClipDecimator::for_filter(decimation_filter_id)
        .decimate_window_from_rolling_buffer(
            rolling_buffer,
            rolling_buffer_first_sample_index,
            window_first_sample_index,
            window_sample_count as usize,
        )
        .map_err(|error| FieldFfiError::ClipCodec { detail: format!("{error:?}") })?;
    encode_field_clip(&decimated_window).map_err(|error| FieldFfiError::ClipCodec { detail: error.to_string() })
}

#[uniffi::export]
pub fn decode_clip(clip_bytes: Vec<u8>) -> Result<FieldDecodedClip, FieldFfiError> {
    let decoded = decode_field_clip(&clip_bytes).map_err(|error| FieldFfiError::ClipCodec { detail: error.to_string() })?;
    Ok(FieldDecodedClip {
        first_sample_index_device_clock: decoded.header.first_sample_index_device_clock,
        decimation_factor: u32::from(decoded.header.decimation_factor),
        output_sample_rate_hz: decoded.header.output_sample_rate_hz,
        pcm16_little_endian_bytes_output_rate: pcm16_little_endian_bytes(&decoded.samples_output_rate),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROLLING_BUFFER_SAMPLES: usize = 48_000;
    const WINDOW_FIRST_SAMPLE_INDEX: u64 = 12_000;
    const WINDOW_SAMPLE_COUNT: u32 = 24_000;
    const TEST_TONE_PERIOD_SAMPLES: usize = 48;
    const TEST_TONE_AMPLITUDE: f64 = 8_000.0;

    fn tone_rolling_buffer() -> Vec<u8> {
        let samples: Vec<i16> = (0..ROLLING_BUFFER_SAMPLES)
            .map(|index| {
                let phase = 2.0 * std::f64::consts::PI * (index % TEST_TONE_PERIOD_SAMPLES) as f64
                    / TEST_TONE_PERIOD_SAMPLES as f64;
                (TEST_TONE_AMPLITUDE * phase.sin()) as i16
            })
            .collect();
        pcm16_little_endian_bytes(&samples)
    }

    #[test]
    fn encoded_clip_decodes_with_exact_device_clock_anchor() {
        let clip_bytes = encode_clip_from_rolling_buffer(
            tone_rolling_buffer(),
            0,
            WINDOW_FIRST_SAMPLE_INDEX,
            WINDOW_SAMPLE_COUNT,
            FieldDecimationFilter::BirdBandWithRumbleCut,
        )
        .expect("encode");
        let decoded = decode_clip(clip_bytes).expect("decode");
        assert_eq!(decoded.first_sample_index_device_clock, WINDOW_FIRST_SAMPLE_INDEX);
        assert_eq!(decoded.decimation_factor, 2);
        assert_eq!(decoded.pcm16_little_endian_bytes_output_rate.len(), WINDOW_SAMPLE_COUNT as usize);
    }

    #[test]
    fn window_outside_or_overrunning_the_buffer_is_rejected_before_allocating() {
        let filter = FieldDecimationFilter::BirdBandWithRumbleCut;
        let oversized_window = encode_clip_from_rolling_buffer(tone_rolling_buffer(), 0, 0, u32::MAX, filter);
        assert!(matches!(oversized_window, Err(FieldFfiError::InvalidArgument { .. })));
        let overflowing_window = encode_clip_from_rolling_buffer(tone_rolling_buffer(), 0, u64::MAX, 1, filter);
        assert!(matches!(overflowing_window, Err(FieldFfiError::InvalidArgument { .. })));
        let empty_window = encode_clip_from_rolling_buffer(tone_rolling_buffer(), 0, 0, 0, filter);
        assert!(matches!(empty_window, Err(FieldFfiError::InvalidArgument { .. })));
    }

    #[test]
    fn garbage_bytes_are_a_clip_codec_error() {
        assert!(matches!(decode_clip(vec![1, 2, 3]), Err(FieldFfiError::ClipCodec { .. })));
    }
}
