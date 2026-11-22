//! Glue: decimated window <-> FLAC-in-MMCL clip bytes.

use super::clip_container::{
    ClipContainerError, FieldClipContainer, FieldClipHeader, Pcm16IntegritySection,
};
use super::decimation_filter_design::{
    FIELD_CLIP_OUTPUT_SAMPLE_RATE_HZ, FIELD_DEVICE_CLOCK_SAMPLE_RATE_HZ,
};
use super::flac_clip_codec::{
    decode_flac_to_pcm16, encode_pcm16_to_flac, FlacClipCodecError,
    FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL,
};
use super::linear_phase_decimator::DecimatedClipWindow;
use super::{ClipCodecId, DecimationFilterId};

const FIELD_CLIP_MONO_CHANNEL_COUNT: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldClipError {
    Container(ClipContainerError),
    Flac(FlacClipCodecError),
    UnsupportedCodec(u8),
    ClipTooLong,
    /// Decoded audio disagrees with the header (rate, channels or length).
    HeaderAudioMismatch(&'static str),
}

impl std::fmt::Display for FieldClipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "field clip error: {self:?}")
    }
}

impl std::error::Error for FieldClipError {}

impl From<ClipContainerError> for FieldClipError {
    fn from(error: ClipContainerError) -> Self {
        Self::Container(error)
    }
}

impl From<FlacClipCodecError> for FieldClipError {
    fn from(error: FlacClipCodecError) -> Self {
        Self::Flac(error)
    }
}

/// A decoded clip: PCM16 at the output rate plus its device-clock anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFieldClip {
    pub header: FieldClipHeader,
    pub decimation_filter: Option<DecimationFilterId>,
    pub samples_output_rate: Vec<i16>,
}

impl DecodedFieldClip {
    /// Device-clock (48 kHz) index of output sample `output_index`.
    pub fn device_clock_index_of_output_sample(&self, output_index: usize) -> u64 {
        self.header.first_sample_index_device_clock
            + output_index as u64 * u64::from(self.header.decimation_factor)
    }
}

/// Encodes a mono decimated window as FLAC inside an MMCL container.
/// Windows under FLAC's 16-sample minimum block go as raw PCM16 (exact; at
/// that size compression is irrelevant).
pub fn encode_field_clip(window: &DecimatedClipWindow) -> Result<Vec<u8>, FieldClipError> {
    let output_samples_per_channel =
        u32::try_from(window.samples_output_rate.len()).map_err(|_| FieldClipError::ClipTooLong)?;
    let (codec, encoded_audio) =
        if window.samples_output_rate.len() < FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL {
            let raw_little_endian = window
                .samples_output_rate
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            (ClipCodecId::Pcm16LittleEndian, raw_little_endian)
        } else {
            let flac_bytes = encode_pcm16_to_flac(
                &window.samples_output_rate,
                FIELD_CLIP_MONO_CHANNEL_COUNT,
                FIELD_CLIP_OUTPUT_SAMPLE_RATE_HZ,
            )?;
            (ClipCodecId::Flac, flac_bytes)
        };
    let container = FieldClipContainer {
        header: FieldClipHeader {
            codec_id: codec.wire_byte(),
            decimation_filter_id: window.decimation_filter_id.wire_byte(),
            channel_count: FIELD_CLIP_MONO_CHANNEL_COUNT,
            output_sample_rate_hz: FIELD_CLIP_OUTPUT_SAMPLE_RATE_HZ,
            device_clock_sample_rate_hz: FIELD_DEVICE_CLOCK_SAMPLE_RATE_HZ,
            decimation_factor: u8::try_from(window.decimation_factor)
                .map_err(|_| FieldClipError::HeaderAudioMismatch("decimation factor"))?,
            first_sample_index_device_clock: window.first_sample_index_device_clock,
            output_samples_per_channel,
            pcm16_integrity: Some(Pcm16IntegritySection {
                clipped_output_sample_count: window.clipped_output_sample_count,
                filter_context_was_zero_padded: window.filter_context_was_zero_padded,
            }),
        },
        encoded_audio,
    };
    Ok(container.to_bytes()?)
}

/// Parses an MMCL container and decodes its audio, checking it against the
/// header so a corrupted or mislabelled clip never reaches the solver.
pub fn decode_field_clip(clip_bytes: &[u8]) -> Result<DecodedFieldClip, FieldClipError> {
    let container = FieldClipContainer::from_bytes(clip_bytes)?;
    let header = container.header;
    let samples_output_rate = match ClipCodecId::from_wire_byte(header.codec_id) {
        Some(ClipCodecId::Flac) => {
            let decoded = decode_flac_to_pcm16(&container.encoded_audio)?;
            if decoded.sample_rate_hz != header.output_sample_rate_hz {
                return Err(FieldClipError::HeaderAudioMismatch("sample rate"));
            }
            if decoded.channel_count != header.channel_count {
                return Err(FieldClipError::HeaderAudioMismatch("channel count"));
            }
            decoded.interleaved_samples
        }
        Some(ClipCodecId::Pcm16LittleEndian) => container
            .encoded_audio
            .chunks_exact(std::mem::size_of::<i16>())
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect(),
        None => return Err(FieldClipError::UnsupportedCodec(header.codec_id)),
    };
    let expected_total_samples =
        header.output_samples_per_channel as usize * usize::from(header.channel_count);
    if samples_output_rate.len() != expected_total_samples {
        return Err(FieldClipError::HeaderAudioMismatch("sample count"));
    }
    Ok(DecodedFieldClip {
        decimation_filter: DecimationFilterId::from_wire_byte(header.decimation_filter_id),
        header,
        samples_output_rate,
    })
}
