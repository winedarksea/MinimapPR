//! Binary clip container, framed with the MMB3 conventions used by the
//! firmware -> sidecar ingest (`audio_payload.rs`): 4-byte magic + version
//! byte, little-endian fixed fields, then a `u16` section-flag word whose set
//! bits each carry a `u16`-length-delimited section. Unknown sections are
//! skipped (forward compatibility); known sections must parse exactly.
//!
//! The magic is `MMCL`, not `MMB3`: a clip is not a live ingest frame, and a
//! distinct magic keeps `decode_audio_payload_segments` from misreading one.
//!
//! Layout (version 1):
//! ```text
//! "MMCL" u8 version
//! u8 codec_id  u8 decimation_filter_id  u8 channel_count
//! u32 output_sample_rate_hz  u32 device_clock_sample_rate_hz  u8 decimation_factor
//! u64 first_sample_index_device_clock  u32 output_samples_per_channel
//! u16 section_flags  { u16 length, payload }*   (ascending bit order)
//! u32 encoded_audio_length  encoded_audio_bytes
//! ```

pub const FIELD_CLIP_CONTAINER_MAGIC: &[u8; 4] = b"MMCL";
pub const FIELD_CLIP_CONTAINER_VERSION: u8 = 1;
/// Section 0x0001: PCM16 integrity (clipped count, zero-padded context flag).
pub const CLIP_SECTION_PCM16_INTEGRITY: u16 = 0x0001;
const SECTION_FLAG_BIT_COUNT: u32 = u16::BITS;

/// Fixed header fields of a clip container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldClipHeader {
    /// Raw `ClipCodecId` byte (kept raw so unknown codecs still parse).
    pub codec_id: u8,
    /// Raw `DecimationFilterId` byte.
    pub decimation_filter_id: u8,
    pub channel_count: u8,
    pub output_sample_rate_hz: u32,
    pub device_clock_sample_rate_hz: u32,
    pub decimation_factor: u8,
    /// Device-clock index of output sample 0 (the timing anchor for TDOA).
    pub first_sample_index_device_clock: u64,
    pub output_samples_per_channel: u32,
    pub pcm16_integrity: Option<Pcm16IntegritySection>,
}

/// Section 0x0001 payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pcm16IntegritySection {
    pub clipped_output_sample_count: u32,
    pub filter_context_was_zero_padded: bool,
}

/// Header plus encoded audio bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldClipContainer {
    pub header: FieldClipHeader,
    pub encoded_audio: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipContainerError {
    InvalidMagic,
    UnsupportedContainerVersion(u8),
    Truncated,
    SectionTooLong(u16),
    EncodedAudioTooLong,
    SectionHasTrailingBytes(u16),
    TrailingBytes,
}

impl std::fmt::Display for ClipContainerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "field clip container error: {self:?}")
    }
}

impl std::error::Error for ClipContainerError {}

impl FieldClipContainer {
    pub fn to_bytes(&self) -> Result<Vec<u8>, ClipContainerError> {
        let header = &self.header;
        let mut bytes = Vec::with_capacity(64 + self.encoded_audio.len());
        bytes.extend_from_slice(FIELD_CLIP_CONTAINER_MAGIC);
        bytes.push(FIELD_CLIP_CONTAINER_VERSION);
        bytes.push(header.codec_id);
        bytes.push(header.decimation_filter_id);
        bytes.push(header.channel_count);
        bytes.extend_from_slice(&header.output_sample_rate_hz.to_le_bytes());
        bytes.extend_from_slice(&header.device_clock_sample_rate_hz.to_le_bytes());
        bytes.push(header.decimation_factor);
        bytes.extend_from_slice(&header.first_sample_index_device_clock.to_le_bytes());
        bytes.extend_from_slice(&header.output_samples_per_channel.to_le_bytes());

        let mut section_flags = 0_u16;
        let mut section_bytes = Vec::new();
        if let Some(integrity) = header.pcm16_integrity {
            section_flags |= CLIP_SECTION_PCM16_INTEGRITY;
            let mut payload = Vec::new();
            payload.extend_from_slice(&integrity.clipped_output_sample_count.to_le_bytes());
            payload.push(u8::from(integrity.filter_context_was_zero_padded));
            append_length_delimited_section(
                &mut section_bytes,
                CLIP_SECTION_PCM16_INTEGRITY,
                &payload,
            )?;
        }
        bytes.extend_from_slice(&section_flags.to_le_bytes());
        bytes.extend_from_slice(&section_bytes);

        let encoded_audio_length = u32::try_from(self.encoded_audio.len())
            .map_err(|_| ClipContainerError::EncodedAudioTooLong)?;
        bytes.extend_from_slice(&encoded_audio_length.to_le_bytes());
        bytes.extend_from_slice(&self.encoded_audio);
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ClipContainerError> {
        let mut reader = LittleEndianReader { remaining: bytes };
        if reader.take(FIELD_CLIP_CONTAINER_MAGIC.len())? != FIELD_CLIP_CONTAINER_MAGIC {
            return Err(ClipContainerError::InvalidMagic);
        }
        let version = reader.u8()?;
        if version != FIELD_CLIP_CONTAINER_VERSION {
            return Err(ClipContainerError::UnsupportedContainerVersion(version));
        }
        let mut header = FieldClipHeader {
            codec_id: reader.u8()?,
            decimation_filter_id: reader.u8()?,
            channel_count: reader.u8()?,
            output_sample_rate_hz: reader.u32()?,
            device_clock_sample_rate_hz: reader.u32()?,
            decimation_factor: reader.u8()?,
            first_sample_index_device_clock: reader.u64()?,
            output_samples_per_channel: reader.u32()?,
            pcm16_integrity: None,
        };
        let section_flags = reader.u16()?;
        for bit_index in 0..SECTION_FLAG_BIT_COUNT {
            let section_bit = 1_u16 << bit_index;
            if section_flags & section_bit == 0 {
                continue;
            }
            let section_length = usize::from(reader.u16()?);
            let mut section_reader = LittleEndianReader {
                remaining: reader.take(section_length)?,
            };
            if section_bit == CLIP_SECTION_PCM16_INTEGRITY {
                header.pcm16_integrity = Some(Pcm16IntegritySection {
                    clipped_output_sample_count: section_reader.u32()?,
                    filter_context_was_zero_padded: section_reader.u8()? != 0,
                });
                if !section_reader.remaining.is_empty() {
                    return Err(ClipContainerError::SectionHasTrailingBytes(section_bit));
                }
            }
            // Unknown sections: already consumed as a length-delimited block.
        }
        let encoded_audio_length = reader.u32()? as usize;
        let encoded_audio = reader.take(encoded_audio_length)?.to_vec();
        if !reader.remaining.is_empty() {
            return Err(ClipContainerError::TrailingBytes);
        }
        Ok(Self {
            header,
            encoded_audio,
        })
    }
}

fn append_length_delimited_section(
    destination: &mut Vec<u8>,
    section_bit: u16,
    payload: &[u8],
) -> Result<(), ClipContainerError> {
    let length = u16::try_from(payload.len())
        .map_err(|_| ClipContainerError::SectionTooLong(section_bit))?;
    destination.extend_from_slice(&length.to_le_bytes());
    destination.extend_from_slice(payload);
    Ok(())
}

struct LittleEndianReader<'bytes> {
    remaining: &'bytes [u8],
}

impl<'bytes> LittleEndianReader<'bytes> {
    fn take(&mut self, length: usize) -> Result<&'bytes [u8], ClipContainerError> {
        if self.remaining.len() < length {
            return Err(ClipContainerError::Truncated);
        }
        let (taken, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(taken)
    }

    fn array<const LENGTH: usize>(&mut self) -> Result<[u8; LENGTH], ClipContainerError> {
        let mut array = [0_u8; LENGTH];
        array.copy_from_slice(self.take(LENGTH)?);
        Ok(array)
    }

    fn u8(&mut self) -> Result<u8, ClipContainerError> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, ClipContainerError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, ClipContainerError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, ClipContainerError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
}
