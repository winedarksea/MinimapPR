//! Lossless FLAC for PCM16 clips: `flacenc` encodes, `claxon` decodes.
//!
//! Both crates are pure Rust with no C toolchain or threads (flacenc's `par`
//! feature is off), so this builds for wasm32, Android and iOS unchanged.

use flacenc::component::BitRepr;
use flacenc::error::Verify;

const PCM16_BITS_PER_SAMPLE: usize = 16;
/// FLAC forbids blocks under 16 samples except the final one, and decoders
/// (claxon) reject a STREAMINFO minimum below 16, so neither the final block
/// nor a whole stream may be shorter than this.
pub const FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL: usize = 16;
/// flacenc's config verifier rejects block sizes below this.
const FLACENC_MINIMUM_CONFIGURED_BLOCK_SIZE: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlacClipCodecError {
    EmptyClip,
    /// Fewer than `FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL` samples per channel.
    ClipShorterThanMinimumFlacBlock,
    ZeroChannels,
    SampleCountNotMultipleOfChannels,
    EncoderConfiguration(String),
    Encode(String),
    Decode(String),
    UnsupportedBitsPerSample(u32),
    /// The stream would decode past the caller's sample bound.
    DecodedSampleLimitExceeded,
}

impl std::fmt::Display for FlacClipCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "FLAC clip codec error: {self:?}")
    }
}

impl std::error::Error for FlacClipCodecError {}

/// PCM16 samples decoded from a FLAC stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPcm16 {
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    /// Interleaved when `channel_count > 1`.
    pub interleaved_samples: Vec<i16>,
}

/// Encodes interleaved PCM16 into a complete FLAC stream (with STREAMINFO).
pub fn encode_pcm16_to_flac(
    interleaved_samples: &[i16],
    channel_count: u8,
    sample_rate_hz: u32,
) -> Result<Vec<u8>, FlacClipCodecError> {
    if channel_count == 0 {
        return Err(FlacClipCodecError::ZeroChannels);
    }
    if interleaved_samples.is_empty() {
        return Err(FlacClipCodecError::EmptyClip);
    }
    if !interleaved_samples
        .len()
        .is_multiple_of(usize::from(channel_count))
    {
        return Err(FlacClipCodecError::SampleCountNotMultipleOfChannels);
    }
    let samples_per_channel = interleaved_samples.len() / usize::from(channel_count);
    if samples_per_channel < FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL {
        return Err(FlacClipCodecError::ClipShorterThanMinimumFlacBlock);
    }
    let mut encoder_configuration = flacenc::config::Encoder::default();
    encoder_configuration.block_size =
        block_size_without_short_final_block(samples_per_channel, encoder_configuration.block_size);
    let encoder_configuration = encoder_configuration
        .into_verified()
        .map_err(|(_, error)| FlacClipCodecError::EncoderConfiguration(format!("{error:?}")))?;
    let widened_samples: Vec<i32> = interleaved_samples
        .iter()
        .map(|&sample| i32::from(sample))
        .collect();
    let source = flacenc::source::MemSource::from_samples(
        &widened_samples,
        usize::from(channel_count),
        PCM16_BITS_PER_SAMPLE,
        sample_rate_hz as usize,
    );
    let flac_stream = flacenc::encode_with_fixed_block_size(
        &encoder_configuration,
        source,
        encoder_configuration.block_size,
    )
    .map_err(|error| FlacClipCodecError::Encode(format!("{error:?}")))?;
    let mut byte_sink = flacenc::bitsink::ByteSink::new();
    flac_stream
        .write(&mut byte_sink)
        .map_err(|error| FlacClipCodecError::Encode(format!("{error:?}")))?;
    Ok(byte_sink.as_slice().to_vec())
}

/// Largest block size <= `preferred` whose final block is either empty or at
/// least the FLAC minimum, so arbitrary clip lengths stay decodable.
/// A stream no longer than `preferred` is one block (>= 16 checked by caller).
fn block_size_without_short_final_block(samples_per_channel: usize, preferred: usize) -> usize {
    if samples_per_channel <= preferred {
        return preferred;
    }
    (FLACENC_MINIMUM_CONFIGURED_BLOCK_SIZE..=preferred)
        .rev()
        .find(|&block_size| {
            let final_block = samples_per_channel % block_size;
            final_block == 0 || final_block >= FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL
        })
        // Never reached for clip lengths (unit test sweeps 4097..=2_000_000).
        .unwrap_or(preferred)
}

/// FLAC stream marker and metadata-block header layout (RFC 9639 §8).
const FLAC_STREAM_MARKER: &[u8; 4] = b"fLaC";
const FLAC_METADATA_BLOCK_HEADER_LENGTH_BYTES: usize = 4;
const FLAC_METADATA_LAST_BLOCK_FLAG: u8 = 0x80;
const FLAC_METADATA_BLOCK_TYPE_MASK: u8 = 0x7F;
const FLAC_METADATA_BLOCK_TYPE_STREAMINFO: u8 = 0;
const FLAC_METADATA_BLOCK_TYPE_PADDING: u8 = 1;

/// Rejects metadata other than STREAMINFO/PADDING and any block whose
/// declared length overruns the input. Clips arrive from untrusted peers and
/// claxon pre-allocates from VORBIS_COMMENT/APPLICATION length fields (up to
/// ~80 MB from a few dozen bytes); our encoder never writes those blocks.
fn validate_flac_metadata_blocks(flac_bytes: &[u8]) -> Result<(), FlacClipCodecError> {
    let malformed = |reason: &str| Err(FlacClipCodecError::Decode(reason.to_string()));
    let Some(mut remaining) = flac_bytes.strip_prefix(FLAC_STREAM_MARKER.as_slice()) else {
        return malformed("missing fLaC stream marker");
    };
    loop {
        if remaining.len() < FLAC_METADATA_BLOCK_HEADER_LENGTH_BYTES {
            return malformed("truncated metadata block header");
        }
        let flags_and_type = remaining[0];
        let block_length =
            u32::from_be_bytes([0, remaining[1], remaining[2], remaining[3]]) as usize;
        remaining = &remaining[FLAC_METADATA_BLOCK_HEADER_LENGTH_BYTES..];
        let block_type = flags_and_type & FLAC_METADATA_BLOCK_TYPE_MASK;
        if block_type != FLAC_METADATA_BLOCK_TYPE_STREAMINFO
            && block_type != FLAC_METADATA_BLOCK_TYPE_PADDING
        {
            return malformed("unexpected FLAC metadata block type");
        }
        if block_length > remaining.len() {
            return malformed("metadata block overruns stream");
        }
        remaining = &remaining[block_length..];
        if flags_and_type & FLAC_METADATA_LAST_BLOCK_FLAG != 0 {
            return Ok(());
        }
    }
}

/// Decodes a FLAC stream produced by `encode_pcm16_to_flac`, refusing to
/// produce more than `maximum_interleaved_samples`. The bound is mandatory
/// because FLAC CONSTANT subframes expand ~10^5x: a few KB from an untrusted
/// peer would otherwise decode to gigabytes.
pub fn decode_flac_to_pcm16(
    flac_bytes: &[u8],
    maximum_interleaved_samples: usize,
) -> Result<DecodedPcm16, FlacClipCodecError> {
    validate_flac_metadata_blocks(flac_bytes)?;
    let mut flac_reader = claxon::FlacReader::new(std::io::Cursor::new(flac_bytes))
        .map_err(|error| FlacClipCodecError::Decode(error.to_string()))?;
    let stream_info = flac_reader.streaminfo();
    if stream_info.bits_per_sample != PCM16_BITS_PER_SAMPLE as u32 {
        return Err(FlacClipCodecError::UnsupportedBitsPerSample(
            stream_info.bits_per_sample,
        ));
    }
    let channel_count = u8::try_from(stream_info.channels)
        .map_err(|_| FlacClipCodecError::Decode("channel count exceeds u8".to_string()))?;
    let declared_interleaved_samples = stream_info
        .samples
        .map(|samples_per_channel| samples_per_channel.saturating_mul(u64::from(channel_count)));
    if declared_interleaved_samples
        .is_some_and(|declared| declared > maximum_interleaved_samples as u64)
    {
        return Err(FlacClipCodecError::DecodedSampleLimitExceeded);
    }
    let mut interleaved_samples = Vec::new();
    for sample in flac_reader.samples() {
        if interleaved_samples.len() >= maximum_interleaved_samples {
            return Err(FlacClipCodecError::DecodedSampleLimitExceeded);
        }
        let value = sample.map_err(|error| FlacClipCodecError::Decode(error.to_string()))?;
        interleaved_samples.push(value as i16);
    }
    Ok(DecodedPcm16 {
        sample_rate_hz: stream_info.sample_rate,
        channel_count,
        interleaved_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_clip_length_gets_a_block_size_with_a_legal_final_block() {
        const DEFAULT_BLOCK_SIZE: usize = 4096;
        const LONGEST_CHECKED_SAMPLES: usize = 2_000_000;
        for samples_per_channel in DEFAULT_BLOCK_SIZE + 1..=LONGEST_CHECKED_SAMPLES {
            let block_size =
                block_size_without_short_final_block(samples_per_channel, DEFAULT_BLOCK_SIZE);
            let final_block = samples_per_channel % block_size;
            assert!(
                final_block == 0 || final_block >= FLAC_MINIMUM_BLOCK_SAMPLES_PER_CHANNEL,
                "{samples_per_channel} -> block {block_size} leaves {final_block}"
            );
        }
    }
}
