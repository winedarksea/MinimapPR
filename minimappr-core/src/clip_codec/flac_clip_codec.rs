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

/// Decodes a FLAC stream produced by `encode_pcm16_to_flac`.
pub fn decode_flac_to_pcm16(flac_bytes: &[u8]) -> Result<DecodedPcm16, FlacClipCodecError> {
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
    let interleaved_samples = flac_reader
        .samples()
        .map(|sample| {
            sample
                .map(|value| value as i16)
                .map_err(|error| FlacClipCodecError::Decode(error.to_string()))
        })
        .collect::<Result<Vec<i16>, _>>()?;
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
