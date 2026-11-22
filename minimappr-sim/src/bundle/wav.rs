//! Interleaved PCM16 WAV encoding matching what `bundle.py` reads
//! (`int16 / 32767.0`, channel i == `channel_sensor_ids[i]`).

const PCM16_FULL_SCALE: f32 = 32767.0;
const RIFF_HEADER_BYTES: usize = 44;
const BITS_PER_SAMPLE: u16 = 16;
const BYTES_PER_SAMPLE: u16 = BITS_PER_SAMPLE / 8;
const PCM_FORMAT_TAG: u16 = 1;
const FMT_CHUNK_BYTES: u32 = 16;

/// Encode channels-first full-scale f32 audio as a PCM16 WAV. Channels of
/// unequal length (silent dropouts) are zero-padded to the longest. Errors
/// when the RIFF 32-bit size fields would overflow (long multichannel
/// sessions) instead of silently writing a corrupt header.
pub fn encode_pcm16_wav(channels: &[Vec<f32>], sample_rate_hz: u32) -> Result<Vec<u8>, String> {
    let channel_count = u16::try_from(channels.len().max(1))
        .map_err(|_| format!("{} channels exceed the WAV limit", channels.len()))?;
    let frames = channels.iter().map(Vec::len).max().unwrap_or(0);
    let data_bytes = frames * channel_count as usize * BYTES_PER_SAMPLE as usize;
    if data_bytes + RIFF_HEADER_BYTES > u32::MAX as usize {
        return Err(format!(
            "{data_bytes} bytes of PCM exceed the 4 GiB RIFF limit; shorten the scenario"
        ));
    }
    let block_align = channel_count * BYTES_PER_SAMPLE;
    let mut out = Vec::with_capacity(RIFF_HEADER_BYTES + data_bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((RIFF_HEADER_BYTES - 8 + data_bytes) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&FMT_CHUNK_BYTES.to_le_bytes());
    out.extend_from_slice(&PCM_FORMAT_TAG.to_le_bytes());
    out.extend_from_slice(&channel_count.to_le_bytes());
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&(sample_rate_hz * block_align as u32).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for frame in 0..frames {
        for channel in channels {
            let value = channel.get(frame).copied().unwrap_or(0.0);
            let quantized = (value.clamp(-1.0, 1.0) * PCM16_FULL_SCALE).round() as i16;
            out.extend_from_slice(&quantized.to_le_bytes());
        }
    }
    Ok(out)
}
