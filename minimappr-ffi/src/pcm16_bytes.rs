//! PCM16 crosses the FFI as little-endian bytes, not `Vec<i16>`: UniFFI
//! lowers `Vec<i16>` to a boxed `List<Short>` in Kotlin (one object per
//! sample, 48k allocations per second), while `Vec<u8>` maps to `ByteArray`.

use crate::error::FieldFfiError;

const PCM16_BYTES_PER_SAMPLE: usize = 2;

pub fn pcm16_samples_from_little_endian_bytes(pcm16_little_endian_bytes: &[u8]) -> Result<Vec<i16>, FieldFfiError> {
    if pcm16_little_endian_bytes.len() % PCM16_BYTES_PER_SAMPLE != 0 {
        return Err(FieldFfiError::InvalidArgument {
            detail: format!("PCM16 byte count {} is odd", pcm16_little_endian_bytes.len()),
        });
    }
    Ok(pcm16_little_endian_bytes
        .chunks_exact(PCM16_BYTES_PER_SAMPLE)
        .map(|sample_bytes| i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]))
        .collect())
}

pub fn pcm16_little_endian_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_extremes() {
        let samples = vec![i16::MIN, -1, 0, 1, i16::MAX];
        let bytes = pcm16_little_endian_bytes(&samples);
        assert_eq!(pcm16_samples_from_little_endian_bytes(&bytes).unwrap(), samples);
    }

    #[test]
    fn odd_byte_count_is_rejected() {
        assert!(pcm16_samples_from_little_endian_bytes(&[0, 1, 2]).is_err());
    }
}
