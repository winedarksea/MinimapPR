//! FLAC bit-exact round trips, MMCL header round trips, and the compression
//! ratio on the committed field-audio fixtures.

mod common;

use minimappr_core::clip_codec::clip_container::{
    ClipContainerError, FieldClipContainer, FieldClipHeader, Pcm16IntegritySection,
};
use minimappr_core::clip_codec::field_clip::{
    decode_field_clip, encode_field_clip, FieldClipError, FIELD_CLIP_MAXIMUM_SAMPLES_PER_CHANNEL,
};
use minimappr_core::clip_codec::flac_clip_codec::{
    decode_flac_to_pcm16, encode_pcm16_to_flac, FlacClipCodecError,
};
use minimappr_core::clip_codec::{ClipCodecId, DecimationFilterId, FieldClipDecimator};

const CLIP_SAMPLE_RATE_HZ: u32 = 24_000;

fn assert_flac_round_trip(samples: &[i16], channel_count: u8) -> usize {
    let encoded =
        encode_pcm16_to_flac(samples, channel_count, CLIP_SAMPLE_RATE_HZ).expect("encodes");
    let decoded = decode_flac_to_pcm16(&encoded, samples.len()).expect("decodes");
    assert_eq!(decoded.sample_rate_hz, CLIP_SAMPLE_RATE_HZ);
    assert_eq!(decoded.channel_count, channel_count);
    assert_eq!(
        decoded.interleaved_samples, samples,
        "FLAC round trip must be bit-exact"
    );
    encoded.len()
}

#[test]
fn flac_round_trip_is_bit_exact_for_noise_at_odd_lengths() {
    let mut noise = common::SeededGaussianNoise::new(17);
    // 4097 / 4111 / 8193 leave a <16-sample final block at the default
    // block size: the encoder must pick a block size that avoids it.
    for length in [16_usize, 17, 4095, 4097, 4111, 8193, 36_001] {
        let samples: Vec<i16> = noise
            .samples(length, 3_000.0)
            .iter()
            .map(|value| value.round().clamp(-32_768.0, 32_767.0) as i16)
            .collect();
        assert_flac_round_trip(&samples, 1);
    }
}

#[test]
fn flac_round_trip_silence_and_full_scale() {
    let silence = vec![0_i16; 24_001];
    let silence_bytes = assert_flac_round_trip(&silence, 1);
    assert!(
        silence_bytes < 1_000,
        "silence should compress to almost nothing ({silence_bytes} B)"
    );
    let full_scale_alternating: Vec<i16> = (0..9_999)
        .map(|index| if index % 2 == 0 { i16::MAX } else { i16::MIN })
        .collect();
    assert_flac_round_trip(&full_scale_alternating, 1);
    let extremes_and_steps: Vec<i16> = (0..5_003)
        .map(|index| match index % 5 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            3 => 0,
            _ => 1,
        })
        .collect();
    assert_flac_round_trip(&extremes_and_steps, 1);
}

#[test]
fn flac_round_trip_stereo_interleaved() {
    let samples: Vec<i16> = (0..2 * 3_001)
        .map(|index| ((index * 7919) % 65_536) as i16)
        .collect();
    assert_flac_round_trip(&samples, 2);
}

#[test]
fn flac_rejects_empty_and_ragged_input() {
    assert_eq!(
        encode_pcm16_to_flac(&[], 1, CLIP_SAMPLE_RATE_HZ),
        Err(FlacClipCodecError::EmptyClip)
    );
    assert_eq!(
        encode_pcm16_to_flac(&[1; 15], 1, CLIP_SAMPLE_RATE_HZ),
        Err(FlacClipCodecError::ClipShorterThanMinimumFlacBlock)
    );
    assert_eq!(
        encode_pcm16_to_flac(&[1, 2, 3], 2, CLIP_SAMPLE_RATE_HZ),
        Err(FlacClipCodecError::SampleCountNotMultipleOfChannels)
    );
    assert!(decode_flac_to_pcm16(b"not flac", usize::MAX).is_err());
}

fn sample_header() -> FieldClipHeader {
    FieldClipHeader {
        codec_id: ClipCodecId::Flac.wire_byte(),
        decimation_filter_id: DecimationFilterId::BirdBandWithRumbleCut.wire_byte(),
        channel_count: 1,
        output_sample_rate_hz: 24_000,
        device_clock_sample_rate_hz: 48_000,
        decimation_factor: 2,
        first_sample_index_device_clock: 0x0123_4567_89AB_CDEF,
        output_samples_per_channel: 36_000,
        pcm16_integrity: Some(Pcm16IntegritySection {
            clipped_output_sample_count: 3,
            filter_context_was_zero_padded: true,
        }),
    }
}

#[test]
fn container_header_round_trips_and_skips_unknown_sections() {
    let container = FieldClipContainer {
        header: sample_header(),
        encoded_audio: vec![9, 8, 7, 6, 5],
    };
    let bytes = container.to_bytes().expect("serializes");
    assert_eq!(&bytes[..4], b"MMCL");
    assert_eq!(
        FieldClipContainer::from_bytes(&bytes).expect("parses"),
        container
    );

    // Splice in a future section (bit 0x0004) after the known one.
    let section_flags_offset = 4 + 1 + 3 + 4 + 4 + 1 + 8 + 4;
    let mut future = bytes.clone();
    future[section_flags_offset..section_flags_offset + 2]
        .copy_from_slice(&0x0005_u16.to_le_bytes());
    let known_section_end = section_flags_offset + 2 + 2 + 5;
    let future_section = [3_u8, 0, 0xAA, 0xBB, 0xCC];
    future.splice(known_section_end..known_section_end, future_section);
    assert_eq!(
        FieldClipContainer::from_bytes(&future).expect("forward compatible"),
        container
    );

    let mut wrong_version = bytes.clone();
    wrong_version[4] = 2;
    assert_eq!(
        FieldClipContainer::from_bytes(&wrong_version),
        Err(ClipContainerError::UnsupportedContainerVersion(2))
    );
    assert_eq!(
        FieldClipContainer::from_bytes(&bytes[..bytes.len() - 1]),
        Err(ClipContainerError::Truncated)
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        FieldClipContainer::from_bytes(&trailing),
        Err(ClipContainerError::TrailingBytes)
    );
    assert_eq!(
        FieldClipContainer::from_bytes(b"MMB3\x03"),
        Err(ClipContainerError::InvalidMagic)
    );
}

#[test]
fn field_clip_round_trip_preserves_samples_and_device_clock_anchor() {
    let path = common::field_audio_fixture_directory().join("wtsp_whistle_xc569523.wav");
    let rolling_buffer = common::load_pcm16_mono_wav(&path);
    let buffer_first_index = 5_000_000_u64;
    let decimator = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut);
    let window_first = buffer_first_index + 48_001; // odd index on purpose
    let window = decimator
        .decimate_window_from_rolling_buffer(
            &rolling_buffer,
            buffer_first_index,
            window_first,
            72_001,
        )
        .expect("inside");
    let clip_bytes = encode_field_clip(&window).expect("encodes");
    let decoded = decode_field_clip(&clip_bytes).expect("decodes");
    assert_eq!(decoded.samples_output_rate, window.samples_output_rate);
    assert_eq!(decoded.header.first_sample_index_device_clock, window_first);
    assert_eq!(
        decoded.decimation_filter,
        Some(DecimationFilterId::BirdBandWithRumbleCut)
    );
    assert_eq!(
        decoded.device_clock_index_of_output_sample(10),
        window_first + 20
    );
    assert_eq!(decoded.header.output_samples_per_channel, 36_001);

    let mut corrupted = clip_bytes.clone();
    let last = corrupted.len() - 3;
    corrupted[last] ^= 0xFF;
    // FLAC frame CRCs catch payload corruption (sealing also authenticates it).
    assert!(decode_field_clip(&corrupted).is_err());
}

#[test]
fn tiny_field_clips_fall_back_to_raw_pcm16_and_round_trip() {
    let decimator =
        FieldClipDecimator::for_filter(DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot);
    let rolling_buffer: Vec<i16> = (0..2_000).map(|index| (index * 13) as i16).collect();
    for window_length in [1_usize, 2, 3, 29, 31, 32, 33] {
        let window = decimator
            .decimate_window_from_rolling_buffer(&rolling_buffer, 0, 1_000, window_length)
            .expect("inside");
        let decoded =
            decode_field_clip(&encode_field_clip(&window).expect("encodes")).expect("decodes");
        assert_eq!(
            decoded.samples_output_rate, window.samples_output_rate,
            "length {window_length}"
        );
        let expected_codec = if window.samples_output_rate.len() < 16 {
            ClipCodecId::Pcm16LittleEndian
        } else {
            ClipCodecId::Flac
        };
        assert_eq!(decoded.header.codec_id, expected_codec.wire_byte());
    }
}

#[test]
fn compression_ratio_on_field_audio_fixtures() {
    let decimator = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut);
    let mut total_raw_48k_bytes = 0_usize;
    let mut total_raw_24k_bytes = 0_usize;
    let mut total_flac_bytes = 0_usize;
    for path in common::field_audio_fixture_paths() {
        let rolling_buffer = common::load_pcm16_mono_wav(&path);
        let window = decimator
            .decimate_window_from_rolling_buffer(&rolling_buffer, 0, 0, rolling_buffer.len())
            .expect("whole fixture");
        let clip_bytes = encode_field_clip(&window).expect("encodes");
        let decoded = decode_field_clip(&clip_bytes).expect("decodes");
        assert_eq!(decoded.samples_output_rate, window.samples_output_rate);
        let raw_48k = rolling_buffer.len() * 2;
        let raw_24k = window.samples_output_rate.len() * 2;
        println!(
            "{:<40} FLAC {:>7} B  vs 24k PCM {:.2}x  vs 48k PCM {:.2}x  ({:.1} KB per 1.5 s)",
            path.file_name().unwrap().to_string_lossy(),
            clip_bytes.len(),
            raw_24k as f64 / clip_bytes.len() as f64,
            raw_48k as f64 / clip_bytes.len() as f64,
            clip_bytes.len() as f64 / (rolling_buffer.len() as f64 / 48_000.0) * 1.5 / 1_000.0,
        );
        total_raw_48k_bytes += raw_48k;
        total_raw_24k_bytes += raw_24k;
        total_flac_bytes += clip_bytes.len();
    }
    let ratio_vs_24k = total_raw_24k_bytes as f64 / total_flac_bytes as f64;
    println!(
        "TOTAL FLAC {total_flac_bytes} B: {ratio_vs_24k:.2}x vs 24 kHz PCM16, {:.2}x vs 48 kHz PCM16",
        total_raw_48k_bytes as f64 / total_flac_bytes as f64
    );
    assert!(
        ratio_vs_24k > 1.2,
        "FLAC should beat raw PCM on outdoor audio ({ratio_vs_24k:.2}x)"
    );
}

/// Wraps `encoded_audio` in a container whose header claims `samples`.
fn hostile_clip_bytes(codec: ClipCodecId, samples: u32, encoded_audio: Vec<u8>) -> Vec<u8> {
    FieldClipContainer {
        header: FieldClipHeader {
            codec_id: codec.wire_byte(),
            output_samples_per_channel: samples,
            first_sample_index_device_clock: 0,
            ..sample_header()
        },
        encoded_audio,
    }
    .to_bytes()
    .expect("serializes")
}

#[test]
fn flac_decompression_bomb_is_bounded_by_the_header_sample_count() {
    // CONSTANT subframes compress silence ~10^4x: 2 M samples in a few KB.
    let silent_samples = vec![0_i16; 2_000_000];
    let bomb = encode_pcm16_to_flac(&silent_samples, 1, CLIP_SAMPLE_RATE_HZ).expect("encodes");
    assert!(bomb.len() < 64 * 1024, "bomb is {} B", bomb.len());
    assert_eq!(
        decode_flac_to_pcm16(&bomb, 1_000),
        Err(FlacClipCodecError::DecodedSampleLimitExceeded)
    );
    // A clip that lies about its length is rejected before decoding past it.
    let lying_clip = hostile_clip_bytes(ClipCodecId::Flac, 1_000, bomb.clone());
    assert!(matches!(
        decode_field_clip(&lying_clip),
        Err(FieldClipError::Flac(
            FlacClipCodecError::DecodedSampleLimitExceeded
        ))
    ));
    // A clip that claims more than the rolling buffer never reaches FLAC.
    let oversized_clip = hostile_clip_bytes(
        ClipCodecId::Flac,
        FIELD_CLIP_MAXIMUM_SAMPLES_PER_CHANNEL + 1,
        bomb,
    );
    assert!(matches!(
        decode_field_clip(&oversized_clip),
        Err(FieldClipError::ImplausibleHeader(_))
    ));
}

#[test]
fn flac_metadata_blocks_that_claxon_preallocates_from_are_rejected() {
    let mut flac = encode_pcm16_to_flac(&[7_i16; 64], 1, CLIP_SAMPLE_RATE_HZ).expect("encodes");
    // Our encoder's metadata must still decode.
    assert!(decode_flac_to_pcm16(&flac, 64).is_ok());
    // Clear STREAMINFO's last-block flag and splice in a VORBIS_COMMENT
    // header whose 24-bit length points far past the input.
    const STREAMINFO_HEADER_OFFSET: usize = 4;
    const STREAMINFO_BODY_LENGTH: usize = 34;
    flac[STREAMINFO_HEADER_OFFSET] &= 0x7F;
    let insert_at = STREAMINFO_HEADER_OFFSET + 4 + STREAMINFO_BODY_LENGTH;
    flac.splice(insert_at..insert_at, [0x84_u8, 0x9F, 0xFF, 0xFF]);
    assert!(matches!(
        decode_flac_to_pcm16(&flac, usize::MAX),
        Err(FlacClipCodecError::Decode(_))
    ));
}

#[test]
fn hostile_container_headers_are_rejected_without_panicking() {
    // Odd raw PCM16 byte count that would otherwise floor to the header count.
    let odd_raw = hostile_clip_bytes(ClipCodecId::Pcm16LittleEndian, 2, vec![1, 0, 2, 0, 3]);
    assert!(matches!(
        decode_field_clip(&odd_raw),
        Err(FieldClipError::HeaderAudioMismatch(_))
    ));
    let mut overflowing_anchor = FieldClipContainer::from_bytes(&hostile_clip_bytes(
        ClipCodecId::Pcm16LittleEndian,
        2,
        vec![1, 0, 2, 0],
    ))
    .expect("parses");
    overflowing_anchor.header.first_sample_index_device_clock = u64::MAX;
    assert!(matches!(
        decode_field_clip(&overflowing_anchor.to_bytes().expect("serializes")),
        Err(FieldClipError::ImplausibleHeader(_))
    ));
    // Every truncation of a valid clip is an error, never a panic.
    let window = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut)
        .decimate_window_from_rolling_buffer(&[100_i16; 4_000], 0, 1_000, 2_000)
        .expect("inside");
    let clip_bytes = encode_field_clip(&window).expect("encodes");
    for truncated_length in 0..clip_bytes.len() {
        assert!(decode_field_clip(&clip_bytes[..truncated_length]).is_err());
    }
}
