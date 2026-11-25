//! Per-frame clock evidence survives the decode's frame merging (R5).

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::json;

use super::decode_audio_payload_segments;
use super::tests::{push_binary_node, push_binary_section};
use crate::localization_sensor_entry::NodeTimeQuality;

const SAMPLE_RATE_HZ: u32 = 16_000;
const SAMPLES_PER_FRAME: u64 = 512;
const FRAME_DURATION_NS: u64 = 32_000_000;
const FIRST_FRAME_TIME_NS: u64 = 1_764_000_000_000_000_000;
/// `push_binary_node` writes this boot counter.
const FIXTURE_BOOT_SESSION: u32 = 7;
const IN_UPLOAD_STEP_NS: u64 = 21_000_000;
const GPS_LOCKED_WIRE_CODE: u8 = 0;
const GPS_HOLDOVER_WIRE_CODE: u8 = 1;
const HOLDOVER_SECTION_BIT: u16 = 0x0010;

struct TestFrame {
    start_time_ns: u64,
    start_sample_index: u64,
    time_quality_wire_code: u8,
    holdover: Option<(bool, u32)>,
}

fn holdover_section(active: bool, predicted_error_ns: u32) -> Vec<u8> {
    let mut section = vec![u8::from(active)];
    section.extend_from_slice(&34_000_u32.to_le_bytes());
    section.extend_from_slice(&predicted_error_ns.to_le_bytes());
    for value in [0.1_f32, 0.01, 0.0, 0.0] {
        section.extend_from_slice(&value.to_le_bytes());
    }
    section
}

fn push_frame(payload: &mut Vec<u8>, version: u8, frame: &TestFrame, sequence: u64) {
    payload.extend_from_slice(&frame.start_time_ns.to_le_bytes());
    payload.extend_from_slice(&(frame.start_time_ns + FRAME_DURATION_NS).to_le_bytes());
    payload.extend_from_slice(&frame.start_sample_index.to_le_bytes());
    payload.extend_from_slice(&(frame.start_sample_index + SAMPLES_PER_FRAME).to_le_bytes());
    payload.extend_from_slice(&SAMPLE_RATE_HZ.to_le_bytes());
    payload.push(1);
    if version == 3 {
        payload.push(3);
    }
    payload.extend_from_slice(&sequence.to_le_bytes());
    payload.extend_from_slice(&frame.start_time_ns.to_le_bytes());
    payload.extend_from_slice(&(frame.start_time_ns + FRAME_DURATION_NS).to_le_bytes());
    payload.push(frame.time_quality_wire_code);
    let samples_per_frame = u32::try_from(SAMPLES_PER_FRAME).unwrap();
    if version == 3 {
        payload.extend_from_slice(&samples_per_frame.to_le_bytes());
        match frame.holdover {
            Some((active, predicted_error_ns)) => {
                payload.extend_from_slice(&HOLDOVER_SECTION_BIT.to_le_bytes());
                push_binary_section(payload, &holdover_section(active, predicted_error_ns));
            }
            None => payload.extend_from_slice(&0_u16.to_le_bytes()),
        }
    } else {
        payload.push(0);
        payload.push(0);
        payload.extend_from_slice(&samples_per_frame.to_le_bytes());
    }
    for sample_index in 0..SAMPLES_PER_FRAME {
        let sample = i16::try_from(sample_index).unwrap();
        payload.extend_from_slice(&sample.to_le_bytes());
    }
}

fn binary_upload(version: u8, frames: &[TestFrame]) -> Vec<u8> {
    let mut payload = if version == 3 {
        b"MMB3".to_vec()
    } else {
        b"MMB2".to_vec()
    };
    payload.push(version);
    payload.push(0);
    payload.extend_from_slice(&u16::try_from(frames.len()).unwrap().to_le_bytes());
    push_binary_node(&mut payload);
    for (frame_index, frame) in frames.iter().enumerate() {
        push_frame(&mut payload, version, frame, frame_index as u64 + 1);
    }
    payload
}

/// Sample-contiguous frames; frames from `step_frame` on carry `step_ns` more.
fn contiguous_frames_with_step(
    frame_count: usize,
    step_frame: usize,
    step_ns: u64,
) -> Vec<TestFrame> {
    (0..frame_count)
        .map(|frame_index| {
            let nominal_ns = FIRST_FRAME_TIME_NS + frame_index as u64 * FRAME_DURATION_NS;
            TestFrame {
                start_time_ns: nominal_ns
                    + if frame_index >= step_frame {
                        step_ns
                    } else {
                        0
                    },
                start_sample_index: frame_index as u64 * SAMPLES_PER_FRAME,
                time_quality_wire_code: GPS_LOCKED_WIRE_CODE,
                holdover: None,
            }
        })
        .collect()
}

#[test]
fn merged_binary_segment_keeps_every_frames_timestamp_including_an_in_upload_step() {
    for version in [2, 3] {
        let frames = contiguous_frames_with_step(6, 3, IN_UPLOAD_STEP_NS);
        let segments = decode_audio_payload_segments(&binary_upload(version, &frames)).unwrap();
        assert_eq!(segments.len(), 1, "sample-contiguous frames still merge");
        let segment = &segments[0];
        assert_eq!(segment.start_time_ns, Some(i128::from(FIRST_FRAME_TIME_NS)));
        assert_eq!(segment.channels[0].len(), 6 * SAMPLES_PER_FRAME as usize);
        assert_eq!(segment.frame_timing.len(), 6);
        for (frame_index, (timing, frame)) in segment.frame_timing.iter().zip(&frames).enumerate() {
            assert_eq!(
                timing.sample_offset,
                frame_index * SAMPLES_PER_FRAME as usize
            );
            assert_eq!(
                timing.packet_timestamp_ns,
                Some(i128::from(frame.start_time_ns))
            );
            assert_eq!(timing.time_quality, NodeTimeQuality::GpsLocked);
            assert_eq!(timing.boot_session, Some(FIXTURE_BOOT_SESSION));
            assert_eq!(timing.reported_holdover_error_ns, None);
        }
        let step_ns = segment.frame_timing[3].packet_timestamp_ns.unwrap()
            - segment.frame_timing[2].packet_timestamp_ns.unwrap();
        assert_eq!(step_ns, i128::from(FRAME_DURATION_NS + IN_UPLOAD_STEP_NS));
    }
}

#[test]
fn split_binary_segments_restart_their_frame_offsets() {
    let mut frames = contiguous_frames_with_step(3, 3, 0);
    frames[2].start_sample_index += 4 * SAMPLES_PER_FRAME;
    let segments = decode_audio_payload_segments(&binary_upload(2, &frames)).unwrap();
    assert_eq!(segments.len(), 2);
    let offsets: Vec<Vec<usize>> = segments
        .iter()
        .map(|segment| {
            segment
                .frame_timing
                .iter()
                .map(|timing| timing.sample_offset)
                .collect()
        })
        .collect();
    assert_eq!(offsets, vec![vec![0, SAMPLES_PER_FRAME as usize], vec![0]]);
}

#[test]
fn mmb3_holdover_section_reports_predicted_error_only_while_holdover_is_active() {
    let mut frames = contiguous_frames_with_step(2, 2, 0);
    frames[0].time_quality_wire_code = GPS_HOLDOVER_WIRE_CODE;
    frames[0].holdover = Some((true, 85_000));
    frames[1].holdover = Some((false, 85_000));
    let segments = decode_audio_payload_segments(&binary_upload(3, &frames)).unwrap();
    let timing = &segments[0].frame_timing;
    assert_eq!(timing[0].time_quality, NodeTimeQuality::GpsHoldover);
    assert_eq!(timing[0].reported_holdover_error_ns, Some(85_000));
    assert_eq!(timing[1].time_quality, NodeTimeQuality::GpsLocked);
    assert_eq!(timing[1].reported_holdover_error_ns, None);
}

#[test]
fn store_forward_frames_keep_their_own_time_quality_and_timestamp() {
    let pcm = STANDARD.encode(
        [0_i16, 1_i16]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let payload = json!({
        "buffered_frames": [
            {"frame": {"sample_rate_hz": 16_000, "channels": 1, "start_time_ns": 1_000,
                "start_sample_index": 0, "end_sample_index": 2, "samples_per_channel": 2,
                "samples_b64": pcm, "time_quality": "ntp_sync"}},
            {"frame": {"sample_rate_hz": 16_000, "channels": 1, "utc_start_ns": 22_001_125,
                "start_sample_index": 2, "end_sample_index": 4, "samples_per_channel": 2,
                "samples_b64": pcm, "time_quality": "free_running"}},
            {"frame": {"sample_rate_hz": 16_000, "channels": 1,
                "start_sample_index": 4, "end_sample_index": 6, "samples_per_channel": 2,
                "samples_b64": pcm}}
        ]
    });
    let segments = decode_audio_payload_segments(&serde_json::to_vec(&payload).unwrap()).unwrap();
    assert_eq!(segments.len(), 1);
    let timing = &segments[0].frame_timing;
    assert_eq!(
        timing
            .iter()
            .map(|frame| frame.sample_offset)
            .collect::<Vec<_>>(),
        [0, 2, 4]
    );
    assert_eq!(timing[0].time_quality, NodeTimeQuality::NtpDisciplined);
    assert_eq!(timing[1].time_quality, NodeTimeQuality::FreeRunning);
    assert_eq!(timing[2].time_quality, NodeTimeQuality::Unknown);
    assert_eq!(timing[1].packet_timestamp_ns, Some(22_001_125));
    assert_eq!(timing[2].packet_timestamp_ns, None);
    assert!(timing.iter().all(|frame| frame.boot_session.is_none()));
}
