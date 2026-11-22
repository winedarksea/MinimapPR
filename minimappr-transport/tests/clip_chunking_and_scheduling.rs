//! Chunking/reassembly, control-over-clip priority, late-clip abandonment
//! and queue caps — all sans-IO.

mod common;

use common::{clip_payload, readiness_query};
use minimappr_core::protocol::{decode_field_message_frame, DecodedFrame, NodeId};
use minimappr_relay_wire::RelayFrameKind;
use minimappr_transport::clip_chunking::{
    split_payload_frame_into_chunks, ClipChunkingError, ClipReassembler, ClipReassemblyProgress,
    ClipReassemblyRejection,
};
use minimappr_transport::outbound_scheduler::{
    OutboundQueueError, OutboundScheduler, QueuedControlFrame, ScheduledOutbound,
};
use minimappr_transport::RelayTransportConfig;

const SENDER: NodeId = NodeId(4);

fn encoded_clip(request_id: u32, container_bytes: usize) -> Vec<u8> {
    clip_payload(request_id, SENDER, container_bytes).encode_frame().unwrap()
}

fn control_frame(query_id: u32) -> QueuedControlFrame {
    QueuedControlFrame {
        recipient_peer_id: 1,
        relay_frame_kind: RelayFrameKind::SealedEnvelope,
        plaintext_field_frame: readiness_query(query_id).encode_frame().unwrap(),
    }
}

#[test]
fn chunks_reassemble_in_any_order_to_the_identical_clip() {
    let frame = encoded_clip(7, 40_000);
    let mut chunks = split_payload_frame_into_chunks(7, SENDER, &frame, 4096).unwrap();
    assert_eq!(chunks.len(), frame.len().div_ceil(4096));
    assert!(chunks.iter().all(|chunk| chunk.chunk_count as usize == chunks.len()));
    chunks.reverse();
    let mut reassembler = ClipReassembler::new(&RelayTransportConfig::default());
    let last = chunks.pop().unwrap();
    for chunk in chunks.clone() {
        assert_eq!(reassembler.accept_chunk(SENDER, chunk, 10), Ok(ClipReassemblyProgress::Incomplete));
    }
    // A duplicate mid-clip is ignored.
    assert_eq!(
        reassembler.accept_chunk(SENDER, chunks[0].clone(), 11),
        Ok(ClipReassemblyProgress::IgnoredDuplicate)
    );
    let Ok(ClipReassemblyProgress::Complete(reassembled)) = reassembler.accept_chunk(SENDER, last, 12) else {
        panic!("clip should be complete");
    };
    assert_eq!(reassembled, frame);
    let DecodedFrame::Known { message, .. } = decode_field_message_frame(&reassembled).unwrap() else {
        panic!("known kind");
    };
    assert_eq!(message, clip_payload(7, SENDER, 40_000));
    // Chunks resent after a reconnect never deliver the clip twice.
    assert_eq!(
        reassembler.accept_chunk(SENDER, chunks[1].clone(), 13),
        Ok(ClipReassemblyProgress::IgnoredDuplicate)
    );
}

#[test]
fn chunking_rejects_empty_and_oversized_payloads() {
    assert_eq!(
        split_payload_frame_into_chunks(1, SENDER, &[], 16),
        Err(ClipChunkingError::EmptyPayloadFrame)
    );
    let too_many = vec![0_u8; (u16::MAX as usize) + 1];
    assert!(matches!(
        split_payload_frame_into_chunks(1, SENDER, &too_many, 1),
        Err(ClipChunkingError::TooManyChunks { .. })
    ));
}

#[test]
fn reassembly_rejects_forged_and_inconsistent_chunks_and_abandons_stale_ones() {
    let config = RelayTransportConfig {
        reassembly_abandon_after_milliseconds: 1_000,
        maximum_reassembled_clip_bytes: 10_000,
        maximum_partial_clips_in_reassembly: 2,
        ..RelayTransportConfig::default()
    };
    let mut reassembler = ClipReassembler::new(&config);
    let chunks = split_payload_frame_into_chunks(1, SENDER, &encoded_clip(1, 8_000), 1_000).unwrap();
    assert_eq!(
        reassembler.accept_chunk(NodeId(9), chunks[0].clone(), 0),
        Err(ClipReassemblyRejection::ChunkNodeIdDiffersFromSender)
    );
    reassembler.accept_chunk(SENDER, chunks[0].clone(), 0).unwrap();
    let mut wrong_count = chunks[1].clone();
    wrong_count.chunk_count += 1;
    assert_eq!(
        reassembler.accept_chunk(SENDER, wrong_count, 1),
        Err(ClipReassemblyRejection::ChunkCountChangedMidClip)
    );
    // Too large in total.
    let big = split_payload_frame_into_chunks(2, SENDER, &encoded_clip(2, 12_000), 4_000).unwrap();
    let mut outcome = Ok(ClipReassemblyProgress::Incomplete);
    for chunk in big {
        outcome = reassembler.accept_chunk(SENDER, chunk, 2);
        if outcome.is_err() {
            break;
        }
    }
    assert_eq!(outcome, Err(ClipReassemblyRejection::ReassembledClipTooLarge));
    // Stale partials are dropped at the deadline.
    reassembler.accept_chunk(SENDER, chunks[0].clone(), 100).unwrap();
    assert_eq!(reassembler.partial_clip_count(), 1);
    reassembler.abandon_expired(1_100);
    assert_eq!(reassembler.partial_clip_count(), 0);
    // A flood of first chunks evicts the oldest instead of growing.
    for request_id in 10..15 {
        let chunks = split_payload_frame_into_chunks(request_id, SENDER, &[1, 2, 3, 4], 2).unwrap();
        reassembler.accept_chunk(SENDER, chunks[0].clone(), 2_000 + request_id as u64).unwrap();
    }
    assert_eq!(reassembler.partial_clip_count(), 2);
    assert!(reassembler.abandoned_partial_clip_count() >= 6);
}

#[test]
fn control_frames_overtake_a_clip_already_in_progress() {
    let config = RelayTransportConfig::default();
    let mut scheduler = OutboundScheduler::new(&config);
    let chunks = split_payload_frame_into_chunks(5, SENDER, &encoded_clip(5, 20_000), 4096).unwrap();
    let chunk_count = chunks.len();
    scheduler.enqueue_clip(1, chunks, 10_000).unwrap();
    assert!(matches!(scheduler.next_outbound(0), Some(ScheduledOutbound::ClipChunk { .. })));
    scheduler.enqueue_control(control_frame(1)).unwrap();
    scheduler.enqueue_control(control_frame(2)).unwrap();
    assert!(matches!(scheduler.next_outbound(1), Some(ScheduledOutbound::Control(_))));
    assert!(matches!(scheduler.next_outbound(2), Some(ScheduledOutbound::Control(_))));
    let mut remaining_chunks = 0;
    while let Some(ScheduledOutbound::ClipChunk { chunk, .. }) = scheduler.next_outbound(3) {
        remaining_chunks += 1;
        assert_eq!(chunk.request_id, 5);
    }
    assert_eq!(remaining_chunks, chunk_count - 1);
    let statistics = scheduler.statistics();
    assert_eq!((statistics.control_frames_sent, statistics.clips_fully_sent), (2, 1));
}

#[test]
fn earliest_deadline_clip_goes_first_and_late_clips_are_abandoned() {
    let config = RelayTransportConfig::default();
    let mut scheduler = OutboundScheduler::new(&config);
    let relaxed = split_payload_frame_into_chunks(1, SENDER, &encoded_clip(1, 9_000), 4096).unwrap();
    let urgent = split_payload_frame_into_chunks(2, SENDER, &encoded_clip(2, 9_000), 4096).unwrap();
    scheduler.enqueue_clip(1, relaxed, 5_000).unwrap();
    scheduler.enqueue_clip(1, urgent, 1_000).unwrap();
    let Some(ScheduledOutbound::ClipChunk { chunk, .. }) = scheduler.next_outbound(0) else {
        panic!("expected a chunk");
    };
    assert_eq!(chunk.request_id, 2, "earliest deadline first");
    // The urgent clip misses its deadline mid-transfer and is dropped whole.
    let Some(ScheduledOutbound::ClipChunk { chunk, .. }) = scheduler.next_outbound(1_000) else {
        panic!("expected the relaxed clip");
    };
    assert_eq!(chunk.request_id, 1);
    assert_eq!(scheduler.statistics().clips_abandoned_late, 1);
    scheduler.abandon_late_clips(5_000);
    assert_eq!(scheduler.queued_clip_count(), 0);
    assert_eq!(scheduler.next_outbound(5_001), None);
}

#[test]
fn reconnect_rewinds_unexpired_clips_including_fully_written_ones() {
    let config = RelayTransportConfig::default();
    let mut scheduler = OutboundScheduler::new(&config);
    let chunks = split_payload_frame_into_chunks(3, SENDER, &encoded_clip(3, 5_000), 4096).unwrap();
    assert_eq!(chunks.len(), 2);
    scheduler.enqueue_clip(1, chunks, 2_000).unwrap();
    while scheduler.next_outbound(0).is_some() {}
    scheduler.rewind_clips_after_reconnect();
    let mut resent = 0;
    while scheduler.next_outbound(10).is_some() {
        resent += 1;
    }
    assert_eq!(resent, 2);
    // After the deadline nothing comes back.
    scheduler.rewind_clips_after_reconnect();
    assert_eq!(scheduler.next_outbound(2_000), None);
}

#[test]
fn queue_caps_push_back_instead_of_growing() {
    let config = RelayTransportConfig {
        maximum_queued_control_frames: 2,
        maximum_queued_clip_bytes: 10_000,
        ..RelayTransportConfig::default()
    };
    let mut scheduler = OutboundScheduler::new(&config);
    scheduler.enqueue_control(control_frame(1)).unwrap();
    scheduler.enqueue_control(control_frame(2)).unwrap();
    assert_eq!(scheduler.enqueue_control(control_frame(3)), Err(OutboundQueueError::ControlQueueFull));
    let small = split_payload_frame_into_chunks(1, SENDER, &encoded_clip(1, 6_000), 4096).unwrap();
    let again = small.clone();
    scheduler.enqueue_clip(1, small, 1_000).unwrap();
    assert_eq!(scheduler.enqueue_clip(1, again, 1_000), Err(OutboundQueueError::DuplicateClipRequestId));
    let large = split_payload_frame_into_chunks(2, SENDER, &encoded_clip(2, 6_000), 4096).unwrap();
    assert_eq!(
        scheduler.enqueue_clip(1, large, 1_000),
        Err(OutboundQueueError::ClipQueueByteBudgetExceeded)
    );
}
