//! Outbound priority queue: control frames always go before clip chunks, and
//! clip chunks past their deadline are abandoned instead of sent.
//!
//! One relay WebSocket is one ordered stream, so a 40 KB clip on a weak link
//! would otherwise hold every detection event behind it (plan §Online relay,
//! Ordering). The scheduler hands the adapter one frame at a time; the
//! adapter asks again only once the previous frame is written, so a control
//! frame enqueued mid-clip waits for at most one chunk.

use std::collections::VecDeque;

use minimappr_core::protocol::detection_messages::ClipPayloadChunk;
use minimappr_core::protocol::NodeId;
use minimappr_relay_wire::RelayFrameKind;

use crate::transport_config::RelayTransportConfig;

/// A small, latency-sensitive frame (any message except clip chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedControlFrame {
    pub recipient_peer_id: u32,
    pub relay_frame_kind: RelayFrameKind,
    /// Encoded `FieldMessage` frame; sealed when dequeued so sequence
    /// numbers follow the actual send order.
    pub plaintext_field_frame: Vec<u8>,
}

/// One frame the scheduler chose to send next.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduledOutbound {
    Control(QueuedControlFrame),
    ClipChunk {
        recipient_peer_id: u32,
        chunk: ClipPayloadChunk,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutboundQueueError {
    ControlQueueFull,
    ClipQueueByteBudgetExceeded,
    DuplicateClipRequestId,
}

struct QueuedClipTransfer {
    recipient_peer_id: u32,
    sending_node_id: NodeId,
    request_id: u32,
    chunks: Vec<ClipPayloadChunk>,
    next_chunk_index: usize,
    abandon_at_milliseconds: u64,
    total_chunk_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutboundSchedulerStatistics {
    pub control_frames_sent: u64,
    pub clip_chunks_sent: u64,
    pub clips_fully_sent: u64,
    /// Clips dropped because their deadline passed before the last chunk.
    pub clips_abandoned_late: u64,
}

pub struct OutboundScheduler {
    control_frames: VecDeque<QueuedControlFrame>,
    clip_transfers: Vec<QueuedClipTransfer>,
    /// Fully written clips kept until their deadline: if the socket dies
    /// before the tail chunks reach the relay, a reconnect resends them.
    sent_clips_kept_for_resend: Vec<QueuedClipTransfer>,
    queued_clip_bytes: usize,
    maximum_queued_control_frames: usize,
    maximum_queued_clip_bytes: usize,
    statistics: OutboundSchedulerStatistics,
}

impl OutboundScheduler {
    pub fn new(config: &RelayTransportConfig) -> Self {
        Self {
            control_frames: VecDeque::new(),
            clip_transfers: Vec::new(),
            sent_clips_kept_for_resend: Vec::new(),
            queued_clip_bytes: 0,
            maximum_queued_control_frames: config.maximum_queued_control_frames,
            maximum_queued_clip_bytes: config.maximum_queued_clip_bytes,
            statistics: OutboundSchedulerStatistics::default(),
        }
    }

    pub fn statistics(&self) -> OutboundSchedulerStatistics {
        self.statistics
    }

    pub fn queued_control_frame_count(&self) -> usize {
        self.control_frames.len()
    }

    pub fn queued_clip_count(&self) -> usize {
        self.clip_transfers.len()
    }

    pub fn enqueue_control(&mut self, frame: QueuedControlFrame) -> Result<(), OutboundQueueError> {
        if self.control_frames.len() >= self.maximum_queued_control_frames {
            return Err(OutboundQueueError::ControlQueueFull);
        }
        self.control_frames.push_back(frame);
        Ok(())
    }

    /// Queues every chunk of one clip, to be abandoned at `abandon_at_milliseconds`.
    pub fn enqueue_clip(
        &mut self,
        recipient_peer_id: u32,
        chunks: Vec<ClipPayloadChunk>,
        abandon_at_milliseconds: u64,
    ) -> Result<(), OutboundQueueError> {
        let Some(first_chunk) = chunks.first() else {
            return Ok(());
        };
        let (sending_node_id, request_id) = (first_chunk.node_id, first_chunk.request_id);
        let duplicate = self.clip_transfers.iter().chain(&self.sent_clips_kept_for_resend).any(|transfer| {
            transfer.request_id == request_id
                && transfer.sending_node_id == sending_node_id
                && transfer.recipient_peer_id == recipient_peer_id
        });
        if duplicate {
            return Err(OutboundQueueError::DuplicateClipRequestId);
        }
        let total_chunk_bytes: usize = chunks.iter().map(|chunk| chunk.chunk_bytes.len()).sum();
        if self.queued_clip_bytes + total_chunk_bytes > self.maximum_queued_clip_bytes {
            return Err(OutboundQueueError::ClipQueueByteBudgetExceeded);
        }
        self.queued_clip_bytes += total_chunk_bytes;
        self.clip_transfers.push(QueuedClipTransfer {
            recipient_peer_id,
            sending_node_id,
            request_id,
            chunks,
            next_chunk_index: 0,
            abandon_at_milliseconds,
            total_chunk_bytes,
        });
        Ok(())
    }

    /// Drops every clip whose deadline has passed.
    pub fn abandon_late_clips(&mut self, now_milliseconds: u64) {
        let mut released_bytes = 0;
        let mut abandoned = 0;
        self.clip_transfers.retain(|transfer| {
            let late = now_milliseconds >= transfer.abandon_at_milliseconds;
            if late {
                released_bytes += transfer.total_chunk_bytes;
                abandoned += 1;
            }
            !late
        });
        self.sent_clips_kept_for_resend.retain(|transfer| {
            let expired = now_milliseconds >= transfer.abandon_at_milliseconds;
            if expired {
                released_bytes += transfer.total_chunk_bytes;
            }
            !expired
        });
        self.queued_clip_bytes -= released_bytes;
        self.statistics.clips_abandoned_late += abandoned;
    }

    /// After a reconnect, chunks written to the dead socket may be lost, so
    /// every clip still inside its deadline (unfinished or recently sent)
    /// restarts from chunk 0; the receiver ignores duplicates and clips it
    /// already delivered. Deadlines are unchanged, so a late clip still dies.
    pub fn rewind_clips_after_reconnect(&mut self) {
        self.clip_transfers.append(&mut self.sent_clips_kept_for_resend);
        for transfer in &mut self.clip_transfers {
            transfer.next_chunk_index = 0;
        }
    }

    /// The next frame to write: any control frame first, then one chunk of
    /// the clip with the earliest deadline.
    pub fn next_outbound(&mut self, now_milliseconds: u64) -> Option<ScheduledOutbound> {
        if let Some(control_frame) = self.control_frames.pop_front() {
            self.statistics.control_frames_sent += 1;
            return Some(ScheduledOutbound::Control(control_frame));
        }
        self.abandon_late_clips(now_milliseconds);
        let most_urgent_position = self
            .clip_transfers
            .iter()
            .enumerate()
            .min_by_key(|(_, transfer)| transfer.abandon_at_milliseconds)
            .map(|(position, _)| position)?;
        let transfer = &mut self.clip_transfers[most_urgent_position];
        let chunk = transfer.chunks[transfer.next_chunk_index].clone();
        let recipient_peer_id = transfer.recipient_peer_id;
        transfer.next_chunk_index += 1;
        self.statistics.clip_chunks_sent += 1;
        if transfer.next_chunk_index == transfer.chunks.len() {
            let finished = self.clip_transfers.remove(most_urgent_position);
            self.sent_clips_kept_for_resend.push(finished);
            self.statistics.clips_fully_sent += 1;
        }
        Some(ScheduledOutbound::ClipChunk {
            recipient_peer_id,
            chunk,
        })
    }
}
