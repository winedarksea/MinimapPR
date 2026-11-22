//! Operational knobs of the relay client (AGENTS.md: no magic numbers).

/// Tuning for one `RelayClientSession`. `Default` gives field values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayTransportConfig {
    /// Bytes of encoded clip frame per `ClipPayloadChunk`. A 40 KB clip is
    /// ~10 chunks: small enough that a detection event waits at most one
    /// chunk (~0.3 s at 100 kbit/s) behind a clip, large enough that the
    /// per-message relay billing (20:1) stays a few requests per clip.
    pub clip_chunk_body_bytes: usize,
    /// Control frames queued while disconnected or backlogged; beyond this
    /// `send_message` fails so the caller sees backpressure, not silent loss.
    pub maximum_queued_control_frames: usize,
    /// Encoded clip bytes queued for sending across all clips.
    pub maximum_queued_clip_bytes: usize,
    /// Sender-side clip lifetime when the request gives none: a clip later
    /// than this is abandoned, not retried (plan §Online relay, Ordering).
    pub default_clip_abandon_after_milliseconds: u64,
    /// Receiver-side: a partially received clip older than this is dropped.
    pub reassembly_abandon_after_milliseconds: u64,
    /// Receiver-side bound on concurrently reassembling clips (oldest is
    /// evicted first), so a flood of first chunks cannot exhaust memory.
    pub maximum_partial_clips_in_reassembly: usize,
    /// Largest reassembled clip frame accepted.
    pub maximum_reassembled_clip_bytes: usize,
    /// Completed (sender, request id) pairs remembered so chunks resent
    /// after a reconnect do not deliver a clip twice.
    pub remembered_completed_clip_count: usize,
    /// Concurrent join-code key exchanges a host keeps state for (one per
    /// relay peer; the oldest is evicted when full, never refused).
    pub maximum_pending_join_code_exchanges: usize,
    pub reconnect_initial_backoff_milliseconds: u64,
    pub reconnect_maximum_backoff_milliseconds: u64,
    pub reconnect_backoff_multiplier: u32,
}

impl Default for RelayTransportConfig {
    fn default() -> Self {
        Self {
            clip_chunk_body_bytes: 4 * 1024,
            maximum_queued_control_frames: 512,
            maximum_queued_clip_bytes: 512 * 1024,
            default_clip_abandon_after_milliseconds: 5_000,
            reassembly_abandon_after_milliseconds: 8_000,
            maximum_partial_clips_in_reassembly: 16,
            maximum_reassembled_clip_bytes: 1024 * 1024,
            remembered_completed_clip_count: 256,
            maximum_pending_join_code_exchanges: 8,
            reconnect_initial_backoff_milliseconds: 250,
            reconnect_maximum_backoff_milliseconds: 15_000,
            reconnect_backoff_multiplier: 2,
        }
    }
}
