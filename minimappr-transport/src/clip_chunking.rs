//! Splitting an encoded `ClipPayload`/`TapToHearPayload` frame into
//! `ClipPayloadChunk`s, and reassembling them on the receiver.
//!
//! Chunks are slices of the *encoded* payload frame, so the reassembled bytes
//! decode with the ordinary `decode_field_message_frame` and the clip keeps
//! every field (onsets, sensor frames) without a second format.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use minimappr_core::protocol::detection_messages::ClipPayloadChunk;
use minimappr_core::protocol::NodeId;

use crate::transport_config::RelayTransportConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipChunkingError {
    EmptyPayloadFrame,
    ZeroChunkBodySize,
    /// `chunk_count` is a u16 on the wire.
    TooManyChunks { required_chunk_count: usize },
}

/// Cuts `payload_frame` into ordered chunks of at most `chunk_body_bytes`.
pub fn split_payload_frame_into_chunks(
    request_id: u32,
    sending_node_id: NodeId,
    payload_frame: &[u8],
    chunk_body_bytes: usize,
) -> Result<Vec<ClipPayloadChunk>, ClipChunkingError> {
    if payload_frame.is_empty() {
        return Err(ClipChunkingError::EmptyPayloadFrame);
    }
    if chunk_body_bytes == 0 {
        return Err(ClipChunkingError::ZeroChunkBodySize);
    }
    let required_chunk_count = payload_frame.len().div_ceil(chunk_body_bytes);
    let chunk_count = u16::try_from(required_chunk_count)
        .map_err(|_| ClipChunkingError::TooManyChunks { required_chunk_count })?;
    Ok(payload_frame
        .chunks(chunk_body_bytes)
        .enumerate()
        .map(|(chunk_index, chunk_bytes)| ClipPayloadChunk {
            request_id,
            node_id: sending_node_id,
            chunk_index: chunk_index as u16,
            chunk_count,
            chunk_bytes: chunk_bytes.to_vec(),
        })
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipReassemblyRejection {
    /// The chunk claims another node than the authenticated sender.
    ChunkNodeIdDiffersFromSender,
    ChunkIndexOutOfRange,
    ChunkCountChangedMidClip,
    ReassembledClipTooLarge,
    ZeroChunkCount,
    /// The splitter never emits an empty chunk; accepting them would let a
    /// peer park up to 65535 map entries per clip at zero byte cost.
    EmptyChunk,
}

/// What one received chunk did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipReassemblyProgress {
    Incomplete,
    /// Duplicate chunk, or a chunk of a clip already delivered.
    IgnoredDuplicate,
    /// All chunks present: the encoded payload frame.
    Complete(Vec<u8>),
}

type ClipKey = (NodeId, u32);

struct PartialClip {
    chunk_count: u16,
    /// Keyed by chunk index and filled as chunks arrive: a peer-declared
    /// `chunk_count` (up to 65535) must not size an allocation up front.
    received_chunks: BTreeMap<u16, Vec<u8>>,
    received_bytes: usize,
    first_chunk_received_at_milliseconds: u64,
}

/// Receiver-side reassembly keyed by (authenticated sender, request id).
pub struct ClipReassembler {
    partial_clips: HashMap<ClipKey, PartialClip>,
    completed_clip_keys: HashSet<ClipKey>,
    completed_clip_order: VecDeque<ClipKey>,
    abandon_after_milliseconds: u64,
    maximum_partial_clips: usize,
    maximum_reassembled_bytes: usize,
    remembered_completed_clip_count: usize,
    abandoned_partial_clip_count: u64,
}

impl ClipReassembler {
    pub fn new(config: &RelayTransportConfig) -> Self {
        Self {
            partial_clips: HashMap::new(),
            completed_clip_keys: HashSet::new(),
            completed_clip_order: VecDeque::new(),
            abandon_after_milliseconds: config.reassembly_abandon_after_milliseconds,
            maximum_partial_clips: config.maximum_partial_clips_in_reassembly,
            maximum_reassembled_bytes: config.maximum_reassembled_clip_bytes,
            remembered_completed_clip_count: config.remembered_completed_clip_count,
            abandoned_partial_clip_count: 0,
        }
    }

    /// Partial clips dropped for age, eviction, size, or inconsistency.
    pub fn abandoned_partial_clip_count(&self) -> u64 {
        self.abandoned_partial_clip_count
    }

    pub fn partial_clip_count(&self) -> usize {
        self.partial_clips.len()
    }

    /// Drops partial clips older than the reassembly deadline.
    pub fn abandon_expired(&mut self, now_milliseconds: u64) {
        let deadline = self.abandon_after_milliseconds;
        let before = self.partial_clips.len();
        self.partial_clips.retain(|_, partial| {
            now_milliseconds.saturating_sub(partial.first_chunk_received_at_milliseconds) < deadline
        });
        self.abandoned_partial_clip_count += (before - self.partial_clips.len()) as u64;
    }

    pub fn accept_chunk(
        &mut self,
        authenticated_sender: NodeId,
        chunk: ClipPayloadChunk,
        now_milliseconds: u64,
    ) -> Result<ClipReassemblyProgress, ClipReassemblyRejection> {
        self.abandon_expired(now_milliseconds);
        if chunk.node_id != authenticated_sender {
            return Err(ClipReassemblyRejection::ChunkNodeIdDiffersFromSender);
        }
        if chunk.chunk_count == 0 {
            return Err(ClipReassemblyRejection::ZeroChunkCount);
        }
        if chunk.chunk_index >= chunk.chunk_count {
            return Err(ClipReassemblyRejection::ChunkIndexOutOfRange);
        }
        if chunk.chunk_bytes.is_empty() {
            return Err(ClipReassemblyRejection::EmptyChunk);
        }
        let key = (authenticated_sender, chunk.request_id);
        if self.completed_clip_keys.contains(&key) {
            return Ok(ClipReassemblyProgress::IgnoredDuplicate);
        }
        if !self.partial_clips.contains_key(&key) {
            self.evict_oldest_if_full();
            self.partial_clips.insert(
                key,
                PartialClip {
                    chunk_count: chunk.chunk_count,
                    received_chunks: BTreeMap::new(),
                    received_bytes: 0,
                    first_chunk_received_at_milliseconds: now_milliseconds,
                },
            );
        }
        let partial = self.partial_clips.get_mut(&key).expect("inserted above");
        if partial.chunk_count != chunk.chunk_count {
            self.drop_partial(&key);
            return Err(ClipReassemblyRejection::ChunkCountChangedMidClip);
        }
        if partial.received_chunks.contains_key(&chunk.chunk_index) {
            return Ok(ClipReassemblyProgress::IgnoredDuplicate);
        }
        partial.received_bytes += chunk.chunk_bytes.len();
        if partial.received_bytes > self.maximum_reassembled_bytes {
            self.drop_partial(&key);
            return Err(ClipReassemblyRejection::ReassembledClipTooLarge);
        }
        partial.received_chunks.insert(chunk.chunk_index, chunk.chunk_bytes);
        if partial.received_chunks.len() < partial.chunk_count as usize {
            return Ok(ClipReassemblyProgress::Incomplete);
        }
        let complete = self.partial_clips.remove(&key).expect("present");
        self.remember_completed(key);
        let mut payload_frame = Vec::with_capacity(complete.received_bytes);
        for chunk_bytes in complete.received_chunks.into_values() {
            payload_frame.extend_from_slice(&chunk_bytes);
        }
        Ok(ClipReassemblyProgress::Complete(payload_frame))
    }

    fn drop_partial(&mut self, key: &ClipKey) {
        if self.partial_clips.remove(key).is_some() {
            self.abandoned_partial_clip_count += 1;
        }
    }

    fn evict_oldest_if_full(&mut self) {
        if self.partial_clips.len() < self.maximum_partial_clips {
            return;
        }
        let oldest_key = self
            .partial_clips
            .iter()
            .min_by_key(|(_, partial)| partial.first_chunk_received_at_milliseconds)
            .map(|(key, _)| *key);
        if let Some(oldest_key) = oldest_key {
            self.drop_partial(&oldest_key);
        }
    }

    fn remember_completed(&mut self, key: ClipKey) {
        self.completed_clip_keys.insert(key);
        self.completed_clip_order.push_back(key);
        while self.completed_clip_order.len() > self.remembered_completed_clip_count {
            if let Some(forgotten) = self.completed_clip_order.pop_front() {
                self.completed_clip_keys.remove(&forgotten);
            }
        }
    }
}
