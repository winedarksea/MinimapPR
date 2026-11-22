//! The host's append-only session event log and every node's mirror of it
//! (plan §Host handover: `events?since=N`, the pcplayerpicker cursor).
//!
//! Entries hold encoded `FieldMessage` frames. Detections are logged at most
//! once per `(node_id, local_sequence)`, which makes queued uploads after
//! solo fallback idempotent. A mirror applies entries strictly in sequence;
//! a gap is reported so the node asks for `since = head`.

use std::collections::BTreeSet;

use crate::protocol::mesh_messages::{EventLogAppend, EventLogEntry};
use crate::protocol::{
    decode_field_message_frame, detection_messages::DetectionEvent, DecodedFrame, FieldMessage,
};

use super::DetectionKey;

fn detection_key_of(frame: &[u8]) -> Option<DetectionKey> {
    match decode_field_message_frame(frame) {
        Ok(DecodedFrame::Known {
            message: FieldMessage::DetectionEvent(event),
            ..
        }) => Some((event.node_id, event.local_sequence)),
        _ => None,
    }
}

/// Decode one entry's record; `None` for kinds this build does not know.
pub fn decode_entry(entry: &EventLogEntry) -> Option<FieldMessage> {
    match decode_field_message_frame(&entry.field_message_frame) {
        Ok(DecodedFrame::Known { message, .. }) => Some(message),
        _ => None,
    }
}

/// Host side: the source of truth.
#[derive(Debug, Clone, Default)]
pub struct SessionEventLog {
    entries: Vec<EventLogEntry>,
    logged_detection_keys: BTreeSet<DetectionKey>,
    /// Entries up to this sequence were already broadcast.
    broadcast_sequence: u64,
}

impl SessionEventLog {
    /// Successor host: continue from a mirror (sequence numbers carry on).
    pub fn from_mirror(entries: &[EventLogEntry]) -> Self {
        let logged_detection_keys = entries
            .iter()
            .filter_map(|entry| detection_key_of(&entry.field_message_frame))
            .collect();
        let head = entries.last().map_or(0, |entry| entry.sequence);
        Self {
            entries: entries.to_vec(),
            logged_detection_keys,
            broadcast_sequence: head,
        }
    }

    pub fn head_sequence(&self) -> u64 {
        self.entries.last().map_or(0, |entry| entry.sequence)
    }

    pub fn entries(&self) -> &[EventLogEntry] {
        &self.entries
    }

    /// Append any record. Encoding failure is impossible for well-formed
    /// bodies; it is reported rather than panicking.
    pub fn append(&mut self, host_term: u32, record: &FieldMessage) -> Option<u64> {
        let field_message_frame = record.encode_frame().ok()?;
        let sequence = self.head_sequence() + 1;
        self.entries.push(EventLogEntry {
            sequence,
            host_term,
            field_message_frame,
        });
        Some(sequence)
    }

    pub fn contains_detection(&self, key: DetectionKey) -> bool {
        self.logged_detection_keys.contains(&key)
    }

    /// Idempotent on `(node_id, local_sequence)`; true when newly logged.
    pub fn append_detection_if_new(&mut self, host_term: u32, detection: &DetectionEvent) -> bool {
        let key = (detection.node_id, detection.local_sequence);
        if self.logged_detection_keys.contains(&key) {
            return false;
        }
        let record = FieldMessage::DetectionEvent(detection.clone());
        if self.append(host_term, &record).is_none() {
            return false;
        }
        self.logged_detection_keys.insert(key);
        true
    }

    /// Entries with `sequence > since_sequence`, at most `maximum_entries`.
    pub fn entries_since(&self, since_sequence: u64, maximum_entries: usize) -> &[EventLogEntry] {
        let start = self
            .entries
            .partition_point(|entry| entry.sequence <= since_sequence);
        let end = start.saturating_add(maximum_entries).min(self.entries.len());
        &self.entries[start..end]
    }

    /// Entries appended since the last call (for the live broadcast).
    pub fn take_unbroadcast(&mut self) -> Vec<EventLogEntry> {
        let fresh = self.entries_since(self.broadcast_sequence, usize::MAX).to_vec();
        self.broadcast_sequence = self.head_sequence();
        fresh
    }

    pub fn logged_detection_count(&self) -> usize {
        self.logged_detection_keys.len()
    }
}

/// What applying an append did to a mirror.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MirrorApplyOutcome {
    /// Records newly applied, in order (decoded; unknown kinds skipped).
    pub applied_records: Vec<FieldMessage>,
    /// Entries removed because a newer term rewrote their sequence numbers.
    pub discarded_entries: Vec<EventLogEntry>,
    /// Set when the mirror is behind: request catch-up `since` this.
    pub catch_up_since: Option<u64>,
}

/// Node side: an exact prefix of the host log (or of a past host's log,
/// until a newer term rewrites the tail).
#[derive(Debug, Clone, Default)]
pub struct MirroredEventLog {
    entries: Vec<EventLogEntry>,
}

impl MirroredEventLog {
    pub fn head_sequence(&self) -> u64 {
        self.entries.last().map_or(0, |entry| entry.sequence)
    }

    pub fn entries(&self) -> &[EventLogEntry] {
        &self.entries
    }

    /// Drop every entry above `head` (a new host's log ends there).
    pub fn truncate_to(&mut self, head: u64) -> Vec<EventLogEntry> {
        let keep = self.entries.partition_point(|entry| entry.sequence <= head);
        self.entries.split_off(keep)
    }

    /// Drop every entry written in `host_term` or later. Two hosts that
    /// claimed one term at once write entries with equal `(sequence, term)`,
    /// so once the tie is decided the contested term cannot be told apart
    /// and is re-fetched from the winner.
    pub fn truncate_from_term(&mut self, host_term: u32) -> Vec<EventLogEntry> {
        let keep = self
            .entries
            .iter()
            .position(|entry| entry.host_term >= host_term)
            .unwrap_or(self.entries.len());
        self.entries.split_off(keep)
    }

    pub fn apply(&mut self, append: &EventLogAppend) -> MirrorApplyOutcome {
        let mut outcome = MirrorApplyOutcome::default();
        for entry in &append.entries {
            let head = self.head_sequence();
            // Sequences start at 1; a 0 is malformed and would underflow below.
            if entry.sequence == 0 {
                continue;
            }
            if entry.sequence <= head {
                let index = (entry.sequence - 1) as usize;
                let conflicting = self
                    .entries
                    .get(index)
                    .is_some_and(|stored| stored.host_term < entry.host_term);
                if !conflicting {
                    continue;
                }
                outcome
                    .discarded_entries
                    .extend(self.truncate_to(entry.sequence - 1));
            } else if entry.sequence > head + 1 {
                outcome.catch_up_since = Some(head);
                return outcome;
            }
            self.entries.push(entry.clone());
            if let Some(record) = decode_entry(entry) {
                outcome.applied_records.push(record);
            }
        }
        if append.host_head_sequence > self.head_sequence() {
            outcome.catch_up_since = Some(self.head_sequence());
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::mesh_messages::EventLogAppend;
    use crate::protocol::NodeId;

    fn detection(node: u32, sequence: u64) -> DetectionEvent {
        DetectionEvent {
            node_id: NodeId(node),
            local_sequence: sequence,
            window_first_device_sample_index: 0,
            window_sample_count: 1,
            classifier_model_id: 0,
            species_scores: vec![],
            stage1_contrast_db: 0.0,
            stage1_active_band: None,
        }
    }

    fn append_of(log: &SessionEventLog, since: u64) -> EventLogAppend {
        EventLogAppend {
            host_node_id: NodeId(0),
            host_term: 1,
            host_head_sequence: log.head_sequence(),
            entries: log.entries_since(since, usize::MAX).to_vec(),
        }
    }

    #[test]
    fn detections_are_logged_once_per_key() {
        let mut log = SessionEventLog::default();
        assert!(log.append_detection_if_new(1, &detection(2, 7)));
        assert!(!log.append_detection_if_new(1, &detection(2, 7)));
        assert!(log.append_detection_if_new(1, &detection(3, 7)));
        assert_eq!(log.head_sequence(), 2);
        let restored = SessionEventLog::from_mirror(log.entries());
        assert!(restored.contains_detection((NodeId(2), 7)));
    }

    #[test]
    fn mirror_reports_gap_and_catches_up_with_cursor() {
        let mut log = SessionEventLog::default();
        for sequence in 0..5 {
            log.append_detection_if_new(1, &detection(1, sequence));
        }
        let mut mirror = MirroredEventLog::default();
        let mut partial = append_of(&log, 0);
        partial.entries.remove(1);
        let outcome = mirror.apply(&partial);
        assert_eq!(outcome.catch_up_since, Some(1));
        let outcome = mirror.apply(&append_of(&log, 1));
        assert_eq!(outcome.catch_up_since, None);
        assert_eq!(mirror.entries(), log.entries());
        // A duplicate delivery changes nothing.
        assert!(mirror.apply(&append_of(&log, 0)).applied_records.is_empty());
    }

    #[test]
    fn newer_term_rewrites_conflicting_tail() {
        let mut log = SessionEventLog::default();
        log.append_detection_if_new(1, &detection(1, 1));
        log.append_detection_if_new(1, &detection(1, 2));
        let mut mirror = MirroredEventLog::default();
        mirror.apply(&append_of(&log, 0));
        let mut successor = SessionEventLog::from_mirror(&log.entries()[..1]);
        successor.append_detection_if_new(2, &detection(4, 9));
        let mut append = append_of(&successor, 1);
        append.host_term = 2;
        let outcome = mirror.apply(&append);
        assert_eq!(outcome.discarded_entries.len(), 1);
        assert_eq!(mirror.entries(), successor.entries());
    }

    #[test]
    fn malformed_zero_sequence_is_ignored() {
        let mut log = SessionEventLog::default();
        log.append_detection_if_new(1, &detection(1, 1));
        let mut mirror = MirroredEventLog::default();
        mirror.apply(&append_of(&log, 0));
        let mut bogus = append_of(&log, 0);
        bogus.entries[0].sequence = 0;
        bogus.entries[0].host_term = 9;
        let outcome = mirror.apply(&bogus);
        assert!(outcome.applied_records.is_empty() && outcome.discarded_entries.is_empty());
        assert_eq!(mirror.entries(), log.entries());
    }

    #[test]
    fn contested_term_is_dropped_from_the_mirror() {
        let mut log = SessionEventLog::default();
        log.append_detection_if_new(1, &detection(1, 1));
        log.append_detection_if_new(2, &detection(1, 2));
        log.append_detection_if_new(2, &detection(1, 3));
        let mut mirror = MirroredEventLog::default();
        mirror.apply(&append_of(&log, 0));
        let discarded = mirror.truncate_from_term(2);
        assert_eq!(discarded.len(), 2);
        assert_eq!(mirror.head_sequence(), 1);
    }
}
