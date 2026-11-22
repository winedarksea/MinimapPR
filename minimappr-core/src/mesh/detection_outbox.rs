//! A node's unacknowledged detections. The same queue serves live sending
//! and solo fallback: a detection leaves it only when the node sees it in
//! its event-log mirror (the log echo is the acknowledgement), so a lost
//! message, a host death or a 60 s walk out of range all end in an
//! idempotent `(node_id, local_seq)` re-upload rather than a silent loss.

use std::collections::VecDeque;

use crate::protocol::detection_messages::DetectionEvent;

use super::{DetectionKey, MeshMilliseconds};

#[derive(Debug, Clone)]
struct QueuedDetection {
    detection: DetectionEvent,
    last_sent_ms: Option<MeshMilliseconds>,
}

#[derive(Debug, Clone)]
pub struct DetectionOutbox {
    queued: VecDeque<QueuedDetection>,
    capacity: usize,
    dropped_oldest_count: u64,
}

impl DetectionOutbox {
    pub fn new(capacity: usize) -> Self {
        Self {
            queued: VecDeque::new(),
            capacity: capacity.max(1),
            dropped_oldest_count: 0,
        }
    }

    /// Queue a new detection; beyond capacity the oldest is dropped and counted.
    pub fn push(&mut self, detection: DetectionEvent) {
        if self.queued.len() >= self.capacity {
            self.queued.pop_front();
            self.dropped_oldest_count += 1;
        }
        self.queued.push_back(QueuedDetection {
            detection,
            last_sent_ms: None,
        });
    }

    /// Mark one as sent now (the live path sends it immediately).
    pub fn mark_sent(&mut self, key: DetectionKey, now_ms: MeshMilliseconds) {
        if let Some(item) = self
            .queued
            .iter_mut()
            .find(|item| (item.detection.node_id, item.detection.local_sequence) == key)
        {
            item.last_sent_ms = Some(now_ms);
        }
    }

    /// The mirror echoed it: done.
    pub fn acknowledge(&mut self, key: DetectionKey) {
        self.queued
            .retain(|item| (item.detection.node_id, item.detection.local_sequence) != key);
    }

    /// Detections never sent or not echoed within `resend_after_ms`, at most
    /// `maximum_count`, marked as sent now.
    pub fn take_due(
        &mut self,
        now_ms: MeshMilliseconds,
        resend_after_ms: MeshMilliseconds,
        maximum_count: usize,
    ) -> Vec<DetectionEvent> {
        let mut due = Vec::new();
        for item in self.queued.iter_mut() {
            if due.len() >= maximum_count {
                break;
            }
            let is_due = item
                .last_sent_ms
                .is_none_or(|sent| now_ms.saturating_sub(sent) >= resend_after_ms);
            if is_due {
                item.last_sent_ms = Some(now_ms);
                due.push(item.detection.clone());
            }
        }
        due
    }

    pub fn len(&self) -> usize {
        self.queued.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queued.is_empty()
    }

    pub fn dropped_oldest_count(&self) -> u64 {
        self.dropped_oldest_count
    }
}
