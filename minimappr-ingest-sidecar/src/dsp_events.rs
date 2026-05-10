use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use tokio::sync::{broadcast, RwLock};

use crate::{dsp_worker::SharedDspState, manifests::DspManifest};

#[derive(Clone, Debug)]
pub struct ReplayableDspEvent {
    pub event_id: u64,
    pub manifest: DspManifest,
}

#[derive(Clone, Debug, Default)]
pub struct ReplayWindow {
    pub events: Vec<ReplayableDspEvent>,
    pub gap_detected: bool,
    pub oldest_available_event_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct DspEventPublisher {
    tx: broadcast::Sender<ReplayableDspEvent>,
    recent_events: Arc<RwLock<VecDeque<ReplayableDspEvent>>>,
    state: SharedDspState,
    next_event_id: Arc<AtomicU64>,
    replay_capacity: usize,
    recent_results_capacity: usize,
}

impl DspEventPublisher {
    pub fn new(
        state: SharedDspState,
        channel_capacity: usize,
        replay_capacity: usize,
        recent_results_capacity: usize,
    ) -> Self {
        let (tx, _rx) = broadcast::channel(channel_capacity);
        Self {
            tx,
            recent_events: Arc::new(RwLock::new(VecDeque::with_capacity(replay_capacity))),
            state,
            next_event_id: Arc::new(AtomicU64::new(0)),
            replay_capacity,
            recent_results_capacity,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ReplayableDspEvent> {
        self.tx.subscribe()
    }

    pub async fn publish(&self, manifest: DspManifest) -> ReplayableDspEvent {
        let event = ReplayableDspEvent {
            event_id: self.next_event_id.fetch_add(1, Ordering::Relaxed) + 1,
            manifest: manifest.clone(),
        };

        {
            let mut recent_events = self.recent_events.write().await;
            recent_events.push_back(event.clone());
            while recent_events.len() > self.replay_capacity {
                recent_events.pop_front();
            }
        }

        {
            let mut state = self.state.write().await;
            state.recent_results.push(manifest);
            while state.recent_results.len() > self.recent_results_capacity {
                state.recent_results.remove(0);
            }
        }

        let _ = self.tx.send(event.clone());
        event
    }

    pub async fn replay_after(&self, last_event_id: Option<u64>, limit: usize) -> ReplayWindow {
        let recent_events = self.recent_events.read().await;
        let oldest_available_event_id = recent_events.front().map(|event| event.event_id);
        let Some(last_event_id) = last_event_id else {
            return ReplayWindow {
                oldest_available_event_id,
                ..ReplayWindow::default()
            };
        };

        let events = recent_events
            .iter()
            .filter(|event| event.event_id > last_event_id)
            .take(limit)
            .cloned()
            .collect::<Vec<_>>();

        ReplayWindow {
            events,
            gap_detected: oldest_available_event_id
                .is_some_and(|oldest| oldest > last_event_id.saturating_add(1)),
            oldest_available_event_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dsp_worker::DspWorkerState;

    fn test_manifest(manifest_id: &str) -> DspManifest {
        DspManifest {
            manifest_id: manifest_id.to_string(),
            manifest_type: "classifier_render".to_string(),
            created_ns: 1,
            source_handles: vec![],
            derived_handle: None,
            localization: None,
            classifier_render: None,
            birdnet: None,
            coverage_stats: None,
            promotion_ready: false,
            env_samples: None,
            node_context: None,
            raw_payload: None,
            raw_render_bytes: None,
            raw_audio_frame: None,
            raw_audio_bytes: None,
        }
    }

    #[tokio::test]
    async fn replay_after_detects_gap_when_cursor_falls_behind_ring() {
        let state = Arc::new(RwLock::new(DspWorkerState::default()));
        let publisher = DspEventPublisher::new(state, 8, 2, 50);

        publisher.publish(test_manifest("m1")).await;
        publisher.publish(test_manifest("m2")).await;
        publisher.publish(test_manifest("m3")).await;
        publisher.publish(test_manifest("m4")).await;

        let replay_window = publisher.replay_after(Some(1), 10).await;

        assert!(replay_window.gap_detected);
        assert_eq!(replay_window.oldest_available_event_id, Some(3));
        assert_eq!(replay_window.events.len(), 2);
        assert_eq!(replay_window.events[0].event_id, 3);
        assert_eq!(replay_window.events[1].event_id, 4);
    }

    #[tokio::test]
    async fn replay_after_returns_only_events_newer_than_cursor() {
        let state = Arc::new(RwLock::new(DspWorkerState::default()));
        let publisher = DspEventPublisher::new(state, 8, 4, 50);

        publisher.publish(test_manifest("m1")).await;
        publisher.publish(test_manifest("m2")).await;
        publisher.publish(test_manifest("m3")).await;

        let replay_window = publisher.replay_after(Some(2), 10).await;

        assert!(!replay_window.gap_detected);
        assert_eq!(replay_window.oldest_available_event_id, Some(1));
        assert_eq!(replay_window.events.len(), 1);
        assert_eq!(replay_window.events[0].event_id, 3);
        assert_eq!(replay_window.events[0].manifest.manifest_id, "m3");
    }
}
