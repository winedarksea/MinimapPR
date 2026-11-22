//! One-line, human-readable summaries of received `FieldMessage`s for the
//! on-screen relay log (and the browser checks that read it).

use minimappr_core::protocol::FieldMessage;

pub fn describe_field_message(message: &FieldMessage) -> String {
    match message {
        FieldMessage::Heartbeat(heartbeat) => format!(
            "Heartbeat node {} seq {} sample {}",
            heartbeat.node_id.0, heartbeat.local_sequence, heartbeat.device_sample_index_now
        ),
        FieldMessage::DetectionEvent(detection) => {
            let best = detection
                .species_scores
                .iter()
                .max_by(|left, right| left.score.total_cmp(&right.score));
            match best {
                Some(score) => format!(
                    "DetectionEvent node {} seq {} label_id {} score {:.2}",
                    detection.node_id.0, detection.local_sequence, score.label_id, score.score
                ),
                None => format!("DetectionEvent node {} seq {} (no scores)", detection.node_id.0, detection.local_sequence),
            }
        }
        other => format!("message kind {}", other.message_kind_id()),
    }
}
