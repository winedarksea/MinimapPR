//! `pipeline` cases of the wildlife-tracking oracle: scripted detections and
//! maintenance ticks through `WildlifeEntityPipeline`, reported in the same
//! shape as `tests/wildlife_pipeline_reference.py`.

use minimappr_core::bird_entity::config::BirdEntityConfig;
use minimappr_core::bird_entity::entity_pipeline::{
    EntityPipelineDetection, WildlifeEntityPipeline,
};
use minimappr_core::bird_entity::model::{BirdEntityRecord, SpatialDisplayMode};
use minimappr_core::bird_entity::registry::BirdEntityChangeKind;
use minimappr_core::bird_entity::reporting_decision::EntityReportingAction;
use minimappr_core::wildlife_tracking::tracker_config::PerchedTrackerConfig;
use minimappr_core::wildlife_tracking::wildlife_track::WildlifeTrackStatus;
use minimappr_core::wildlife_tracking::{Covariance3, Position3};
use serde_json::{json, Value};

use super::label_scores;

fn change_kind_name(kind: BirdEntityChangeKind) -> &'static str {
    match kind {
        BirdEntityChangeKind::Created => "created",
        BirdEntityChangeKind::Updated => "updated",
        BirdEntityChangeKind::Absorbed => "absorbed",
        BirdEntityChangeKind::StateChanged => "state_changed",
        BirdEntityChangeKind::Expired => "expired",
    }
}

fn action_name(action: EntityReportingAction) -> &'static str {
    match action {
        EntityReportingAction::Insert => "insert",
        EntityReportingAction::EnrichExisting => "enrich_existing",
        EntityReportingAction::UpgradeExisting => "upgrade_existing",
    }
}

fn status_name(status: WildlifeTrackStatus) -> &'static str {
    match status {
        WildlifeTrackStatus::Tentative => "tentative",
        WildlifeTrackStatus::Confirmed => "confirmed",
        WildlifeTrackStatus::Coasting => "coasting",
        WildlifeTrackStatus::Dropped => "dropped",
    }
}

fn spatial_mode(name: &str) -> Result<SpatialDisplayMode, String> {
    match name {
        "localized" => Ok(SpatialDisplayMode::Localized),
        "bearing_only" => Ok(SpatialDisplayMode::BearingOnly),
        "node_only" => Ok(SpatialDisplayMode::NodeOnly),
        other => Err(format!("unknown spatial mode {other:?}")),
    }
}

fn detection_from(
    step: &Value,
) -> Result<EntityPipelineDetection, Box<dyn std::error::Error + Send + Sync>> {
    let optional_position = |key: &str| -> Result<Option<Position3>, serde_json::Error> {
        serde_json::from_value(step[key].clone())
    };
    let covariance: Option<Covariance3> = serde_json::from_value(step["covariance_m2"].clone())?;
    Ok(EntityPipelineDetection {
        detection_id: step["detection_id"]
            .as_str()
            .ok_or("detection_id")?
            .to_string(),
        event_time_ns: step["event_time_ns"].as_i64().ok_or("event_time_ns")?,
        label: step["label"].as_str().ok_or("label")?.to_string(),
        confidence: step["confidence"].as_f64().ok_or("confidence")?,
        species_evidence: label_scores(&step["species_evidence"]),
        spatial_display_mode: spatial_mode(step["mode"].as_str().ok_or("mode")?)?,
        reporting_node_id: step["node_id"].as_str().map(str::to_string),
        reporting_node_position_m: optional_position("node_position_m")?,
        position_m: optional_position("position_m")?,
        covariance_m2: covariance,
        range_projection_mode: step["range_projection_mode"].as_str().map(str::to_string),
    })
}

fn entity_summary(entity: &BirdEntityRecord) -> Value {
    json!({
        "entity_id": entity.entity_id,
        "state": entity.state.as_str(),
        "top_label": entity.species_top_label,
        "top_probability": entity.species_top_probability,
        "position_m": entity.position_m,
        "horizontal_uncertainty_m": entity.horizontal_uncertainty_m,
        "bearing_deg": entity.bearing.as_ref().map(|bearing| bearing.bearing_deg),
        "half_width_deg": entity.bearing.as_ref().map(|bearing| bearing.half_width_deg),
        "track_id": entity.track_id,
        "detection_count": entity.detection_count,
        "absorbed_detection_count": entity.absorbed_detection_count,
        "recent_fix_count": entity.recent_fixes.len(),
    })
}

pub fn run_pipeline_case(case: &Value) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let mut pipeline =
        WildlifeEntityPipeline::new(PerchedTrackerConfig::default(), BirdEntityConfig::default());
    let mut outputs = Vec::new();
    for step in case["steps"].as_array().ok_or("steps")? {
        if step["type"].as_str() == Some("maintain") {
            let changes = pipeline.maintain(step["now_ns"].as_i64().ok_or("now_ns")?);
            let changes: Vec<Value> = changes
                .iter()
                .map(|change| {
                    json!([
                        change_kind_name(change.kind),
                        change.entity_id,
                        change.state.as_str()
                    ])
                })
                .collect();
            outputs.push(json!({ "changes": changes }));
            continue;
        }
        let outcome = pipeline.process(&detection_from(step)?);
        let revert = outcome
            .track_update
            .and_then(|update| update.relocation_revert);
        outputs.push(json!({
            "action": action_name(outcome.decision.action),
            "decision_entity_id": outcome.decision.entity_id,
            "track_id": outcome.track_update.map(|update| update.track_id),
            "track_created": outcome.track_update.is_some_and(|update| update.created),
            "revert": revert.map(|revert| json!([revert.track_id, revert.split_track_id, revert.hop_time_ns])),
            "released": outcome.revert_fixes_released,
            "entity_id": outcome.entity_id(),
            "change": outcome.change.as_ref().map(|change| change_kind_name(change.kind)),
            "entity_state": outcome.change.as_ref().map(|change| change.state.as_str()),
        }));
    }
    let mut entities: Vec<&BirdEntityRecord> = pipeline.registry.entities();
    entities.sort_by_key(|entity| entity.entity_id);
    let tracks: Vec<Value> = pipeline
        .tracker
        .tracks()
        .map(|track| {
            json!({
                "track_id": track.id,
                "status": status_name(track.status),
                "position_m": track.position_m,
                "label": track.label,
                "confidence": track.confidence,
                "update_count": track.update_count,
            })
        })
        .collect();
    Ok(json!({
        "steps": outputs,
        "entities": entities.into_iter().map(entity_summary).collect::<Vec<_>>(),
        "tracks": tracks,
    }))
}
