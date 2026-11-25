//! Hidden `wildlife-tracking-oracle` subcommand: runs the ported S3/S6
//! server-track pieces and the per-category tracking modes
//! (`minimappr_core::wildlife_tracking`,
//! `minimappr_core::bird_entity`) on cases supplied as JSON on stdin and
//! prints one JSON result per case. Used by
//! `tests/test_wildlife_tracking_cross_language_parity.py` against goldens
//! from the Python reference (`tests/wildlife_tracking_parity_cases.py`).

#[path = "wildlife_pipeline_oracle.rs"]
mod wildlife_pipeline_oracle;

use minimappr_core::bird_entity::absorption_geometry::{
    covariance_gated_fix_score, fixes_are_two_birds,
};
use minimappr_core::bird_entity::config::BirdEntityConfig;
use minimappr_core::bird_entity::model::{
    bearing_estimate_toward, horizontal_major_sigma_m, uncertainty_ellipse, BearingFanLimits,
    EntityFixSample,
};
use minimappr_core::wildlife_tracking::nearest_neighbor_association::{
    associate_nearest_neighbor, AssociationCandidate, NearestNeighborAssociationConfig,
};
use minimappr_core::wildlife_tracking::perched_concurrency::{
    concurrently_active_perch_conflicts, ConcurrentPerchGuardConfig, PerchGuardTrack,
};
use minimappr_core::wildlife_tracking::perched_filter::{
    PerchedFilterConfig, PerchedRandomWalkFilter,
};
use minimappr_core::wildlife_tracking::species_association_gate::species_claims_contradict;
use minimappr_core::wildlife_tracking::species_log_odds::{
    LabelScores, SpeciesBeliefConfig, SpeciesLogOddsBelief,
};
use minimappr_core::wildlife_tracking::tracking_mode::TrackingModeConfig;
use minimappr_core::wildlife_tracking::{Covariance3, Position3};
use serde::Deserialize;
use serde_json::{json, Value};

/// The reference's base timestamp for geometry cases.
const T0_NS: i64 = 1_700_000_000_000_000_000;
/// Horizon (s) of the perched filter's reported prediction.
const PREDICTION_HORIZON_SECONDS: f64 = 3.0;
const SPECIES_GATE_CONFIDENT_PROBABILITY: f64 = 0.5;
const SPECIES_GATE_CLAIM_PROBABILITY: f64 = 0.15;
const ORACLE_TRACK_ID: u64 = 1;

pub(crate) fn label_scores(value: &Value) -> LabelScores {
    value
        .as_array()
        .map(|pairs| {
            pairs
                .iter()
                .filter_map(|pair| {
                    Some((pair.get(0)?.as_str()?.to_string(), pair.get(1)?.as_f64()?))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn run_belief_case(case: &Value) -> Value {
    let mut belief = SpeciesLogOddsBelief::new(SpeciesBeliefConfig::default());
    let steps: Vec<Value> = case["updates"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|update| {
            belief.update(&label_scores(update));
            let top = belief.top_label();
            json!({
                "exposed": belief.exposed_probabilities().into_iter().map(|(l, p)| json!([l, p])).collect::<Vec<_>>(),
                "top_label": top.as_ref().map(|(label, _)| label.clone()),
                "top_probability": top.map_or(0.0, |(_, probability)| probability),
                "passes_display": belief.passes_display_threshold(),
            })
        })
        .collect();
    json!({ "steps": steps })
}

#[derive(Deserialize)]
struct PerchedStep {
    measurement_m: Position3,
    covariance_m2: Option<Covariance3>,
    dt_s: f64,
}

fn run_perched_filter_case(case: &Value) -> Result<Value, serde_json::Error> {
    let initial: Position3 = serde_json::from_value(case["initial_position_m"].clone())?;
    let steps: Vec<PerchedStep> = serde_json::from_value(case["steps"].clone())?;
    let mut filter = PerchedRandomWalkFilter::new(PerchedFilterConfig::default());
    filter.initialize_track(ORACLE_TRACK_ID, &initial);
    let mut current = initial;
    let mut outputs = Vec::new();
    for step in steps {
        let before = filter.relocation_count;
        let estimate = filter.update(
            ORACLE_TRACK_ID,
            &current,
            &step.measurement_m,
            step.dt_s,
            step.covariance_m2.as_ref(),
        );
        current = estimate.position_m;
        let predicted = filter
            .predict(ORACLE_TRACK_ID, PREDICTION_HORIZON_SECONDS)
            .expect("track initialized");
        outputs.push(json!({
            "position_m": estimate.position_m,
            "covariance_m2": estimate.covariance_m2,
            "relocated": filter.relocation_count > before,
            "predicted_covariance_m2": predicted.covariance_m2,
        }));
    }
    Ok(json!({ "steps": outputs }))
}

#[derive(Deserialize)]
struct OracleTrack {
    track_id: u64,
    position_m: Position3,
    covariance_m2: Option<Covariance3>,
    #[serde(default)]
    last_seen_ns: i64,
    #[serde(default)]
    top_species: Vec<String>,
}

fn run_association_case(case: &Value) -> Result<Value, serde_json::Error> {
    let position: Position3 = serde_json::from_value(case["position_m"].clone())?;
    let covariance: Option<Covariance3> = serde_json::from_value(case["covariance_m2"].clone())?;
    let tracks: Vec<OracleTrack> = serde_json::from_value(case["tracks"].clone())?;
    let candidates: Vec<AssociationCandidate> = tracks
        .iter()
        .map(|track| AssociationCandidate {
            track_id: track.track_id,
            position_m: track.position_m,
            covariance_m2: track.covariance_m2,
        })
        .collect();
    let matched = associate_nearest_neighbor(
        &position,
        covariance.as_ref(),
        &candidates,
        &NearestNeighborAssociationConfig::default(),
    );
    Ok(json!({ "matched_track_id": matched }))
}

fn run_species_gate_case(case: &Value) -> Value {
    let contradicts: Vec<bool> = case["pairs"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|pair| {
            species_claims_contradict(
                &label_scores(&pair[0]),
                &label_scores(&pair[1]),
                SPECIES_GATE_CONFIDENT_PROBABILITY,
                SPECIES_GATE_CLAIM_PROBABILITY,
            )
        })
        .collect();
    json!({ "contradicts": contradicts })
}

fn run_concurrency_case(case: &Value) -> Result<Value, serde_json::Error> {
    let position: Position3 = serde_json::from_value(case["position_m"].clone())?;
    let covariance: Option<Covariance3> = serde_json::from_value(case["covariance_m2"].clone())?;
    let tracks: Vec<OracleTrack> = serde_json::from_value(case["tracks"].clone())?;
    let guard_tracks: Vec<PerchGuardTrack> = tracks
        .into_iter()
        .map(|track| PerchGuardTrack {
            track_id: track.track_id,
            position_m: track.position_m,
            covariance_m2: track.covariance_m2,
            last_seen_ns: track.last_seen_ns,
            is_perched: true,
            top_species: track.top_species,
        })
        .collect();
    let conflicts = concurrently_active_perch_conflicts(
        case["timestamp_ns"].as_i64().unwrap_or_default(),
        &position,
        covariance.as_ref(),
        case["species"].as_str(),
        &guard_tracks,
        &ConcurrentPerchGuardConfig::default(),
    );
    Ok(json!({ "conflicts": conflicts.into_iter().collect::<Vec<_>>() }))
}

#[derive(Deserialize)]
struct GeometrySample {
    node_m: Position3,
    target_m: Position3,
    covariance_m2: Option<Covariance3>,
    other_m: Position3,
    other_covariance_m2: Option<Covariance3>,
    dt_ns: i64,
    range_observable: bool,
}

fn run_entity_geometry_case(case: &Value) -> Result<Value, serde_json::Error> {
    let samples: Vec<GeometrySample> = serde_json::from_value(case["samples"].clone())?;
    let config = BirdEntityConfig::default();
    let limits = BearingFanLimits {
        default_half_width_deg: config.bearing_default_half_width_deg,
        min_half_width_deg: config.bearing_min_half_width_deg,
        max_half_width_deg: config.bearing_max_half_width_deg,
        sigma_multiplier: config.uncertainty_sigma_multiplier,
    };
    let outputs: Vec<Value> = samples
        .iter()
        .map(|sample| {
            let bearing = bearing_estimate_toward("n", &sample.node_m, &sample.target_m, sample.covariance_m2.as_ref(), sample.range_observable, &limits);
            let ellipse = uncertainty_ellipse(sample.covariance_m2.as_ref(), config.uncertainty_sigma_multiplier);
            let first = EntityFixSample { event_time_ns: T0_NS, position_m: sample.target_m, covariance_m2: sample.covariance_m2 };
            let second = EntityFixSample { event_time_ns: T0_NS + sample.dt_ns, position_m: sample.other_m, covariance_m2: sample.other_covariance_m2 };
            json!({
                "horizontal_sigma_m": horizontal_major_sigma_m(sample.covariance_m2.as_ref()),
                "bearing_deg": bearing.bearing_deg,
                "half_width_deg": bearing.half_width_deg,
                "range_min_m": bearing.range_min_m,
                "range_max_m": bearing.range_max_m,
                "ellipse": ellipse.map(|(major, minor, orientation)| vec![major, minor, orientation]),
                "fix_score": covariance_gated_fix_score(&sample.target_m, sample.covariance_m2.as_ref(), &sample.other_m, sample.other_covariance_m2.as_ref(), &config),
                "two_birds": fixes_are_two_birds(&first, &second, &config),
            })
        })
        .collect();
    Ok(json!({ "samples": outputs }))
}

/// Each configuration: `mapping` as `[category, mode]` pairs (null = shipped
/// defaults) and the categories to resolve; an invalid mapping resolves none.
fn run_tracking_mode_case(case: &Value) -> Result<Value, serde_json::Error> {
    #[derive(Deserialize)]
    struct ModeConfiguration {
        mapping: Option<Vec<(String, String)>>,
        categories: Vec<String>,
    }
    let configurations: Vec<ModeConfiguration> =
        serde_json::from_value(case["configurations"].clone())?;
    let results: Vec<Value> = configurations
        .iter()
        .map(|configuration| {
            let parsed = match &configuration.mapping {
                None => Ok(TrackingModeConfig::default()),
                Some(pairs) => TrackingModeConfig::from_category_modes(
                    pairs
                        .iter()
                        .map(|(category, mode)| (category.as_str(), mode.as_str())),
                ),
            };
            match parsed {
                Ok(modes) => json!({
                    "valid": true,
                    "modes": configuration
                        .categories
                        .iter()
                        .map(|category| modes.mode_for_category(category).name())
                        .collect::<Vec<_>>(),
                }),
                Err(_) => json!({ "valid": false, "modes": Value::Null }),
            }
        })
        .collect();
    Ok(json!({ "results": results }))
}

pub fn run_wildlife_tracking_oracle() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let value: Value = serde_json::from_str(&input)?;
    let mut results = Vec::new();
    for case in value["cases"].as_array().ok_or("cases")? {
        let result = match case["kind"].as_str().ok_or("kind")? {
            "belief" => run_belief_case(case),
            "perched_filter" => run_perched_filter_case(case)?,
            "association" => run_association_case(case)?,
            "species_gate" => run_species_gate_case(case),
            "concurrency" => run_concurrency_case(case)?,
            "entity_geometry" => run_entity_geometry_case(case)?,
            "tracking_mode" => run_tracking_mode_case(case)?,
            "pipeline" => wildlife_pipeline_oracle::run_pipeline_case(case)?,
            other => return Err(format!("unknown case kind {other:?}").into()),
        };
        results.push(result);
    }
    println!("{}", serde_json::to_string(&json!({ "results": results }))?);
    Ok(())
}
