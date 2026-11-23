//! Hidden `attribution-oracle` subcommand: the S5 port (noise-floor-relative
//! trigger, attribution cascade and LOCO decision) on JSON cases from stdin,
//! for `tests/test_attribution_trigger_cross_language_parity.py`. The
//! host-owned masking and classifier rerun are scripted from the case.

use std::collections::BTreeMap;

use minimappr_core::onset_clustering::attribution_cascade::{
    attribute_cluster, AttributionCascadeConfig, AttributionTier, ClusterAttributionContext,
    LocoRerunBudget, LocoRerunHost,
};
use minimappr_core::onset_clustering::attribution_decision::{
    loco_decision, AttributionThresholds, ClassificationWindow, ClassifierAnswer, InterfererOnset,
};
use minimappr_core::onset_clustering::attribution_evidence::{
    ClusterAttributionEvidence, ClusterOnsetArrival,
};
use minimappr_core::mesh::call_attribution::reconcile_window_labels;
use minimappr_core::protocol::wire_codes::SpeciesScore;
use minimappr_core::stage1::noise_floor_relative_trigger::{
    NoiseFloorRelativeTrigger, RelativeTriggerConfig,
};
use minimappr_core::wildlife_tracking::species_log_odds::round_decimal;
use serde_json::{json, Value};

/// Stop reasons the reference writes into its annotation's `reason` key (the
/// others are implied by the tier and the gradient / LOCO reasons).
const REFERENCE_ANNOTATED_REASONS: [&str; 3] = ["loco_disabled", "interferers_silent_in_window", "loco_rerun_cap"];

/// The reference annotates dominance rounded to this many decimals.
const DOMINANCE_DECIMALS: usize = 2;

type OracleResult = Result<Value, Box<dyn std::error::Error + Send + Sync>>;

fn run_relative_trigger_case(case: &Value, with_probes: bool) -> OracleResult {
    let mut trigger = NoiseFloorRelativeTrigger::new(RelativeTriggerConfig::default());
    let mut steps = Vec::new();
    for frame in case["frames"].as_array().ok_or("frames")? {
        let node = frame[0].as_str().ok_or("node")?;
        let rms = frame[1].as_f64().unwrap_or(f64::NAN);
        let triggered = trigger.observe(node, rms, frame[2].as_f64().ok_or("seconds")?);
        steps.push(json!([triggered, trigger.floor_rms(node)]));
    }
    if !with_probes {
        return Ok(json!({ "steps": steps }));
    }
    let probe_floors: Vec<Option<f64>> = case["probes"]
        .as_array()
        .ok_or("probes")?
        .iter()
        .map(|probe| probe.as_str().and_then(|node| trigger.floor_rms(node)))
        .collect();
    Ok(json!({ "tracked": trigger.tracked_node_count(), "probe_floors": probe_floors }))
}

fn classifier_answer(value: &Value) -> ClassifierAnswer {
    let scores = value["scores"]
        .as_array()
        .map(|pairs| {
            pairs
                .iter()
                .filter_map(|pair| Some((pair[0].as_str()?.to_string(), pair[1].as_f64()?)))
                .collect()
        })
        .unwrap_or_default();
    ClassifierAnswer {
        label: value["label"].as_str().unwrap_or_default().to_string(),
        confidence: value["confidence"].as_f64().unwrap_or_default(),
        scores,
    }
}

fn float_map(value: &Value) -> BTreeMap<String, f64> {
    value
        .as_object()
        .map(|map| {
            map.iter()
                .filter_map(|(key, v)| Some((key.clone(), v.as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

fn evidence_from(raw: &Value) -> ClusterAttributionEvidence {
    let arrivals = raw["node_onset_arrivals"]
        .as_object()
        .map(|nodes| {
            nodes
                .iter()
                .map(|(node, arrivals)| {
                    let parsed = arrivals
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|a| {
                            Some(ClusterOnsetArrival {
                                span_sample: a[0].as_i64()?,
                                band_min_hz: a[1].as_f64()?,
                                band_max_hz: a[2].as_f64()?,
                            })
                        })
                        .collect();
                    (node.clone(), parsed)
                })
                .collect()
        })
        .unwrap_or_default();
    ClusterAttributionEvidence {
        node_band_snr_db: float_map(&raw["node_band_snr_db"]),
        node_distances_m: float_map(&raw["node_distances_m"]),
        node_onset_arrivals: arrivals,
        fitted_source_level_db: raw["fitted_source_level_db"].as_f64(),
        spreading_residual_db: raw["spreading_residual_db"].as_f64(),
    }
}

/// Masking and rerun scripted by the case (the real host owns the signal work).
struct ScriptedHost {
    masked_cell_fraction: f64,
    masked_answer: ClassifierAnswer,
    /// What the cascade asked to mask (compared with the reference's seeds).
    masked_seeds: Option<Vec<InterfererOnset>>,
}

impl LocoRerunHost for ScriptedHost {
    fn mask_interferers(&mut self, seeds: &[InterfererOnset]) -> f64 {
        self.masked_seeds = Some(seeds.to_vec());
        self.masked_cell_fraction
    }
    fn rerun_masked(&mut self) -> ClassifierAnswer {
        self.masked_answer.clone()
    }
}

fn tier_name(tier: AttributionTier) -> &'static str {
    match tier {
        AttributionTier::SingleCluster => "single_cluster",
        AttributionTier::SnrGradient => "snr_gradient",
        AttributionTier::Loco => "loco",
        AttributionTier::Unsettled => "unsettled",
    }
}

fn run_attribution_batch(case: &Value) -> OracleResult {
    let mut budget = LocoRerunBudget::default();
    let mut results = Vec::new();
    for case in case["cases"].as_array().ok_or("cases")? {
        let config = AttributionCascadeConfig {
            loco_enabled: case["loco_enabled"].as_bool().unwrap_or(true),
            ..AttributionCascadeConfig::default()
        };
        let cluster_evidence: BTreeMap<usize, ClusterAttributionEvidence> = case["clusters"]
            .as_object()
            .ok_or("clusters")?
            .iter()
            .filter_map(|(index, raw)| Some((index.parse().ok()?, evidence_from(raw))))
            .collect();
        let context = ClusterAttributionContext {
            window_id: case["window_id"].as_str().unwrap_or_default().to_string(),
            target_index: case["target_index"].as_u64().ok_or("target_index")? as usize,
            cluster_evidence,
            window: ClassificationWindow {
                classifying_node_id: case["classifying_node_id"]
                    .as_str()
                    .ok_or("classifying_node_id")?,
                start_sample: case["window_start_sample"]
                    .as_i64()
                    .ok_or("window_start_sample")?,
                end_sample: case["window_end_sample"]
                    .as_i64()
                    .ok_or("window_end_sample")?,
                sample_rate_hz: case["sample_rate_hz"].as_u64().ok_or("sample_rate_hz")? as u32,
                shared_onset_max_lag_samples: case["shared_onset_max_lag_samples"].as_i64(),
            },
        };
        let mut host = ScriptedHost {
            masked_cell_fraction: case["masked_cell_fraction"].as_f64().unwrap_or_default(),
            masked_answer: classifier_answer(&case["masked"]),
            masked_seeds: None,
        };
        let outcome = attribute_cluster(
            &classifier_answer(&case["original"]),
            &context,
            &config,
            &mut budget,
            &mut host,
        );
        let dominance: BTreeMap<String, f64> = outcome
            .gradient
            .as_ref()
            .map(|gradient| {
                gradient
                    .dominance_db
                    .iter()
                    .map(|(index, value)| {
                        (index.to_string(), round_decimal(*value, DOMINANCE_DECIMALS))
                    })
                    .collect()
            })
            .unwrap_or_default();
        results.push(json!({
            "tier": tier_name(outcome.tier),
            "interferers": outcome.interferer_count,
            "gradient_reason": outcome.gradient.as_ref().map(|gradient| gradient.reason),
            "loco_reason": outcome.loco.as_ref().map(|loco| loco.reason),
            "label": outcome.label.unwrap_or_else(|| "unknown".to_string()),
            "confidence": outcome.confidence,
            "dominance_db": dominance,
            "collapsed": outcome.loco.as_ref().map(|loco| loco.collapsed_labels.clone()),
            "reason": REFERENCE_ANNOTATED_REASONS.contains(&outcome.reason).then_some(outcome.reason),
            "mask_seeds": host.masked_seeds.as_ref().map(|seeds| {
                seeds.iter().map(|seed| json!([seed.onset_seconds, seed.band_min_hz, seed.band_max_hz])).collect::<Vec<_>>()
            }),
        }));
    }
    Ok(json!({ "results": results }))
}

fn tier_from_name(name: &str) -> Result<AttributionTier, String> {
    match name {
        "single_cluster" => Ok(AttributionTier::SingleCluster),
        "snr_gradient" => Ok(AttributionTier::SnrGradient),
        "loco" => Ok(AttributionTier::Loco),
        "unsettled" => Ok(AttributionTier::Unsettled),
        other => Err(format!("unknown tier {other:?}")),
    }
}

/// The mesh host's reconcile on numeric label ids; returns each cluster's kept labels in order.
fn run_window_label_reconciliation_case(case: &Value) -> OracleResult {
    let mut kept = Vec::new();
    for window in case["windows"].as_array().ok_or("windows")? {
        let mut decisions = Vec::new();
        for (index, cluster) in window.as_array().ok_or("window")?.iter().enumerate() {
            let tier = tier_from_name(cluster["tier"].as_str().ok_or("tier")?)?;
            let species = cluster["scores"]
                .as_array()
                .ok_or("scores")?
                .iter()
                .map(|pair| -> Result<SpeciesScore, String> {
                    let label_id = pair[0].as_str().and_then(|label| label.parse().ok()).ok_or("label id")?;
                    Ok(SpeciesScore { label_id, score: pair[1].as_f64().ok_or("score")? as f32 })
                })
                .collect::<Result<Vec<_>, _>>()?;
            decisions.push((index, tier, species));
        }
        reconcile_window_labels(&mut decisions);
        kept.push(
            decisions
                .iter()
                .map(|(_, _, species)| species.iter().map(|entry| entry.label_id.to_string()).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        );
    }
    Ok(json!({ "kept_labels": kept }))
}

fn run_loco_decision_case(case: &Value) -> OracleResult {
    let thresholds = AttributionThresholds::default();
    let decisions: Vec<Value> = case["pairs"]
        .as_array()
        .ok_or("pairs")?
        .iter()
        .map(|pair| {
            let decision = loco_decision(
                &classifier_answer(&pair["original"]),
                &classifier_answer(&pair["masked"]),
                &thresholds,
            );
            json!([
                decision.settled,
                decision.reason,
                decision.label,
                decision.confidence,
                decision.collapsed_labels
            ])
        })
        .collect();
    Ok(json!({ "decisions": decisions }))
}

pub fn run_attribution_oracle() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read;

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let value: Value = serde_json::from_str(&input)?;
    let mut results = Vec::new();
    for case in value["cases"].as_array().ok_or("cases")? {
        results.push(match case["kind"].as_str().ok_or("kind")? {
            "relative_trigger" => run_relative_trigger_case(case, false)?,
            "relative_trigger_lru" => run_relative_trigger_case(case, true)?,
            "attribution_batch" => run_attribution_batch(case)?,
            "loco_decision" => run_loco_decision_case(case)?,
            "window_label_reconciliation" => run_window_label_reconciliation_case(case)?,
            other => return Err(format!("unknown case kind {other:?}").into()),
        });
    }
    println!("{}", serde_json::to_string(&json!({ "results": results }))?);
    Ok(())
}
