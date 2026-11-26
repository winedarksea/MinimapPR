//! Markdown rendering of a bake-off report: the effective solver and gate
//! settings it ran, the headline held-out table,
//! slices by site and event, the decision with each criterion, and the
//! commonest reasons a solve published nothing.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::bakeoff_candidate_fix::BakeoffCandidate;
use crate::bakeoff_decision::BakeoffDecision;
use crate::bakeoff_localizers::PointBakeoffReport;
use crate::bakeoff_scenes::BakeoffEventKind;
use crate::bakeoff_scoring::{SceneClass, SceneScore};
use crate::bakeoff_settings_profile::effective_settings_markdown;
use crate::bakeoff_summary::{summarize, CandidateSummary};

/// Site spread bands (m): the upper edge of the middle band is the distance
/// sound covers within production's 1 s event-span lag cap.
const SPREAD_BAND_EDGES_M: [f64; 2] = [100.0, 343.0];
const FAILURE_REASONS_SHOWN: usize = 4;

fn optional(value: Option<f64>, digits: usize) -> String {
    value.map_or_else(|| "–".to_string(), |value| format!("{value:.digits$}"))
}

fn summary_row(label: &str, candidate: BakeoffCandidate, summary: &CandidateSummary) -> String {
    format!(
        "| {label} | {} | {} | {:.3} | {} | {} | {} | {} | {} | {} |\n",
        candidate.label(),
        summary.call_count,
        summary.recall,
        summary.fixed_call_count,
        optional(summary.median_error_m, 1),
        optional(summary.p90_error_m, 1),
        optional(summary.coverage, 3),
        optional(summary.attribution_error_rate, 3),
        optional(summary.median_solve_ms, 1),
    )
}

const TABLE_HEADER: &str = "| slice | candidate | calls | recall | calls fixed | median err (m) | p90 err (m) | 95 % coverage | 2-caller attribution err | median solve (ms) |\n|---|---|---|---|---|---|---|---|---|---|\n";

fn spread_band(class: &SceneClass) -> &'static str {
    if class.aperture_m < SPREAD_BAND_EDGES_M[0] {
        "spread < 100 m"
    } else if class.aperture_m < SPREAD_BAND_EDGES_M[1] {
        "spread 100–343 m"
    } else {
        "spread >= 343 m"
    }
}

fn node_band(class: &SceneClass) -> &'static str {
    match class.node_count {
        0..=3 => "3 nodes",
        4..=5 => "4–5 nodes",
        _ => "6–8 nodes",
    }
}

fn slice_keys(class: &SceneClass) -> Vec<String> {
    vec![
        spread_band(class).to_string(),
        format!("event {}", class.event_kind.label()),
        format!("first call {}", class.waveform_label),
        format!("clocks {}", class.clock_regime.label()),
        if class.has_hand_measured_node { "has hand-measured node".to_string() } else { "all surveyed".to_string() },
        node_band(class).to_string(),
    ]
}

fn slice_table(scores: &[SceneScore]) -> String {
    let mut slices: BTreeMap<String, Vec<&SceneScore>> = BTreeMap::new();
    for score in scores {
        for key in slice_keys(&score.class) {
            slices.entry(key).or_default().push(score);
        }
    }
    let mut table = String::from(TABLE_HEADER);
    for (key, members) in &slices {
        for candidate in BakeoffCandidate::ALL {
            table.push_str(&summary_row(&format!("{key} ({} seeds)", members.len()), candidate, &summarize(members, candidate)));
        }
    }
    table
}

fn decision_section(decision: &BakeoffDecision) -> String {
    let mut text = String::from("| criterion | passed | evidence |\n|---|---|---|\n");
    for criterion in &decision.criteria {
        let _ = writeln!(text, "| {} | {} | {} |", criterion.name, if criterion.passed { "yes" } else { "no" }, criterion.detail);
    }
    let _ = writeln!(text, "\nGuard compared against: {}.", decision.strongest_production_window.label());
    let _ = writeln!(text, "\n**Verdict: {}.**", decision.verdict.label());
    text
}

fn failure_reasons(scores: &[SceneScore]) -> String {
    let mut text = String::new();
    for candidate in BakeoffCandidate::ALL {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for score in scores {
            for reason in score.by_candidate.get(&candidate).map(|score| score.failure_reasons.as_slice()).unwrap_or_default() {
                // Onset reasons carry per-window counts; keep the stable prefix.
                let stable = reason.split(" (").next().unwrap_or(reason).to_string();
                *counts.entry(stable).or_default() += 1;
            }
        }
        let mut ordered: Vec<(String, usize)> = counts.into_iter().collect();
        ordered.sort_by(|a, b| b.1.cmp(&a.1));
        let shown: Vec<String> = ordered.iter().take(FAILURE_REASONS_SHOWN).map(|(reason, count)| format!("{reason} ×{count}")).collect();
        let _ = writeln!(text, "- {}: {}", candidate.label(), if shown.is_empty() { "none".to_string() } else { shown.join(", ") });
    }
    text
}

pub fn render_markdown(report: &PointBakeoffReport) -> String {
    let held_out: Vec<&SceneScore> = report.held_out_scores.iter().collect();
    let mut text = String::new();
    let call_count: usize = report.held_out_scores.iter().map(|score| score.call_count).sum();
    let two_caller_scenes = report.held_out_scores.iter().filter(|score| score.class.event_kind == BakeoffEventKind::TwoBirds).count();
    let _ = writeln!(
        text,
        "Held-out: {} seeds ({} scenes scored, {} calls, {} two-caller scenes); tuning: {} seeds; failed seeds: {}; wall time {:.0} s.\n",
        report.seeds.held_out_seeds.len(),
        report.held_out_scores.len(),
        call_count,
        two_caller_scenes,
        report.seeds.tuning_seeds.len(),
        report.failed_seeds.len(),
        report.elapsed_seconds
    );
    let _ = writeln!(text, "{}", effective_settings_markdown(&report.config));
    let _ = writeln!(text, "Onset grid reach tuned on the tuning seeds (multiple of aperture, recall, p90 m):");
    for (multiple, recall, p90) in &report.tuning.variants {
        let _ = writeln!(text, "- {multiple:.1}: recall {recall:.3}, p90 {}", optional(*p90, 1));
    }
    let _ = writeln!(text, "- chosen: {:.1}\n", report.tuning.chosen_grid_range_aperture_multiple);
    text.push_str("### Held-out results\n\n");
    text.push_str(TABLE_HEADER);
    for candidate in BakeoffCandidate::ALL {
        text.push_str(&summary_row("all", candidate, &summarize(&held_out, candidate)));
    }
    let _ = writeln!(text, "\n### Decision (held-out seeds only)\n\n{}", decision_section(&report.decision));
    let _ = writeln!(text, "### Held-out slices\n\n{}", slice_table(&report.held_out_scores));
    let _ = writeln!(text, "### Why solves published nothing (held-out)\n\n{}", failure_reasons(&report.held_out_scores));
    for (seed, message) in &report.failed_seeds {
        let _ = writeln!(text, "- seed {seed} failed: {message}");
    }
    text
}
