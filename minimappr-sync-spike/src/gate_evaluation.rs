//! Phase 1 gate items evaluated from run reports. A run is evaluated only
//! on the items its data can speak to; `merge_gate_items` combines runs:
//! any FAIL fails the item; passes covering fewer phone models than the
//! gate asks for (three) are PARTIAL.

use std::collections::BTreeSet;

use crate::analysis_config::GateThresholds;
use crate::run_analysis::RunReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum GateStatus {
    Pass,
    Partial,
    Fail,
    Info,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GateItem {
    pub id: String,
    pub matrix_run: String,
    pub run_label: String,
    pub models: Vec<String>,
    pub status: GateStatus,
    pub evidence: String,
}

/// Tape distances within this of the gate distance count as "at" it.
const GATE_DISTANCE_TOLERANCE_M: f64 = 1.0;
const MICROSECONDS_PER_SECOND: f64 = 1e6;

fn item(report: &RunReport, id: &str, models: Vec<String>, status: GateStatus, evidence: String) -> GateItem {
    GateItem { id: id.into(), matrix_run: report.matrix_run.clone(), run_label: report.run_label.clone(), models, status, evidence }
}

fn pass_or_fail(pass: bool) -> GateStatus {
    if pass { GateStatus::Pass } else { GateStatus::Fail }
}

pub fn evaluate_run(report: &RunReport, gates: &GateThresholds) -> Vec<GateItem> {
    let mut items = Vec::new();
    let both_models: Vec<String> = report.phones.iter().map(|p| p.platform_name.clone()).collect();
    for phone in &report.phones {
        let model = vec![phone.platform_name.clone()];
        let onset = &phone.onset;
        if onset.segments.len() > 1 {
            let shift = onset.max_segment_shift_us.unwrap_or(f64::INFINITY);
            let sigma = onset.overall_sigma_us.unwrap_or(f64::INFINITY);
            let limit_us = gates.self_onset_stability_s * MICROSECONDS_PER_SECOND;
            items.push(item(report, "self_onset_stable_10us", model.clone(), pass_or_fail(shift <= limit_us && sigma <= limit_us),
                format!("max shift {shift:.1} µs, σ {sigma:.1} µs over {} segments; clipped {}", onset.segments.len(), onset.clipped_count)));
        }
        if phone.volume_zero_seen {
            items.push(item(report, "muted_phone_inaudible", model.clone(), pass_or_fail(onset.detected == 0),
                format!("volume 0 seen; self chirps detected {} of {}", onset.detected, onset.detected + onset.missed)));
        }
        if phone.screen_off_events > 0 {
            let allowed_missing = gates.maximum_missing_periods_while_locked.floor() as usize;
            items.push(item(report, "locked_screen_playback", model.clone(), pass_or_fail(onset.missed <= allowed_missing),
                format!("{} screen-off events; self chirps missed {} of {}", phone.screen_off_events, onset.missed, onset.detected + onset.missed)));
        }
        if let Some(leakage) = phone.remote_leakage.as_ref().and_then(|l| l.median_over_floor_db) {
            items.push(item(report, "bird_band_leakage", model.clone(), pass_or_fail(leakage <= gates.maximum_leakage_over_floor_db),
                format!("median {leakage:.1} dB over floor in {:?} Hz (listener)", phone.remote_leakage.as_ref().map(|l| l.band_hz))));
        }
        items.push(item(report, "capture_path", model, GateStatus::Info,
            format!("source {}, self σ {:?} µs, median self SNR {:?} dB",
                phone.capture_source, onset.overall_sigma_us.map(|s| (s * 10.0).round() / 10.0),
                onset.segments.first().and_then(|s| s.median_self_snr_db).map(|s| s.round()))));
    }
    let best = report.distance.iter().find(|d| d.mode == "double_sided").or(report.distance.first());
    if let (Some(tape), Some(stats)) = (report.tape_distance_m, best) {
        let at_gate = (tape - gates.range_gate_distance_m).abs() <= GATE_DISTANCE_TOLERANCE_M;
        let error = stats.error_vs_tape_m.unwrap_or(f64::INFINITY);
        let within = error.abs() <= gates.range_error_at_20_m_m && stats.robust_sigma_m <= gates.range_error_at_20_m_m;
        let status = if at_gate { pass_or_fail(within) } else { GateStatus::Info };
        items.push(item(report, "range_10cm_at_20m", both_models.clone(), status,
            format!("tape {tape:.3} m: median {:.3} m ({}), error {error:+.3} m, robust σ {:.3} m, {} rounds, {} outliers",
                stats.median_m, stats.mode, stats.robust_sigma_m, stats.count, stats.outlier_count)));
        if let Some(implied) = report.implied_mean_self_path_m {
            items.push(item(report, "self_path_sum_calibration", both_models.clone(), GateStatus::Info,
                format!("implied (d_AA + d_BB)/2 = {implied:.4} m")));
        }
    }
    if let Some(clock) = &report.clock {
        if let Some(heartbeat) = &clock.filter_heartbeat_cadence {
            let long_enough = clock.duration_s >= gates.clock_residual_minimum_duration_s;
            let within = heartbeat.p95_abs_s <= gates.clock_residual_s;
            let status = if long_enough { pass_or_fail(within) } else { GateStatus::Info };
            items.push(item(report, "clock_residual_50us_30min", both_models.clone(), status,
                format!("{:.0} s run, {:.0} s cadence: p95 {:.1} µs, rms {:.1} µs ({} predictions); linear-fit rms {:?} µs, rate {:?} ppm",
                    clock.duration_s, clock.heartbeat_cadence_s, heartbeat.p95_abs_s * 1e6, heartbeat.rms_s * 1e6, heartbeat.count,
                    clock.linear_fit.as_ref().map(|f| (f.rms_s * 1e7).round() / 10.0), clock.rate_offset_ppm.map(|r| (r * 100.0).round() / 100.0))));
        }
    }
    for observation in &report.channel_map {
        items.push(item(report, "speaker_channel_map", vec![observation.emitter_platform.clone()], GateStatus::Info,
            format!("{} channel {:?} in {:?}: heard by {} at median {:?} dB ({}/{} chirps)",
                observation.emitter_platform, observation.emitter_channel, observation.emitter_orientation,
                observation.listener_platform, observation.median_remote_snr_db.map(|s| s.round()),
                observation.remote_detected, observation.remote_expected)));
    }
    items
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MergedGateItem {
    pub id: String,
    pub status: GateStatus,
    pub models: Vec<String>,
    pub runs: Vec<String>,
}

pub fn merge_gate_items(items: &[GateItem], required_model_count: usize) -> Vec<MergedGateItem> {
    let ids: BTreeSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
    ids.into_iter()
        .map(|id| {
            let members: Vec<&GateItem> = items.iter().filter(|i| i.id == id).collect();
            let models: BTreeSet<String> = members.iter().flat_map(|i| i.models.iter().cloned()).collect();
            let statuses: BTreeSet<GateStatus> = members.iter().map(|i| i.status).collect();
            let status = if statuses.contains(&GateStatus::Fail) {
                GateStatus::Fail
            } else if statuses.iter().all(|s| *s == GateStatus::Info) {
                GateStatus::Info
            } else if models.len() >= required_model_count {
                GateStatus::Pass
            } else {
                GateStatus::Partial
            };
            MergedGateItem {
                id: id.into(),
                status,
                models: models.into_iter().collect(),
                runs: members.iter().map(|i| format!("{} {}", i.matrix_run, i.run_label)).collect(),
            }
        })
        .collect()
}
