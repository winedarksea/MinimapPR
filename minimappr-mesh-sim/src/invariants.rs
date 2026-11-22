//! Assertions shared by every scenario (test strategy §Layer 2 and §What to
//! assert): no duplicate uploads, no silent detection loss, eventual
//! consistency of every mirror, bounded clip traffic, no TDOA across clock
//! domains, no stale clip against a new clock model, fixes inside their
//! reported region. Each returns a message naming what broke.

use std::collections::BTreeSet;

use minimappr_core::mesh::DetectionKey;
use minimappr_core::protocol::FieldMessage;

use crate::scenario_outcome::{ScenarioOutcome, ScoredFix};

pub type InvariantResult = Result<(), String>;

pub fn ensure(condition: bool, message: impl FnOnce() -> String) -> InvariantResult {
    if condition { Ok(()) } else { Err(message()) }
}

fn logged_detection_keys(outcome: &ScenarioOutcome) -> Vec<DetectionKey> {
    outcome
        .host_log_records()
        .into_iter()
        .filter_map(|record| match record {
            FieldMessage::DetectionEvent(event) => Some((event.node_id, event.local_sequence)),
            _ => None,
        })
        .collect()
}

pub fn no_duplicate_detections(outcome: &ScenarioOutcome) -> InvariantResult {
    let keys = logged_detection_keys(outcome);
    let unique: BTreeSet<_> = keys.iter().collect();
    ensure(unique.len() == keys.len(), || format!("{} duplicate detection entries in the host log", keys.len() - unique.len()))
}

/// Every detection made by a device still alive at the end is in the host
/// log (it may have travelled live, been resent, or queued through solo).
pub fn no_silent_detection_loss(outcome: &ScenarioOutcome) -> InvariantResult {
    let logged: BTreeSet<DetectionKey> = logged_detection_keys(outcome).into_iter().collect();
    let missing: Vec<_> = outcome
        .metrics
        .generated_detections
        .iter()
        .filter(|(key, device, _)| outcome.alive(*device) && !logged.contains(key))
        .collect();
    let dropped: u64 = outcome.devices.iter().map(|device| device.node.outbox.dropped_oldest_count()).sum();
    ensure(missing.is_empty() && dropped == 0, || format!("{} detections never reached the host log (outbox drops {dropped}): {:?}", missing.len(), missing.iter().take(4).collect::<Vec<_>>()))
}

/// Mirrors are exact prefixes of the host log, behind it by no more than
/// what the host appended in the last `allowed_lag_s`.
pub fn event_logs_converge(outcome: &ScenarioOutcome, allowed_lag_s: f64) -> InvariantResult {
    let Some((host_device, host)) = outcome.active_host() else { return Err("no live host at the end".into()) };
    let host_entries = host.event_log.entries();
    let required_head = outcome.metrics.host_log_head_at(outcome.end_s - allowed_lag_s);
    for device in &outcome.devices {
        if !outcome.alive(device.index) || device.index == host_device || device.node.link.solo || !device.node.link.link_up {
            continue;
        }
        let mirror = device.node.mirror.entries();
        ensure(mirror.len() <= host_entries.len() && mirror == &host_entries[..mirror.len()], || {
            format!("device {} mirror diverges from the host log ({} vs {} entries)", device.index, mirror.len(), host_entries.len())
        })?;
        ensure(device.node.mirror.head_sequence() >= required_head, || {
            format!("device {} mirror head {} behind required {}", device.index, device.node.mirror.head_sequence(), required_head)
        })?;
    }
    Ok(())
}

pub fn clip_traffic_bounded(outcome: &ScenarioOutcome) -> InvariantResult {
    let cap = outcome.spec.mesh.clip_fetch.readiness_clip_cap;
    let records = outcome.call_records();
    for record in &records {
        ensure(record.requested_nodes.len() <= cap, || format!("call {} requested {} clips over cap {cap}", record.call_id, record.requested_nodes.len()))?;
    }
    let requested: usize = records.iter().map(|record| record.requested_nodes.len()).sum();
    let in_flight_allowance = outcome.all_hosts().iter().map(|(_, h)| h.calls.pending_call_count() * cap).sum::<usize>();
    let sent = outcome.network_stats().clip_payloads_delivered_or_lost as usize;
    ensure(sent <= requested + in_flight_allowance, || format!("{sent} clip payloads sent for {requested} requested"))
}

pub fn no_cross_island_tdoa(outcome: &ScenarioOutcome) -> InvariantResult {
    for record in outcome.call_records() {
        let domains: BTreeSet<(u32, u64)> = record.clip_domains.iter().map(|d| (d.1, d.2)).collect();
        ensure(domains.len() <= 1, || format!("call {} mixed clock domains {domains:?}", record.call_id))?;
    }
    Ok(())
}

pub fn no_stale_clip_used(outcome: &ScenarioOutcome) -> InvariantResult {
    for record in outcome.call_records() {
        for (node, _, _, at_request, at_use) in &record.clip_domains {
            ensure(at_request == at_use, || format!("call {} used node {node:?} clip across clock epochs {at_request}->{at_use}", record.call_id))?;
        }
    }
    Ok(())
}

/// Below this reported chirality confidence a mirrored map is "flagged":
/// the system said it might be the mirror image (test strategy: "or reports
/// low chirality confidence"). Stricter than the UI's 0.9 so the gate also
/// checks the confidence is roughly calibrated.
pub const FLAGGED_CHIRALITY_CONFIDENCE: f64 = 0.99;

fn mirror_was_flagged(fix: &ScoredFix) -> bool {
    fix.layout_mirrored == Some(true) && !fix.chirality_locked && fix.chirality_confidence.is_none_or(|c| c < FLAGGED_CHIRALITY_CONFIDENCE)
}

/// Never a confident wrong side: a mirrored fix must come with low reported
/// chirality confidence and never after chirality was locked by a cue.
pub fn no_confident_mirror(fixes: &[ScoredFix]) -> InvariantResult {
    for fix in fixes {
        ensure(fix.layout_mirrored != Some(true) || mirror_was_flagged(fix), || {
            format!("call {} mirrored with confidence {:?} (locked {})", fix.call_index, fix.chirality_confidence, fix.chirality_locked)
        })?;
    }
    Ok(())
}

/// "No fix worse than its reported region": error within k·σ plus a floor.
/// A flagged mirror is exempt (its region is drawn on both sides).
pub fn fixes_within_reported_region(fixes: &[ScoredFix], sigma_multiplier: f64, floor_m: f64) -> InvariantResult {
    no_confident_mirror(fixes)?;
    for fix in fixes.iter().filter(|fix| !mirror_was_flagged(fix)) {
        let allowed = sigma_multiplier * fix.reported_major_sigma_m + floor_m;
        ensure(fix.horizontal_error_m <= allowed, || {
            format!("call {} error {:.2} m outside reported region {:.2} m (σ {:.2})", fix.call_index, fix.horizontal_error_m, allowed, fix.reported_major_sigma_m)
        })?;
    }
    Ok(())
}

pub fn fixes_within_gate(fixes: &[ScoredFix], gate_m: f64) -> InvariantResult {
    for fix in fixes.iter().filter(|fix| !mirror_was_flagged(fix)) {
        ensure(fix.horizontal_error_m <= gate_m, || format!("call {} error {:.2} m over gate {gate_m} m", fix.call_index, fix.horizontal_error_m))?;
    }
    Ok(())
}

/// The invariants every scenario runs.
pub fn common_invariants(outcome: &ScenarioOutcome) -> InvariantResult {
    no_duplicate_detections(outcome)?;
    clip_traffic_bounded(outcome)?;
    no_cross_island_tdoa(outcome)?;
    no_stale_clip_used(outcome)
}
