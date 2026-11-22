//! Network scenarios: faults (loss, a late clip, a mid-call disconnect),
//! offline host loss (solo + idempotent queued upload), transport fallback
//! (relay drops, message cap, host cell loss) and a constrained cell link.
//! Arrival-time abstraction throughout.

use minimappr_core::mesh::host_transport_policy::{CellLinkQuality, SessionTransportPreference};
use minimappr_core::mesh::node_state_machine::NodeUserNotice;
use minimappr_core::protocol::wire_codes::{SessionChangeReasonCode, SessionTransportCode};
use minimappr_core::protocol::FieldMessage;

use crate::device_spec::Trajectory;
use crate::invariants::{ensure, event_logs_converge, fixes_within_reported_region, no_silent_detection_loss};
use crate::scenario::{ring_of_phones, ScenarioSpec, ScriptAction};
use crate::scenario_outcome::ScenarioOutcome;
use crate::simulated_network::ActiveTransport;

use super::call_scenarios::{healthy_path_checks, CONVERGENCE_LAG_S, REPORTED_REGION_FLOOR_M, REPORTED_REGION_SIGMA_MULTIPLIER};
use super::{report, run_checked, ScenarioReport};

const PHONE_HEIGHT_M: f64 = 1.4;
const OFFLINE_WANDERERS: [usize; 2] = [3, 4];

fn transport_notice_reasons(outcome: &ScenarioOutcome) -> Vec<(SessionTransportCode, SessionChangeReasonCode)> {
    outcome
        .host_log_records()
        .into_iter()
        .filter_map(|record| match record {
            FieldMessage::TransportNotice(notice) => Some((notice.transport, notice.reason)),
            _ => None,
        })
        .collect()
}

fn notice_seen(outcome: &ScenarioOutcome, device: usize, notice: NodeUserNotice) -> Result<f64, String> {
    outcome.metrics.first_notice(device, notice).ok_or_else(|| format!("device {device} never showed {notice:?}"))
}

/// 10 % loss, one phone's clips 800 ms late, a joiner disconnecting mid
/// call: every call is localized inside its region or cleanly abandoned,
/// never solved with a stale clip, and the logs converge once loss stops.
pub fn network_faults_spec(seed: u64) -> ScenarioSpec {
    const LATE_PHONE: usize = 3;
    const DISCONNECTING_PHONE: usize = 4;
    const LOSS_PROBABILITY: f64 = 0.1;
    ScenarioSpec::new("network_faults", 110.0, ring_of_phones(5, [0.0, 0.0], 12.0, seed))
        .at(5.0, ScriptAction::SetLossProbability(Some(LOSS_PROBABILITY)))
        .at(5.0, ScriptAction::ExtraBulkDelay { device: LATE_PHONE, extra_s: 0.8 })
        .call(20.0, [2.0, 3.0, 6.0])
        .call(30.0, [-4.0, 1.0, 7.0])
        .call(40.0, [3.0, -4.0, 5.0])
        .at(42.4, ScriptAction::Disconnect { device: DISCONNECTING_PHONE, until_s: 56.0 })
        .call(50.0, [-2.0, -5.0, 8.0])
        .call(60.0, [5.0, 5.0, 6.0])
        .at(75.0, ScriptAction::SetLossProbability(None))
}

pub fn network_faults(seed: u64) -> Result<ScenarioReport, String> {
    let spec = network_faults_spec(seed);
    report("network_faults", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let records = outcome.call_records();
        let localized = records.iter().filter(|r| r.fix.is_some()).count();
        ensure(localized >= 2, || format!("only {localized} of {} calls localized under faults", records.len()))?;
        fixes_within_reported_region(&outcome.scored_fixes(), REPORTED_REGION_SIGMA_MULTIPLIER, REPORTED_REGION_FLOOR_M)?;
        no_silent_detection_loss(&outcome)?;
        event_logs_converge(&outcome, CONVERGENCE_LAG_S)?;
        let lost = outcome.network_stats().messages_lost;
        Ok(format!("{localized}/{} localized, {} abandoned cleanly, {lost} messages lost", records.len(), records.len() - localized))
    })())
}

/// Two phones walk out of an offline host's hotspot for ~60 s, detect birds,
/// and return: solo after the timeout, queued detections uploaded once
/// (idempotent under resend), chirp sync resumes.
pub fn offline_host_loss_spec(seed: u64) -> ScenarioSpec {
    let mut devices = ring_of_phones(5, [0.0, 0.0], 8.0, seed);
    for (offset, device) in OFFLINE_WANDERERS.iter().enumerate() {
        let home = devices[*device].trajectory.position_at(0.0);
        let away = [70.0, 4.0 * offset as f64, PHONE_HEIGHT_M];
        devices[*device].trajectory = Trajectory::stationary(home).then_move(30.0, 50.0, away).then_move(110.0, 130.0, home);
    }
    let mut spec = ScenarioSpec::new("offline_host_loss", 200.0, devices)
        .call(20.0, [2.0, 2.0, 6.0])
        .call(80.0, [74.0, 8.0, 5.0])
        .call(95.0, [66.0, -4.0, 6.0])
        .at(128.0, ScriptAction::SetLossProbability(Some(0.3)))
        .at(150.0, ScriptAction::SetLossProbability(None))
        .call(170.0, [-2.0, 3.0, 7.0]);
    spec.transport_preference = SessionTransportPreference::LocalOnly;
    spec
}

pub fn offline_host_loss(seed: u64) -> Result<ScenarioReport, String> {
    let spec = offline_host_loss_spec(seed);
    report("offline_host_loss", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        for device in OFFLINE_WANDERERS {
            let solo = notice_seen(&outcome, device, NodeUserNotice::EnteredSolo)?;
            let rejoined = notice_seen(&outcome, device, NodeUserNotice::Rejoined)?;
            ensure(solo < rejoined, || format!("device {device} rejoined at {rejoined:.0} before solo {solo:.0}"))?;
            let away_detections = outcome
                .metrics
                .generated_detections
                .iter()
                .filter(|(_, d, t)| *d == device && *t > solo && *t < rejoined)
                .count();
            ensure(away_detections > 0, || format!("device {device} detected nothing while solo"))?;
        }
        let (_, host) = outcome.active_host().ok_or("no host")?;
        for device in OFFLINE_WANDERERS {
            ensure(host.sync.engine.clock_mapping(minimappr_core::protocol::NodeId(device as u32)).is_some(), || format!("device {device} not resynced after rejoining"))?;
        }
        let worst = healthy_path_checks(&outcome, 2, None)?;
        let uploads: u64 = OFFLINE_WANDERERS.iter().map(|d| outcome.devices[*d].node.counters.queued_uploads_sent).sum();
        Ok(format!("solo + rejoin for both, {uploads} queued uploads, no duplicates, worst {worst:.2} m"))
    })())
}

pub fn relay_drop_and_cap_spec(seed: u64) -> ScenarioSpec {
    ScenarioSpec::new("transport_fallback_relay", 140.0, ring_of_phones(5, [0.0, 0.0], 10.0, seed))
        .at(30.0, ScriptAction::DropFromRelay { device: 3, until_s: 55.0 })
        .at(30.0, ScriptAction::DropFromRelay { device: 4, until_s: 55.0 })
        .call(38.0, [2.0, 1.0, 6.0])
        .call(46.0, [-3.0, 2.0, 5.0])
        .at(80.0, ScriptAction::ReachRelayMessageCap)
        .call(100.0, [1.0, -3.0, 7.0])
}

pub fn host_cell_loss_spec(seed: u64) -> ScenarioSpec {
    const FAR_PHONE: usize = 4;
    let mut devices = ring_of_phones(5, [0.0, 0.0], 9.0, seed);
    let near = devices[FAR_PHONE].trajectory.position_at(0.0);
    devices[FAR_PHONE].trajectory = Trajectory::stationary([50.0, 6.0, PHONE_HEIGHT_M]).then_move(85.0, 100.0, near);
    ScenarioSpec::new("transport_fallback_cell", 150.0, devices)
        .at(40.0, ScriptAction::SetCellQuality { device: 0, quality: CellLinkQuality::NoSignal })
        .call(60.0, [2.0, 2.0, 6.0])
        .call(70.0, [48.0, 8.0, 5.0])
        .call(120.0, [-2.0, 1.0, 6.0])
}

/// Relay drops 2 of 5 phones for 25 s, then the session hits its relay
/// message cap; separately the host loses cell. Dropped phones catch up
/// from the log with no duplicates; each switch is announced and logged;
/// a phone out of hotspot range runs solo and catches up when back.
pub fn transport_fallback(seed: u64) -> Result<ScenarioReport, String> {
    report("transport_fallback", seed, (|| {
        let relay = run_checked(relay_drop_and_cap_spec(seed), seed)?;
        for device in [3, 4] {
            notice_seen(&relay, device, NodeUserNotice::EnteredSolo)?;
            ensure(relay.devices[device].node.counters.catch_up_requests_sent > 0, || format!("device {device} never asked for catch-up"))?;
        }
        let reasons = transport_notice_reasons(&relay);
        ensure(reasons.contains(&(SessionTransportCode::LocalHotspot, SessionChangeReasonCode::RelayMessageCapReached)), || format!("cap switch not logged: {reasons:?}"))?;
        ensure(matches!(relay.network.transport, ActiveTransport::LocalHotspot { .. }), || "session not on the local hotspot after the cap".into())?;
        healthy_path_checks(&relay, 2, None)?;

        let cell = run_checked(host_cell_loss_spec(seed), seed)?;
        let reasons = transport_notice_reasons(&cell);
        ensure(reasons.contains(&(SessionTransportCode::LocalHotspot, SessionChangeReasonCode::HostCellLinkLost)), || format!("cell-loss switch not logged: {reasons:?}"))?;
        let solo = notice_seen(&cell, 4, NodeUserNotice::EnteredSolo)?;
        let rejoined = notice_seen(&cell, 4, NodeUserNotice::Rejoined)?;
        ensure(solo < rejoined, || "far phone rejoined before going solo".into())?;
        let worst = healthy_path_checks(&cell, 2, None)?;
        Ok(format!("relay drop + cap + cell loss handled, far phone solo {solo:.0}-{rejoined:.0} s, worst {worst:.2} m"))
    })())
}

/// The host's cell link degrades to a constrained (satellite-like) link:
/// Auto treats it as no signal, starts the hotspot, and no clip ever
/// crosses the constrained link.
pub fn constrained_link(seed: u64) -> Result<ScenarioReport, String> {
    let spec = ScenarioSpec::new("constrained_link", 90.0, ring_of_phones(5, [0.0, 0.0], 10.0, seed))
        .call(20.0, [2.0, 2.0, 6.0])
        .at(40.0, ScriptAction::SetCellQuality { device: 0, quality: CellLinkQuality::Constrained })
        .call(55.0, [-3.0, 1.0, 5.0])
        .call(70.0, [1.0, -4.0, 7.0]);
    report("constrained_link", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let constrained_bytes = outcome.network_stats().clip_bytes_over_constrained_links;
        ensure(constrained_bytes == 0, || format!("{constrained_bytes} clip bytes crossed the constrained link"))?;
        let reasons = transport_notice_reasons(&outcome);
        ensure(reasons.contains(&(SessionTransportCode::LocalHotspot, SessionChangeReasonCode::HostCellLinkConstrained)), || format!("constrained switch not logged: {reasons:?}"))?;
        let fixes = outcome.scored_fixes();
        ensure(fixes.iter().any(|f| f.call_index == 1) && fixes.iter().any(|f| f.call_index == 2), || "calls after the switch not localized".into())?;
        let worst = healthy_path_checks(&outcome, 3, None)?;
        Ok(format!("hotspot after constrained link, 0 clip bytes over it, worst {worst:.2} m"))
    })())
}
