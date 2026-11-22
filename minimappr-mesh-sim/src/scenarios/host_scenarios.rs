//! Host lifecycle scenarios: host death (mid chirp round, then mid clip
//! fetch) and thermal shedding (online handover vs hotspot shedding).
//! Arrival-time abstraction throughout.

use minimappr_core::mesh::config::ChirpScheduleMode;
use minimappr_core::mesh::host_transport_policy::SessionTransportPreference;
use minimappr_core::mesh::node_state_machine::NodeUserNotice;
use minimappr_core::protocol::NodeId;

use crate::device_spec::ThermalRamp;
use crate::invariants::ensure;
use crate::scenario::{ring_of_phones, ScenarioSpec, ScriptAction};
use crate::scenario_outcome::ScenarioOutcome;

use super::call_scenarios::healthy_path_checks;
use super::{report, run_checked, ScenarioReport};

/// A successor must start its first round within this of taking over, and
/// finish it within one round's worth of slots and report wait.
const TAKEOVER_FIRST_ROUND_START_BUDGET_S: f64 = 1.0;
const TAKEOVER_FIRST_ROUND_FINISH_BUDGET_S: f64 = 6.0;
const MILLISECONDS_PER_SECOND: f64 = 1_000.0;

fn successor_resurveyed_quickly(outcome: &ScenarioOutcome) -> Result<(), String> {
    for tenure in outcome.metrics.host_tenures.iter().skip(1) {
        let (_, host) = outcome
            .all_hosts()
            .into_iter()
            .find(|(device, host)| *device == tenure.device && host.host_term == tenure.host_term)
            .ok_or_else(|| format!("tenure {tenure:?} has no host"))?;
        let first_start = host.sync.rounds_started.first().map(|r| r.started_ms as f64 / MILLISECONDS_PER_SECOND);
        let first_finish = host.sync.round_completion_ms.first().map(|ms| *ms as f64 / MILLISECONDS_PER_SECOND);
        ensure(first_start.is_some_and(|t| t - tenure.started_s <= TAKEOVER_FIRST_ROUND_START_BUDGET_S), || {
            format!("host {} term {} first round started {first_start:?} after takeover at {:.1}", tenure.device, tenure.host_term, tenure.started_s)
        })?;
        ensure(first_finish.is_some_and(|t| t - tenure.started_s <= TAKEOVER_FIRST_ROUND_FINISH_BUDGET_S), || {
            format!("host {} first round finished {first_finish:?} after takeover at {:.1}", tenure.device, tenure.started_s)
        })?;
    }
    Ok(())
}

/// Kill the host mid chirp round, then the successor mid clip fetch. The
/// next eligible node takes over from its mirror each time, re-surveys
/// within one round, and later calls localize again.
pub fn host_death_spec(seed: u64) -> ScenarioSpec {
    ScenarioSpec::new("host_death", 90.0, ring_of_phones(5, [0.0, 0.0], 12.0, seed))
        .call(10.0, [2.0, 3.0, 6.0])
        .at(11.0, ScriptAction::KillHostDuringRound)
        .call(32.0, [-4.0, 2.0, 7.0])
        .at(40.0, ScriptAction::KillHostDuringClipFetch)
        .call(42.0, [3.0, -5.0, 5.0])
        .call(72.0, [5.0, 4.0, 8.0])
}

pub fn host_death(seed: u64) -> Result<ScenarioReport, String> {
    let spec = host_death_spec(seed);
    report("host_death", seed, (|| {
        let outcome = run_checked(spec, seed)?;
        let tenures: Vec<(usize, u32)> = outcome.metrics.host_tenures.iter().map(|t| (t.device, t.host_term)).collect();
        // Succession follows join order (arrival order of join requests), not device index.
        let (_, final_host) = outcome.active_host().ok_or("no live host")?;
        let order: Vec<usize> = final_host.membership.succession_order().iter().map(|n| n.0 as usize).collect();
        let expected = vec![(order[0], 1), (order[1], 2), (order[2], 3)];
        ensure(tenures == expected, || format!("host succession {tenures:?}, join order {order:?}"))?;
        let live_hosts = outcome.devices.iter().filter(|d| d.host.is_some() && outcome.alive(d.index)).count();
        ensure(live_hosts == 1, || format!("{live_hosts} live hosts at the end"))?;
        successor_resurveyed_quickly(&outcome)?;
        let fixes = outcome.scored_fixes();
        for call in [1usize, 3] {
            ensure(fixes.iter().any(|f| f.call_index == call), || format!("call {call} after takeover not localized"))?;
        }
        let worst = healthy_path_checks(&outcome, 2, None)?;
        Ok(format!("succession {tenures:?}, post-takeover fixes ok, worst {worst:.2} m"))
    })())
}

pub fn thermal_spec(seed: u64, preference: SessionTransportPreference) -> ScenarioSpec {
    const SESSION_S: f64 = 600.0;
    const HOT_PHONE: usize = 2;
    const HOST: usize = 0;
    let mut devices = ring_of_phones(5, [0.0, 0.0], 10.0, seed);
    devices[HOT_PHONE].thermal_ramp = Some(ThermalRamp { start_s: 0.0, end_s: 420.0, recovered_s: 520.0 });
    devices[HOST].thermal_ramp = Some(ThermalRamp { start_s: 0.0, end_s: 520.0, recovered_s: 620.0 });
    let mut spec = ScenarioSpec::new("thermal_shedding", SESSION_S, devices);
    spec.tick_interval_ms = 250;
    spec.transport_preference = preference;
    spec.mesh.scheduler.mode = ChirpScheduleMode::Periodic;
    spec.mesh.scheduler.periodic_round_interval_ms = 30_000;
    let mut call_time = 30.0;
    let mut index = 0.0;
    while call_time < SESSION_S - 20.0 {
        spec = spec.call(call_time, [4.0 * (index * 1.3f64).cos(), 4.0 * (index * 1.3f64).sin(), 6.0]);
        call_time += 37.0;
        index += 1.0;
    }
    spec
}

/// Over a compressed 10-minute session (30 min in the plan) one phone and
/// the host heat up. The phone goes Mic-only before its first stall; online
/// the host hands over before throttling; on the hotspot it sheds
/// classification instead. Stalls (reported or silent) never yield a fix
/// outside its region, and the hot phone is resynced afterwards.
pub fn thermal_shedding(seed: u64) -> Result<ScenarioReport, String> {
    const HOT_PHONE: usize = 2;
    const HOST: usize = 0;
    report("thermal_shedding", seed, (|| {
        let online = run_checked(thermal_spec(seed, SessionTransportPreference::Auto), seed)?;
        let first_stall = |outcome: &ScenarioOutcome, device: usize| {
            outcome.world.devices[device].stalls.first().map(|stall| stall.time_s)
        };
        let mic_only_at = online.metrics.first_notice(HOT_PHONE, NodeUserNotice::MicOnlyForThermal).ok_or("hot phone never went Mic-only")?;
        let phone_stall = first_stall(&online, HOT_PHONE).ok_or("hot phone never stalled")?;
        ensure(mic_only_at < phone_stall, || format!("Mic-only at {mic_only_at:.0} s, first stall {phone_stall:.0} s"))?;
        let handover = online.metrics.host_tenures.get(1).ok_or("online host never handed over")?;
        let host_stall = first_stall(&online, HOST).unwrap_or(f64::INFINITY);
        ensure(handover.started_s < host_stall, || format!("handover at {:.0} s after host stall {host_stall:.0} s", handover.started_s))?;
        ensure(handover.device != HOT_PHONE, || "handed over to the hot phone".into())?;
        let (_, host) = online.active_host().ok_or("no host")?;
        ensure(host.sync.engine.clock_mapping(NodeId(HOT_PHONE as u32)).is_some(), || "hot phone not resynced after its stalls".into())?;
        let worst_online = healthy_path_checks(&online, 1, None)?;

        let hotspot = run_checked(thermal_spec(seed, SessionTransportPreference::LocalOnly), seed)?;
        ensure(hotspot.metrics.host_tenures.len() == 1, || "hotspot host handed over instead of shedding".into())?;
        let (_, hotspot_host) = hotspot.active_host().ok_or("no hotspot host")?;
        ensure(hotspot_host.shed_classification && hotspot.devices[HOST].node.mic_only, || "hot hotspot host did not shed classification".into())?;
        let worst_hotspot = healthy_path_checks(&hotspot, 1, None)?;
        Ok(format!(
            "Mic-only {mic_only_at:.0} s < stall {phone_stall:.0} s; handover {:.0} s → {}; hotspot shed; worst {:.2} m",
            handover.started_s,
            handover.device,
            worst_online.max(worst_hotspot)
        ))
    })())
}
